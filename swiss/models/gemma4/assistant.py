"""The Gemma 4 assistant: a drafter whose layers attend to the target's KV pages."""

import dataclasses

import numpy as np
from carbon.kernels.ops import sampling

from ...core.model import Unsupported
from ...core.speculative import Attached, Chain
from ...core.weights import Uploads
from ...kernels import linear, normalization, rope
from .model import Layers


class Run:
    """The target's page table under the drafter's own lengths: one row at the last
    stored position."""

    def __init__(self, target, device):
        self.table, self.lengths = target.table, device.buffer(16)

    def max_pages(self, buffers):
        return 0

    def last(self, buffers, rows):
        return rows


class Gemma4Assistant(Attached, Layers, Uploads):
    """Drafts from [target embedding of the next token, target hidden state]; each step
    feeds its projected hidden state back, at the same position.

    Its sliding layers see `sliding_window` + 1 keys, as the reference mask
    |query - key| <= sliding_window does.
    """

    def __init__(self, spec, weights, target, bits=None):
        if spec.first_shared or spec.per_layer:
            raise Unsupported("the assistant must share the KV of every layer")
        spec = dataclasses.replace(spec, window=spec.window + 1)
        self.spec, self.target, self.device = spec, target, target.device
        self.weights, self.quantization, self.prefix = weights, None, "model."
        self.requantize = (bits, 64) if bits else None
        self.blocks = [self.block(layer) for layer in range(spec.layers)]
        self.final_norm = self.vector("norm.weight")
        self.head = self.matrix("embed_tokens")
        width = target.spec.width
        pre = weights["pre_projection.weight"]
        if pre.shape[1] != 2 * width:
            raise Unsupported("the assistant was made for another target width")
        weights["embedded.weight"] = np.ascontiguousarray(pre[:, :width])
        weights["context.weight"] = np.ascontiguousarray(pre[:, width:])
        self.pre_embedded = self.matrix("embedded", prefix="")
        self.pre_hidden = self.matrix("context", prefix="")
        self.post = self.matrix("post_projection", prefix="")
        del self.weights
        self.run = Run(target.run, self.device)
        self.sources = {
            kind: max(i for i, k in enumerate(target.spec.kinds) if k == kind)
            for kind in set(spec.kinds)
        }
        self.pages = {
            layer: target.pages[self.sources[kind]]
            for layer, kind in enumerate(spec.kinds)
        }
        self.ropes = {
            kind: [linear.raw(self.device, table[1:]) for table in tables]
            for kind, tables in self.rope_tables().items()
        }
        self.chain = Chain(self.device, self.sizes(), self.steps)

    @classmethod
    def attach(cls, spec, checkpoint, target, bits=None):
        """The assistant of `checkpoint`, drafting for the Gemma 4 model `target`;
        `bits` quantizes its matrices as they load."""
        drafter = cls(spec, dict(checkpoint.tensors()), target, bits)
        target.run.follow(drafter)
        return drafter

    @property
    def capacity(self):
        return self.target.capacity

    def rope_tables(self):
        """Tables one position ahead: row p turns a query at position p + 1."""
        s = self.spec
        return {
            "full_attention": rope.rope_tables(
                self.capacity + 1,
                s.global_head_dim,
                s.theta,
                s.angles,
            ),
            "sliding_attention": rope.rope_tables(
                self.capacity + 1,
                s.head_dim,
                s.sliding_theta,
            ),
        }

    def geometry(self, layer):
        return self.target.geometry(self.sources[self.spec.kinds[layer]])

    def sizes(self):
        s, head_dim = self.spec, max(self.spec.head_dim, self.spec.global_head_dim)
        return {
            "embedded": self.target.spec.width * 2,
            "context": self.target.spec.width * 2,
            "x": s.width * 2,
            "normed": s.width * 2,
            "qkv": max(w["qkv"].outputs for w in self.blocks) * 2,
            "q": s.heads * head_dim * 2,
            "attended": s.heads * head_dim * 2,
            "update": s.width * 2,
            "hidden": s.hidden * 2,
            "last": s.width * 2,
            "logits": s.vocab * 2,
            "settings": 6 * 4,
        }

    def steps(self, b):
        s, device, target = self.spec, self.device, self.target
        b["settings"].write(sampling.settings([0], [0.0]).tobytes())
        steps = [
            linear.embedding(
                device,
                b["ids"],
                target.embed,
                b["embedded"],
                1,
                scale=target.spec.width**0.5,
            ),
        ]
        steps += linear.product(device, b["embedded"], self.pre_embedded, b["x"], 1)
        steps += linear.product(device, b["context"], self.pre_hidden, b["update"], 1)
        steps += normalization.residual_norm(
            device,
            b["x"],
            b["update"],
            weight=self.blocks[0]["input_norm"],
            output=b["normed"],
            rows=1,
            columns=s.width,
            epsilon=s.eps,
        )
        for layer in range(s.layers):
            steps += self.layer_steps(layer, 1, b)
        steps += linear.product(device, b["last"], self.post, b["context"], 1)
        steps += linear.product(device, b["last"], self.head, b["logits"], 1)
        return steps + [
            sampling.sample(
                device,
                b["logits"],
                b["settings"],
                self.run.lengths,
                b["token"],
                rows=1,
                vocab=s.vocab,
            ),
        ]

    def propose(self, sequence, token, count):
        """`count` greedy drafts after `token`, the next input of `sequence`."""
        run = self.target.run
        buffers, row = run.outputs[sequence]
        size = self.target.spec.width * 2
        self.chain.buffers["context"].write(
            buffers["last"].read()[row * size : (row + 1) * size],
        )
        run.write_tables([sequence])
        position = np.array([run.length(sequence) - 1, 0, 0, 0], np.uint32)
        self.run.lengths.write(position.tobytes())
        return self.chain.run(token, count)

"""Tiny random-weight checkpoints made from any config in tests/model_config.json."""

import copy
import functools
import json
import re
from pathlib import Path

import numpy as np

CONFIGS = json.loads((Path(__file__).parent / "model_config.json").read_text())
VOCAB = 1024
SIZES = {
    "hidden_size": 256,
    "intermediate_size": 512,
    "hidden_size_per_layer_input": 64,
    "vocab_size": VOCAB,
    "vocab_size_per_layer_input": VOCAB,
    "sliding_window": 32,
}
FORMATS = {
    "bf16": None,
    "4bit": {"group_size": 64, "bits": 4},
    "8bit": {"group_size": 32, "bits": 8},
}
DRAFTERS = {"google/gemma-4-12B-it-assistant": "google/gemma-4-12B-it"}
TAPPED = {
    "z-lab/Qwen3.5-9B-DFlash": "Qwen/Qwen3.5-0.8B",
    "BLR2/Eagle3-Qwen3.5-9B": "Qwen/Qwen3.5-0.8B",
}
TAPS = (0, 1, 2)
DRAFT_VOCAB = 512
TOWERS = (
    r"(model\.)?(visual|vision_tower|vision_embedder|embed_vision|audio_tower|"
    r"embed_audio|multi_modal_projector)\."
)


def text(config):
    return config.get("text_config") or config


def generates(config):
    return any(
        name.endswith(("ForCausalLM", "ForConditionalGeneration"))
        for name in config.get("architectures", [])
    )


def shrink(config):
    """`config` with few layers, narrow widths and a small vocabulary; head dims, head
    ratios and layer kinds stay."""
    config = copy.deepcopy(config)
    section = text(config)
    section.update({key: value for key, value in SIZES.items() if section.get(key)})
    kinds = section.get("layer_types")
    layers = (
        next(n for n in range(2, len(kinds) + 1) if set(kinds[:n]) == set(kinds))
        if kinds
        else 2
    )
    shared = section.get("num_kv_shared_layers")
    if shared and shared < section["num_hidden_layers"]:
        section["num_kv_shared_layers"] = layers
        layers *= 2
    elif shared:
        section["num_kv_shared_layers"] = layers
    if config.get("backbone_hidden_size"):
        config["backbone_hidden_size"] = SIZES["hidden_size"]
    section["num_hidden_layers"] = layers
    for part in (config, section):
        if (part.get("pad_token_id") or 0) >= VOCAB:
            part["pad_token_id"] = 0
    if kinds:
        section["layer_types"] = kinds[:layers]
    heads = [
        key
        for key, value in section.items()
        if key.endswith("heads") and isinstance(value, int) and value
    ]
    factor = max(1, min(section[key] for key in heads) // 2) if heads else 1
    if all(section[key] % factor == 0 for key in heads):
        section.update({key: section[key] // factor for key in heads})
    for tower in ("vision_config", "audio_config"):
        for key in ("num_hidden_layers", "depth"):
            if key in (config.get(tower) or {}):
                config[tower][key] = 1
        if "layer_types" in (config.get(tower) or {}):
            config[tower]["layer_types"] = config[tower]["layer_types"][:1]
    return config


def cases():
    """(repo, format) for each generating checkpoint and each weight format."""
    return [
        (repo, name)
        for repo, config in CONFIGS.items()
        if generates(config) and repo not in DRAFTERS
        for name in FORMATS
    ]


def noisy(key, value, normal, factors):
    """Noise on constant tensors, so a skipped tensor changes the output; query and key
    norms times `factors`."""
    if value.size == 1 or float(value.max() - value.min()) == 0.0:
        value = value + normal(value.shape) * (0.1 * abs(value) + 0.05)
    return value * factors.get(key, 1.0)


def norm_factors(modules, attribute):
    """{q/k norm weight name: (scale · √head_dim)^-1/2}, so attention scores stay near
    unit size for any scale."""
    factors = {}
    for name, module in modules:
        scale = getattr(module, attribute, None)
        if isinstance(scale, (int, float)) and hasattr(module, "q_norm"):
            size = module.q_norm.weight.shape[-1]
            for part in ("q_norm", "k_norm"):
                factors[f"{name}.{part}.weight"] = (scale * size**0.5) ** -0.5
    return factors


def save(repo, form, folder, seed=7):
    """Write the tiny checkpoint of `repo` in weight format `form` into `folder`."""
    config = shrink(CONFIGS[repo])
    quantization = FORMATS[form]
    folder.mkdir(parents=True, exist_ok=True)
    save_bf16(config, folder, seed)
    if quantization is not None:
        quantize(folder, **quantization)
        config["quantization"] = {**quantization, "mode": "affine"}
    (folder / "config.json").write_text(json.dumps(config))
    return folder


def save_bf16(config, folder, seed):
    """Write a random bf16 transformers model of `config`."""
    import torch
    import transformers

    torch.manual_seed(seed)
    model = getattr(transformers, config["architectures"][0])._from_config(
        transformers.AutoConfig.for_model(**config),
        dtype=torch.float32,
    )
    factors = norm_factors(model.named_modules(), "scaling")
    with torch.no_grad():
        for key, value in model.state_dict().items():
            if value.is_floating_point():
                value.copy_(
                    noisy(key, value, lambda shape: torch.randn(shape), factors),
                )
    model.to(torch.bfloat16).save_pretrained(folder)


def quantize(folder, group_size, bits):
    """Replace each matrix whose rows split into groups with MLX affine `bits`-bit
    weights, scales and biases."""
    import torch
    from safetensors.torch import load_file, save_file

    levels, per_word = 2**bits - 1, 32 // bits
    shifts = torch.arange(per_word, dtype=torch.int64) * bits
    for path in folder.glob("*.safetensors"):
        tensors = load_file(path)
        for name in [n for n in tensors if n.endswith(".weight")]:
            weight = tensors[name].float()
            if weight.ndim != 2 or weight.shape[1] % group_size:
                continue
            groups = weight.view(weight.shape[0], -1, group_size)
            low, high = groups.amin(-1), groups.amax(-1)
            scales = ((high - low) / levels).clamp_min(1e-7).bfloat16()
            biases = low.bfloat16()
            values = (groups - biases.float()[..., None]) / scales.float()[..., None]
            values = values.round().clamp(0, levels).to(torch.int64)
            words = (values.view(weight.shape[0], -1, per_word) << shifts).sum(-1)
            tensors[name] = words.to(torch.uint32)
            tensors[name.removesuffix("weight") + "scales"] = scales
            tensors[name.removesuffix("weight") + "biases"] = biases
        save_file(tensors, path, metadata={"format": "pt"})


def dequantized(folder, bits):
    """A copy of the checkpoint in `folder` with f32 weights in place of quantized
    ones."""
    import torch
    from safetensors.torch import load_file, save_file

    out = folder / "dequantized"
    if out.exists():
        return out
    out.mkdir()
    shifts = torch.arange(32 // bits, dtype=torch.int64) * bits
    for path in folder.glob("*.safetensors"):
        tensors = {name: value for name, value in load_file(path).items()}
        for name in [n for n in tensors if n.endswith(".scales")]:
            base = name.removesuffix("scales")
            scales, biases = tensors.pop(name).float(), tensors.pop(base + "biases")
            words = tensors[base + "weight"].to(torch.int64)
            values = ((words[..., None] >> shifts) & (2**bits - 1)).float()
            values = values.view(scales.shape[0], scales.shape[1], -1)
            weight = values * scales[..., None] + biases.float()[..., None]
            tensors[base + "weight"] = weight.view(scales.shape[0], -1)
        save_file(
            {
                name: value.float() if value.is_floating_point() else value
                for name, value in tensors.items()
            },
            out / path.name,
            metadata={"format": "pt"},
        )
    config = json.loads((folder / "config.json").read_text())
    config.pop("quantization")
    (out / "config.json").write_text(json.dumps(config))
    return out


@functools.cache
def torch_model(folder):
    """The f32 transformers model of the checkpoint in `folder`."""
    import torch
    import transformers

    config = json.loads((folder / "config.json").read_text())
    if "quantization" in config:
        folder = dequantized(folder, config["quantization"]["bits"])
    kind = getattr(transformers, config["architectures"][0])
    return kind.from_pretrained(folder, dtype=torch.float32).eval()


def reference(folder, tokens):
    """(len(tokens), vocab) f32 logits of the transformers model of `folder`."""
    import torch

    with torch.no_grad():
        out = torch_model(folder)(input_ids=torch.tensor([tokens]))
    return out.logits[0].float().numpy()


def unused(name):
    return re.match(TOWERS, name) is not None


def relative(got, want):
    """RMS of got - want over the RMS of want."""
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    return float(np.sqrt(np.mean((got - want) ** 2) / np.mean(want**2)))


def shared_kv():
    """Make transformers' Gemma 4 targets store the KV of the last layer of each kind,
    and its assistants reuse it, as before huggingface/transformers#47290."""
    from transformers.models.gemma4_unified import configuration_gemma4_unified as gemma

    # https://github.com/huggingface/transformers/issues/50316
    def modes(config):
        kinds = config.layer_types
        first = config.num_hidden_layers - config.num_kv_shared_layers
        chosen = [None] * first + ["reuse"] * config.num_kv_shared_layers
        for kind in set(kinds[:first]):
            chosen[max(i for i in range(first) if kinds[i] == kind)] = "store"
        return chosen

    gemma.Gemma4UnifiedTextConfig.kv_sharing_modes = property(modes)


def save_drafter(repo, folder, seed=7):
    """Write the tiny bf16 drafter of `repo`, with the heads of its tiny target."""
    config = shrink(CONFIGS[repo])
    target = text(shrink(CONFIGS[DRAFTERS[repo]]))
    text(config)["num_attention_heads"] = target["num_attention_heads"]
    folder.mkdir(parents=True, exist_ok=True)
    shared_kv()
    save_bf16(config, folder, seed)
    (folder / "config.json").write_text(json.dumps(config))
    return folder


def drafter_reference(target, drafter, tokens, steps):
    """f32 logits of `steps` greedy drafts after `tokens`, as transformers drafts them
    from the target's last hidden state and shared KV."""
    import torch
    import transformers

    def model(folder):
        config = json.loads((folder / "config.json").read_text())
        kind = getattr(transformers, config["architectures"][0])
        return kind.from_pretrained(folder, dtype=torch.float32).eval()

    shared_kv()
    big, small = model(target), model(drafter)
    with torch.no_grad():
        out = big(
            input_ids=torch.tensor([tokens]),
            output_hidden_states=True,
            return_shared_kv_states=True,
        )
        hidden, token = out.hidden_states[-1][:, -1:], out.logits[:, -1:].argmax(-1)
        position, rows = torch.tensor([[len(tokens)]]), []
        for _ in range(steps):
            embedded = big.get_input_embeddings()(token)
            step = small(
                inputs_embeds=torch.cat([embedded, hidden], -1),
                position_ids=position,
                shared_kv_states=out.shared_kv_states,
            )
            hidden, token = step.last_hidden_state, step.logits.argmax(-1)
            rows.append(step.logits[0, -1].float().numpy())
    return int(out.logits[0, -1].argmax()), np.stack(rows)


def tapped_config(repo):
    """The tiny config of drafter `repo`: it reads layers TAPS of its tiny target."""
    config = shrink(CONFIGS[repo])
    target = text(shrink(CONFIGS[TAPPED[repo]]))
    if "dflash_config" in config:
        config["dflash_config"].update(
            target_layer_ids=list(TAPS),
            mask_token_id=VOCAB - 1,
            num_target_layers=target["num_hidden_layers"],
        )
        config["sliding_window"] = 4096
    else:
        config["eagle_config"]["eagle_aux_hidden_state_layer_ids"] = list(TAPS)
        config["draft_vocab_size"] = DRAFT_VOCAB
        config["num_hidden_layers"] = 1
        config["rope_parameters"]["partial_rotary_factor"] = 1.0
        config.pop("partial_rotary_factor", None)
    return config


def tapped_tensors(config, rng):
    """Random f32 tensors of a DFlash or EAGLE-3 checkpoint of `config`, by name."""
    width, hidden = config["hidden_size"], config["intermediate_size"]
    heads, kv_heads = config["num_attention_heads"], config["num_key_value_heads"]
    head_dim = config.get("head_dim") or width // heads

    def dense(outputs, inputs):
        return rng.standard_normal((outputs, inputs)) * inputs**-0.5

    def norm(size):
        return 1.0 + 0.1 * rng.standard_normal(size)

    def layer(prefix, inputs):
        return {
            prefix + "input_layernorm.weight": norm(width),
            prefix + "post_attention_layernorm.weight": norm(width),
            prefix + "self_attn.q_proj.weight": dense(heads * head_dim, inputs),
            prefix + "self_attn.k_proj.weight": dense(kv_heads * head_dim, inputs),
            prefix + "self_attn.v_proj.weight": dense(kv_heads * head_dim, inputs),
            prefix + "self_attn.o_proj.weight": dense(width, heads * head_dim),
            prefix + "mlp.gate_proj.weight": dense(hidden, width),
            prefix + "mlp.up_proj.weight": dense(hidden, width),
            prefix + "mlp.down_proj.weight": dense(width, hidden),
        }

    taps = len(TAPS) * width
    if "dflash_config" in config:
        tensors = {"fc.weight": dense(width, taps), "hidden_norm.weight": norm(width)}
        for index in range(config["num_hidden_layers"]):
            prefix = f"layers.{index}."
            tensors.update(layer(prefix, width))
            tensors[prefix + "self_attn.q_norm.weight"] = norm(head_dim)
            tensors[prefix + "self_attn.k_norm.weight"] = norm(head_dim)
    else:
        tensors = {
            "fc.weight": dense(width, taps),
            "lm_head.weight": dense(DRAFT_VOCAB, width),
            "layers.0.hidden_norm.weight": norm(width),
            **layer("layers.0.", 2 * width),
        }
        tensors["d2t"] = rng.integers(0, VOCAB - DRAFT_VOCAB, DRAFT_VOCAB)
    tensors["norm.weight"] = norm(width)
    return tensors


def save_tapped(repo, folder, seed=7):
    """Write the tiny bf16 drafter `repo` that reads layers TAPS of its tiny target."""
    import torch
    from safetensors.torch import save_file

    config = tapped_config(repo)
    tensors = tapped_tensors(config, np.random.default_rng(seed))
    folder.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            name: torch.tensor(value).to(
                torch.int64 if name == "d2t" else torch.bfloat16,
            )
            for name, value in tensors.items()
        },
        folder / "model.safetensors",
    )
    (folder / "config.json").write_text(json.dumps(config))
    return folder


def rms(x, weight, eps):
    import torch

    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def rotate(x, positions, theta):
    """`x` (heads, rows, head_dim) turned by RoPE at `positions`, halves paired."""
    import torch

    size = x.shape[-1]
    angles = positions[:, None].double() / theta ** (
        torch.arange(0, size, 2).double() / size
    )
    cos, sin = angles.cos().float().repeat(1, 2), angles.sin().float().repeat(1, 2)
    half = size // 2
    return x * cos + torch.cat([-x[..., half:], x[..., :half]], -1) * sin


def attend(q, k, v, mask):
    """Grouped attention of q (heads, rows, d) over k, v (kv_heads, keys, d)."""
    import torch

    groups = q.shape[0] // k.shape[0]
    out = torch.nn.functional.scaled_dot_product_attention(
        q,
        k.repeat_interleave(groups, 0),
        v.repeat_interleave(groups, 0),
        attn_mask=mask,
    )
    return out.transpose(0, 1).reshape(q.shape[1], -1)


def mlp(x, w, prefix):
    import torch

    gate = torch.nn.functional.silu(x @ w[prefix + "gate_proj.weight"].T)
    return (gate * (x @ w[prefix + "up_proj.weight"].T)) @ w[
        prefix + "down_proj.weight"
    ].T


def geometry(config):
    heads, kv_heads = config["num_attention_heads"], config["num_key_value_heads"]
    head_dim = config.get("head_dim") or config["hidden_size"] // heads
    rope = config.get("rope_parameters") or {}
    theta = float(rope.get("rope_theta", config.get("rope_theta", 10000.0)))
    return heads, kv_heads, head_dim, theta, config["rms_norm_eps"]


def dflash_reference(config, w, taps, embed, head, token, steps):
    """Logits of block rows 1 to `steps` after `token`, as z-lab/dflash `model.py`
    drafts them: sliding layers are causal unless the config sets `is_causal`."""
    import torch

    heads, kv_heads, head_dim, theta, eps = geometry(config)
    layers = config["num_hidden_layers"]
    kinds = config.get("layer_types") or ["full_attention"] * layers
    context = rms(taps @ w["fc.weight"].T, w["hidden_norm.weight"], eps)
    ids = [token] + [config["dflash_config"]["mask_token_id"]] * steps
    x, stored = embed[ids], len(context)
    positions = torch.arange(stored + len(ids))
    for layer in range(layers):
        p, a = f"layers.{layer}.", f"layers.{layer}.self_attn."
        h = rms(x, w[p + "input_layernorm.weight"], eps)
        both = torch.cat([context, h])
        q = rms(
            (h @ w[a + "q_proj.weight"].T).view(len(ids), heads, head_dim),
            w[a + "q_norm.weight"],
            eps,
        )
        k = rms(
            (both @ w[a + "k_proj.weight"].T).view(-1, kv_heads, head_dim),
            w[a + "k_norm.weight"],
            eps,
        )
        v = (both @ w[a + "v_proj.weight"].T).view(-1, kv_heads, head_dim)
        causal = config.get("is_causal")
        causal = kinds[layer] == "sliding_attention" if causal is None else causal
        mask = positions[None, :] <= positions[stored:, None] if causal else None
        attended = attend(
            rotate(q.transpose(0, 1), positions[stored:], theta),
            rotate(k.transpose(0, 1), positions, theta),
            v.transpose(0, 1),
            mask,
        )
        x = x + attended @ w[a + "o_proj.weight"].T
        x = x + mlp(
            rms(x, w[p + "post_attention_layernorm.weight"], eps),
            w,
            p + "mlp.",
        )
    return (rms(x, w["norm.weight"], eps) @ head.T)[1:]


def eagle3_reference(config, w, taps, embed, tokens, token, steps):
    """The hidden state of the first draft after `token`, and the draft-vocabulary
    logits of the next `steps` - 1, as SpecForge's `LlamaForCausalLMEagle3` drafts
    them: entry i holds token i + 1 with the taps of token i."""
    import torch

    heads, kv_heads, head_dim, theta, eps = geometry(config)
    p, a = "layers.0.", "layers.0.self_attn."
    keys, values = [], []

    def step(ids, hidden, positions):
        x = torch.cat(
            [
                rms(embed[ids], w[p + "input_layernorm.weight"], eps),
                rms(hidden, w[p + "hidden_norm.weight"], eps),
            ],
            -1,
        )
        q = (x @ w[a + "q_proj.weight"].T).view(len(ids), heads, head_dim)
        k = (x @ w[a + "k_proj.weight"].T).view(len(ids), kv_heads, head_dim)
        v = (x @ w[a + "v_proj.weight"].T).view(len(ids), kv_heads, head_dim)
        keys.append(rotate(k.transpose(0, 1), positions, theta))
        values.append(v.transpose(0, 1))
        k, v = torch.cat(keys, 1), torch.cat(values, 1)
        mask = torch.arange(k.shape[1])[None, :] <= positions[:, None]
        out = hidden + attend(
            rotate(q.transpose(0, 1), positions, theta),
            k,
            v,
            mask,
        ) @ (w[a + "o_proj.weight"].T)
        return out + mlp(
            rms(out, w[p + "post_attention_layernorm.weight"], eps),
            w,
            p + "mlp.",
        )

    def draft(hidden):
        logits = rms(hidden, w["norm.weight"], eps) @ w["lm_head.weight"].T
        index = int(logits.argmax())
        return logits, index + int(w["d2t"][index])

    ids = [*tokens[1:], token]
    hidden = step(ids, taps @ w["fc.weight"].T, torch.arange(len(ids)))[-1:]
    first = hidden[0].numpy()
    _, drafted = draft(hidden[0])
    rows = []
    for index in range(steps - 1):
        hidden = step([drafted], hidden, torch.tensor([len(ids) + index]))
        logits, drafted = draft(hidden[0])
        rows.append(logits.numpy())
    return first, np.stack(rows) if rows else None


def tapped_reference(target, drafter, tokens, steps):
    """The next token after `tokens`, and f32 drafts after it from the target's layers
    TAPS: DFlash block logits of rows 1 to `steps`, or the EAGLE-3 hidden state of its
    first draft and the logits of its next `steps` - 1."""
    import torch
    from safetensors.torch import load_file

    big = torch_model(target)
    config = json.loads((drafter / "config.json").read_text())
    w = {
        name: value if name == "d2t" else value.float()
        for name, value in load_file(drafter / "model.safetensors").items()
    }
    with torch.no_grad():
        out = big(input_ids=torch.tensor([tokens]), output_hidden_states=True)
        token = int(out.logits[0, -1].argmax())
        taps = torch.cat([out.hidden_states[layer + 1][0] for layer in TAPS], -1)
        embed = big.get_input_embeddings().weight
        if "dflash_config" in config:
            head = big.get_output_embeddings().weight
            rows = dflash_reference(config, w, taps, embed, head, token, steps)
            return token, None, rows.numpy()
        return token, *eagle3_reference(config, w, taps, embed, tokens, token, steps)

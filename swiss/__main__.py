import argparse
import sys
from time import perf_counter

from .core import Embedder, Engine, RuntimeConfig, architectures, attach, load
from .server import Server


def serve(args):
    start = perf_counter()
    runtime = RuntimeConfig(
        capacity=args.capacity,
        max_sequences=args.max_sequences,
        draft=args.draft,
    )
    model, checkpoint = load(args.checkpoint, runtime)
    if args.drafter:
        attach(model, args.drafter, args.drafter_bits or None)
    name = args.name or checkpoint.name
    engine = (
        Embedder(model, checkpoint.tokenizer, name=name)
        if hasattr(model, "embed_tokens")
        else Engine(
            model.run,
            checkpoint.tokenizer,
            eos_token_ids=checkpoint.eos_token_ids(),
            name=name,
        )
    )
    if not args.no_warm:
        engine.warm()
    print(
        f"Loaded {engine.name} in {perf_counter() - start:.1f}s; {model.sequences} "
        "sequences of "
        f"{engine.capacity} tokens",
        file=sys.stderr,
    )
    server = Server((args.host, args.port), engine, verbose=args.verbose)
    print(
        f"Serving on http://{args.host}:{server.server_address[1]}/v1",
        file=sys.stderr,
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def models(args):
    for model_type, architecture in sorted(architectures().items()):
        print(
            f"{model_type:16} "
            f"{architecture.model.__module__}.{architecture.model.__name__}",
        )


def main():
    parser = argparse.ArgumentParser(prog="swiss", description="Swiss inference engine")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "serve",
        help="Serve a local checkpoint over OpenAI-compatible HTTP",
    )
    run.add_argument(
        "checkpoint",
        help="Directory with config.json, tokenizer and safetensors weights",
    )
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=8080)
    run.add_argument(
        "--name",
        help="Model ID reported to clients (default: directory name)",
    )
    run.add_argument(
        "--capacity",
        type=int,
        default=4096,
        help="Context tokens per sequence",
    )
    run.add_argument(
        "--max-sequences",
        type=int,
        default=8,
        help="Requests decoded together in one batch",
    )
    run.add_argument(
        "--draft",
        type=int,
        default=0,
        help=(
            "Most drafted tokens per verify step, used when a request decodes alone; "
            "prompt lookup drafts without --drafter (8 to 15 suits copying)"
        ),
    )
    run.add_argument(
        "--drafter",
        help=(
            "A drafter checkpoint for the model: a Gemma 4 assistant, a DFlash or an "
            "EAGLE-3 drafter"
        ),
    )
    run.add_argument(
        "--drafter-bits",
        type=int,
        choices=(0, 4, 8),
        default=4,
        help="Quantize the drafter's bf16 matrices as they load; 0 keeps them",
    )
    run.add_argument(
        "--no-warm",
        action="store_true",
        help="Compile prefill programs on first use",
    )
    run.add_argument("--verbose", action="store_true")
    run.set_defaults(handler=serve)
    listing = commands.add_parser("models", help="List supported model types")
    listing.set_defaults(handler=models)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

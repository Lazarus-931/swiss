# swiss
jack of all trades -- and a master of all as well :)

Swiss is an _experimental_ inference engine running on [carbon](https://github.com/Lazarus-931/carbon), a metal DSL for apple silicon.

## Requirements

- An Apple silicon Mac with macOS 26 or newer.
- Python 3.12 or newer.
- [Carbon](https://github.com/Lazarus-931/carbon), installed in one of two ways:
  - **Download** the prebuilt package from [Carbon's releases](https://github.com/Lazarus-931/carbon/releases) and install it with `pip install carbon_metal-*.whl`.
  - **Build** it from source with its compiler, as [Carbon's quick start](https://github.com/Lazarus-931/carbon/blob/main/docs/quickstart.md) shows.

## Install

With Carbon installed in the same environment:

```sh
git clone https://github.com/Lazarus-931/swiss
pip install -e "./swiss[models]"
```

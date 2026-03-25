# Book

The companion book is built with [MkDocs](https://www.mkdocs.org/) and the
[Material theme](https://squidfunk.github.io/mkdocs-material/). It combines
test reports, executed Marimo notebooks, API reference, and project
documentation into a single static site.

## Building

```bash
make book
```

The output is written to `_book/`.

## What gets included

| Section | Source | How it's generated |
|---|---|---|
| Home | `README.md` | Included via snippet |
| Marimo Notebooks | `book/marimo/notebooks/*.py` | Exported to HTML with `marimo export html --sandbox` |
| Reports | `_tests/` | Copied after running `make test`, `make benchmark`, `make stress`, `make hypothesis-test` |
| API Reference | `src/` | Rendered by mkdocstrings from source docstrings |

## Adding a notebook

Place a Marimo notebook (`.py`) in `book/marimo/notebooks/`. It will be
picked up automatically on the next `make book`. Notebooks are executed in
a sandbox using their inline [PEP 723](https://peps.python.org/pep-0723/)
dependencies, so each notebook is self-contained.

## Configuration

The MkDocs configuration lives in `.rhiza/mkdocs.yml`.

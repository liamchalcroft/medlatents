# MedLatents Documentation

This directory contains the Sphinx documentation for MedLatents.

## Building Documentation

Install documentation dependencies:
```bash
pip install -e ".[docs]"
```

Build HTML documentation:
```bash
cd docs
make html
```

View the documentation:
```bash
open _build/html/index.html
```

## Documentation Structure

- `conf.py` - Sphinx configuration
- `index.rst` - Landing page
- `getting_started.rst` - Quick start guide
- `guides/` - Module-specific guides
- `api/` - Auto-generated API reference
- `tutorials/` - Step-by-step guides/tutorials
- `research/` - Research paper documentation
- `_static/` - Static assets (CSS, images)
- `_templates/` - Custom Jinja templates

## Online Documentation

Documentation is automatically built and deployed to GitHub Pages:

https://liamchalcroft.github.io/medlatents/

## Contributing to Documentation

1. Edit source files in `docs/`
2. Build with `make html`
3. Review locally
4. Submit pull request

When adding new modules:
1. Create `api/<module>.rst` with `automodule` directive
2. Create `guides/<module>.rst` with usage examples
3. Update `index.rst` to include new pages

For more details, see `contributing.rst`.

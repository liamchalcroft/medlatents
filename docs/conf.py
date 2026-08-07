"""Configuration file for the Sphinx documentation builder."""

from __future__ import annotations

import inspect
import os
import sys
from datetime import datetime

# --- Path Setup -------------------------------------------------------------

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("src"))

import medlatents  # noqa: E402

# --- Project Information -----------------------------------------------------

project = "MedLatents"
author = "Liam Chalcroft"
copyright = f"{datetime.now().year}, {author}"
version = medlatents.__version__
release = version

language = "en"

# Files Sphinx should not treat as source documents (README.md is for GitHub).
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]

# Short symbol names (e.g. ``loss``) recur across many autodoc'd trainer classes;
# silence the resulting cross-reference ambiguity warnings (these are not errors).
suppress_warnings = ["ref.python"]

# --- Extensions --------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.linkcode",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.mermaid",
    "myst_parser",
]

# --- Napoleon Configuration (Google/NumPy docstrings) -------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# --- Type Hints -------------------------------------------------------------

autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_typehints_format = "fully-qualified"
typehints_defaults = "comma"
always_document_param_types = True

# --- Autodoc Configuration ---------------------------------------------------

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": False,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
    "inherited-members": True,
    "private-members": False,
    "protected-members": False,
}

# Autosummary settings
autosummary_generate = True
autosummary_generate_overwrite = True
autosummary_imported_members = True

# --- Intersphinx (Cross-references to other projects) -----------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# --- Link to Source Code ----------------------------------------------------


def linkcode_resolve(domain, info):
    """Determine URL corresponding to Python object."""
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    try:
        fn = inspect.getsourcefile(obj)
    except TypeError:
        fn = None
    if not fn:
        return None

    try:
        source_lines, lineno = inspect.getsourcelines(obj)
    except (OSError, TypeError):
        lineno = None
        source_lines = []

    if lineno:
        linespec = f"#L{lineno}-L{lineno + len(source_lines) - 1}"
    else:
        linespec = ""

    fn = os.path.relpath(fn, start=os.path.dirname(medlatents.__file__))
    return f"https://github.com/liamchalcroft/medlatents/blob/main/src/medlatents/{fn}{linespec}"


# --- MathJax ---------------------------------------------------------------

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

# --- MyST Parser (Markdown support) -----------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

myst_enable_checkboxes = True
myst_heading_anchors = 3

# --- Mermaid Diagrams -------------------------------------------------------

mermaid_version = "latest"
mermaid_init_js = "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"

# --- HTML Theme -------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "logo_only": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
    "vcs_pageview_mode": "",
    "style_nav_header_background": "#2980B9",
    "navigation_depth": 4,
}

html_static_path = ["_static"]
html_extra_path = []

html_css_files = [
    "css/custom.css",
]

templates_path = ["_templates"]

html_last_updated_fmt = "%b %d, %Y"
html_show_sphinx = True
html_show_copyright = True
html_show_sigla = True

html_context = {
    "display_github": True,
    "github_user": "liamchalcroft",
    "github_repo": "medlatents",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# --- Table of Contents -------------------------------------------------------

html_sidebars = {
    "**": [
        "about.html",
        "navigation.html",
        "relations.html",
        "searchbox.html",
        "donate.html",
    ]
}

# --- Index -----------------------------------------------------------------

master_doc = "index"
html_use_index = True
html_split_index = False

# --- Doctest ----------------------------------------------------------------

doctest_default_flags = 0

# --- Misc -------------------------------------------------------------------

add_module_names = True
show_authors = True
show_warning_types = True

nitpick_ignore = [
    ("py:class", "torch.dtype"),
    ("py:class", "torch.device"),
    ("py:class", "torch.nn.Module"),
]

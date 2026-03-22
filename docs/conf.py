"""Sphinx configuration for OpenPKPD."""

import os
import sys

# Make src/ importable so autodoc can import the package
sys.path.insert(0, os.path.abspath("../src"))

# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------
project = "OpenPKPD"
author = "OpenPKPD contributors"
copyright = "2025, OpenPKPD contributors"
release = "0.2.1"
version = "0.2.1"

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",           # Google-style docstrings
    "sphinx.ext.viewcode",           # [source] links in API docs
    "sphinx.ext.intersphinx",        # Cross-link numpy / scipy / pandas docs
    "sphinx_autodoc_typehints",      # Render PEP 604 type annotations cleanly
    "myst_parser",                   # Markdown (.md) content files
    "sphinx_copybutton",             # Copy button on every code block
    "sphinx_design",                 # Grid cards, tabs on landing pages
]

# ---------------------------------------------------------------------------
# MyST (Markdown) configuration
# ---------------------------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",   # ::: fenced directives (cleaner than ```)
    "deflist",       # Definition lists
    "tasklist",      # - [ ] checkboxes in contributing.md
    "attrs_inline",  # Inline attributes {.class}
]
myst_heading_anchors = 3  # Auto-anchor h1 / h2 / h3

# ---------------------------------------------------------------------------
# autodoc / autosummary
# ---------------------------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",   # Preserve chaining order in ModelBuilder
    "undoc-members": False,
    "show-inheritance": True,
    "special-members": "__init__",
}
autodoc_typehints = "description"    # Types in description, not signature
autodoc_typehints_format = "short"   # numpy.ndarray → ndarray
autosummary_generate = True

# Packages that are optional at runtime — mock them so autodoc doesn't fail
# if they aren't installed in the docs build environment.
autodoc_mock_imports = [
    "jax", "jaxlib", "diffrax", "numpyro", "pymc",
    "optimagic", "dask", "sympy",
]

# ---------------------------------------------------------------------------
# Napoleon (Google-style docstring) settings
# ---------------------------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_attr_annotations = True

# ---------------------------------------------------------------------------
# Intersphinx: cross-link to upstream docs
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3.12", None),
    "numpy":  ("https://numpy.org/doc/stable", None),
    "scipy":  ("https://docs.scipy.org/doc/scipy", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

# ---------------------------------------------------------------------------
# Theme: sphinx-rtd-theme
# ---------------------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "logo_only": False,
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
    "style_external_links": True,
}
html_title = f"OpenPKPD {version}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# ---------------------------------------------------------------------------
# Source file settings
# ---------------------------------------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst",
}
root_doc = "index"

# ---------------------------------------------------------------------------
# LaTeX / PDF output
# ---------------------------------------------------------------------------
latex_engine = "xelatex"  # Unicode-safe; handles emoji without errors
latex_use_xindy = False  # Prefer makeindex for broader local toolchain support

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
suppress_warnings = ["autodoc.import_object"]
nitpicky = False

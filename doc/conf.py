# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pennylane-calculquebec"
copyright = "2026, CalculQuébec"
author = "CalculQuébec"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.apidoc",
    "sphinx.ext.autosummary",
    "myst_parser",
]

apidoc_modules = [
    {
        'path': '../pennylane_calculquebec',
        'destination': 'rtd/code_ref/',
        'exclude_patterns': ['**/test*'],
        'max_depth': 3,
        'follow_links': False,
        'separate_modules': True,
        'include_private': False,
        'no_headings': False,
        'module_first': False,
        'implicit_namespaces': False,
        'automodule_options': {
            'members', 'show-inheritance', 'member-order : groupwise',
        },
    },
]

# Autodoc and Autosummary configuration
# autodoc_default_options = {
#     "members": True,
#     "inherited-members": True,
#     "show-inheritance": True,
#     "special-members": "__init__",
# }
# Include both class docstring and __init__ docstring
autoclass_content = "both"

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**/*.md",
]

autosummary_generate = True


language = "en"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ["_static"]
html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/calculquebec/pennylane-calculquebec",
    "use_repository_button": True,
}

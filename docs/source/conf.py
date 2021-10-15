# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'Recurrent Intensity Model Experiments'
copyright = '2021, Yifei Ma, Ge Liu, Anoop Deoras'
author = 'Yifei Ma, Ge Liu, Anoop Deoras'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
]

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.append(os.path.abspath('../../'))
import cr_mech_coli

project = 'cr_mech_coli'
copyright = '2024, Jonas Pleyer'
author = 'Jonas Pleyer'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser',
    'sphinx.ext.napoleon',
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosectionlabel',
    'sphinxcontrib.bibtex',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

autosummary_generate = True
add_module_names = False
autodoc_typehints_format = 'short'

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

bibtex_bibfiles = ['references.bib']
bibtex_default_style = 'unsrt'
bibtex_reference_style = 'label'

html_theme_options = {
    "show_nav_level": 2,
    "logo": {
        "image_dark": '_static/cellular_raza_dark_mode.png',
        "image_light": '_static/cellular_raza.png'
    },
}

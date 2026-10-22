# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

import inspect
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../"))
import interpax

project = "interpax"
copyright = "2023, Rory Conlin"
author = "Rory Conlin"
version = interpax.__version__
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_github_style",
]
# options for sphinx_github_style
top_level = "interpax"
linkcode_blob = "head"
linkcode_url = r"https://github.com/f0uriest/interpax/"
linkcode_link_text = "Source"


def linkcode_resolve(domain, info):
    """Returns a link to the source code on GitHub, with appropriate lines highlighted"""

    if domain != "py" or not info["module"]:
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

    # for jitted stuff, get the original src
    if hasattr(obj, "__wrapped__"):
        obj = obj.__wrapped__

    # get the link to HEAD
    cmd = "git log -n1 --pretty=%H"
    try:
        # get most recent commit hash
        head = subprocess.check_output(cmd.split()).strip().decode("utf-8")

        # if head is a tag, use tag as reference
        cmd = "git describe --exact-match --tags " + head
        try:
            tag = subprocess.check_output(cmd.split(" ")).strip().decode("utf-8")
            blob = tag

        except subprocess.CalledProcessError:
            blob = head

    except subprocess.CalledProcessError:
        print("Failed to get head")  # so no head?
        blob = "main"

    linkcode_url = r"https://github.com/f0uriest/interpax/"
    linkcode_url = linkcode_url.strip("/") + f"/blob/{blob}/"
    linkcode_url += "{filepath}#L{linestart}-L{linestop}"

    # get a Path object representing the working directory of the repository.
    try:
        cmd = "git rev-parse --show-toplevel"
        repo_dir = Path(subprocess.check_output(cmd.split(" ")).strip().decode("utf-8"))

    except subprocess.CalledProcessError as e:
        raise RuntimeError("Unable to determine the repository directory") from e

    # For ReadTheDocs, repo is cloned to /path/to/<repo_dir>/checkouts/<version>/
    if repo_dir.parent.stem == "checkouts":
        repo_dir = repo_dir.parent.parent

    # path to source file
    try:
        filepath = os.path.relpath(inspect.getsourcefile(obj), repo_dir)
        if filepath is None:
            return
    except Exception:
        return None

    # lines in source file
    try:
        source, lineno = inspect.getsourcelines(obj)
    except OSError:
        return None
    else:
        linestart, linestop = lineno, lineno + len(source) - 1

    # Fix links with "../../../" or "..\\..\\..\\"
    filepath = "/".join(filepath[filepath.find(top_level) :].split("\\"))

    final_link = linkcode_url.format(
        filepath=filepath, linestart=linestart, linestop=linestop
    )
    print(f"Final Link for {fullname}: {final_link}")
    return final_link


# numpydoc_class_members_toctree = False
# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = False

autodoc_default_options = {
    "member-order": "bysource",
    "exclude-members": "__init__",
}
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = [".rst", ".md"]
# source_suffix = {
#     '.rst': 'restructuredtext',
#     '.md': 'markdown',
# }
# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.rst"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    #    'canonical_url': '',
    #    'analytics_id': 'UA-XXXXXXX-1',  #  Provided by Google in your dashboard
    "logo_only": True,
    "prev_next_buttons_location": "both",
    "style_external_links": False,
    "style_nav_header_background": "#3c4142",
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 2,
    "includehidden": True,
    "titles_only": False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_logo = "_static/images/logo_small_clear.png"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = "_static/images/desc_icon.ico"


# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = "%b %d, %Y"

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
# html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
# html_additional_pages = {}

# If false, no module index is generated.
html_domain_indices = True

# If false, no index is generated.
html_use_index = True

# If true, the index is split into individual pages for each letter.
html_split_index = False

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
# html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
# html_file_suffix = None

# Output file base name for HTML help builder.
htmlhelp_basename = "interpax"


# -- Options for LaTeX output ---------------------------------------------
latex_engine = "pdflatex"
latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    "papersize": "letterpaper",
    "fncychap": "\\usepackage{fncychap}",
    "fontpkg": "\\usepackage{amsmath,amsfonts,amssymb,amsthm}",
    "figure_align": "htbp",
    # The font size ('10pt', '11pt' or '12pt').
    #
    "pointsize": "10pt",
    # Additional stuff for the LaTeX preamble.
    #
    "preamble": r"""
    %% %% %% %% %% %% %% %% %% %% Meher %% %% %% %% %% %% %% %% %%
    %% %add number to subsubsection 2=subsection, 3=subsubsection
    %% % below subsubsection is not good idea.
    \setcounter{secnumdepth}{3}
    %% %% Table of content upto 2=subsection, 3=subsubsection
    \setcounter{tocdepth}{3}
    \usepackage{amsmath,amsfonts,amssymb,amsthm}
    \usepackage{graphicx}

    %% % reduce spaces for Table of contents, figures and tables
    %% % it is used "\addtocontents{toc}{\vskip -1.2cm}" etc. in the document
    \usepackage[notlot,nottoc,notlof]{}

    \usepackage{color}
    \usepackage{transparent}
    \usepackage{eso-pic}
    \usepackage{lipsum}
    \usepackage{hyperref}
    \usepackage{footnotebackref} %% link at the footnote to go to the place of footnote in the text
    %% spacing between line
    \usepackage{setspace}
    %% %% \onehalfspacing
    %% %% \doublespacing
    \singlespacing
    %% %% %% %% %% % datetime
    \usepackage{datetime}
    %% RO, LE will not work for 'oneside' layout.
    %% Change oneside to twoside in document class
    \usepackage{fancyhdr}
    \pagestyle{fancy}
    \fancyhf{}
    %% % Alternating Header for oneside
    \fancyhead[L]{\ifthenelse{\isodd{\value{page}}}{ \small \nouppercase{\leftmark} }{}}
    \fancyhead[R]{\ifthenelse{\isodd{\value{page}}}{}{ \small \nouppercase{\rightmark} }}
    %% % Alternating Header for two side
    %\fancyhead[RO]{\small \nouppercase{\rightmark}}
    %\fancyhead[LE]{\small \nouppercase{\leftmark}}
    %% for oneside: change footer at right side. If you want to use Left and right then use same as header defined above.
    %% % Alternating Footer for two side
    %\fancyfoot[RO, RE]{\scriptsize Meher Krishna Patel (mekrip@gmail.com)}
    %% % page number
    \fancyfoot[CO, CE]{\thepage}
    \renewcommand{\headrulewidth}{0.5pt}
    \renewcommand{\footrulewidth}{0.5pt}
    %\RequirePackage{tocbibind} %%% comment this to remove page number for following
    \addto\captionsenglish{\renewcommand{\contentsname}{Table of contents}}
    %\addto\captionsenglish{\renewcommand{\listfigurename}{List of figures}}
    %\addto\captionsenglish{\renewcommand{\listtablename}{List of tables}}
     \addto\captionsenglish{\renewcommand{\chaptername}{Chapter}}
    %% reduce spacing for itemize
    \usepackage{enumitem}
    \setlist{nosep}
    %% %% %% %% %% % Quote Styles at the top of chapter
    \usepackage{epigraph}
    \setlength{\epigraphwidth}{0.8\columnwidth}
    \newcommand{\chapterquote}[2]{\epigraphhead[60]{\epigraph{\textit{#1}}{\textbf {\textit{--#2}}}}}
    %% %% %% %% %% % Quote for all places except Chapter
    \newcommand{\sectionquote}[2]{{\quote{\textit{``#1''}}{\textbf {\textit{--#2}}}}}
    """,
    "maketitle": r"""
    \pagenumbering{Roman} %% % to avoid page 1 conflict with actual page 1
    \begin{titlepage}

        \vspace*{80mm} %% % * is used to give space from top

        \centering
        \textbf{\Huge {interpax Documentation}}

        \vspace*{2mm}

        \vspace*{10mm}

        \centering
        \textbf{ \Large {Rory Conlin}}

        \vspace*{10mm}

        \centering
        \small {Created: \today}
        %% \vfill adds at the bottom

        \vfill

    \end{titlepage}
    \clearpage
    %\pagenumbering{roman}
    %\sphinxtableofcontents
    %\clearpage
    \pagenumbering{arabic}
    """,
    "sphinxsetup": "hmargin={0.7in,0.7in}, vmargin={1in,1in}, \
    verbatimwithframe=true, \
    TitleColor={rgb}{0,0,0}, \
    HeaderFamily=\\rmfamily\\bfseries, \
    InnerLinkColor={rgb}{0,0,1}, \
    OuterLinkColor={rgb}{0,0,1}",
    #   'tableofcontents': '',
}
# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        "interpax.tex",
        "interpax Documentation",
        "Rory Conlin",
        "report",
        True,
    ),
]

latex_toplevel_sectioning = "chapter"

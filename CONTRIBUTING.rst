Contributing
============

:+1::tada: First off, thanks for taking the time to contribute! :tada::+1:

The following is a set of guidelines for contributing to interpax These are
mostly guidelines, not rules. Use your best judgment, and feel free to
propose changes to this document in a pull request.

Table Of Contents
^^^^^^^^^^^^^^^^^

* `I don’t want to read this whole thing, I just have a question!!! <#i-dont-want-to-read-this-whole-thing-i-just-have-a-question>`__

* `How Can I Contribute? <#how-can-i-contribute>`__

  - `Reporting Bugs <#reporting-bugs>`__
  - `Suggesting Enhancements <#suggesting-enhancements>`__
  - `Your First Code Contribution <#your-first-code-contribution>`__
  - `Pull Requests <#pull-requests>`__

* `Styleguides <#styleguides>`__

  - `Python <#python-styleguide>`__
  - `Git Commit Messages <#git-commit-messages>`__
  - `Documentation Styleguide <#documentation-styleguide>`__


I don’t want to read this whole thing I just have a question!!!
***************************************************************

If you just want to ask a question, the simplest method is to `create an issue
on github <https://github.com/f0uriest/interpax/issues/new>`__ and begin the
subject line with ``Question:`` That way it will be seen by all developers, and
the answer will be viewable by other users.

How Can I Contribute?
^^^^^^^^^^^^^^^^^^^^^

Reporting Bugs
**************

How Do I Submit A (Good) Bug Report?
------------------------------------

Bugs are tracked as `GitHub issues <https://github.com/PlasmaControl/f0uriest/interpax/>`__.

Explain the problem and include additional details to help maintainers
reproduce the problem:

-  **Use a clear and descriptive title** for the issue to identify the
   problem.
-  **Describe the exact steps which reproduce the problem** in as many
   details as possible. When listing steps, *don’t just say what you did, but explain how you did it*.
-  **Provide specific examples to demonstrate the steps**. Include links
   to files or copy/pasteable snippets, which you use in those examples.
   If you’re providing snippets in the issue, use
   `Markdown code blocks <https://help.github.com/articles/markdown-basics/#multiple-lines>`__.
-  **Describe the behavior you observed after following the steps** and
   point out what exactly is the problem with that behavior.
-  **Explain which behavior you expected to see instead and why.**
-  **Include plots** of results that you believe to be wrong.

Provide more context by answering these questions:

-  **Did the problem start happening recently** (e.g. after updating to
   a new version) or was this always a problem?
-  If the problem started happening recently, **can you reproduce the problem in an older version?**
   What’s the most recent version in which the problem doesn’t happen?
-  **Can you reliably reproduce the issue?** If not, provide details
   about how often the problem happens and under which conditions it
   normally happens.

Include details about your configuration and environment:

-  **Which version of interpax are you using?**
-  **Which version of JAX (and other dependencies) are you using**
-  **What’s the name and version of the OS you’re using**?
-  **What hardware are you running on?** Which CPU, GPU, RAM, etc. If on
   a cluster, what resources are you allocating?

Suggesting Enhancements
***********************

This section guides you through submitting an enhancement suggestion for
interpax, including completely new features and minor improvements to
existing functionality.

Before creating enhancement suggestions, please check `this list <#before-submitting-an-enhancement-suggestion>`__
as you might find out that you don’t need to create one. When you are creating an
enhancement suggestion, please `include as many details as possible <#how-do-i-submit-a-good-enhancement-suggestion>`__,
including the steps that you imagine you would take if the feature you’re
requesting existed.

Before Submitting An Enhancement Suggestion
-------------------------------------------

-  `Check the documentation <https://interpax.readthedocs.io/en/latest/>`__
   for tips — you might discover that the enhancement is already available.
-  `Perform a cursory search <https://github.com/f0uriest/interpax/issues?q=is%3Aopen+is%3Aissue+label%3Aenhancement>`__
   to see if the enhancement has already been suggested. If it has, add
   a comment to the existing issue instead of opening a new one.

How Do I Submit A (Good) Enhancement Suggestion?
------------------------------------------------

Enhancement suggestions are tracked as `GitHub issues <https://guides.github.com/features/issues/>`__.
After you’ve followed the above steps and verified that an issue does not already
exist, create an issue and provide the following information:

-  **Use a clear and descriptive title** for the issue to identify the
   suggestion.
-  **Provide a step-by-step description of the suggested enhancement**
   in as many details as possible.
-  **Provide specific examples to demonstrate the steps**. Include
   copy/pasteable snippets which you use in those examples, as
   `Markdown code blocks <https://help.github.com/articles/markdown-basics/#multiple-lines>`__.
-  **Describe the current behavior** and **explain which behavior you expected to see instead** and why.
-  **Explain why this enhancement would be useful** to other users.
-  **Provide references** (if relevant) to papers that discuss the
   physics behind the enhancement, or describe the enhancement in some
   way.

Your First Code Contribution
****************************

Unsure where to begin contributing? You can start by looking
through these ``good first issue`` and ``help wanted`` issues:

-  `Good first issues <https://github.com/f0uriest/interpax/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22>`__ - issues which should only require a few lines of code, and a test or two.
-  `Help wanted issues <https://github.com/f0uriest/interpax/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22>`__ - issues which should be a bit more involved than beginner issues.

Pull Requests
*************

Once you've made your changes on a local branch, `open a pull request <https://github.com/f0uriest/interpax/pulls>`_
on github. In the description, give a summary of what is being changed and why. Try to keep pull requests small and atomic,
with each PR focused on a adding or fixing a single thing. Large PRs will generally take much longer to review and approve.

Opening a PR will trigger a suite of tests and style/formatting checks that must pass before new code can be merged.
We also require approval from at least one (ideally multiple) of the main developers, who may have suggested changes
or edits to your PR.


Styleguides
^^^^^^^^^^^

Python Styleguide
*****************

-  `Follow the PEP8 format <https://www.python.org/dev/peps/pep-0008/>`__ where possible
-  Format code using `black <https://github.com/psf/black>`__ before committing - with formatting, consistency is better than "correctness." We use version ``22.10.0`` (there are small differences between versions). Install with ``pip install "black==22.10.0"``.
-  Check code with ``flake8``, settings are in ``setup.cfg``
-  We recommend installing ``pre-commit`` with ``pip install pre-commit`` and then running ``pre-commit install`` from the root of the repository. This will automatically run a number of checks every time you commit new code, reducing the likelihood of committing bad code.
-  -  Use `Numpy Style Docstrings <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html#example-numpy>`__ - see the code for plenty of examples. At a minimum, the docstring should include a description of inputs and outputs, and a short description of what the function or method does. Code snippets showing example usage strongly encouraged.
-  **Readability** and **usability** are more important than speed 99%
   of the time.
-  If it takes more than 30 seconds to understand what a line or block
   of code is doing, include a comment summarizing what it does.
-  If a function has more than ~5 inputs and/or return values, consider
   packaging them in a dictionary or custom class.
-  Make things modular. Focus on small functions that `do one thing and do it well <https://en.wikipedia.org/wiki/Unix_philosophy#Origin>`__,
   and then combine them together. Don’t try to shove everything into a
   single function.
-  *It’s not Fortran*! You are not limited to 6 character variable
   names. Please no variables or functions like ``ma00ab`` or
   ``psifac``. Make names descriptive and clear. If the name and meaning
   of a variable is not immediately apparent, the name is probably
   wrong.
-  Sometimes, a shorter, less descriptive name may make the code more
   readable. If you want to use an abbreviation or shorthand, include a
   comment with the keyword ``notation:`` explaining the notation at the
   beginning of the function or method explaining it, eg
   ``# notation: v = vartheta, straight field line poloidal angle in radians``.


``pytest``
----------

The testing suite is based on `pytest <https://docs.pytest.org/>`__, and makes use of several plugins for specialized testing. You can install all the necessary tools with ``pip install -r requirements-dev.txt``. You can run the tests from the root of the repository with ``pytest -m unit``. To only run selected tests you can use ``pytest -k foo`` which will only run tests that have ``foo`` in the test or file name.

Additional useful flags include:

- ``--cov`` will tell it to also report how much of the code is covered by tests using `pytest-cov <https://pytest-cov.readthedocs.io/en/latest/>`__. A summary of the coverage is printed to the terminal at the end of the tests, and detailed information is saved to a ``.coverage`` file, which can then be turned into a simple HTML page with ``coverage html``. This will create a ``htmlcov/`` directory in the root of the repository that can be viewed in a browser to see line by line coverage.


`Git Commit Messages <https://chris.beams.io/posts/git-commit/>`__
*******************************************************************

-  A commit message template is included in the repository, ``.gitmessagetemplate``
-  You can set the template to be the default with ``git config commit.template .gitmessagetemplate``

Some helpful rules to follow (also included in the template):

-  Separate subject line from body with a single blank line.
-  Limit the subject line to 50 characters or less, and wrap body lines
   at 72 characters.
-  Capitalize the subject line.
-  Use the present tense (“Add feature” not “Added feature”) and the
   imperative mood (“Fix issue…” not “Fixes issue…”) in the subject
   line.
-  Reference issues and pull requests liberally in the body, including
   specific issue numbers. If the commit resolves an issue, note that at
   the bottom like ``Resolves: #123``.
-  Explain *what* and *why* vs *how*. Leave implementation details in
   the code. The commit message should be about what was changed and
   why.

Documentation Styleguide
************************

-  Use `SphinxDoc <https://www.sphinx-doc.org/en/master/index.html>`__.
-  Use `Numpy Style Docstrings <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html#example-numpy>`__.
-  Use `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`__.

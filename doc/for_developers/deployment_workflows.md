# Quantum plugins deployment

This documents how quantum software projects are published by Calcul Quebec on PyPI.

The deployment workflow assumes all QA steps have been succeeded and a production ready version of the software has been merged on the default branch.

## Reliance on tags in VCS

This workflow is intended to fully rely on tags as a single source of truth for versioning the project releases. Multiple tools within this workflow already have as a default behavior to refer to Version Control Software (VCS) tags.

It is recommended to not deviate from this philosophy, since specifying versions anywhere else might break the default behavior from those same tools. Version specifiers in `toml` and `_version.py` files are to be avoided since they can essentially be generated from tools relying on VCS tags.

## Which tools retrieve tags as references?

In order:

1. `draft-release.yml` **workflow** : this workflow consists of only one step that create a release using `softprops/action-gh-release` . [`action-gh-release`](https://github.com/softprops/action-gh-release?tab=readme-ov-file#inputs) has an attribute `name` defaulting to its `tag_name` attribute, which in turn defaults to `github.ref_name` . Since the workflow is triggered by a tag push event on main, the `github.ref_name` is equal to that tag.
2. `pypi-publish.yml` **workflow** : this workflow is triggered by the publication of a release and will therefore have `github.ref_name` [equal to the tag associated with the release](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#release). It contains two components that rely on that `ref_name`:
    - `actions/checkout` : the `ref` configuration [defaults to the reference, aka. the trigger](https://github.com/actions/checkout?tab=readme-ov-file#usage) 
    - the build step that uses the `setuptools-scm` build system (it is configured in the `pyproject.toml` file), [which relies on the latest tag it can fetch in the git history](https://setuptools-scm.readthedocs.io/en/latest/usage/#default-versioning-scheme) 

## Typical deployment steps

1. The developer tags a commit on the default branch locally, then pushes that tag on the remote repository. The `draft-release.yml` workflow then drafts a release on Github.
2. The developer reviews and complement the draft release if necessary.
3. The developer publishes the release, which triggers the `pypi-publish.yml` that will build the project and publish it on PyPI.
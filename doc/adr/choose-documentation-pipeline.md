# Decision Record: Choice of Documentation Pipeline

## Context

The plugin is a software development project for which some basic documentation exists in [`doc/`](https://github.com/calculquebec/pennylane-calculquebec/tree/main/doc).

This documentation is currently made up of Markdown pages and Jupyter notebooks, which are maintained manually. However, this process has several drawbacks:
- Pages written as notebooks produce diffs that can’t easily be peer-reviewed in a Pull Request.
- The API documentation is not updated automatically, which makes it likely to become outdated if it is missed during a refactor, for example.
- The documentation is susceptible to be neglected, especially if the developer resources are limited, as they will likely prioritize the development and maintenance of the plugin itself over the documentation.
- The documentation is not easily discoverable, as it is located in the source code repository and not published on a dedicated platform. This can make it difficult for users to find and access the documentation, especially if they are not familiar with the project or its structure.

As the project grows, those pain points are becoming more significant, making the use, but also the maintenance, of the plugin as a whole more difficult.

In accordance with [PennyLane's documentation guidelines](https://docs.pennylane.ai/en/stable/development/guide/documentation.html), the code already contains docstrings using reStructuredText (reST) markup, formatted Google style. This means that the documentation pipeline should be able to parse and render these docstrings correctly, ensuring that the API documentation is generated accurately and consistently.

This decision aims to select a free documentation pipeline that automates documentation generation, enforces a consistent structure and format, and reduces the maintenance burden on developers. It should support Python and Jupyter notebooks, integrate with existing developer tools and workflows, and remain easy to use and maintain even for contributors with limited experience with documentation tooling.

## Decision

After evaluating several documentation pipelines, it has been decided to use [Sphinx](https://www.sphinx-doc.org/en/master/) as the documentation generator and [Read the Docs](https://readthedocs.org/) as the hosting platform for the plugin's documentation.

Sphinx is a widely used documentation generator that supports Python, but also Jupyter notebooks through extensions like `nbsphinx`. It can parse reST docstrings and generate API documentation automatically, ensuring that the documentation remains inline with the code and is less likely to become outdated. Sphinx allows for a high level of customization and the technical depth required to set it up and maintain it can be a burden. However, Read the Docs provides a simple and easy to digest documentation on its support for Sphinx, allowing to lower the learning curve and maintenance overhead.

Read the Docs (RTD) is popular for hosting documentation and provides seamless integration with Sphinx and GitHub. It offers features like versioning, localization using multiple RTD projects, and automatic builds triggered by commits to the repository. This means the documentation will be automatically updated whenever changes are made to the codebase and the changes will be easy to review in a Pull Request, as the generated documentation will be added as a check in the PR. Read the Docs is free of charge for the usage we expect for this project, and it provides a user-friendly interface for browsing the documentation, making it more accessible to users.

## Consequences

**Positive:**
- Automation of documentation generation, ensuring that the API documentation is always up-to-date with the codebase.
- Improved discoverability and accessibility of the documentation through hosting on Read the Docs.
- Support for multi-version documentation and localization, allowing users to access the documentation in their preferred language.

**Negative/Neutral:**
- Sphinx has a steeper learning curve compared to simpler tools, which may require additional time to set up properly. However, Read the Docs provides good examples and documentation to help with this process.
- While Read the Docs is free for the expected usage, it may have limitations in terms of build time and storage, which could become an issue if the documentation grows significantly. This is however unlikely to be a problem for the plugin's documentation in the foreseeable future.

## Alternatives Considered

- GitHub Pages: While GitHub Pages is a popular choice and very simple to set up, it does not provide the support for multi-version documentation unless the archive is manually maintained and stored in the repository. Additionally, it makes localization, the same way as multi-version documentation, more difficult to maintain, as it would require the use of branches for each language.
- Zensical: Zensical is the new kid on the block, and is a promising tool for documentation. It is built on top of MkDocs, and only supports Markdown, which is not ideal for our docstrings written in reST. Additionally, it is still in early development, and requires a heavy maintenance effort to keep up with the latest changes, which is not ideal for a project with limited developer resources.

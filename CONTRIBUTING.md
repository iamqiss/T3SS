# Contributing to T3SS

First off, thank you for considering a contribution to T3SS! 🎉 We're on a mission to build the next generation of web search, and we welcome your help. Every contribution, from reporting a bug to implementing a major feature, is valuable.

This document provides guidelines to ensure a smooth and effective collaboration process.

***

## Code of Conduct

This project and everyone participating in it is governed by the [T3SS Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior.

***

## How Can I Contribute?

There are many ways to contribute to the project, and all are welcome.

### 🐛 Reporting Bugs

If you find a bug, please ensure it hasn't already been reported by searching the issues on GitHub. If you can't find an open issue addressing the problem, open a new one. Be sure to include:

* A **clear and descriptive title**.
* **Detailed steps to reproduce** the bug.
* A description of the **expected behavior** versus the **actual behavior**.
* Details about your environment (OS, Kubernetes version, etc.).

### ✨ Suggesting Enhancements

If you have an idea for a new feature or an improvement to an existing one:

1.  Search the issues to see if the enhancement has already been suggested.
2.  If not, open a new issue with a clear title and a detailed description of the proposed functionality and its potential benefits.

### 💻 Your First Code Contribution

Unsure where to begin? Look for issues tagged with `good first issue` or `help wanted`. These are specifically chosen as good entry points into the project.

***

## Development Workflow

To ensure consistency and quality, we follow a standard development workflow.

1.  **Fork the Repository**
    Start by forking the main T3SS repository to your own GitHub account.

2.  **Clone Your Fork**
    Clone your fork to your local machine:
    ```bash
    git clone [https://github.com/iamqiss/T3SS.git](https://github.com/iamqiss/T3SS.git)
    cd T3SS
    ```

3.  **Create a Branch**
    Create a new branch for your changes. The branch name should be descriptive and follow this convention:
    * For new features: `feature/<feature-name>` (e.g., `feature/image-caption-generator`)
    * For bug fixes: `bugfix/<issue-number>` (e.g., `bugfix/issue-421`)
    ```bash
    # Branch from the 'develop' branch
    git checkout develop
    git pull origin develop
    git checkout -b feature/your-new-feature
    ```

4.  **Make Your Changes**
    Write your code! Ensure your code adheres to the style guidelines for the respective language. Add or update tests as necessary.

5.  **Run Tests**
    Before submitting, ensure all relevant tests pass. Each service has its own test suite.
    ```bash
    # Conceptual command to run tests for a specific service
    make test SERVICE=core/indexing
    ```

6.  **Commit Your Changes**
    We use the **Conventional Commits** specification for our commit messages. This helps us automate changelogs and makes the project history more readable.

    The format is `type(scope): subject`.
    * **type**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
    * **scope**: The part of the project you're changing (e.g., `querying`, `frontend`, `auth`)

    *Example commit:*
    ```
    feat(querying): Implement spell correction for multi-word queries
    ```

7.  **Push to Your Fork**
    Push your changes to your forked repository.
    ```bash
    git push origin feature/your-new-feature
    ```

8.  **Submit a Pull Request (PR)**
    Open a Pull Request from your branch to the `develop` branch of the main T3SS repository. In your PR description, please:
    * Provide a clear explanation of the changes.
    * Link to the issue it resolves (e.g., `Closes #123`).
    * Wait for a review from the maintainers. Be prepared to make changes based on feedback.

***

## 🎨 Style Guides & Conventions

Consistency is key in a large project. Please adhere to the following standards.

* **Go**: Code should be formatted with `gofmt`. We use `golangci-lint` for linting.
* **Rust**: Code should be formatted with `rustfmt`. We use `clippy` for linting.
* **Python**: We use `Black` for auto-formatting and `flake8` for linting.
* **Infrastructure**: All YAML and Terraform files should be consistently formatted.
* **APIs**: Internal services communicate via gRPC. Public-facing APIs follow RESTful principles.

Thank you again for your contribution!

# Contributing to VectorDB

Thank you for your interest in contributing to VectorDB! We welcome contributions from the community and appreciate your efforts to improve this project.

## How to Contribute

There are many ways to contribute to VectorDB:

- Submit bug reports or feature requests via [GitHub Issues](https://github.com/avnlp/vectordb/issues).
- Fix bugs or implement features via [Pull Requests](https://github.com/avnlp/vectordb/pulls).
- Improve documentation.
- Review existing pull requests.
- Help answer questions from other users.

## Development Setup

1. Fork the repository on GitHub.

2. Clone your fork locally and navigate to the project directory:

    ```bash
    cd vectordb
    ```

    Note: This project requires Python 3.11 or higher.

3. The project uses [uv](https://github.com/astral-sh/uv) for dependency management. Ensure uv is installed:

   ```bash
   pip install uv
   ```

4. Sync the project and install all dependencies:

     ```bash
     make sync
     ```

     Then activate the virtual environment:

     ```bash
     source .venv/bin/activate
     ```

     This creates an isolated virtual environment with project dependencies and the project installed in editable mode.

## Code Quality Standards

### Testing

Before submitting a pull request, ensure all tests pass:

```bash
make test
```

To run tests with coverage:

```bash
make test-cov
```

To run tests with coverage and generate reports:

```bash
make cov
```

To generate coverage reports from existing coverage data:

```bash
make cov-report
```

Add tests for any new functionality or bug fixes.

### Code Quality

This project uses:

- [Ruff](https://docs.astral.sh/ruff/) for code formatting and linting.
- [mypy](https://mypy.readthedocs.io/) for type checking.

Format and lint your code before submitting:

```bash
make lint-all
```

This command will format code with Ruff, apply auto-fixes with Ruff, and check typing with mypy.

### Useful Development Commands

**Setup:**

- `make sync` - Create/sync development environment (installs all dependencies).

**Testing:**

- `make test` - Run unit tests (excludes integration tests).
- `make test-cov` - Run tests with coverage collection.
- `make test-ci` - Run tests with coverage + XML/junit output (used in CI).
- `make cov` - Run tests and generate coverage reports (html + xml).
- `make cov-report` - Generate coverage reports (html + xml) from existing coverage data.

**Code Quality:**

- `make lint-all` - Run all code quality checks and formatting (format + lint + type check).
- `make lint-fmt` - Format code and apply auto-fixes (ruff).
- `make lint-check` - Check formatting and lint without modifying files.
- `make lint-style` - Lint only (no changes).
- `make lint-typing` - Type check only (mypy).
- `make lint-typos` - Check for typos.

**Security:**

- `make security-bandit` - Run Bandit security scan.
- `make security-audit` - Run pip-audit for dependency vulnerabilities.
- `make security` - Run all security scans.

Run `make help` to see all available targets.

## Community Guidelines

Please be respectful and constructive in all interactions.

By contributing to this project, you agree that your contributions will be licensed under the MIT license as specified in the [LICENSE](LICENSE) file.

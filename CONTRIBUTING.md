# Contributing

Thank you for your interest in contributing to AI EDA Platform. We welcome bug reports, improvements, documentation updates, and feature proposals.

## Setup

1. Fork the repository and create a feature branch:

```bash
git checkout -b feat/short-description
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
pip install -r requirements.txt
```

## Branch naming

- `feat/<short-description>` — new features
- `fix/<short-description>` — bug fixes
- `docs/<short-description>` — documentation updates
- `chore/<short-description>` — repository maintenance

## Pull Requests

- Open a pull request with a clear title and description.
- Reference any relevant issues using `#<issue-number>`.
- Keep PRs focused and small when possible.

## Code style

- Follow PEP8 for Python code.
- Use meaningful variable and function names.
- Add tests where appropriate and describe manual test steps in the PR description.

## Tests

- This repository does not include an automated test framework by default. Add unit or integration tests when introducing non-trivial logic.

## Review process

- PRs will be reviewed for clarity, tests, and backward-compatible behavior.
- Significant changes may require a design discussion before implementation.

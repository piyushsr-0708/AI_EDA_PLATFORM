# Security

Security is important. Please follow these guidelines when contributing or operating this project.

- NEVER commit credentials or secrets into the repository. In particular, do not commit `credentials/service_account.json` or any `.env` files.
- Add any new secrets to `.gitignore` and provide a template in `.env.example`.
- Report security vulnerabilities by opening an issue with the `security` label or contacting the repository owner directly.

Recommended actions for repository owners:
- Rotate credentials immediately if a secret is accidentally committed.
- Use secret scanning tools (GitHub secret scanning) and pre-commit hooks to prevent accidental commits.
- Limit service account permissions to the minimum required for BigQuery or other cloud services.

# Dependency Audit — AI_EDA_PLATFORM

Date: 2026-06-22

This document summarizes a dependency audit for the AI EDA Platform repository and provides a recommended, stable dependency strategy (requirements_clean.txt).

---

## 1) Third-party packages detected (import scan)

The codebase imports the following third-party packages:

- streamlit (UI)
- pandas (dataframes)
- numpy (numerical ops)
- scikit-learn (model training)
- plotly (visualizations)
- reportlab (PDF generation)
- joblib (model persistence)
- prophet (time-series forecasting)
- statsmodels (Holt-Winters fallback)
- google-cloud-bigquery (BigQuery client in `modules/bigquery_connector.py`)
- google.api-core, grpcio, protobuf, pyarrow (indirect/BigQuery related)
- python-dotenv (present in requirements but not actively used in modules)
- pandas-gbq (NOT imported anywhere today; optional)

Files scanned: `app.py` and all files under `modules/`.

---

## 2) Observations / detected issues

1. Current `requirements.txt` has had multiple edits and pins that created resolver backtracking during installs.
   - Example: `google-cloud-bigquery` moved between 3.42.x and 3.12.x in prior workflows which causes incompatibilities with `pandas-gbq` and newer `google-api-core` builds.

2. `python-dotenv` was added but is not referenced in the code. It is useful for a local `.env` workflow but is optional and should be kept as an opt-in dev dependency.

3. `pandas-gbq` is not used by the codebase. It provides convenience helpers for uploading pandas DataFrames to BigQuery. The code uses the BigQuery Python client directly (`google-cloud-bigquery`), so `pandas-gbq` is optional.

4. Core transitive conflicts to watch:
   - `google-cloud-bigquery` requires specific ranges of `google-api-core`, `grpcio`, and `protobuf`. Using an older bigquery client while newer pandas-gbq (or vice versa) can force pip to downgrade/upgrade these transitive deps.
   - `protobuf` major releases (4.x) frequently introduce breaking changes for some Google libraries. Use conservative ranges (>=4.21,<5.0) to avoid edge cases.

5. `pyarrow` is not directly imported but often required when transferring large tables to/from BigQuery or for parquet/arrow-based IO. Keep a compatible pyarrow pinned range.

---

## 3) Packages classified

- Required and used:
  - streamlit, pandas, numpy, scikit-learn, plotly, reportlab, joblib, prophet, statsmodels, google-cloud-bigquery, google-api-core, grpcio, protobuf, pyarrow

- Optional (not currently used in code):
  - pandas-gbq (convenience wrapper)
  - python-dotenv (env loader)

- Not present / not required:
  - Any other cloud clients besides BigQuery are not used.

---

## 4) Detected conflicts

- google-cloud-bigquery (older pinned releases like 3.12.0) are incompatible with some versions of `pandas-gbq` and `google-api-core` that expect newer APIs.
- Repeated pinning of `protobuf` or `grpcio` to exact versions causes resolver backtracking. Avoid exact pins unless necessary.

---

## 5) Recommendations (summary)

1. Use the curated ranges in `requirements_clean.txt`. These ranges aim to keep BigQuery, Prophet, Statsmodels, and scikit-learn working together while avoiding the resolver back-and-forth.

2. Prefer `google-cloud-bigquery>=3.42.0,<4.0`. Reason: newer bigquery client releases stabilize APIs and require modern `google-api-core` and `protobuf` versions. If you currently rely on a specific older client behavior, test before upgrading.

3. Keep `protobuf` in the `>=4.21.0,<5.0` range to avoid unexpected breaking changes from a hypothetical 5.x release.

4. Use `pip check` and `pipdeptree` (or `pip-compile` from `pip-tools`) to produce a lockfile. For production deployments, create a lockfile (`requirements-lock.txt`) via `pip-compile` and use `pip-sync` on servers.

5. Keep `pandas-gbq` optional. The application currently uses `google-cloud-bigquery` directly. If you prefer convenience `to_gbq`/`from_gbq` APIs, add `pandas-gbq` but ensure its required `google-cloud-bigquery` minimum is satisfied.

---

## 6) Final approved package versions (recommended ranges)

See `requirements_clean.txt` included with this audit. Key entries:

- streamlit>=1.18.0,<2.0
- pandas>=2.1.0,<3.0
- numpy>=2.2.0,<3.0
- scikit-learn>=1.2.2,<2.0
- prophet>=1.1.1,<2.0
- statsmodels>=0.14.0,<1.0
- google-cloud-bigquery>=3.42.0,<4.0
- google-api-core>=2.11.0,<3.0
- protobuf>=4.21.0,<5.0
- grpcio>=1.49.0,<2.0
- pyarrow>=11.0.0,<13.0.0

Rationale: these ranges are conservative but allow security updates and avoid forcing downgrades of transitive dependencies. They were chosen to be compatible with Prophet and Statsmodels while supporting BigQuery.

---

## 7) pandas-gbq vs google-cloud-bigquery — which do you need?

Short answer: You only strictly need `google-cloud-bigquery` for programmatic BigQuery operations. `pandas-gbq` is a convenience wrapper built on top of `google-cloud-bigquery`.

Advantages of `google-cloud-bigquery` (client library):
- Full control over BigQuery APIs (table creation, DDL/DML, streaming inserts, job management).
- Better for production-grade flows where you need explicit control of schemas, table partitioning, and job handling.
- Officially supported and receives updates in parallel with the Google Cloud Platform.

Advantages of `pandas-gbq`:
- Simple convenience functions: `to_gbq()` and `read_gbq()` that map pandas DataFrames to BigQuery tables quickly.
- Good for quick prototyping and one-off uploads without writing BigQuery client logic.

Recommendation for this project: keep `google-cloud-bigquery` as the canonical dependency (the code already uses it). Add `pandas-gbq` only if you want to offer a simpler DataFrame-based API; if added, ensure its required `google-cloud-bigquery` minimum version is satisfied.

---

## 8) Commands — fresh environment, install, verify

1) Create and activate a fresh virtual environment

- Windows (PowerShell):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

- macOS / Linux:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

2) Install the curated requirements (use the included `requirements_clean.txt`)

```bash
pip install -r requirements_clean.txt
```

3) Verify dependency resolution

```bash
pip check
pip install pipdeptree
pipdeptree --warn silence
```

4) (Optional) Create a locked environment for production using pip-tools

```bash
pip install pip-tools
pip-compile requirements_clean.txt --output-file=requirements-lock.txt
pip install pip-sync requirements-lock.txt
```

5) Test critical flows

- Start Streamlit and run a small training flow to ensure scikit-learn and Prophet work:

```bash
streamlit run app.py
```

- Verify BigQuery connectivity (use your service account env var):

```bash
python -c "from modules.bigquery_connector import connect_bigquery; print(connect_bigquery())"
```

---

## 9) Additional notes and next steps

- If you continue to experience resolver backtracking, produce a `pipdeptree` output and attach it to a follow-up ticket. I can analyze it and propose specific pins.
- For reproducible deployments, prefer lockfiles (`requirements-lock.txt`) or a packaging tool like `poetry`/`pipx`/containers.
- Consider CI job that runs `pip install -r requirements_clean.txt` and `pip check` to catch regressions early.

---

If you want, I can now:

- Produce a `requirements-lock.txt` using `pip-compile` from the current environment and the approved ranges, or
- Run a simulated install in an isolated environment to validate the resolver and report back with exact versions chosen by the resolver.

---

## Note on pyarrow

- `pyarrow` is optional for the currently verified application workflows (EDA, training, prediction, forecasting, PDF generation, and BigQuery Phase 1 uploads). The `pip-compile` step failed due to building/wheel issues for `pyarrow` on Windows in this environment.
- Recommendation: do not block development on `pyarrow`. If you need `pyarrow` later (parquet exports or advanced BigQuery storage paths), install a platform-compatible wheel or use `conda`:

  - Windows wheel install (match Python version):

    ```powershell
    pip install pyarrow==<suitable-version>-cpXXX-...whl
    ```

  - Or with conda:

    ```bash
    conda install -c conda-forge pyarrow
    ```

- For now we've kept `pyarrow` out of the lockfile; the `requirements_clean.txt` contains a compatible range and `pyarrow` can be added separately when needed.

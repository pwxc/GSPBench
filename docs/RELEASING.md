# Releasing GSPBench

This runbook is the source of truth for maintainers releasing GSPBench. It is
designed around token-free PyPI Trusted Publishing and must be followed in
order: CI, TestPyPI, clean installation, tag, GitHub Release, production PyPI,
then a second clean installation.

## Release infrastructure

| Purpose | Value |
| --- | --- |
| Repository | `pwxc/GSPBench` |
| Package name | `gspbench` |
| Workflow | `.github/workflows/publish.yml` |
| TestPyPI environment | `testpypi` |
| PyPI environment | `pypi` |
| TestPyPI project | <https://test.pypi.org/project/gspbench/> |
| PyPI project | <https://pypi.org/project/gspbench/> |

Both package indexes trust the GitHub repository, workflow filename, and
environment listed above. Renaming the repository, workflow, or environment
breaks OIDC publishing until the corresponding Trusted Publisher is updated.

The workflow behaves as follows:

- A manual `workflow_dispatch` builds and uploads to TestPyPI only.
- Publishing a GitHub Release builds and uploads to production PyPI only.
- Every build runs `twine check` before an upload.
- No long-lived PyPI API token is stored in GitHub or the repository.

## Security rules

- Keep 2FA enabled on PyPI and TestPyPI and store recovery codes offline.
- Never commit, paste into issues, or record passwords, TOTP seeds, recovery
  codes, API tokens, OIDC tokens, or browser session data.
- Treat changes to `publish.yml` as credential-sensitive changes. Review them
  before merging because that workflow has authority to publish the package.
- Use the dedicated `pypi` and `testpypi` environments. Configure environment
  reviewers in GitHub when more maintainers receive write access.
- A production filename and version cannot be replaced. A bad release must be
  yanked and followed by a new version; never try to overwrite it.

## 1. Prepare the version

Use semantic versioning. Update the version in all of these files:

- `pyproject.toml`
- `src/gspbench/__init__.py`
- `CITATION.cff`
- `CHANGELOG.md`

For releases that add or change packaged data, also complete every item below:

- Record each dataset's source, license status, notices, and raw/processed
  checksums in `src/gspbench/data/DATA_LICENSES.json`.
- Keep raw provider downloads and exploratory scripts out of Git.
- Regenerate reference bandlimitedness results on the final weighted graph.
- Confirm that official analysis uses the symmetric normalized Laplacian,
  does not center the signal, and retains mode zero at every spectral budget.
- Update README methodology, limitations, node counts, signal names, and
  citations when any of them change.

## 2. Validate locally

From a clean checkout or virtual environment:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python -m pytest
.venv/bin/python -m build
.venv/bin/python -m twine check dist/*
```

Inspect the wheel and source distribution before uploading. They must contain
the processed package data and license manifest, and must not contain raw NOAA
CSV files, notebooks, exploratory scripts, credentials, or local reports.

Commit and push `main`, then wait for `.github/workflows/ci.yml` to pass on all
supported Python versions.

## 3. Publish and verify TestPyPI

Open the repository's **Actions > Publish** page, choose **Run workflow**, and
run it from `main`. This invokes the `publish-testpypi` job.

TestPyPI is a separate index and does not reliably provide runtime
dependencies. Validate the uploaded wheel by installing dependencies from
PyPI and the package itself from TestPyPI:

```bash
python -m venv /tmp/gspbench-testpypi
/tmp/gspbench-testpypi/bin/python -m pip install numpy scipy scikit-learn
/tmp/gspbench-testpypi/bin/python -m pip install \
  --no-deps --index-url https://test.pypi.org/simple/ \
  gspbench==VERSION
```

In that clean environment, load every packaged dataset, run bandlimitedness,
and run at least one small benchmark. Confirm node counts, daily matrix shapes,
reference K values, and zero-mode retention.

Do not continue if this step fails. TestPyPI files cannot be overwritten under
the same version and filename; fix the problem, increment the version, and run
the complete validation again.

## 4. Tag and publish production

Only after the TestPyPI artifact passes:

```bash
git tag -a vVERSION -m "gspbench VERSION"
git push origin vVERSION
```

Create a GitHub Release from that exact tag. Release notes should summarize
datasets, graph construction, analysis or benchmark changes, licensing, and
the validation performed. Publishing the GitHub Release triggers the
`publish-pypi` job through OIDC.

The GitHub Release is a public announcement and production PyPI versions are
immutable. Get explicit maintainer confirmation immediately before clicking
**Publish release**.

## 5. Verify production

Wait for every production workflow job to finish, then verify the index and a
normal user installation:

```bash
python -m venv /tmp/gspbench-pypi
/tmp/gspbench-pypi/bin/python -m pip install \
  "gspbench[benchmarks]==VERSION"
```

Repeat the dataset, bandlimitedness, and benchmark smoke tests. Also verify:

- `https://pypi.org/pypi/gspbench/VERSION/json` reports the expected metadata.
- The wheel and sdist are both present and not yanked.
- The GitHub Release points to the intended commit and is marked latest.
- `main` is clean and synchronized with `origin/main`.

## Failure handling

- Before production upload: fix the issue, increment the version if TestPyPI
  already received files, and repeat the full process.
- After production upload: do not delete or overwrite artifacts. Yank the bad
  version with a clear reason, fix the issue, and publish a new patch release.
- If OIDC reports `invalid-publisher`, compare the owner, repository,
  `publish.yml` filename, environment name, and package name character for
  character with the Trusted Publisher configuration.
- If a workflow stalls, inspect individual job status before retrying; avoid a
  second upload attempt when the first upload may already have succeeded.

## Release record: 0.0.1

Version 0.0.1 was published on 2026-07-19. Both Trusted Publishers were
configured successfully, TestPyPI and production workflows passed, and clean
installs from both indexes loaded the US 144-node and Australia 126-node
datasets and ran the compression benchmark with mode zero retained.

# GSPBench Agent Notes

Read `docs/RELEASING.md` before changing versions, release automation, package
data, or publishing a release.

Read `docs/RWTH_HPC.md` before accessing RWTH CLAIX, transferring research
data, selecting a partition, or submitting a Slurm job.

- The package name is `gspbench`; the repository is `pwxc/GSPBench`.
- `publish.yml` is bound to PyPI Trusted Publishing. Its filename and the
  `pypi`/`testpypi` environment names are part of the authorization contract.
- Manual workflow dispatch publishes to TestPyPI. A published GitHub Release
  publishes to production PyPI.
- Never publish production before CI and a clean TestPyPI installation pass.
- Ask for explicit confirmation immediately before publishing a GitHub
  Release, because it triggers an immutable production PyPI upload.
- Never store or request passwords, TOTP seeds, recovery codes, API tokens, or
  OIDC tokens in the repository or conversation.
- Update versions in `pyproject.toml`, `src/gspbench/__init__.py`,
  `CITATION.cff`, and `CHANGELOG.md` together.
- Raw provider files and exploratory scripts do not enter Git. Every packaged
  dataset needs its own source, license decision, notices, and checksums in
  `src/gspbench/data/DATA_LICENSES.json`.
- Official spectral results use the weighted symmetric normalized Laplacian,
  do not subtract the signal mean, and always retain the zero-frequency mode.
- Do not reuse a version after files reach TestPyPI or PyPI. Fixes require a
  new version; a faulty production version should be yanked, not overwritten.
- Never ask for or record RWTH/HPC/VPN passwords, MFA codes, TOTP seeds,
  recovery codes, or private SSH keys. Let the user complete authentication
  locally and use a registered public key plus an unlocked SSH agent.
- Never run graph eigensolvers on a CLAIX login node. Test on `devel`, measure
  memory/time, and ask for explicit approval before submitting billed jobs.
- Do not assume CLAIX-2025 access: its compute partitions currently require an
  NHR or WestAI project. Discover allowed partitions with `r_wlm_usage -q`.
- Large graph analysis defaults to sparse partial or approximate spectral
  methods. Full dense eigendecomposition requires an explicit feasibility and
  memory review.

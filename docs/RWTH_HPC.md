# RWTH CLAIX HPC Preparation

This document prepares GSPBench graph-spectrum workloads for RWTH Aachen's
CLAIX cluster. The official documentation was last checked on 2026-07-19.
CLAIX-2025 is new and its pilot documentation is still changing, so partitions
and access rules must be checked again immediately before a real submission.

## Current system picture

CLAIX uses Slurm. Work is prepared on login nodes, submitted with `sbatch`, and
executed on allocated compute nodes. Heavy eigensolvers must never be run on a
login node.

There are currently two relevant compute generations:

- CLAIX-2023 is the practical default for general RWTH access. CPU partitions
  are `c23ms`, `c23mm`, and `c23ml`; `c23g` provides H100 GPUs. The `devel`
  partition is free, CPU-only, limited to one hour, and intended for small
  correctness tests without a project account.
- CLAIX-2025 has AMD CPU nodes in `c25ms` and `c25ml` and H100 GPU nodes in
  `c25g`. Compute-job access is currently restricted to NHR or WestAI projects,
  even though its login nodes can be accessed more broadly.

Do not assume that the newest cluster is automatically available. Select the
cluster from the user's approved project and `r_wlm_usage -q` output.

## Access requirements

The user needs all of the following before Codex can work on CLAIX:

1. An RWTH account and a separately registered HPC account.
2. HPC MFA configured in RegApp. RWTH IdM MFA and HPC MFA are separate.
3. Access from the RWTH/approved partner network, or an active RWTH VPN
   connection when connecting from an external network.
4. An approved computing-time project for substantial jobs. Basic/free time
   and `devel` are appropriate only for initial tests.
5. A suitable project ID and membership when project accounting or shared
   project storage will be used.

The normal SSH targets are:

```text
login23-1.hpc.itc.rwth-aachen.de   CLAIX-2023, Rocky 9, general use
login23-4.hpc.itc.rwth-aachen.de   CLAIX-2023, Rocky 9, general use
login25-1.hpc.itc.rwth-aachen.de   CLAIX-2025, general use
copy23-1.hpc.itc.rwth-aachen.de    large file transfers
```

Use a passphrase-protected SSH key registered through RegApp and unlock it in a
local SSH agent. The user may still need to complete password/MFA prompts
periodically. Do not set `PreferredAuthentications publickey` for CLAIX because
RWTH warns that it prevents the required password request.

## Information the user must provide

Safe to provide in the conversation:

- HPC username.
- RWTH role: student, doctoral researcher, employee, or senior scientist.
- Whether the HPC account, HPC MFA, and VPN are already working.
- Computing project type (`RWTH Small`, `NHR`, `WestAI`, `Thesis`, etc.),
  project ID, project end date, and whether the user is already a member.
- Output of `r_wlm_usage -q` and `r_quota` after login. These show allowed
  partitions, time/core limits, accounting, and storage capacity.
- Intended personal or project storage path, such as `$HPCWORK` or
  `/hpcwork/<project-id>`.
- Whether a dedicated public SSH key has been registered in RegApp.
- Dataset licenses or confidentiality restrictions that affect uploading.

Never provide in the conversation or repository:

- RWTH, HPC, or VPN passwords.
- MFA/TOTP codes, TOTP seeds, or recovery codes.
- Private SSH keys or SSH-agent contents.
- Personal access tokens or confidential dataset credentials.

When authentication is required, the user should complete it locally. Codex
may use the resulting authenticated SSH connection, registered public key, or
unlocked SSH agent, but must not ask the user to paste secrets.

For each graph-analysis run, also provide:

- Exact Git commit/tag to execute.
- Graph count, node count `N`, undirected edge count `E` or sparse `nnz`, data
  type, and number of graph signals per graph.
- Input format and current location: packaged dataset, local path, URL, or
  already present on CLAIX.
- Required result: full eigenspectrum, first `K` eigenpairs, K90/K95/K99,
  approximate spectral density, filtered signals, or benchmark outputs.
- Accuracy/tolerance requirements and whether approximate results are allowed.
- Desired deadline and an upper resource budget in core-hours/GPU-hours.
- Output format, retention period, and where results should be copied back.
- Explicit approval immediately before resource-consuming `sbatch` jobs are
  submitted.

## First authenticated discovery session

Do not hard-code a Python module or partition before checking the live account.
Run these commands after the first successful SSH login:

```bash
hostname
cat /etc/os-release
r_wlm_usage -q
r_quota
module reset
module spider Python
module spider PETSc
module spider SLEPc
module spider Apptainer
sinfo
```

Record only non-secret environment facts in the run log. `module spider` must
be repeated on CLAIX-2023 and CLAIX-2025 because their optimized software trees
and CPU architectures differ.

## Storage and transfer plan

- Keep source code, small configuration files, and important compact results
  in `$HOME`, which is backed up. Avoid heavy I/O there.
- Use `$HPCWORK` for large sparse matrices, cached eigenvectors, checkpoints,
  and high-throughput job I/O. It is not backed up.
- `$WORK` is persistent but not backed up and is suitable for intermediate
  work with lower I/O demand.
- Use project paths `/home/<project-id>`, `/work/<project-id>`, and
  `/hpcwork/<project-id>` when results must be shared with project members.
- Use `copy23-1` or `copy23-2` and `rsync`, `sftp`, or `rclone` for large
  transfers. Bundle many tiny files into an archive before transfer.
- Run `r_quota` before large uploads and back up irreplaceable results outside
  CLAIX. CLAIX is not a long-term archive.

The public GSPBench release can normally be installed from PyPI. Development
code should be transferred as a specific Git commit or immutable source
archive, not as an unrecorded working tree.

## Python environment

Prefer a clean `venv` built from a Python module discovered with
`module spider Python`. Avoid `pip --user` because it can mix packages across
toolchains. RWTH blocks Anaconda package repositories; if conda is genuinely
needed, use Miniforge/conda-forge instead.

For NumPy/SciPy workloads, record `numpy.show_config()` and
`scipy.show_config()` and verify the BLAS implementation. Match thread counts
to allocated Slurm CPUs:

```bash
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
```

Do not combine host and container virtual environments. Apptainer is the
fallback when a reproducible native environment cannot be built. Use the
cluster's optimized modules first for MPI and linear algebra.

## Graph-spectrum execution strategy

Full dense eigendecomposition is not the default for large graphs. One dense
float64 `N x N` array requires `8*N^2` bytes before eigensolver workspace: at
`N=100,000`, one array alone is about 80 GB. The compute cost grows cubically.

Use this escalation path:

1. Store adjacency and Laplacian matrices as sparse CSR/CSC.
2. For the lowest `K` frequencies, use a sparse iterative solver such as
   ARPACK through `scipy.sparse.linalg.eigsh` or LOBPCG. Save the basis once per
   graph and reuse it for all signals.
3. For very large graphs where only bandlimitedness summaries are required,
   use polynomial filtering, stochastic Lanczos quadrature, or another method
   that avoids materializing every eigenvector.
4. Use PETSc/SLEPc for a genuinely distributed eigenproblem after confirming
   live module availability and validating a small MPI job.
5. Use GPUs only after a GPU-capable sparse backend shows a measured benefit.
   A SciPy/ARPACK job does not become GPU-accelerated merely by requesting an
   H100.

Approximate sparse memory before submission. With float64 values and int32
indices, a CSR matrix is roughly `12*nnz + 4*(N+1)` bytes, while `K` stored
float64 eigenvectors require another `8*N*K` bytes. Solver workspaces and
copies add substantial overhead, so measure peak RSS on a representative
sample and request headroom rather than relying only on the formula.

For many independent graphs, use a Slurm job array. For many signals on one
graph, compute the eigensystem once and parallelize signal projections or
benchmarks without recomputing the basis.

## Slurm workflow

First run a small interactive correctness test on `devel` without an account:

```bash
salloc --partition=devel --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem-per-cpu=1000M --time=00:30:00
```

The production script must be generated from measured requirements. A starting
template is:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=gsp-spectrum
#SBATCH --account=<project-id>
#SBATCH --partition=<allowed-partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=<threads>
#SBATCH --mem=<measured-memory-with-headroom>
#SBATCH --time=<measured-walltime-with-headroom>
#SBATCH --output=<writable-log-dir>/%x-%j.out
#SBATCH --error=<writable-log-dir>/%x-%j.err

set -euo pipefail
module reset
module load <verified-python-module>
source <venv>/bin/activate

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

cd "$SLURM_SUBMIT_DIR"
srun python -m <entry-point> <arguments>
```

Use `sbatch job.sh`, `squeue --me`, `squeue --me --start`, and
`scancel <job-id>` to manage jobs. Capture the job ID, Git commit, arguments,
random seed, module list, `pip freeze`, and output checksums in every run
manifest.

Start with one node. SciPy's sparse eigensolvers do not automatically scale
across nodes. Request multiple nodes only for a tested PETSc/SLEPc or explicit
MPI implementation.

## Partition selection for this project

- Use `devel` only for short CPU correctness checks, without `--account`.
- Use `c23ms` when the measured memory fits its per-core and per-node limits.
- Escalate to `c23mm` or `c23ml` for memory-bound eigenproblems, subject to the
  account's allowed partitions.
- Use `c23g` only for a validated H100 implementation.
- Use `c25ms`, `c25ml`, or `c25g` only if the user has an NHR or WestAI
  allocation and after rebuilding/testing the environment for CLAIX-2025.

Partition limits and billing rules change. The official partitions page and
`r_wlm_usage -q` override this document.

## Official sources

- [RWTH HPC service documentation](https://help.itc.rwth-aachen.de/en/service/rhr4fjjutttf/)
- [First steps in accessing CLAIX](https://help.itc.rwth-aachen.de/en/service/rhr4fjjutttf/article/b7221149cca349e196cc0a28fcba8cc9/)
- [Terminal login](https://help.itc.rwth-aachen.de/en/service/rhr4fjjutttf/article/fb61d6c86ae245b5b7bba8c0cb7db6eb/)
- [Slurm partitions](https://help.itc.rwth-aachen.de/en/service/rhr4fjjutttf/article/9108f4a6f43c40a3a168919afd36839d/)
- [CLAIX-2025 introduction](https://help.itc.rwth-aachen.de/en/service/rhr4fjjutttf/article/783f1418454e4cac84e4a230060b351b/)
- [Computing-time project catalogue](https://help.itc.rwth-aachen.de/en/service/rhr4fjjutttf/article/45825b06afb647e194be4a5b9f5b8768/)
- [Submitting jobs](https://help.itc.rwth-aachen.de/en/service/rhr4fjjutttf/article/13ace46cfbb84e92a64c1361e0e4c104/)
- [Testing applications](https://help.itc.rwth-aachen.de/en/service/rhr4fjjutttf/article/5b115949502a40c0819005719a70e49e/)
- [File systems](https://help.itc.rwth-aachen.de/en/service/rhr4fjjutttf/article/da307ec2c60940b29bd42ac483fc3ea7/)
- [Python environments](https://help.itc.rwth-aachen.de/en/service/rhr4fjjutttf/article/7230ed5050e94aacbbe1db14cada5b56/)
- [Module system](https://help.itc.rwth-aachen.de/en/service/rhr4fjjutttf/article/450d33cc19fd4e50b1dd07027e9b55bd/)
- [Data transfer](https://help.itc.rwth-aachen.de/en/service/rhr4fjjutttf/article/db3e5fd39d1d42c9815b4fa689719ac9/)

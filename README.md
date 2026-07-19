# GSPBench

GSPBench is a small Python library for studying graph-signal bandlimitedness on
real weather observations and measuring how spectral compression affects
downstream tasks. Version 0.0.1 contains two processed 2025 NOAA GHCN-Daily
datasets:

| Dataset | Nodes | Winter signal | Summer signal |
| --- | ---: | --- | --- |
| `us_weather_2025` | 144 | January mean | July mean |
| `australia_weather_2025` | 126 | July mean | January mean |

The primary signals are raw absolute temperatures. GSPBench never subtracts
their graph-wide mean before the GFT, and every spectral truncation retains
the zero-frequency mode.

## Installation

```bash
pip install gspbench
```

Install the classical machine-learning benchmark dependencies with:

```bash
pip install "gspbench[benchmarks]"
```

## Loading data

```python
from gspbench import available_datasets, load_dataset

print(available_datasets())
dataset = load_dataset("us_weather_2025")

winter = dataset.signals["winter_temperature_midrange_c"]
daily = dataset.temporal_signals["daily_temperature_midrange_c"]
observed = dataset.temporal_signals["observation_mask"]

print(dataset.adjacency.shape)  # (144, 144), SciPy CSR
print(daily.shape)              # (365, 144), NaNs are retained
```

Each dataset also exposes station identifiers and names, latitude/longitude,
elevation, observation counts, Haversine edge distances, and the local scale
used by the edge-weight kernel.

## Weighted graph definition

Each node first selects its six nearest neighbors by Haversine distance. The
directed neighbor sets are combined into an undirected edge set. An included
edge receives the self-tuning weight

```text
w_ij = exp(-d_ij^2 / (sigma_i * sigma_j))
```

where `sigma_i` is the distance from node `i` to its sixth neighbor. The
official analyses use the symmetric normalized Laplacian. The combinatorial
Laplacian is also available:

```python
normalized = dataset.laplacian("normalized")
combinatorial = dataset.laplacian("combinatorial")
```

## Bandlimitedness

```python
from gspbench.analysis import bandlimitedness

result = bandlimitedness(dataset, winter)
print(result.effective_bandwidth)       # K90, K95, K99
print(result.zero_mode_energy_ratio)    # mode zero is included
print(result.auc_energy_concentration)
```

The result also contains the eigenvalues, GFT coefficients, per-mode energy,
cumulative energy, knee index, and graph total variation.

The packaged 0.0.1 reference results use the final weighted normalized graph:

| Dataset and signal | K95 | AUC-EC | Zero-mode energy |
| --- | ---: | ---: | ---: |
| US winter | 27 | 0.9570 | 0.1445 |
| US summer | 1 | 0.9953 | 0.9725 |
| Australia winter | 3 | 0.9942 | 0.8988 |
| Australia summer | 1 | 0.9961 | 0.9585 |

They can be loaded without recomputation:

```python
from gspbench.analysis import load_reference_results

reference = load_reference_results("us_weather_2025")
```

## Benchmarks

```python
from gspbench.benchmarks import run_benchmark

result = run_benchmark(
    "denoising",
    dataset="us_weather_2025",
    test_repeats=5,
)
print(result.summary)
```

Available tasks are:

- `denoising`: identity, Tikhonov, heat-kernel, and low-pass GFT baselines.
- `interpolation`: mean, geographic nearest-neighbor, Tikhonov, and
  bandlimited least-squares recovery.
- `compression`: graph low-pass, zero-mode-retaining GFT oracle, PCA, and
  random-projection controls.
- `season_classification`: four local seasons with grouped month folds and
  dummy, logistic, RBF-SVM, and random-forest models.
- `next_day_forecasting`: seven-day-to-next-day regression with persistence,
  ridge VAR, random forest, and graph-diffusion ridge baselines.
- `anomaly_detection`: controlled synthetic node perturbations over real
  daily signals, evaluated with robust Z-scores, Isolation Forest, and
  reconstruction residuals.

ML tasks compare full signals with graph Fourier, PCA, and Gaussian random
projection budgets. Feature scaling is fitted on training folds only. The
zero-frequency GFT coefficient remains present at every graph-spectral budget.

## Data processing

- Source: NOAA Global Historical Climatology Network - Daily, Version 3.
- Year: 2025.
- Daily value: `(TMAX + TMIN) / 2`, reported as temperature midrange rather
  than as an observed `TAVG`.
- Quality: blank GHCN `QFLAG`, at least 300 paired annual days, and at least
  25 paired days in January and July.
- Selection: deterministic 10 by 6 geographic grid, up to three of the most
  complete stations per occupied cell.
- Monthly signals: arithmetic mean over valid observations, without hidden
  imputation.
- Daily signals: missing observations remain `NaN` and are accompanied by an
  explicit boolean mask.

Raw NOAA files are not redistributed. Dataset-specific source URLs, checksums,
notices, and redistribution status are recorded in
`src/gspbench/data/DATA_LICENSES.json`.

## Limitations

The station observations are not homogenized climate normals. The graphs
encode geographic proximity, not atmospheric transport or causal weather
relationships. The anomaly labels are controlled synthetic perturbations on
real observations, not verified historical weather anomalies. Results should
be interpreted as benchmark measurements, not operational forecasts.

## Citation

Please cite GSPBench and the upstream dataset:

> Menne, M. J. et al. (2012). Global Historical Climatology Network - Daily
> (GHCN-Daily), Version 3. NOAA National Climatic Data Center.
> <https://doi.org/10.7289/V5D21VHZ>

## License

The GSPBench source code is BSD-3-Clause. Packaged data remain subject to the
upstream NOAA GHCN-Daily citation, use, and warranty notices described in the
dataset license manifest.

## Maintainer documentation

The repeatable CI, TestPyPI, Trusted Publishing, tagging, production release,
and rollback procedure is documented in [`docs/RELEASING.md`](docs/RELEASING.md).

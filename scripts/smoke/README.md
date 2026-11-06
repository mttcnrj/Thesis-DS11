# Smoke tests and compute benchmark

Run these on the chosen laptop before the full pipeline exists. They use
synthetic data with the same shape and quirks as the MMDA/PAGASA records, so
they need no Drive access.

| Script | What it does | Time on 2 reference cores |
| --- | --- | --- |
| `smoke_pipeline.py` | Every phase of Chapter III end to end (align → link analysis → tide → Granger → windows → hybrid, LSTM, ridge, KNN, logistic → RMSE, DM + Holm, PR-AUC, block bootstrap) plus 7 checks | ~30–50 s |
| `bench_compute.py` | Seconds per epoch for all 24 hybrid architecture × batch combinations, the standalone LSTM and the scikit-learn baselines; projects total hours for the experiment plan | ~5 min at 4,000 cases; longer at full size |
| `common.py` | Shared code: synthetic data, alignment, harmonic tide, windows, `build_hybrid`, pooled Granger, DM test, hardware info | — |

## 1. Pick the laptop (PowerShell, on both Windows laptops)

```powershell
Get-CimInstance Win32_VideoController | Select-Object Name
```

* Only NVIDIA GPUs count; TensorFlow cannot use Intel or AMD GPUs.
* Both NVIDIA: the higher-tier card wins. Neither: pick the laptop with more CPU cores
  (`Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores`).

## 2a. Chosen laptop HAS an NVIDIA GPU → WSL2

TensorFlow 2.10 was the last release with GPU support on native Windows, so
use WSL2.

```powershell
# administrator PowerShell, after updating the NVIDIA Windows driver
wsl --install -d Ubuntu      # then restart
```

```bash
# inside Ubuntu (VS Code: "WSL: Connect to WSL"); keep the repo in ~/, not /mnt/c
nvidia-smi                                   # must list the GPU
python3 -m venv .venv && source .venv/bin/activate
pip install "tensorflow[and-cuda]" statsmodels scikit-learn pandas pyarrow scipy psutil
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"   # must not be []

python scripts/smoke/smoke_pipeline.py --device cpu     # must print 7 PASS
python scripts/smoke/smoke_pipeline.py --device gpu     # must print 7 PASS
python scripts/smoke/bench_compute.py  --device cpu --measure-epochs 5
python scripts/smoke/bench_compute.py  --device gpu --measure-epochs 5
python scripts/smoke/bench_compute.py  --device gpu --measure-epochs 5 --no-determinism
```

## 2b. Chosen laptop has NO NVIDIA GPU → native Windows, CPU

```powershell
py -m venv .venv; .venv\Scripts\Activate.ps1
pip install tensorflow statsmodels scikit-learn pandas pyarrow scipy psutil
python scripts/smoke/smoke_pipeline.py --device cpu      # must print 7 PASS
python scripts/smoke/bench_compute.py  --device cpu --measure-epochs 5
```

## Tips

* Plugged in, power mode "Best performance", nothing else heavy running.
* Laptops slow down when hot: run the benchmark twice back to back and use the
  second run's `projection.csv`.
* For long jobs inside WSL, start them in `tmux` so they survive closing VS Code.

## Reading the output

* `runs/smoke_<device>_<size>/summary.json` – checks, alert levels, seconds per epoch, hardware
* `runs/bench_<device>[_nodet]/epoch_times.csv` – one row per architecture × batch
* `runs/bench_<device>[_nodet]/projection.csv` – hours for each block of the plan

Commit `summary.json` and `projection.csv` to `reports/compute/<laptop>/` so
Chapter III can state the hardware and run time, and fill the
"Chosen laptop hours" column of the plan's Section 11.

## Knobs

* `--expected-epochs` (default 60): mean epochs per run with early stopping
  (the October prototype stopped between 32 and 73).
* `--seeds` (default 5): seeds per final configuration.
* `--target-cases` (default 16,158): training cases at lead 1 from Table III-6;
  replace with the recomputed count after Phase 1.
* `--train-cases N`: train on N cases and rescale (only to save time on a slow CPU).

# Laptop runbook (chosen laptop: Ryzen 7 7735HS, 16 GB, RTX 4060 Laptop GPU 8 GB; G0)

Order: 0 setup → 1 smoke test → 2 Phases 1–4 → 3 Notebook 05 (stop at G16) → 4 Notebook 06 (once) →
5 Notebook 07 → 6 commit and report. Each step names what to commit.

Everything that produces final results runs here (D16). Phases 1–4 were built and run in a cloud container;
they are deterministic, so re-running them here must reproduce the committed gates and tables.

## 0. One-time setup (WSL2 + GPU TensorFlow, Section 2.2)

After `wsl --install` and the clone, `bash scripts/setup_wsl.sh` does every step of this section (venv on Python
3.13, the pinned `requirements.txt`, the `libcusolver` link, a GPU check, the fast tests); `--check` only verifies
an existing setup, `--rebuild` recreates `~/.venv`. The manual steps below are what it runs.

```powershell
# administrator PowerShell, after updating the NVIDIA Windows driver
wsl --install -d Ubuntu      # restart when asked
```

```bash
# inside Ubuntu (VS Code: "WSL: Connect to WSL"); keep the repo in ~/, not /mnt/c
nvidia-smi                                      # must list the RTX 4060 (8188 MiB, driver 595.97 in Windows; G0)
sudo apt-get update && sudo apt-get install -y python3-venv tmux git make
git clone --branch feat/pasig-forecast-pipeline https://github.com/mttcnrj/Thesis-DS11.git ~/pasig-forecast
cd ~/pasig-forecast

# Python 3.13 in ~/.venv: TensorFlow 2.21 has wheels for 3.10-3.13 only, and Ubuntu 26.04 ships 3.14.
# uv fetches 3.13 under ~/.local/share/uv (no sudo, no change to the system Python).
python3 -m venv ~/.venv
~/.venv/bin/python -m pip install --target ~/.local/share/uv-bootstrap uv
UV=~/.local/share/uv-bootstrap/bin/uv
$UV python install 3.13
$UV venv --clear --seed --python 3.13 ~/.venv
source ~/.venv/bin/activate

# WSL's /tmp is a RAM disk smaller than the unpacked CUDA packages: keep pip's temp files on disk
export TMPDIR=~/.cache/pip-tmp && mkdir -p "$TMPDIR"
pip install -e . -r requirements-dev.txt "tensorflow[and-cuda]" joblib
rm -rf "$TMPDIR" && unset TMPDIR

# TensorFlow 2.21's library search path omits nvidia/cusolver/lib; link the file into a folder it does search
SP=$(python -c "import site; print(site.getsitepackages()[0])")
ln -sfn ../../cusolver/lib/libcusolver.so.11 "$SP/nvidia/cusparse/lib/libcusolver.so.11"

python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"   # GPUs must not be []
make test                                       # fast tests, < 1 min
```

Activate with `source ~/.venv/bin/activate` in every new shell. In VS Code, open the `~/pasig-forecast` folder
through "WSL: Connect to WSL" and select `~/.venv/bin/python` as the interpreter and notebook kernel.

| Symptom | Cause | Fix |
|---|---|---|
| `nvidia-smi` fails inside Ubuntu | Windows driver or WSL out of date | Update the NVIDIA Windows driver (do **not** install a Linux driver inside WSL) and run `wsl --update` in PowerShell |
| `GPUs: []` with `tensorflow-intel` in `pip list` | the Python is native Windows; TensorFlow after 2.10 has no GPU support there | Use `~/.venv` inside WSL, not a venv in the Windows folder |
| `No matching distribution found for tensorflow` | the venv's Python is 3.14 or newer | Rebuild `~/.venv` on 3.13 (steps above) |
| `No space left on device` during `pip install` | pip is unpacking into the small `/tmp` RAM disk | Set `TMPDIR` as above |
| `GPUs: []` and `Cannot dlopen some GPU libraries`, although `nvidia-smi` works | a CUDA library is not on TensorFlow's search path; reinstalling does not help | Create the `libcusolver` link above, again after any venv rebuild. To see which library failed: `TF_CPP_VMODULE=dso_loader=2 python -c "import tensorflow as tf; tf.config.list_physical_devices('GPU')" 2>&1 \| grep "Could not load"` |

If the GPU still cannot be used, the pipeline runs on CPU: G13 records `device: cpu` and G15 switches to 3 seeds
if the projection exceeds 10 h.

If anything about the machine changes (another laptop, new driver), re-record G0:
`python scripts/record_g0.py --laptop "<name>|<GPU names>|<nvidia-smi name, memory>|<cores>" --choose <name> --ram-gb 16 --driver "<driver, CUDA>"`.

## 1. Smoke test and benchmark (Section 12) — do this as early as possible

```bash
make smoke                                                      # must print 7 PASS
python scripts/smoke/smoke_pipeline.py --device gpu             # must print 7 PASS
```

Notebook 05's `compute` stage runs the benchmark itself (twice per device) and records G13–G15.

## 2. Raw data and Phases 1–4

The raw data come with the clone (`data/raw/`, decision R1); no Google Drive step.

```bash
make data-check                                                 # 465 raw files match the manifest (SHA-256)
python scripts/run_phase.py --phase 1 --tag
python scripts/run_phase.py --phase 2
python scripts/run_phase.py --phase 3
python scripts/run_phase.py --phase 4
make test-data                                                  # tests on the real processed files
git diff --stat gates configs reports/tables                    # decisions must match the committed ones
pip freeze > requirements.txt                                   # commit; never upgrade mid-thesis
```

**Expected differences on this laptop.** The laptop's library versions differ from the cloud container's, so the
rebuild prints `WARNING: the rebuild changed committed results` for the tide files: `artifacts/tide_*.json`,
`gates/G09_tide_fit.json`, `reports/tables/tide_constituents.csv` and `tide_check_2023.json` differ around the 11th
decimal (predicted tide by about 1e-14 m), and the PNG figures differ in their bytes. This is floating-point noise
(F33). What must match is every decision: `configs/`, `gates/G08*`, `gates/G10*`, `artifacts/scalers.json`,
`reports/tables/table_III-6_recomputed.csv`, `granger_results.csv` and `windows_summary.csv`. Commit the laptop's
versions, because G17 hashes the files on this machine.

## 3. Notebook 05 (long: run inside tmux; resumable)

Plugged in, power mode "Best performance", sleep "Never", Windows Update paused (Section 2.3).

```bash
tmux new -s nb05
source ~/.venv/bin/activate
python scripts/run_phase.py --phase 5 --stage compute     # G14 -> G13 -> G15 (~30-60 min of benchmarks)
python scripts/run_phase.py --phase 5 --stage variants    # G11, G12
python scripts/run_phase.py --phase 5 --stage grid        # 60 hybrid + 18 LSTM configs (the long one)
python scripts/run_phase.py --phase 5 --stage finals      # 5 seeds each, determinism on
python scripts/run_phase.py --phase 5 --stage baselines
python scripts/run_phase.py --phase 5 --stage thresholds
python scripts/run_phase.py --phase 5 --stage checks      # G16 — REVIEW before freezing
python scripts/run_phase.py --phase 5 --stage freeze --tag   # G17, tag settings-frozen
```

(`python scripts/run_phase.py --phase 5 --tag` runs all stages in order; any stage can be re-run and skips
finished configs/seeds.) Detach with `Ctrl-b d`, reattach with `tmux attach -t nb05`.

If G16 reports "beats persistence: no" for the hybrid or LSTM, work through Section 9.7 before freezing.

Commit afterwards: `gates/`, `configs/` (compute, model_inputs, thresholds, frozen_manifest.json),
`reports/compute/<laptop>/`, `reports/tables/`. `artifacts/frozen/` stays on the laptop (hashed in the manifest).

## 4. Notebook 06 (opens the 2024 test season ONCE; G18)

Only after G16 has been reviewed, G17 passed and `settings-frozen` is tagged and pushed. Do not edit anything listed
in `configs/frozen_manifest.json` after the tag: G18 refuses to open the test if a hash differs.

```bash
git status                                                  # clean; HEAD contains settings-frozen
python scripts/run_phase.py --phase 6                       # unseal (G18) -> predict -> evaluate (~10 min, CPU)
```

If a stage crashes after `unseal`, fix the code and re-run from that stage (`--stage predict` or `--stage evaluate`);
this does not re-open the test. Re-running `unseal` needs `--rerun-reason "..."`, and that reason goes in
`reports/transparency_log.md`. Commit `gates/G18_*.json`, `gates/TEST_UNSEALED.json`, `reports/tables/test_*`,
`reports/figures/test_*` (the prediction parquets are a few MB; commit them too).

## 5. Notebook 07 (after G18; long — run inside tmux; resumable)

```bash
tmux new -s nb07
source ~/.venv/bin/activate
python scripts/run_phase.py --phase 7 --stage walkforward   # 3 folds x seeds (G15) + baselines
python scripts/run_phase.py --phase 7 --stage ablations     # D20 weight search (3 runs) + 6 variants x seeds
python scripts/run_phase.py --phase 7 --stage casestudy     # every 2024 flood-risk event, figures
python scripts/run_phase.py --phase 7 --stage aviles        # Aviles June-15 Jul 2024, descriptive
```

(`python scripts/run_phase.py --phase 7` runs all four and then the tests.) Finished runs are skipped on a
re-run (`runs/walkforward.csv`, `runs/ablations.csv`). Commit `reports/tables/` and `reports/figures/case_event_*`.

## 6. Commit and report

```bash
python -m pytest -q                                          # everything must pass (112 passed, 0 skipped)
make report                                                  # Chapter IV tables + figures -> reports/chapter4/
git add gates configs reports artifacts requirements.txt && git commit -m "Laptop run: ..." && git push
```

* Write every outcome into `reports/transparency_log.md` (new F-numbers) and the "Outcome" column of
  `docs/decisions.md`, whichever way it falls — including "beats persistence: no" or a hybrid that loses.
* `data/interim`, `data/processed`, `runs/` and `artifacts/frozen/` are git-ignored and stay on the laptop. `data/raw` is
  already committed (R1) and is never edited.
* `make report` writes `reports/chapter4/chapter4.md`: every Chapter IV table (with a full-precision CSV beside
  each), the H3/H4 verdicts, the figure list, and for Chapter III the hardware and the measured training time
  summed from the run folders' `timing.json` (so run it on the laptop, where `runs/` exists). It only reads
  finished outputs; re-run it after any change to `reports/tables/`.
* Back up `artifacts/frozen/` outside WSL (e.g. `cp -r artifacts/frozen /mnt/c/Users/<you>/pasig_frozen_backup`):
  it is not in git, and G18 checks its hashes.

Done on 10–11 Nov 2026: steps 0–6 (F33–F36).

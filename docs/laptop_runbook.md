# Laptop runbook (chosen laptop: Ryzen 7 7735HS, 16 GB, RTX 4060 Laptop GPU 8 GB; G0)

Order: 0 setup → 1 smoke test → 2 Phases 1–4 → 3 Notebook 05 (stop at G16) → 4 Notebook 06 (once) →
5 Notebook 07 → 6 commit and report. Each step names what to commit.

Everything that produces final results runs here (D16). Phases 1–4 were built and run in a cloud container;
they are deterministic, so re-running them here must reproduce the committed gates and tables.

## 0. One-time setup (WSL2 + GPU TensorFlow, Section 2.2)

```powershell
# administrator PowerShell, after updating the NVIDIA Windows driver
wsl --install -d Ubuntu      # restart when asked
```

```bash
# inside Ubuntu (VS Code: "WSL: Connect to WSL"); keep the repo in ~/, not /mnt/c
nvidia-smi                                      # must list the RTX 4060 (8188 MiB, driver 595.97 in Windows; G0)
sudo apt-get update && sudo apt-get install -y python3-venv tmux git
git clone https://github.com/mttcnrj/Thesis-DS11.git ~/pasig-forecast && cd ~/pasig-forecast
git checkout feat/pasig-forecast-pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install -e . -r requirements-dev.txt "tensorflow[and-cuda]" joblib
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"   # must not be []
make test                                       # fast tests, < 1 min
```

If `nvidia-smi` fails inside Ubuntu, update the NVIDIA Windows driver (do **not** install a Linux driver inside
WSL) and run `wsl --update` in PowerShell. If TensorFlow lists no GPU although `nvidia-smi` works, reinstall
`"tensorflow[and-cuda]"` in the venv. If the GPU still cannot be used, the pipeline runs on CPU: G13 records
`device: cpu` and G15 switches to 3 seeds if the projection exceeds 10 h.

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

## 3. Notebook 05 (long: run inside tmux; resumable)

Plugged in, power mode "Best performance", sleep "Never", Windows Update paused (Section 2.3).

```bash
tmux new -s nb05
source .venv/bin/activate
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
source .venv/bin/activate
python scripts/run_phase.py --phase 7 --stage walkforward   # 3 folds x seeds (G15) + baselines
python scripts/run_phase.py --phase 7 --stage ablations     # D20 weight search (3 runs) + 6 variants x seeds
python scripts/run_phase.py --phase 7 --stage casestudy     # every 2024 flood-risk event, figures
python scripts/run_phase.py --phase 7 --stage aviles        # Aviles June-15 Jul 2024, descriptive
```

(`python scripts/run_phase.py --phase 7` runs all four and then the tests.) Finished runs are skipped on a
re-run (`runs/walkforward.csv`, `runs/ablations.csv`). Commit `reports/tables/` and `reports/figures/case_event_*`.

## 6. Commit and report

```bash
python -m pytest -q                                          # everything must pass
git add gates configs reports requirements.txt && git commit -m "Laptop run: ..." && git push
```

* Write every outcome into `reports/transparency_log.md` (new F-numbers) and the "Outcome" column of
  `docs/decisions.md`, whichever way it falls — including "beats persistence: no" or a hybrid that loses.
* Never commit `data/raw`, `data/interim`, `data/processed`, `runs/` or `artifacts/frozen/` (git-ignored).
* Chapter III needs the hardware and measured run times from `reports/compute/<laptop>/` and the run folders'
  `timing.json`; Chapter IV tables come from `reports/tables/test_*`, `walkforward_*`, `ablation_*`,
  `case_study_*`, `aviles_check*`.

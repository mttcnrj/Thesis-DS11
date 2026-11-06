# Laptop runbook (chosen laptop: Ryzen 7 7735HS, 16 GB, RTX 4060 Laptop GPU; G0)

Everything that produces final results runs here (D16). Phases 1–4 were built and run in a cloud container;
they are deterministic, so re-running them here must reproduce the committed gates and tables.

## 0. One-time setup (WSL2 + GPU TensorFlow, Section 2.2)

```powershell
# administrator PowerShell, after updating the NVIDIA Windows driver
wsl --install -d Ubuntu      # restart when asked
```

```bash
# inside Ubuntu (VS Code: "WSL: Connect to WSL"); keep the repo in ~/, not /mnt/c
nvidia-smi                                      # must list the RTX 4060; note the memory for G0
sudo apt-get update && sudo apt-get install -y python3-venv tmux git
git clone https://github.com/mttcnrj/Thesis-DS11.git ~/pasig-forecast && cd ~/pasig-forecast
git checkout feat/pasig-forecast-pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install -e . -r requirements-dev.txt "tensorflow[and-cuda]" joblib
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"   # must not be []
```

## 1. Smoke test and benchmark (Section 12) — do this as early as possible

```bash
make smoke                                                      # must print 7 PASS
python scripts/smoke/smoke_pipeline.py --device gpu             # must print 7 PASS
```

Notebook 05's `compute` stage runs the benchmark itself (twice per device) and records G13–G15.

## 2. Raw data and Phases 1–4

```bash
python scripts/sync_drive.py --src "/mnt/g/My Drive/Thesis II"  # mount G: first if needed: sudo mount -t drvfs G: /mnt/g
python scripts/run_phase.py --phase 1 --tag
python scripts/run_phase.py --phase 2
python scripts/run_phase.py --phase 3
python scripts/run_phase.py --phase 4
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

## 4. Notebooks 06 and 07

Notebook 06 opens the 2024 test season once (G18). Start it deliberately, after G17.

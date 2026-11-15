# Setup (Windows laptop)

## One-time

1. Install **Python 3.11 or 3.12** from python.org. Check **"Add python.exe to PATH"** on the first installer screen.
2. Open the repo folder in VS Code, then open a terminal (`` Ctrl+` ``).
3. Create and activate a virtual environment, then install the project:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

If PowerShell blocks the activate script, run this once, then try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

The install pulls PyTorch and is a few hundred MB, so give it a few minutes.

## Run Milestone 1 (webcam detection)

```powershell
python -m csop.detect_webcam
```

The first run downloads the model weights (`yolo11n.pt`, ~5 MB). A window opens with labeled boxes and an FPS counter. Press **q** with the window focused to quit.

Useful options:

| Option | What it does |
|---|---|
| `--source 1` | use a second camera, or pass a video file path |
| `--conf 0.6` | only show detections at 60%+ confidence |
| `--imgsz 320` | faster but less accurate |
| `--model yolo11s.pt` | bigger, slower, more accurate model |
| `--log runs/dets.csv` | save every detection to a CSV |

## Run the tests

```powershell
pytest
```

## Note on drones

The pretrained model knows 80 everyday object classes (person, cup, bird, airplane, kite, ...) but has **no "drone" class**. Point it at a drone and you'll likely see "bird," "airplane," "kite," or nothing. Teaching it "drone" means training on a drone dataset, which is a later milestone.

# Quick Start Commands

## Install Everything
```powershell
# Install Python packages
python -m pip install -r requirements.txt
```

## Download Datasets (Already Done ✅)
```powershell
$env:PYTHONPATH = "$PWD"
C:/Users/jagdi/Care++/.venv/Scripts/python.exe scripts/download_datasets.py --all
```

## Train Models

### 1. X-ray Classifier (After preparing train/val folders)
```powershell
$env:PYTHONPATH = "$PWD"
C:/Users/jagdi/Care++/.venv/Scripts/python.exe scripts/train_xray.py `
  --train_dir datasets/unifesp-bodypart/train `
  --val_dir datasets/unifesp-bodypart/val `
  --num_classes 6 `
  --epochs 10 `
  --batch_size 16
```

### 2. T5 Summarizer (After preparing CSV)
```powershell
$env:PYTHONPATH = "$PWD"
C:/Users/jagdi/Care++/.venv/Scripts/python.exe scripts/train_summarizer.py `
  --csv datasets/mimic-iii-demo/notes.csv `
  --output_dir t5_medical
```

## Run API

### Test Mode (Mocks)
```powershell
$env:USE_MOCKS = "1"
C:/Users/jagdi/Care++/.venv/Scripts/python.exe app.py
```

### Production Mode (Real Models)
```powershell
$env:USE_MOCKS = "0"
C:/Users/jagdi/Care++/.venv/Scripts/python.exe app.py
```

## Test Endpoints

```powershell
# Test X-ray
curl -X POST -F "file=@test_xray.jpg" http://localhost:5000/analyze_xray

# Test Summary
curl -X POST -F "file=@test_report.pdf" http://localhost:5000/summarize_report

# Test Drug Safety
curl -X POST -H "Content-Type: application/json" `
  -d '{\"current_meds\": [\"Aspirin\"], \"new_drug\": \"Ibuprofen\"}' `
  http://localhost:5000/check_safety
```

# Care++ Setup & Execution Guide

## ✅ Downloaded Datasets

All datasets are now in `datasets/`:
- **roco-dataset** - Radiology reports + images (6.19GB)
- **mura-v11** - MURA musculoskeletal X-rays (3.14GB)
- **spinal-lesions** - Annotated spinal lesions (9.91GB)
- **nih-chest-xrays** - NIH Chest X-rays (42GB)
- **drug-drug-interactions** - Drug interactions CSV
- **mimic-iii-demo** - Clinical notes demo
- **unifesp-bodypart** - Body part classification X-rays

## 📦 Installation

### 1. Install Python Dependencies
```powershell
python -m pip install -r requirements.txt
```

### 2. Install External Tools (Windows)

**Tesseract OCR** (for PDF text extraction):
- Download: https://github.com/UB-Mannheim/tesseract/wiki
- Install and add to PATH
- Or set in code: `DocumentLoader(tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe')`

**Poppler** (for PDF to image conversion):
- Download: https://github.com/oschwartz10612/poppler-windows/releases
- Extract and add `bin/` folder to PATH

## 🚀 Training Models

### Vision Model (X-ray Classification)

**Prepare Data Structure:**
```
datasets/unifesp-bodypart/
├── train/
│   ├── chest/
│   ├── abdomen/
│   ├── hand/
│   └── ...
└── val/
    └── (same structure)
```

**Train Command:**
```powershell
$env:PYTHONPATH = "$PWD"
C:/Users/jagdi/Care++/.venv/Scripts/python.exe scripts/train_xray.py --train_dir datasets/unifesp-bodypart/train --val_dir datasets/unifesp-bodypart/val --num_classes 6 --epochs 10 --batch_size 16
```

**Output:** `resnet50_chest_xray.pth` (model weights)

### Summarizer Model (T5)

**Prepare CSV:**
Create `datasets/mimic-iii-demo/notes.csv` with columns:
- `full_report` - Original clinical text
- `summary` - Short summary

**Train Command:**
```powershell
$env:PYTHONPATH = "$PWD"
C:/Users/jagdi/Care++/.venv/Scripts/python.exe scripts/train_summarizer.py --csv datasets/mimic-iii-demo/notes.csv --output_dir t5_medical
```

**Output:** `t5_medical/` folder with trained model

## 🌐 Running the API

### Mock Mode (Test without trained models):
```powershell
$env:USE_MOCKS = "1"
C:/Users/jagdi/Care++/.venv/Scripts/python.exe app.py
```

### Production Mode (Use trained models):
```powershell
$env:USE_MOCKS = "0"
C:/Users/jagdi/Care++/.venv/Scripts/python.exe app.py
```

Server runs at: http://localhost:5000

## 📡 API Endpoints

### 1. Analyze X-ray
```bash
POST http://localhost:5000/analyze_xray
Content-Type: multipart/form-data
file: <xray_image.jpg>
```

**Response:**
```json
{"body_part": "Chest", "finding": "Pneumonia", "confidence": 0.95}
```

### 2. Summarize Report
```bash
POST http://localhost:5000/summarize_report
Content-Type: multipart/form-data
file: <report.pdf>
```

**Response:**
```json
{"original_text_len": 1500, "summary": "Patient presents with..."}
```

### 3. Check Drug Safety
```bash
POST http://localhost:5000/check_safety
Content-Type: application/json

{
  "current_meds": ["Aspirin", "Warfarin"],
  "new_drug": "Ibuprofen"
}
```

**Response:**
```json
{
  "is_safe": false,
  "warnings": [
    "Warning: Co-administration of Warfarin and Ibuprofen may lead to potential interaction. Risk level is 0.87."
  ]
}
```

## 🧪 Testing with cURL

```powershell
# Test X-ray analysis
curl -X POST -F "file=@path/to/xray.jpg" http://localhost:5000/analyze_xray

# Test summarization
curl -X POST -F "file=@path/to/report.pdf" http://localhost:5000/summarize_report

# Test drug safety
curl -X POST -H "Content-Type: application/json" -d '{\"current_meds\": [\"Aspirin\"], \"new_drug\": \"Ibuprofen\"}' http://localhost:5000/check_safety
```

## 📊 Dataset-to-Model Mapping

| Module | Model | Dataset | Purpose |
|--------|-------|---------|---------|
| Vision Router | ResNet50 | unifesp-bodypart | Classify body parts |
| Disease Detector | ResNet50 | nih-chest-xrays | Detect chest diseases |
| Summarizer | T5-Small | mimic-iii-demo | Summarize clinical notes |
| Safety Engine | BioBERT | drug-drug-interactions | Detect interactions |

## 🔧 Troubleshooting

**Import errors:**
```powershell
$env:PYTHONPATH = "$PWD"
```

**GPU not detected:**
- Check: `nvidia-smi` (NVIDIA GPU)
- CPU fallback is automatic

**Tesseract errors:**
```python
loader = DocumentLoader(tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe')
```

## 📁 Project Structure
```
Care++/
├── app.py                  # Flask API
├── data_loader.py          # OCR & preprocessing
├── vision_model.py         # ResNet50 classifier
├── summarizer.py           # T5 summarizer
├── safety_engine.py        # BioBERT DDI detector
├── explain_logic.py        # Explanation logic
├── pipeline.py             # High-level wiring
├── config.py               # Paths config
├── datasets.py             # Dataset metadata
├── requirements.txt        # Dependencies
├── scripts/
│   ├── download_datasets.py
│   ├── train_xray.py
│   └── train_summarizer.py
├── datasets/               # Downloaded data
└── models/                 # Trained weights
```

## ⚠️ Notes

- **HIPAA Compliance:** Ensure proper data handling for real patient data
- **DrugBank:** Download manually from https://go.drugbank.com/releases/latest
- **GPU Recommended:** Training on CPU will be slow
- **NIH Dataset:** 42GB - Consider subset for faster training

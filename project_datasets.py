from pathlib import Path
from typing import Dict

from config import DATASETS_DIR

DATASETS: Dict[str, Dict[str, str]] = {
    "roco": {
        "slug": "virajbagal/roco-dataset",
        "folder": "roco-dataset",
        "type": "text+image",
        "notes": "Radiology report and image dataset. Check license terms on Kaggle.",
    },
    "mura": {
        "slug": "cjinny/mura-v11",
        "folder": "mura-v11",
        "type": "xray",
        "notes": "MURA musculoskeletal radiographs. Requires agreement on Kaggle.",
    },
    "spine_lesions": {
        "slug": "siddhale937e92739/annotated-medical-image-dataset-for-spinal-lesions",
        "folder": "spinal-lesions",
        "type": "xray",
        "notes": "Annotated spinal lesion images. Verify folder structure after download.",
    },
    "nih_chest": {
        "slug": "nih-chest-xrays/data",
        "folder": "nih-chest-xrays",
        "type": "xray",
        "notes": "NIH Chest X-ray dataset. Large; consider selective download.",
    },
    "drug_interactions": {
        "slug": "mghobashy/drug-drug-interactions",
        "folder": "drug-drug-interactions",
        "type": "table",
        "notes": "Drug-drug interactions CSV for future fine-tuning.",
    },
    "mimic_demo": {
        "slug": "montassarba/mimic-iii-clinical-database-demo-1-4",
        "folder": "mimic-iii-demo",
        "type": "text",
        "notes": "MIMIC-III demo. Ensure HIPAA compliance for real data.",
    },
    "unifesp_bodypart": {
        "slug": "felipekitamura/unifesp-xray-bodypart-classification",
        "folder": "unifesp-bodypart",
        "type": "xray",
        "notes": "Body part classification dataset; good for router model.",
    },
}


def get_dataset_dir(key: str) -> Path:
    info = DATASETS.get(key)
    if not info:
        raise KeyError(f"Unknown dataset key: {key}")
    return DATASETS_DIR / info["folder"]


def list_datasets() -> Dict[str, Dict[str, str]]:
    return DATASETS

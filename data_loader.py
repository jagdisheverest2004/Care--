import os
from typing import Optional

import numpy as np
import cv2
from pdf2image import convert_from_path
import pytesseract
from PIL import Image


class DocumentLoader:
    def __init__(self, tesseract_cmd: Optional[str] = None) -> None:
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        pages = convert_from_path(pdf_path)
        ocr_text_chunks = []

        for page in pages:
            page_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(page_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(thresh)
            ocr_text_chunks.append(text)

        return "\n".join(ocr_text_chunks).strip()

    def preprocess_xray(self, image_path: str) -> np.ndarray:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        image = image.astype(np.float32) / 255.0

        # Convert to (C, H, W) for PyTorch
        image = np.expand_dims(image, axis=0)
        return image

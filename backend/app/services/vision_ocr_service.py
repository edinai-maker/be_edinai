"""
Google Cloud Vision API service for OCR and document analysis.

This module provides functions to extract text from PDFs and images using 
Google Cloud Vision API, with support for multi-page documents and automatic
language detection.
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.cloud import vision
from google.oauth2 import service_account
from pdf2image import convert_from_path
from PIL import Image

logger = logging.getLogger("uvicorn.error").getChild("vision_ocr_service")

# Configuration
CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ".json")
USE_VISION_API = os.getenv("USE_VISION_API", "false").lower() == "true"


class VisionOCRService:
    """Service for Google Cloud Vision API operations."""

    def __init__(self, credentials_path: Optional[str] = None):
        """
        Initialize Vision API client.

        Args:
            credentials_path: Path to Google Cloud service account JSON file.
                            If None, uses GOOGLE_APPLICATION_CREDENTIALS env var.
        
        Raises:
            FileNotFoundError: If credentials file is not found.
            RuntimeError: If Vision API client initialization fails.
        """
        self.credentials_path = credentials_path or CREDENTIALS_PATH
        
        if not os.path.exists(self.credentials_path):
            raise FileNotFoundError(
                f"Google Cloud credentials file not found: {self.credentials_path}"
            )
        
        try:
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path
            )
            self.client = vision.ImageAnnotatorClient(credentials=credentials)
            logger.info("Google Vision API client initialized successfully.")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize Vision API client: {exc}"
            ) from exc

    def extract_text_from_image(
        self, 
        image_path: str | Path,
        language_hints: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Extract text from a single image using Vision API.

        Args:
            image_path: Path to the image file.
            language_hints: Optional list of language codes (e.g., ['en', 'hi', 'gu'])
                          to hint the OCR engine.

        Returns:
            Dictionary containing:
                - text: Extracted text content
                - confidence: Overall confidence score (0-1)
                - language: Detected language code
                - annotations: Full Vision API response for advanced usage

        Raises:
            FileNotFoundError: If image file doesn't exist.
            RuntimeError: If Vision API call fails.
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            # Read image file
            with io.open(str(image_path), "rb") as image_file:
                content = image_file.read()

            image = vision.Image(content=content)

            # Configure image context with language hints
            image_context = None
            if language_hints:
                image_context = vision.ImageContext(language_hints=language_hints)

            # Perform text detection
            response = self.client.document_text_detection(
                image=image,
                image_context=image_context
            )

            if response.error.message:
                raise RuntimeError(
                    f"Vision API error: {response.error.message}"
                )

            # Extract full text
            full_text = response.full_text_annotation.text if response.full_text_annotation else ""

            # Detect language
            detected_language = None
            if response.full_text_annotation and response.full_text_annotation.pages:
                page = response.full_text_annotation.pages[0]
                if page.property and page.property.detected_languages:
                    detected_language = page.property.detected_languages[0].language_code

            # Calculate average confidence
            confidence = 0.0
            if response.text_annotations:
                # First annotation is the full text, skip it for confidence calculation
                if len(response.text_annotations) > 1:
                    confidences = [
                        annotation.confidence 
                        for annotation in response.text_annotations[1:] 
                        if hasattr(annotation, 'confidence')
                    ]
                    if confidences:
                        confidence = sum(confidences) / len(confidences)

            logger.info(
                "OCR completed for %s: %d characters, language=%s, confidence=%.2f",
                image_path.name,
                len(full_text),
                detected_language or "unknown",
                confidence
            )

            return {
                "text": full_text,
                "confidence": confidence,
                "language": detected_language,
                "annotations": response.text_annotations,
            }

        except Exception as exc:
            logger.error("Vision API OCR failed for %s: %s", image_path.name, exc)
            raise RuntimeError(f"Vision API OCR failed: {exc}") from exc

    def extract_text_from_pdf(
        self,
        pdf_path: str | Path,
        language_hints: Optional[List[str]] = None,
        dpi: int = 300,
        max_pages: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Extract text from a PDF using Vision API.
        
        First tries native PDF processing (no Poppler needed).
        Falls back to image conversion only if native processing fails.

        Args:
            pdf_path: Path to the PDF file.
            language_hints: Optional list of language codes for OCR hints.
            dpi: DPI for PDF to image conversion (only used in fallback).
            max_pages: Maximum number of pages to process (None = all pages).

        Returns:
            Dictionary containing extracted text and metadata.
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info("Attempting native PDF processing with Vision API: %s", pdf_path.name)
        
        try:
            # Try native PDF processing first (no Poppler required!)
            return self.extract_text_from_pdf_native(pdf_path, language_hints, max_pages)
        except Exception as exc:
            logger.warning(
                "Native PDF processing failed for %s: %s. Attempting image conversion fallback.",
                pdf_path.name,
                exc
            )
            
            # Fallback to image conversion (requires Poppler)
            try:
                return self._extract_text_from_pdf_images(pdf_path, language_hints, dpi, max_pages)
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"Both native PDF processing and image conversion failed. "
                    f"Native error: {exc}. Image conversion error: {fallback_exc}"
                ) from fallback_exc

    def _extract_text_from_pdf_images(
        self,
        pdf_path: Path,
        language_hints: Optional[List[str]] = None,
        dpi: int = 300,
        max_pages: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Extract text by converting PDF pages to images (requires Poppler).
        This is a fallback method.
        """
        logger.info("Converting PDF to images: %s", pdf_path.name)

        try:
            # Convert PDF pages to images
            last_page = max_pages if max_pages else None
            images = convert_from_path(
                str(pdf_path),
                dpi=dpi,
                last_page=last_page
            )
            
            logger.info("Processing %d pages from %s", len(images), pdf_path.name)

        except Exception as exc:
            raise RuntimeError(f"Failed to convert PDF to images: {exc}") from exc

        # Process each page
        page_results = []
        all_text_parts = []
        confidences = []
        detected_language = None

        for page_num, image in enumerate(images, start=1):
            try:
                # Save image to temporary bytes buffer
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)

                # Create Vision API image
                vision_image = vision.Image(content=img_byte_arr.getvalue())

                # Configure image context
                image_context = None
                if language_hints:
                    image_context = vision.ImageContext(language_hints=language_hints)

                # Perform OCR
                response = self.client.document_text_detection(
                    image=vision_image,
                    image_context=image_context
                )

                if response.error.message:
                    logger.warning(
                        "Vision API error on page %d: %s",
                        page_num,
                        response.error.message
                    )
                    continue

                # Extract text
                page_text = response.full_text_annotation.text if response.full_text_annotation else ""
                
                # Detect language from first page
                if page_num == 1 and response.full_text_annotation and response.full_text_annotation.pages:
                    page = response.full_text_annotation.pages[0]
                    if page.property and page.property.detected_languages:
                        detected_language = page.property.detected_languages[0].language_code

                # Calculate confidence
                page_confidence = 0.0
                if response.text_annotations and len(response.text_annotations) > 1:
                    page_confidences = [
                        annotation.confidence 
                        for annotation in response.text_annotations[1:] 
                        if hasattr(annotation, 'confidence')
                    ]
                    if page_confidences:
                        page_confidence = sum(page_confidences) / len(page_confidences)

                page_results.append({
                    "page_number": page_num,
                    "text": page_text,
                    "confidence": page_confidence,
                })

                all_text_parts.append(f"\n--- Page {page_num} ---\n{page_text}")
                confidences.append(page_confidence)

                logger.info(
                    "Page %d/%d processed: %d characters, confidence=%.2f",
                    page_num,
                    len(images),
                    len(page_text),
                    page_confidence
                )

            except Exception as exc:
                logger.error("Failed to process page %d: %s", page_num, exc)
                page_results.append({
                    "page_number": page_num,
                    "text": "",
                    "confidence": 0.0,
                    "error": str(exc)
                })

        # Combine results
        combined_text = "\n".join(all_text_parts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        logger.info(
            "PDF OCR completed: %d pages, %d characters, avg_confidence=%.2f",
            len(page_results),
            len(combined_text),
            avg_confidence
        )

        return {
            "text": combined_text,
            "pages": page_results,
            "total_pages": len(page_results),
            "language": detected_language,
            "avg_confidence": avg_confidence,
        }

    def extract_text_from_pdf_native(
        self,
        pdf_path: str | Path,
        language_hints: Optional[List[str]] = None,
        max_pages: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Extract text from PDF using Vision API's native PDF processing (NO POPPLER NEEDED).
        
        This method uses Vision API's batch file annotation to process PDFs directly
        without converting to images first.
        
        Args:
            pdf_path: Path to the PDF file.
            language_hints: Optional language hints for OCR.
            max_pages: Maximum pages to process (None = all pages).
            
        Returns:
            Dictionary containing extracted text and metadata.
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info("Processing PDF natively with Vision API (no Poppler): %s", pdf_path.name)
        
        try:
            # Read PDF file as bytes
            with open(pdf_path, 'rb') as pdf_file:
                pdf_content = pdf_file.read()
            
            # Create input config for PDF
            input_config = vision.InputConfig(
                content=pdf_content,
                mime_type='application/pdf'
            )
            
            # Configure OCR features
            features = [vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)]
            
            # Create image context with language hints
            image_context = None
            if language_hints:
                image_context = vision.ImageContext(language_hints=language_hints)
            
            # Create annotate file request
            request = vision.AnnotateFileRequest(
                input_config=input_config,
                features=features,
                image_context=image_context,
                pages=[i+1 for i in range(max_pages)] if max_pages else None
            )
            
            # Send request to Vision API
            response = self.client.batch_annotate_files(requests=[request])
            
            if not response.responses:
                raise RuntimeError("No response from Vision API")
            
            file_response = response.responses[0]
            
            if file_response.error.message:
                raise RuntimeError(f"Vision API error: {file_response.error.message}")
            
            # Extract text from all pages
            all_text_parts = []
            page_results = []
            detected_language = None
            confidences = []
            
            for page_num, page_response in enumerate(file_response.responses, start=1):
                if page_response.error.message:
                    logger.warning("Error on page %d: %s", page_num, page_response.error.message)
                    continue
                
                # Get full text
                if page_response.full_text_annotation:
                    page_text = page_response.full_text_annotation.text
                    
                    # Detect language from first page
                    if page_num == 1 and page_response.full_text_annotation.pages:
                        page_obj = page_response.full_text_annotation.pages[0]
                        if page_obj.property and page_obj.property.detected_languages:
                            detected_language = page_obj.property.detected_languages[0].language_code
                    
                    # Calculate confidence
                    page_confidence = 0.0
                    if page_response.text_annotations and len(page_response.text_annotations) > 1:
                        confs = [a.confidence for a in page_response.text_annotations[1:] if hasattr(a, 'confidence')]
                        if confs:
                            page_confidence = sum(confs) / len(confs)
                    
                    page_results.append({
                        "page_number": page_num,
                        "text": page_text,
                        "confidence": page_confidence,
                    })
                    
                    all_text_parts.append(f"\n--- Page {page_num} ---\n{page_text}")
                    confidences.append(page_confidence)
                    
                    logger.info("Page %d: %d chars, conf=%.2f", page_num, len(page_text), page_confidence)
            
            # Combine results
            combined_text = "\n".join(all_text_parts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            logger.info(
                "Native PDF OCR done: %d pages, %d chars, conf=%.2f",
                len(page_results),
                len(combined_text),
                avg_confidence
            )
            
            return {
                "text": combined_text,
                "pages": page_results,
                "total_pages": len(page_results),
                "language": detected_language,
                "avg_confidence": avg_confidence,
            }
            
        except Exception as exc:
            logger.error("Native PDF processing failed for %s: %s", pdf_path.name, exc)
            raise RuntimeError(f"Vision API native processing failed: {exc}") from exc

    def detect_document_features(
        self,
        pdf_path: str | Path,
        max_labels: int = 10,
        max_pages: int = 3
    ) -> Dict[str, Any]:
        """
        Detect document features and labels (topics) using Vision API.

        This uses Vision's label detection to identify topics/subjects in the document.

        Args:
            pdf_path: Path to the PDF file.
            max_labels: Maximum number of labels to return per page.
            max_pages: Maximum number of pages to analyze for labels.

        Returns:
            Dictionary containing:
                - labels: List of detected labels with scores
                - topics: Extracted topic strings
                - categories: Detected categories

        Raises:
            FileNotFoundError: If PDF file doesn't exist.
            RuntimeError: If processing fails.
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info("Detecting document features for: %s", pdf_path.name)

        try:
            # Convert first few pages to images
            images = convert_from_path(
                str(pdf_path),
                dpi=200,  # Lower DPI for label detection is sufficient
                last_page=max_pages
            )

        except Exception as exc:
            raise RuntimeError(f"Failed to convert PDF to images: {exc}") from exc

        all_labels = []
        all_topics = set()

        for page_num, image in enumerate(images, start=1):
            try:
                # Convert image to bytes
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)

                vision_image = vision.Image(content=img_byte_arr.getvalue())

                # Detect labels
                response = self.client.label_detection(
                    image=vision_image,
                    max_results=max_labels
                )

                if response.error.message:
                    logger.warning(
                        "Label detection error on page %d: %s",
                        page_num,
                        response.error.message
                    )
                    continue

                # Collect labels
                for label in response.label_annotations:
                    all_labels.append({
                        "description": label.description,
                        "score": label.score,
                        "page": page_num
                    })
                    all_topics.add(label.description)

                logger.info(
                    "Page %d: Detected %d labels",
                    page_num,
                    len(response.label_annotations)
                )

            except Exception as exc:
                logger.error("Failed to detect labels on page %d: %s", page_num, exc)

        # Sort labels by score
        all_labels.sort(key=lambda x: x["score"], reverse=True)

        logger.info(
            "Feature detection completed: %d unique topics from %d labels",
            len(all_topics),
            len(all_labels)
        )

        return {
            "labels": all_labels,
            "topics": list(all_topics),
            "total_labels": len(all_labels),
            "unique_topics": len(all_topics),
        }


# Global instance (lazy initialization)
_vision_service: Optional[VisionOCRService] = None


def get_vision_service() -> VisionOCRService:
    """
    Get or create the global Vision OCR service instance.

    Returns:
        VisionOCRService instance.

    Raises:
        RuntimeError: If service cannot be initialized.
    """
    global _vision_service
    
    if _vision_service is None:
        _vision_service = VisionOCRService()
    
    return _vision_service


def is_vision_api_enabled() -> bool:
    """Check if Vision API is enabled via environment variable."""
    return USE_VISION_API


# Convenience functions
def extract_text_from_image(
    image_path: str | Path,
    language_hints: Optional[List[str]] = None
) -> str:
    """
    Convenience function to extract text from an image.

    Args:
        image_path: Path to the image file.
        language_hints: Optional language hints for OCR.

    Returns:
        Extracted text content.
    """
    service = get_vision_service()
    result = service.extract_text_from_image(image_path, language_hints)
    return result["text"]


def extract_text_from_pdf(
    pdf_path: str | Path,
    language_hints: Optional[List[str]] = None,
    dpi: int = 300,
    max_pages: Optional[int] = None
) -> str:
    """
    Convenience function to extract text from a PDF.

    Args:
        pdf_path: Path to the PDF file.
        language_hints: Optional language hints for OCR.
        dpi: DPI for image conversion.
        max_pages: Maximum pages to process.

    Returns:
        Combined text from all pages.
    """
    service = get_vision_service()
    result = service.extract_text_from_pdf(pdf_path, language_hints, dpi, max_pages)
    return result["text"]


def detect_topics_from_pdf(
    pdf_path: str | Path,
    max_labels: int = 10,
    max_pages: int = 3
) -> List[str]:
    """
    Convenience function to detect topics from a PDF.

    Args:
        pdf_path: Path to the PDF file.
        max_labels: Maximum labels per page.
        max_pages: Maximum pages to analyze.

    Returns:
        List of detected topic strings.
    """
    service = get_vision_service()
    result = service.detect_document_features(pdf_path, max_labels, max_pages)
    return result["topics"]

"""
Vision API Routes for OCR and Topic Detection

This module provides REST API endpoints for Google Cloud Vision API,
including OCR text extraction and document topic detection.
"""

import logging
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import JSONResponse

from app.routes.auth_routes import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vision", tags=["Vision API"])


@router.post("/ocr", summary="Extract text using Google Vision API")
async def vision_ocr_endpoint(
    file: UploadFile = File(..., description="PDF or image file to process"),
    language_hints: Optional[str] = Form(None, description="Comma-separated language codes (e.g., 'en,hi,gu')"),
    max_pages: Optional[int] = Form(None, description="Maximum pages to process for PDFs"),
    current_user: dict = Depends(get_current_user),
):
    """
    Extract text from a PDF or image using Google Vision API.
    
    Requires Vision API to be enabled (USE_VISION_API=true in .env).
    """
    try:
        from app.services.vision_ocr_service import get_vision_service, is_vision_api_enabled
    except ImportError:
        return JSONResponse(
            status_code=501,
            content={
                "success": False,
                "message": "Google Vision API is not installed. Install with: pip install google-cloud-vision"
            }
        )
    
    if not is_vision_api_enabled():
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "message": "Vision API is not enabled. Set USE_VISION_API=true in your .env file."
            }
        )
    
    # Validate file type
    file_ext = Path(file.filename).suffix.lower() if file.filename else ""
    if file_ext not in [".pdf", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"]:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": f"Unsupported file type: {file_ext}. Supported: PDF, PNG, JPG, JPEG, GIF, BMP, TIFF"
            }
        )
    
    # Save uploaded file temporarily
    temp_dir = Path("./uploads/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / f"{uuid4()}{file_ext}"
    
    try:
        # Save file
        content = await file.read()
        temp_file.write_bytes(content)
        logger.info(f"Processing file with Vision API: {file.filename} ({len(content)} bytes)")
        
        # Parse language hints
        lang_hints_list = None
        if language_hints:
            lang_hints_list = [lang.strip() for lang in language_hints.split(",")]
        
        # Get Vision service
        vision_service = get_vision_service()
        
        # Process based on file type
        if file_ext == ".pdf":
            result = vision_service.extract_text_from_pdf(
                temp_file,
                language_hints=lang_hints_list,
                max_pages=max_pages
            )
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "text": result["text"],
                    "total_pages": result["total_pages"],
                    "language": result.get("language"),
                    "avg_confidence": result.get("avg_confidence", 0.0),
                    "pages": [
                        {
                            "page_number": p["page_number"],
                            "text_length": len(p["text"]),
                            "confidence": p["confidence"]
                        }
                        for p in result["pages"]
                    ]
                }
            )
        else:
            # Image file
            result = vision_service.extract_text_from_image(
                temp_file,
                language_hints=lang_hints_list
            )
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "text": result["text"],
                    "language": result.get("language"),
                    "confidence": result.get("confidence", 0.0),
                }
            )
    
    except Exception as exc:
        logger.error(f"Vision API OCR failed: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"OCR processing failed: {str(exc)}"
            }
        )
    finally:
        # Clean up temp file
        if temp_file.exists():
            try:
                temp_file.unlink()
            except Exception:
                pass


@router.post("/topics", summary="Detect topics using Google Vision API")
async def vision_topics_endpoint(
    file: UploadFile = File(..., description="PDF file to analyze"),
    max_labels: int = Form(10, description="Maximum labels to detect per page"),
    max_pages: int = Form(3, description="Maximum pages to analyze"),
    current_user: dict = Depends(get_current_user),
):
    """
    Detect document topics/labels using Google Vision API's label detection.
    
    Requires Vision API to be enabled (USE_VISION_API=true in .env).
    """
    try:
        from app.services.vision_ocr_service import get_vision_service, is_vision_api_enabled
    except ImportError:
        return JSONResponse(
            status_code=501,
            content={
                "success": False,
                "message": "Google Vision API is not installed. Install with: pip install google-cloud-vision"
            }
        )
    
    if not is_vision_api_enabled():
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "message": "Vision API is not enabled. Set USE_VISION_API=true in your .env file."
            }
        )
    
    # Validate file type
    file_ext = Path(file.filename).suffix.lower() if file.filename else ""
    if file_ext != ".pdf":
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": f"Only PDF files are supported for topic detection. Got: {file_ext}"
            }
        )
    
    # Save uploaded file temporarily
    temp_dir = Path("./uploads/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / f"{uuid4()}.pdf"
    
    try:
        # Save file
        content = await file.read()
        temp_file.write_bytes(content)
        logger.info(f"Detecting topics with Vision API: {file.filename} ({len(content)} bytes)")
        
        # Get Vision service
        vision_service = get_vision_service()
        
        # Detect topics
        result = vision_service.detect_document_features(
            temp_file,
            max_labels=max_labels,
            max_pages=max_pages
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "topics": result["topics"],
                "total_labels": result["total_labels"],
                "unique_topics": result["unique_topics"],
                "labels": result["labels"][:20],  # Return top 20 labels
            }
        )
    
    except Exception as exc:
        logger.error(f"Vision API topic detection failed: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Topic detection failed: {str(exc)}"
            }
        )
    finally:
        # Clean up temp file
        if temp_file.exists():
            try:
                temp_file.unlink()
            except Exception:
                pass


@router.post("/analyze", summary="Complete document analysis using Google Vision API")
async def vision_analyze_endpoint(
    file: UploadFile = File(..., description="PDF file to analyze"),
    language_hints: Optional[str] = Form(None, description="Comma-separated language codes"),
    max_pages: Optional[int] = Form(None, description="Maximum pages to process"),
    detect_topics: bool = Form(True, description="Whether to detect topics/labels"),
    current_user: dict = Depends(get_current_user),
):
    """
    Perform complete document analysis: OCR text extraction + topic detection.
    
    Requires Vision API to be enabled (USE_VISION_API=true in .env).
    """
    try:
        from app.services.vision_ocr_service import get_vision_service, is_vision_api_enabled
    except ImportError:
        return JSONResponse(
            status_code=501,
            content={
                "success": False,
                "message": "Google Vision API is not installed. Install with: pip install google-cloud-vision"
            }
        )
    
    if not is_vision_api_enabled():
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "message": "Vision API is not enabled. Set USE_VISION_API=true in your .env file."
            }
        )
    
    # Validate file type
    file_ext = Path(file.filename).suffix.lower() if file.filename else ""
    if file_ext != ".pdf":
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": f"Only PDF files are supported for full analysis. Got: {file_ext}"
            }
        )
    
    # Save uploaded file temporarily
    temp_dir = Path("./uploads/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / f"{uuid4()}.pdf"
    
    try:
        # Save file
        content = await file.read()
        temp_file.write_bytes(content)
        logger.info(f"Analyzing document with Vision API: {file.filename} ({len(content)} bytes)")
        
        # Parse language hints
        lang_hints_list = None
        if language_hints:
            lang_hints_list = [lang.strip() for lang in language_hints.split(",")]
        
        # Get Vision service
        vision_service = get_vision_service()
        
        # Extract text
        ocr_result = vision_service.extract_text_from_pdf(
            temp_file,
            language_hints=lang_hints_list,
            max_pages=max_pages
        )
        
        # Optionally detect topics
        topics_result = None
        if detect_topics:
            topics_result = vision_service.detect_document_features(
                temp_file,
                max_labels=10,
                max_pages=min(max_pages or 3, 3)
            )
        
        response_data = {
            "success": True,
            "ocr": {
                "text": ocr_result["text"],
                "total_pages": ocr_result["total_pages"],
                "language": ocr_result.get("language"),
                "avg_confidence": ocr_result.get("avg_confidence", 0.0),
                "text_length": len(ocr_result["text"]),
            }
        }
        
        if topics_result:
            response_data["topics"] = {
                "topics": topics_result["topics"],
                "total_labels": topics_result["total_labels"],
                "unique_topics": topics_result["unique_topics"],
                "top_labels": topics_result["labels"][:10],
            }
        
        return JSONResponse(status_code=200, content=response_data)
    
    except Exception as exc:
        logger.error(f"Vision API analysis failed: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Document analysis failed: {str(exc)}"
            }
        )
    finally:
        # Clean up temp file
        if temp_file.exists():
            try:
                temp_file.unlink()
            except Exception:
                pass

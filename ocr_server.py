#!/usr/bin/env python3
"""
Local OCR Server for Other Skies Inventory System
Runs entirely locally using Tesseract for text extraction
"""

import os
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import requests
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Other Skies OCR Service", version="1.0.0")

# CORS for local Next.js development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
class Config:
    UPLOAD_DIR = Path("./temp/uploads")
    PROCESSED_DIR = Path("./temp/processed")
    OCR_LANGUAGES = ["eng"]  # Add more languages as needed
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    
    # API Keys (from environment)
    GOOGLE_BOOKS_API_KEY = os.getenv("GOOGLE_BOOKS_API_KEY", "")
    ISBNDB_API_KEY = os.getenv("ISBNDB_API_KEY", "")
    
    # Tesseract configuration
    TESSERACT_CONFIG = r'--oem 3 --psm 6'

config = Config()

# Ensure directories exist
config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Pydantic models
class OCRResult(BaseModel):
    raw_text: str
    extracted_isbn: Optional[str]
    extracted_title: Optional[str]
    extracted_author: Optional[str]
    extracted_publisher: Optional[str]
    extracted_year: Optional[int]
    confidence_scores: Dict[str, float]

class MetadataResult(BaseModel):
    source: str
    title: Optional[str]
    author: Optional[str]
    publisher: Optional[str]
    publication_year: Optional[int]
    isbn: Optional[str]
    isbn10: Optional[str]
    description: Optional[str]
    categories: List[str]
    page_count: Optional[int]
    language: Optional[str]
    raw_data: Dict

class ProcessingResult(BaseModel):
    ocr_result: OCRResult
    metadata_results: List[MetadataResult]
    suggested_values: Dict
    processing_time_ms: int

# Image preprocessing functions
class ImagePreprocessor:
    """Prepare images for optimal OCR results"""
    
    @staticmethod
    def preprocess_for_ocr(image_path: str) -> np.ndarray:
        """
        Preprocess image for better OCR accuracy
        Maintains archival quality while optimizing for text extraction
        """
        # Read image
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        
        # Adaptive thresholding for text
        thresh = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        # Deskew if needed
        angle = ImagePreprocessor.get_skew_angle(thresh)
        if abs(angle) > 0.5:
            thresh = ImagePreprocessor.rotate_image(thresh, angle)
        
        return thresh
    
    @staticmethod
    def get_skew_angle(image: np.ndarray) -> float:
        """Detect skew angle of scanned page"""
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        
        if lines is not None:
            angles = []
            for rho, theta in lines[0]:
                angle = (theta * 180 / np.pi) - 90
                if -45 <= angle <= 45:
                    angles.append(angle)
            
            if angles:
                return np.median(angles)
        
        return 0.0
    
    @staticmethod
    def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle"""
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# OCR Engine
class OCREngine:
    """Local OCR using Tesseract"""
    
    @staticmethod
    def extract_text(image_path: str) -> OCRResult:
        """Extract text from image using Tesseract"""
        try:
            # Preprocess image
            processed_img = ImagePreprocessor.preprocess_for_ocr(image_path)
            
            # Run OCR
            raw_text = pytesseract.image_to_string(
                processed_img,
                lang='+'.join(config.OCR_LANGUAGES),
                config=config.TESSERACT_CONFIG
            )
            
            # Get confidence scores
            data = pytesseract.image_to_data(
                processed_img,
                output_type=pytesseract.Output.DICT
            )
            
            # Calculate average confidence
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = np.mean(confidences) if confidences else 0
            
            # Extract specific fields
            extracted_data = OCREngine.extract_book_data(raw_text)
            
            return OCRResult(
                raw_text=raw_text,
                extracted_isbn=extracted_data.get('isbn'),
                extracted_title=extracted_data.get('title'),
                extracted_author=extracted_data.get('author'),
                extracted_publisher=extracted_data.get('publisher'),
                extracted_year=extracted_data.get('year'),
                confidence_scores={
                    'overall': avg_confidence,
                    'text_density': len(confidences) / max(len(data['text']), 1)
                }
            )
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            raise
    
    @staticmethod
    def extract_book_data(text: str) -> Dict:
        """Extract structured book data from raw OCR text"""
        import re
        
        data = {}
        
        # Extract ISBN (10 or 13 digits)
        isbn_patterns = [
            r'ISBN[:\s-]*([0-9X-]{10,})',
            r'978[0-9]{10}',
            r'979[0-9]{10}',
            r'[0-9]{9}[0-9X]'
        ]
        
        for pattern in isbn_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                isbn = re.sub(r'[^0-9X]', '', match.group(1) if match.groups() else match.group(0))
                if len(isbn) in [10, 13]:
                    data['isbn'] = isbn
                    break
        
        # Extract year (look for 4-digit years between 1800-2099)
        year_match = re.search(r'(18|19|20)\d{2}', text)
        if year_match:
            data['year'] = int(year_match.group(0))
        
        # Extract publisher (common patterns)
        publisher_patterns = [
            r'Published by ([^,\n]+)',
            r'Publisher[:\s]+([^,\n]+)',
            r'([A-Z][a-z]+ (?:Press|Books|Publications|Publishers|House))',
        ]
        
        for pattern in publisher_patterns:
            match = re.search(pattern, text)
            if match:
                data['publisher'] = match.group(1).strip()
                break
        
        # Title and author extraction (heuristic - usually at the top)
        lines = text.strip().split('\n')
        if len(lines) > 0:
            # First non-empty line is often the title
            for line in lines[:5]:
                if len(line.strip()) > 3:
                    data['title'] = line.strip()
                    break
            
            # Second substantial line often the author
            found_title = False
            for line in lines[:10]:
                if len(line.strip()) > 3:
                    if found_title:
                        # Check if it looks like an author name
                        if re.search(r'^[A-Z][a-z]+\s+[A-Z]', line):
                            data['author'] = line.strip()
                            break
                    elif line.strip() == data.get('title'):
                        found_title = True
        
        return data

# Metadata API clients
class MetadataClient:
    """Fetch metadata from various APIs"""
    
    @staticmethod
    async def fetch_open_library(isbn: str = None, title: str = None, author: str = None) -> Optional[MetadataResult]:
        """Fetch metadata from Open Library API (free, no key required)"""
        try:
            # Try ISBN first
            if isbn:
                url = f"https://openlibrary.org/api/books?bibkeys=ISBN:{isbn}&format=json&jscmd=data"
                response = requests.get(url, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    if f"ISBN:{isbn}" in data:
                        book_data = data[f"ISBN:{isbn}"]
                        return MetadataClient._parse_open_library(book_data)
            
            # Try title/author search
            if title:
                query = f"{title}"
                if author:
                    query += f" {author}"
                
                url = f"https://openlibrary.org/search.json?q={query}&limit=1"
                response = requests.get(url, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('docs'):
                        book_data = data['docs'][0]
                        return MetadataClient._parse_open_library_search(book_data)
                        
        except Exception as e:
            logger.error(f"Open Library API error: {e}")
        
        return None
    
    @staticmethod
    def _parse_open_library(data: Dict) -> MetadataResult:
        """Parse Open Library response"""
        return MetadataResult(
            source="Open Library",
            title=data.get('title'),
            author=', '.join([a.get('name', '') for a in data.get('authors', [])]),
            publisher=', '.join(data.get('publishers', [])),
            publication_year=int(data.get('publish_date', '0')[:4]) if data.get('publish_date') else None,
            isbn=data.get('identifiers', {}).get('isbn_13', [None])[0] if data.get('identifiers') else None,
            isbn10=data.get('identifiers', {}).get('isbn_10', [None])[0] if data.get('identifiers') else None,
            description=data.get('notes'),
            categories=data.get('subjects', [])[:5] if data.get('subjects') else [],
            page_count=data.get('number_of_pages'),
            language=None,
            raw_data=data
        )
    
    @staticmethod
    def _parse_open_library_search(data: Dict) -> MetadataResult:
        """Parse Open Library search response"""
        return MetadataResult(
            source="Open Library",
            title=data.get('title'),
            author=', '.join(data.get('author_name', [])),
            publisher=', '.join(data.get('publisher', [])),
            publication_year=data.get('first_publish_year'),
            isbn=data.get('isbn', [None])[0] if data.get('isbn') else None,
            isbn10=None,
            description=None,
            categories=data.get('subject', [])[:5] if data.get('subject') else [],
            page_count=data.get('number_of_pages_median'),
            language=', '.join(data.get('language', [])),
            raw_data=data
        )
    
    @staticmethod
    async def fetch_google_books(isbn: str = None, title: str = None, author: str = None) -> Optional[MetadataResult]:
        """Fetch metadata from Google Books API"""
        if not config.GOOGLE_BOOKS_API_KEY:
            return None
            
        try:
            query_parts = []
            if isbn:
                query_parts.append(f"isbn:{isbn}")
            if title:
                query_parts.append(f"intitle:{title}")
            if author:
                query_parts.append(f"inauthor:{author}")
            
            if not query_parts:
                return None
            
            query = "+".join(query_parts)
            url = f"https://www.googleapis.com/books/v1/volumes?q={query}&key={config.GOOGLE_BOOKS_API_KEY}"
            
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('items'):
                    book_data = data['items'][0]['volumeInfo']
                    
                    # Extract ISBNs
                    isbn13 = None
                    isbn10 = None
                    for identifier in book_data.get('industryIdentifiers', []):
                        if identifier['type'] == 'ISBN_13':
                            isbn13 = identifier['identifier']
                        elif identifier['type'] == 'ISBN_10':
                            isbn10 = identifier['identifier']
                    
                    return MetadataResult(
                        source="Google Books",
                        title=book_data.get('title'),
                        author=', '.join(book_data.get('authors', [])),
                        publisher=book_data.get('publisher'),
                        publication_year=int(book_data.get('publishedDate', '')[:4]) if book_data.get('publishedDate') else None,
                        isbn=isbn13,
                        isbn10=isbn10,
                        description=book_data.get('description'),
                        categories=book_data.get('categories', []),
                        page_count=book_data.get('pageCount'),
                        language=book_data.get('language'),
                        raw_data=book_data
                    )
                    
        except Exception as e:
            logger.error(f"Google Books API error: {e}")
        
        return None
    
    @staticmethod
    async def fetch_all_sources(isbn: str = None, title: str = None, author: str = None) -> List[MetadataResult]:
        """Fetch metadata from all available sources"""
        tasks = [
            MetadataClient.fetch_open_library(isbn, title, author),
            MetadataClient.fetch_google_books(isbn, title, author)
        ]
        
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]

# Intelligent suggestion engine
class SuggestionEngine:
    """Combine OCR and API results into best suggestions"""
    
    @staticmethod
    def generate_suggestions(ocr_result: OCRResult, metadata_results: List[MetadataResult]) -> Dict:
        """Generate best suggestions from all sources"""
        suggestions = {}
        
        # Prioritize metadata from APIs over OCR
        for field in ['title', 'author', 'publisher', 'isbn', 'isbn10']:
            # First try to get from metadata
            for result in metadata_results:
                value = getattr(result, field, None)
                if value:
                    suggestions[field] = value
                    break
            
            # Fall back to OCR if no metadata
            if field not in suggestions:
                ocr_value = getattr(ocr_result, f'extracted_{field}', None)
                if ocr_value:
                    suggestions[field] = ocr_value
        
        # Special handling for year
        years = []
        for result in metadata_results:
            if result.publication_year:
                years.append(result.publication_year)
        
        if ocr_result.extracted_year:
            years.append(ocr_result.extracted_year)
        
        if years:
            # Use most common year or latest if tied
            from collections import Counter
            year_counts = Counter(years)
            suggestions['publication_year'] = year_counts.most_common(1)[0][0]
        
        # Aggregate categories
        all_categories = []
        for result in metadata_results:
            all_categories.extend(result.categories)
        
        if all_categories:
            from collections import Counter
            category_counts = Counter(all_categories)
            suggestions['categories'] = [cat for cat, _ in category_counts.most_common(5)]
        
        # Generate description if available
        descriptions = [r.description for r in metadata_results if r.description]
        if descriptions:
            # Use longest description
            suggestions['description'] = max(descriptions, key=len)
        
        return suggestions

# API Endpoints
@app.post("/process-single", response_model=ProcessingResult)
async def process_single_image(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Process a single book image"""
    start_time = datetime.now()
    
    # Validate file
    if file.size > config.MAX_IMAGE_SIZE:
        raise HTTPException(status_code=400, detail="Image too large")
    
    # Save uploaded file
    file_path = config.UPLOAD_DIR / f"{datetime.now().timestamp()}_{file.filename}"
    
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Run OCR
        ocr_result = OCREngine.extract_text(str(file_path))
        
        # Fetch metadata from APIs
        metadata_results = await MetadataClient.fetch_all_sources(
            isbn=ocr_result.extracted_isbn,
            title=ocr_result.extracted_title,
            author=ocr_result.extracted_author
        )
        
        # Generate suggestions
        suggestions = SuggestionEngine.generate_suggestions(ocr_result, metadata_results)
        
        # Clean up uploaded file in background
        background_tasks.add_task(os.unlink, file_path)
        
        # Calculate processing time
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        return ProcessingResult(
            ocr_result=ocr_result,
            metadata_results=metadata_results,
            suggested_values=suggestions,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        # Clean up on error
        if file_path.exists():
            os.unlink(file_path)
        
        logger.error(f"Processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-batch")
async def process_batch(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Process multiple book images in batch"""
    results = []
    
    for file in files:
        try:
            result = await process_single_image(file, background_tasks)
            results.append({
                'filename': file.filename,
                'status': 'success',
                'data': result.dict()
            })
        except Exception as e:
            results.append({
                'filename': file.filename,
                'status': 'error',
                'error': str(e)
            })
    
    return JSONResponse(content={'results': results})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "OCR Service",
        "tesseract_version": pytesseract.get_tesseract_version(),
        "apis_configured": {
            "google_books": bool(config.GOOGLE_BOOKS_API_KEY),
            "open_library": True  # Always available
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )

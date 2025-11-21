#!/usr/bin/env python3
"""
LUT Processing Service for Archival Book Photography
Applies color grading while maintaining archival integrity
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import colour
import imageio
import logging

logger = logging.getLogger(__name__)

class LUTProcessor:
    """
    Professional LUT application for archival book photography
    Maintains color accuracy while enhancing visual presentation
    """
    
    def __init__(self, lut_directory: str = "./luts"):
        self.lut_directory = Path(lut_directory)
        self.lut_directory.mkdir(exist_ok=True)
        self.loaded_luts = {}
        self.load_available_luts()
    
    def load_available_luts(self):
        """Load all available LUT files"""
        lut_extensions = ['.cube', '.3dl', '.mga', '.spi1d', '.spi3d']
        
        for lut_file in self.lut_directory.iterdir():
            if lut_file.suffix.lower() in lut_extensions:
                try:
                    self.loaded_luts[lut_file.stem] = str(lut_file)
                    logger.info(f"Loaded LUT: {lut_file.stem}")
                except Exception as e:
                    logger.error(f"Failed to load LUT {lut_file}: {e}")
    
    def create_archival_luts(self):
        """
        Create custom LUTs optimized for rare book photography
        These maintain color accuracy while enhancing presentation
        """
        
        # 1. Warm Archival - Subtle warmth for older books
        warm_archival = self.generate_warm_archival_lut()
        self.save_lut(warm_archival, "warm_archival")
        
        # 2. Neutral Accurate - Maximum color fidelity
        neutral_accurate = self.generate_neutral_accurate_lut()
        self.save_lut(neutral_accurate, "neutral_accurate")
        
        # 3. Gothic Drama - Enhanced contrast for weird fiction
        gothic_drama = self.generate_gothic_drama_lut()
        self.save_lut(gothic_drama, "gothic_drama")
        
        # 4. Parchment - For aged paper tones
        parchment = self.generate_parchment_lut()
        self.save_lut(parchment, "parchment")
        
        logger.info("Created custom archival LUTs")
    
    def generate_warm_archival_lut(self, size: int = 33) -> np.ndarray:
        """
        Generate LUT with subtle warmth for vintage books
        Maintains color accuracy while adding slight golden tone
        """
        lut = np.zeros((size, size, size, 3))
        
        for r in range(size):
            for g in range(size):
                for b in range(size):
                    # Normalize to 0-1
                    rf = r / (size - 1)
                    gf = g / (size - 1)
                    bf = b / (size - 1)
                    
                    # Apply subtle warm shift
                    # Increase reds slightly, decrease blues
                    rf_out = rf * 1.02  # Very subtle red boost
                    gf_out = gf * 1.01  # Tiny green boost
                    bf_out = bf * 0.98  # Slight blue reduction
                    
                    # Apply gentle S-curve for contrast
                    rf_out = self.apply_gentle_s_curve(rf_out)
                    gf_out = self.apply_gentle_s_curve(gf_out)
                    bf_out = self.apply_gentle_s_curve(bf_out)
                    
                    # Clamp values
                    lut[r, g, b] = [
                        np.clip(rf_out, 0, 1),
                        np.clip(gf_out, 0, 1),
                        np.clip(bf_out, 0, 1)
                    ]
        
        return lut
    
    def generate_neutral_accurate_lut(self, size: int = 33) -> np.ndarray:
        """
        Generate LUT for maximum color accuracy
        Only applies gamma and contrast corrections
        """
        lut = np.zeros((size, size, size, 3))
        
        for r in range(size):
            for g in range(size):
                for b in range(size):
                    # Normalize to 0-1
                    rf = r / (size - 1)
                    gf = g / (size - 1)
                    bf = b / (size - 1)
                    
                    # Apply only gamma correction (2.2 standard)
                    rf_out = np.power(rf, 1/2.2)
                    gf_out = np.power(gf, 1/2.2)
                    bf_out = np.power(bf, 1/2.2)
                    
                    lut[r, g, b] = [rf_out, gf_out, bf_out]
        
        return lut
    
    def generate_gothic_drama_lut(self, size: int = 33) -> np.ndarray:
        """
        Generate LUT for weird fiction aesthetic
        Enhanced contrast with preserved shadow detail
        """
        lut = np.zeros((size, size, size, 3))
        
        for r in range(size):
            for g in range(size):
                for b in range(size):
                    # Normalize to 0-1
                    rf = r / (size - 1)
                    gf = g / (size - 1)
                    bf = b / (size - 1)
                    
                    # Desaturate slightly for moodier tone
                    luminance = 0.299 * rf + 0.587 * gf + 0.114 * bf
                    
                    # Mix original with luminance (partial desaturation)
                    saturation = 0.85
                    rf_out = luminance + saturation * (rf - luminance)
                    gf_out = luminance + saturation * (gf - luminance)
                    bf_out = luminance + saturation * (bf - luminance)
                    
                    # Apply stronger S-curve for drama
                    rf_out = self.apply_dramatic_s_curve(rf_out)
                    gf_out = self.apply_dramatic_s_curve(gf_out)
                    bf_out = self.apply_dramatic_s_curve(bf_out)
                    
                    # Slight blue shift in shadows for atmosphere
                    if luminance < 0.3:
                        bf_out *= 1.05
                    
                    # Clamp values
                    lut[r, g, b] = [
                        np.clip(rf_out, 0, 1),
                        np.clip(gf_out, 0, 1),
                        np.clip(bf_out, 0, 1)
                    ]
        
        return lut
    
    def generate_parchment_lut(self, size: int = 33) -> np.ndarray:
        """
        Generate LUT for aged paper tones
        Adds yellowing effect while preserving detail
        """
        lut = np.zeros((size, size, size, 3))
        
        for r in range(size):
            for g in range(size):
                for b in range(size):
                    # Normalize to 0-1
                    rf = r / (size - 1)
                    gf = g / (size - 1)
                    bf = b / (size - 1)
                    
                    # Shift towards yellow/brown
                    rf_out = rf * 1.05  # Boost reds
                    gf_out = gf * 1.02  # Slight green boost
                    bf_out = bf * 0.90  # Reduce blues
                    
                    # Add subtle sepia
                    sepia_r = 0.393 * rf + 0.769 * gf + 0.189 * bf
                    sepia_g = 0.349 * rf + 0.686 * gf + 0.168 * bf
                    sepia_b = 0.272 * rf + 0.534 * gf + 0.131 * bf
                    
                    # Blend original with sepia (20% sepia)
                    rf_out = rf_out * 0.8 + sepia_r * 0.2
                    gf_out = gf_out * 0.8 + sepia_g * 0.2
                    bf_out = bf_out * 0.8 + sepia_b * 0.2
                    
                    # Clamp values
                    lut[r, g, b] = [
                        np.clip(rf_out, 0, 1),
                        np.clip(gf_out, 0, 1),
                        np.clip(bf_out, 0, 1)
                    ]
        
        return lut
    
    def apply_gentle_s_curve(self, value: float) -> float:
        """Apply gentle S-curve for subtle contrast enhancement"""
        # Sigmoid function for smooth S-curve
        return 1 / (1 + np.exp(-5 * (value - 0.5))) * 1.05 - 0.025
    
    def apply_dramatic_s_curve(self, value: float) -> float:
        """Apply stronger S-curve for dramatic contrast"""
        # Stronger sigmoid for more dramatic effect
        return 1 / (1 + np.exp(-8 * (value - 0.5))) * 1.1 - 0.05
    
    def apply_lut_to_image(self, 
                          image_path: str, 
                          lut_name: str = "neutral_accurate",
                          strength: float = 1.0) -> Image.Image:
        """
        Apply LUT to image with adjustable strength
        
        Args:
            image_path: Path to input image
            lut_name: Name of LUT to apply
            strength: Blend strength (0-1, where 1 is full LUT)
        """
        # Load image
        img = Image.open(image_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Get or generate LUT
        if lut_name in self.loaded_luts:
            lut = self.load_lut_file(self.loaded_luts[lut_name])
        else:
            # Generate on the fly if not found
            if lut_name == "warm_archival":
                lut = self.generate_warm_archival_lut()
            elif lut_name == "gothic_drama":
                lut = self.generate_gothic_drama_lut()
            elif lut_name == "parchment":
                lut = self.generate_parchment_lut()
            else:
                lut = self.generate_neutral_accurate_lut()
        
        # Apply LUT
        result = self.apply_3d_lut(img_array, lut)
        
        # Blend with original based on strength
        if strength < 1.0:
            result = img_array * (1 - strength) + result * strength
        
        # Convert back to PIL Image
        result = (result * 255).astype(np.uint8)
        return Image.fromarray(result)
    
    def apply_3d_lut(self, image: np.ndarray, lut: np.ndarray) -> np.ndarray:
        """Apply 3D LUT to image array"""
        h, w = image.shape[:2]
        lut_size = lut.shape[0]
        
        # Flatten image for processing
        pixels = image.reshape(-1, 3)
        result = np.zeros_like(pixels)
        
        for i, pixel in enumerate(pixels):
            # Scale pixel values to LUT indices
            r_idx = pixel[0] * (lut_size - 1)
            g_idx = pixel[1] * (lut_size - 1)
            b_idx = pixel[2] * (lut_size - 1)
            
            # Trilinear interpolation for smooth results
            result[i] = self.trilinear_interpolation(lut, r_idx, g_idx, b_idx)
        
        return result.reshape(h, w, 3)
    
    def trilinear_interpolation(self, lut: np.ndarray, r: float, g: float, b: float) -> np.ndarray:
        """Perform trilinear interpolation in 3D LUT"""
        # Get integer indices
        r0, g0, b0 = int(r), int(g), int(b)
        r1, g1, b1 = min(r0 + 1, lut.shape[0] - 1), min(g0 + 1, lut.shape[1] - 1), min(b0 + 1, lut.shape[2] - 1)
        
        # Get fractional parts
        rf, gf, bf = r - r0, g - g0, b - b0
        
        # Perform trilinear interpolation
        c000 = lut[r0, g0, b0]
        c001 = lut[r0, g0, b1]
        c010 = lut[r0, g1, b0]
        c011 = lut[r0, g1, b1]
        c100 = lut[r1, g0, b0]
        c101 = lut[r1, g0, b1]
        c110 = lut[r1, g1, b0]
        c111 = lut[r1, g1, b1]
        
        c00 = c000 * (1 - bf) + c001 * bf
        c01 = c010 * (1 - bf) + c011 * bf
        c10 = c100 * (1 - bf) + c101 * bf
        c11 = c110 * (1 - bf) + c111 * bf
        
        c0 = c00 * (1 - gf) + c01 * gf
        c1 = c10 * (1 - gf) + c11 * gf
        
        return c0 * (1 - rf) + c1 * rf
    
    def save_lut(self, lut: np.ndarray, name: str):
        """Save LUT in .cube format"""
        lut_path = self.lut_directory / f"{name}.cube"
        size = lut.shape[0]
        
        with open(lut_path, 'w') as f:
            # Write header
            f.write(f"# {name} LUT for Other Skies Rare Books\n")
            f.write(f"# Created for archival book photography\n")
            f.write(f"LUT_3D_SIZE {size}\n")
            f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
            f.write("DOMAIN_MAX 1.0 1.0 1.0\n\n")
            
            # Write LUT data
            for b in range(size):
                for g in range(size):
                    for r in range(size):
                        rgb = lut[r, g, b]
                        f.write(f"{rgb[0]:.6f} {rgb[1]:.6f} {rgb[2]:.6f}\n")
        
        logger.info(f"Saved LUT: {lut_path}")
    
    def load_lut_file(self, lut_path: str) -> np.ndarray:
        """Load LUT from .cube file"""
        with open(lut_path, 'r') as f:
            lines = f.readlines()
        
        # Parse header
        size = None
        data_start = 0
        
        for i, line in enumerate(lines):
            if line.startswith('LUT_3D_SIZE'):
                size = int(line.split()[1])
            elif not line.startswith('#') and not line.startswith('DOMAIN') and not line.startswith('LUT'):
                data_start = i
                break
        
        if size is None:
            raise ValueError(f"Could not parse LUT size from {lut_path}")
        
        # Parse LUT data
        lut = np.zeros((size, size, size, 3))
        data_lines = lines[data_start:]
        
        idx = 0
        for b in range(size):
            for g in range(size):
                for r in range(size):
                    if idx < len(data_lines):
                        values = data_lines[idx].strip().split()
                        if len(values) >= 3:
                            lut[r, g, b] = [float(values[0]), float(values[1]), float(values[2])]
                    idx += 1
        
        return lut
    
    def batch_process_images(self, 
                            input_dir: str, 
                            output_dir: str, 
                            lut_name: str = "neutral_accurate",
                            strength: float = 1.0):
        """
        Batch process images with LUT
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory for processed images
            lut_name: Name of LUT to apply
            strength: LUT strength (0-1)
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        supported_formats = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
        
        for image_file in input_path.iterdir():
            if image_file.suffix.lower() in supported_formats:
                try:
                    # Apply LUT
                    processed = self.apply_lut_to_image(
                        str(image_file),
                        lut_name,
                        strength
                    )
                    
                    # Save with same filename
                    output_file = output_path / image_file.name
                    processed.save(output_file, quality=95, optimize=True)
                    
                    logger.info(f"Processed: {image_file.name}")
                    
                except Exception as e:
                    logger.error(f"Failed to process {image_file}: {e}")

# FastAPI integration
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import StreamingResponse
import io

app = FastAPI(title="LUT Processing Service")

lut_processor = LUTProcessor()

# Create default LUTs on startup
lut_processor.create_archival_luts()

@app.post("/apply-lut")
async def apply_lut(
    file: UploadFile = File(...),
    lut_name: str = Query("neutral_accurate", description="LUT to apply"),
    strength: float = Query(1.0, ge=0.0, le=1.0, description="LUT strength")
):
    """Apply LUT to uploaded image"""
    # Read image
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    
    # Save temporarily
    temp_path = f"/tmp/{file.filename}"
    img.save(temp_path)
    
    # Apply LUT
    processed = lut_processor.apply_lut_to_image(temp_path, lut_name, strength)
    
    # Return processed image
    img_byte_arr = io.BytesIO()
    processed.save(img_byte_arr, format='JPEG', quality=95)
    img_byte_arr.seek(0)
    
    # Clean up
    os.unlink(temp_path)
    
    return StreamingResponse(
        img_byte_arr,
        media_type="image/jpeg",
        headers={"Content-Disposition": f"attachment; filename=processed_{file.filename}"}
    )

@app.get("/available-luts")
async def get_available_luts():
    """Get list of available LUTs"""
    return {
        "luts": list(lut_processor.loaded_luts.keys()),
        "default": "neutral_accurate"
    }

if __name__ == "__main__":
    # Example usage
    processor = LUTProcessor()
    processor.create_archival_luts()
    
    # Process a single image
    # result = processor.apply_lut_to_image(
    #     "input/book_cover.jpg",
    #     "warm_archival",
    #     strength=0.8
    # )
    # result.save("output/book_cover_processed.jpg")

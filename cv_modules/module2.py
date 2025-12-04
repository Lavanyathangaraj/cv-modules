import cv2
import numpy as np
import time
from typing import Dict, List, Any, Tuple
from scipy.signal import convolve2d
from numpy.fft import fft2, ifft2 
from pathlib import Path
from .utils import resize_for_web, rotate_image 

# Base directory (C:\computerVision\cv_modules)
BASE_DIR = Path(__file__).resolve().parent

# Template storage: REMOVED TEMPLATE_DATABASE
# CRITICAL PATH: This path must match your folder structure
TEMPLATE_FOLDER = BASE_DIR / "images" / "module2" / "objectDetection" / "templates"


def create_gaussian_kernel(sigma: float) -> np.ndarray:
    """Creates a 2D Gaussian kernel using cv2 function."""
    sigma = float(sigma)
    if sigma <= 0:
        return np.array([[1.0]])
    
    kernel_size = 2 * int(3 * sigma) + 1
    g = cv2.getGaussianKernel(kernel_size, sigma)
    gaussian_kernel = g @ g.T
    return gaussian_kernel


# --- Module 2: Image Filter Simulation Logic (No Change) ---

def process_image_filter(img_bgr: np.ndarray, sigma: float, K: float) -> Dict[str, np.ndarray]:
    """Applies Gaussian blur and Wiener deconvolution using Fourier Transform."""
    MAX_PROCESS_DIM = 400
    
    processed_img = resize_for_web(img_bgr, MAX_PROCESS_DIM)
    
    original_float = processed_img.astype(np.float64) / 255.0
    m, n, c = original_float.shape

    # 1. Gaussian Blur (Convolution)
    gaussian_kernel = create_gaussian_kernel(sigma)
    blurred_float = np.zeros_like(original_float)
    for ch in range(c):
        blurred_float[:, :, ch] = convolve2d(original_float[:, :, ch], gaussian_kernel, mode='same', boundary='symm')
    
    # 2. Wiener Deconvolution (Frequency Domain)
    recovered_float = np.zeros_like(original_float)
    
    # Pad size for FFT
    padded_m = int(2 ** np.ceil(np.log2(m)))
    padded_n = int(2 ** np.ceil(np.log2(n)))
    padded_size = (padded_m, padded_n)
    
    # Gaussian kernel in the Frequency Domain
    H = fft2(gaussian_kernel, padded_size)
    
    for ch in range(c):
        F_blur = fft2(blurred_float[:, :, ch], padded_size)
        
        # Wiener Filter formula: F_rec = (conj(H) / (|H|^2 + K)) * F_blur
        F_rec = (np.conj(H) / (np.abs(H)**2 + K)) * F_blur
        
        # Inverse FFT
        rec = np.real(ifft2(F_rec)) 
        rec = rec[:m, :n]           
        recovered_float[:, :, ch] = np.clip(rec, 0, 1) # Clip to [0, 1] range
    
    # Convert back to BGR uint8 format for output
    original_bgr = (original_float * 255).astype(np.uint8)
    blurred_bgr = (blurred_float * 255).astype(np.uint8)
    recovered_bgr = (recovered_float * 255).astype(np.uint8)
    
    return {
        'original_data': original_bgr,
        'blurred_data': blurred_bgr,
        'recovered_data': recovered_bgr
    }

# --- Module 2: Object Detection Logic (MODIFIED FOR FILE SYSTEM ACCESS) ---

def process_template_matching(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Performs multi-scale template matching directly from the file system, 
    eliminating the need for an in-memory database.
    """
    
    scene = img_bgr.copy()
    scene_processed = scene.copy()
    scene_gray = cv2.cvtColor(scene_processed, cv2.COLOR_BGR2GRAY) 

    all_detections = []
    
    # Adopted parameters from the working reference code
    NMS_THRESHOLD = 0.3
    DETECTION_THRESHOLD = 0.65 

    # Multi-Scale Parameters 
    SCALES = np.linspace(0.9, 1.1, 5)[::-1] 
    
    if not TEMPLATE_FOLDER.is_dir():
        print(f"FATAL: Template folder not found at {TEMPLATE_FOLDER}.")
        return {
            'original_data': resize_for_web(img_bgr, 600),
            'detected_data': resize_for_web(img_bgr, 600),
            'match_summary': [f"Error: Template folder not found."],
        }
    
    template_paths = []
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    for ext in image_extensions:
        template_paths.extend(TEMPLATE_FOLDER.glob(ext))
        
    if not template_paths:
        return {
            'original_data': resize_for_web(img_bgr, 600),
            'detected_data': resize_for_web(img_bgr, 600),
            'match_summary': ["No templates found in folder."],
        }


    # Iterate through all template files directly
    for template_path in template_paths:
        name = template_path.stem 
        base_template_gray = cv2.imread(str(template_path), 0)
        
        if base_template_gray is None: continue

        for scale in SCALES:
            h_new = int(base_template_gray.shape[0] * scale)
            w_new = int(base_template_gray.shape[1] * scale)

            if w_new <= 0 or h_new <= 0 or w_new > scene_gray.shape[1] or h_new > scene_gray.shape[0]:
                continue

            template = cv2.resize(base_template_gray, (w_new, h_new))
            w, h = template.shape[::-1]

            # Match using Normalized Cross-Correlation Coefficient (TM_CCOEFF_NORMED)
            res = cv2.matchTemplate(scene_gray, template, cv2.TM_CCOEFF_NORMED)
            
            # Find locations where the score meets the detection threshold
            loc = np.where(res >= DETECTION_THRESHOLD)

            for pt in zip(*loc[::-1]):
                score = res[pt[1], pt[0]]
                
                # Store the full detection data
                all_detections.append({
                    'box': [pt[0], pt[1], w, h],
                    'score': score,
                    'name': name # Store the template name here
                })

    # --- NMS Filtering and Final Output Generation ---

    all_boxes_list = [d['box'] for d in all_detections]
    all_scores = [d['score'] for d in all_detections]
    
    match_summary = []
    
    if not all_boxes_list:
        return {
            'original_data': resize_for_web(img_bgr, 600),
            'detected_data': resize_for_web(img_bgr, 600),
            'match_summary': [f"No detections found above threshold ({DETECTION_THRESHOLD})."],
        }

    # Apply NMS
    indices = cv2.dnn.NMSBoxes(all_boxes_list, all_scores, DETECTION_THRESHOLD, NMS_THRESHOLD)
    
    detection_count = len(indices)

    for i in indices:
        idx = i[0] if isinstance(i, np.ndarray) else i 
        
        # Retrieve the full data for the detection that survived NMS
        detection = all_detections[idx]
        x, y, w, h = detection['box']
        score = detection['score']
        name = detection['name'] # CRITICAL: Use the actual template name

        
        # 1. Apply Gaussian Blur to the detected ROI
        roi = scene_processed[y:y + h, x:x + w].copy()
        if roi.size > 0:
            blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
            scene_processed[y:y + h, x:x + w] = blurred_roi
        
        # 2. Draw Bounding Box (Red color matching the reference)
        cv2.rectangle(scene_processed, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(scene_processed, name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        match_summary.append(f"{name} detected @ {score:.4f} (Region: x={x}, y={y})")


    # Final Summary: Count + Detections
    final_summary = match_summary
    
    final_summary.insert(0, f"Successfully detected {detection_count} unique objects.")
    
    return {
        'original_data': resize_for_web(img_bgr, 600),
        'detected_data': scene_processed,
        'match_summary': final_summary,
    }
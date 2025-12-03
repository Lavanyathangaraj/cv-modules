import cv2
import numpy as np
from typing import Dict
from .utils import resize_for_web 

# --------------------------------------------------------------------
# MODULE 3: EDGE, CORNER, BOUNDARY, AND ARUCO-BASED SEGMENTATION LOGIC
# --------------------------------------------------------------------


# --------------------------------------------------------------
# 1. GRADIENT + LOG PROCESSING
# --------------------------------------------------------------
def process_gradient_log(img_bgr: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Computes:
        - Gradient magnitude (from Sobel)
        - Gradient angle
        - Laplacian of Gaussian (LoG)
    Returns visualization-friendly versions of each output.
    """
    
    # Resize image to standard size for display
    img_bgr_resized = resize_for_web(img_bgr)
    gray = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2GRAY)
    
    # ---- 1. Gradient Calculation using Sobel ----
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)   # Gradient along x
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)   # Gradient along y
    
    magnitude = np.sqrt(gx**2 + gy**2)                # Gradient magnitude
    angle = np.arctan2(gy, gx)                        # Gradient direction
    
    # ---- 2. Laplacian of Gaussian (LoG) ----
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)       # Smooth first
    log_img = cv2.Laplacian(blurred, cv2.CV_64F)      # Apply Laplacian
    
    # ---- 3. Convert outputs for display ----
    mag_vis = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    angle_vis = ((angle + np.pi) * (255.0 / (2 * np.pi))).astype(np.uint8)
    log_vis = cv2.normalize(log_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return {
        'original': img_bgr_resized,
        'gradient_magnitude': mag_vis,
        'gradient_angle': angle_vis,
        'log': log_vis
    }



# --------------------------------------------------------------
# 2. EDGE + CORNER DETECTION
# --------------------------------------------------------------
def process_edge_corner(img_bgr: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Detects:
        - Edges using Canny
        - Corners using Harris Corner Detector
    Produces a final overlay (Cyan edges, Red corners).
    """
    
    img_bgr_resized = resize_for_web(img_bgr)
    gray = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2GRAY)
    
    # ---- 1. Edge Detection (Canny) ----
    blurred_gray_canny = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred_gray_canny, 50, 150)
    
    # ---- 2. Corner Detection (Harris) ----
    gray_float = np.float32(gray)
    harris_blur = cv2.GaussianBlur(gray_float, (3, 3), 0.5)
    
    dst_raw = cv2.cornerHarris(harris_blur, blockSize=2, ksize=3, k=0.04)
    
    # Create mask of corner locations
    corners_mask = np.zeros_like(gray, dtype=np.uint8)
    max_response = dst_raw.max()
    if max_response > 1e-6:
        threshold_val = 0.01 * max_response
        corners_mask[dst_raw > threshold_val] = 255
    
    # Visualization of raw Harris response
    dst_vis = np.zeros_like(gray, dtype=np.uint8)
    dst_vis[dst_raw > 0] = dst_raw[dst_raw > 0]
    dst_vis = cv2.normalize(dst_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # ---- 3. Final Color Overlay ----
    result = img_bgr_resized.copy()
    result[edges > 0] = [255, 255, 0]   # Cyan edges
    result[corners_mask > 0] = [0, 0, 255]  # Red corners
    
    return {
        'original': img_bgr_resized,
        'edges': edges,
        'corners': dst_vis,
        'final': result
    }



# --------------------------------------------------------------
# 3. BOUNDARY DETECTION USING CONTOURS
# --------------------------------------------------------------
def process_boundary(img_bgr: np.ndarray) -> np.ndarray:
    """Extracts object boundaries using Canny + Contours and draws them."""
    
    img_bgr_resized = resize_for_web(img_bgr)
    
    gray = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blur, 100, 200)
    
    # Find contours from edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours
    output = img_bgr_resized.copy()
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
    
    return output



# --------------------------------------------------------------
# 4. ARUCO MARKER–BASED SEGMENTATION
# --------------------------------------------------------------
def process_aruco_segmentation(img_bgr: np.ndarray) -> np.ndarray:
    """
    Detects ArUco markers and segments the object by computing
    a convex hull around marker centers.
    """
    
    img_bgr_resized = resize_for_web(img_bgr)
    
    # Load ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    gray = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2GRAY)
    
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict)
    
    output = img_bgr_resized.copy()
    
    if ids is not None and len(corners) >= 1:
        
        # Compute center of each marker
        centers = np.vstack([np.mean(c[0], axis=0) for c in corners])
        points_for_hull = centers
        
        # Convex hull over all marker centers
        if len(points_for_hull) >= 3:
            hull_indices = cv2.convexHull(points_for_hull.astype(np.float32), returnPoints=False)
            hull_points = points_for_hull[hull_indices.flatten()]
            
            # Draw the hull
            cv2.polylines(
                output,
                [hull_points.astype(np.int32).reshape((-1, 1, 2))],
                isClosed=True,
                color=(0, 255, 0),
                thickness=3
            )
            
            # Draw individual marker centers
            for center in centers:
                cv2.circle(output, tuple(center.astype(int)), 5, (255, 0, 0), -1)
    
    return output



# --------------------------------------------------------------
# 5. SIMULATED SAM2 SEGMENTATION VS ARUCO SEGMENTATION
# --------------------------------------------------------------
def process_segmentation_comparison(img_original: np.ndarray, img_aruco: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Creates a simulated SAM2 segmentation using adaptive thresholding.
    Compares it visually with the ArUco segmentation output.
    """
    
    # Resize both images to same size
    img_original_resized = resize_for_web(img_original)
    img_aruco_resized = cv2.resize(img_aruco, (img_original_resized.shape[1], img_original_resized.shape[0]))
    
    gray_original = cv2.cvtColor(img_original_resized, cv2.COLOR_BGR2GRAY)
    
    # Noise reduction before thresholding
    median_filtered_gray = cv2.medianBlur(gray_original, 11)
    
    # Simulated SAM2 mask (adaptive threshold)
    sam2_mask_simulated = cv2.adaptiveThreshold(
        median_filtered_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        5
    )
    
    # Morphological cleaning
    kernel = np.ones((5, 5), np.uint8)
    sam2_mask_simulated = cv2.morphologyEx(sam2_mask_simulated, cv2.MORPH_CLOSE, kernel)
    
    # Convert mask to red overlay
    red_mask = cv2.merge([np.zeros_like(sam2_mask_simulated),
                          np.zeros_like(sam2_mask_simulated),
                          sam2_mask_simulated])
    
    # Blend with original image
    sam2_output_final = cv2.addWeighted(
        img_original_resized, 0.7,
        red_mask, 0.3,
        0
    )
    
    return {
        'original': img_original_resized,
        'aruco_mask': img_aruco_resized,
        'sam2_mask': sam2_output_final
    }

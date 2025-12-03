import cv2
import numpy as np
from typing import Dict
from .utils import resize_for_web 

# --- Module 3: Edge and Corner Detection Logic ---

def process_gradient_log(img_bgr: np.ndarray) -> Dict[str, np.ndarray]:
    """Computes gradient magnitude, angle, and Laplacian of Gaussian (LoG)."""
    
    img_bgr_resized = resize_for_web(img_bgr)
    gray = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2GRAY)
    
    # 1. Gradient Calculation (Sobel)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx)
    
    # 2. Laplacian of Gaussian (LoG)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    log_img = cv2.Laplacian(blurred, cv2.CV_64F)
    
    # 3. Visualization Mapping
    mag_vis = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    angle_vis = ((angle + np.pi) * (255.0 / (2 * np.pi))).astype(np.uint8)
    log_vis = cv2.normalize(log_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return {
        'original': img_bgr_resized,
        'gradient_magnitude': mag_vis,
        'gradient_angle': angle_vis,
        'log': log_vis
    }

def process_edge_corner(img_bgr: np.ndarray) -> Dict[str, np.ndarray]:
    """Detects edges (Canny) and corners (Harris) and creates an overlay."""
    
    img_bgr_resized = resize_for_web(img_bgr)
    gray = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2GRAY)
    
    # 1. EDGE DETECTION (Canny)
    blurred_gray_canny = cv2.GaussianBlur(gray, (5, 5), 0) 
    edges = cv2.Canny(blurred_gray_canny, threshold1=50, threshold2=150) 
    
    # 2. CORNER DETECTION (Harris)
    gray_float = np.float32(gray)
    harris_blur = cv2.GaussianBlur(gray_float, (3, 3), 0.5)
    
    dst_raw = cv2.cornerHarris(harris_blur, blockSize=2, ksize=3, k=0.04)
    
    corners_mask = np.zeros_like(gray, dtype=np.uint8)
    max_response = dst_raw.max()
    if max_response > 1e-6:
        threshold_val = 0.01 * max_response 
        corners_mask[dst_raw > threshold_val] = 255
        
    dst_vis = np.zeros_like(gray, dtype=np.uint8)
    dst_vis[dst_raw > 0] = dst_raw[dst_raw > 0]
    dst_vis = cv2.normalize(dst_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 3. Final Overlay (Edges: Cyan, Corners: Red)
    result = img_bgr_resized.copy()
    result[edges > 0] = [255, 255, 0]      
    result[corners_mask > 0] = [0, 0, 255]      
    
    return {
        'original': img_bgr_resized,
        'edges': edges,
        'corners': dst_vis,
        'final': result
    }

def process_boundary(img_bgr: np.ndarray) -> np.ndarray:
    """Detects and draws object boundaries (contours) using Canny edges."""
    
    img_bgr_resized = resize_for_web(img_bgr)
    
    gray = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, threshold1=100, threshold2=200)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    
    output = img_bgr_resized.copy()
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2) 
    
    return output

def process_aruco_segmentation(img_bgr: np.ndarray) -> np.ndarray:
    """Segments an object boundary by drawing the convex hull of detected ArUco markers."""
    
    img_bgr_resized = resize_for_web(img_bgr)
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    gray = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2GRAY)
    
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict)
    
    output = img_bgr_resized.copy()
    
    if ids is not None and len(corners) >= 1:
        centers = np.vstack([np.mean(c[0], axis=0) for c in corners])
        points_for_hull = centers
        
        if len(points_for_hull) >= 3:
            hull_indices = cv2.convexHull(points_for_hull.astype(np.float32), returnPoints=False)
            hull_points = points_for_hull[hull_indices.flatten()]
            
            cv2.polylines(
                output, 
                [hull_points.astype(np.int32).reshape((-1, 1, 2))], 
                isClosed=True, 
                color=(0, 255, 0), 
                thickness=3
            )
            
            for center in centers:
                cv2.circle(output, tuple(center.astype(int)), 5, (255, 0, 0), -1) 
    
    return output

def process_segmentation_comparison(img_original: np.ndarray, img_aruco: np.ndarray) -> Dict[str, np.ndarray]:
    """Simulates SAM2 segmentation for comparison with ArUco segmentation."""
    
    img_original_resized = resize_for_web(img_original)
    img_aruco_resized = cv2.resize(img_aruco, (img_original_resized.shape[1], img_original_resized.shape[0])) 
    
    gray_original = cv2.cvtColor(img_original_resized, cv2.COLOR_BGR2GRAY)
    
    median_filtered_gray = cv2.medianBlur(gray_original, 11)
    
    sam2_mask_simulated = cv2.adaptiveThreshold(
        median_filtered_gray, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        5 
    )
    
    kernel = np.ones((5,5),np.uint8)
    sam2_mask_simulated = cv2.morphologyEx(sam2_mask_simulated, cv2.MORPH_CLOSE, kernel)
    
    red_mask = cv2.merge([np.zeros_like(sam2_mask_simulated), np.zeros_like(sam2_mask_simulated), sam2_mask_simulated])
    
    sam2_output_final = cv2.addWeighted(
        img_original_resized, 0.7, 
        red_mask, 
        0.3, 0)
    
    return {
        'original': img_original_resized,
        'aruco_mask': img_aruco_resized, 
        'sam2_mask': sam2_output_final   
    }
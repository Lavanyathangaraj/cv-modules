import cv2
import numpy as np
import time
from typing import List, Dict, Any, Tuple

from .utils import resize_to_max_dim, load_images_from_files_stitch, load_image_from_file

from .sift_logic import compute_sift, ransac_homography, match_descriptors 


def stitch_images(images: List[np.ndarray]) -> np.ndarray:
    """Stitches a list of images into a panorama using OpenCV's Stitcher."""
    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS) 
    status, pano = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        return pano
    elif status == cv2.Stitcher_ERR_NEED_MORE_IMGS:
        raise Exception("Stitching failed: Need more images or overlap is insufficient.")
    else:
        raise Exception(f"Stitching failed. Error code: {status}")

def resize_for_comparison(img1: np.ndarray, img2: np.ndarray, max_height: int = 600) -> np.ndarray:
    """Resizes and stacks two images horizontally for visual comparison."""
    h1, w1 = img1.shape[:2]
    scale1 = min(max_height / h1, 1.0)
    img1_resized = cv2.resize(img1, (int(w1 * scale1), int(h1 * scale1)), interpolation=cv2.INTER_AREA)
    
    h2, w2 = img2.shape[:2]
    scale2 = img1_resized.shape[0] / h2
    img2_resized = cv2.resize(img2, (int(w2 * scale2), img1_resized.shape[0]), interpolation=cv2.INTER_AREA)
    
    comparison = np.hstack((img1_resized, img2_resized))
    return comparison

def process_stitch_compare(sequential_images: List[np.ndarray], reference_pano: np.ndarray) -> Dict[str, Any]:
    """Stitches images, compares with a reference, and returns the result."""
    
    stitched_panorama = stitch_images(sequential_images)
    
    comparison_image = resize_for_comparison(stitched_panorama, reference_pano)

    message = f'Stitching successful. Comparison image shows: (Left: Your Stitch) vs (Right: Mobile Reference)'

    return {
        'comparison_image': comparison_image,
        'message': message
    }


def process_sift_compare(img1_bgr: np.ndarray, img2_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Compares OpenCV SIFT/RANSAC with a scratch implementation using the imported helper functions.
    """
    
    MAX_SIFT_DIM = 400 
    img1_bgr, _ = resize_to_max_dim(img1_bgr, MAX_SIFT_DIM)
    img2_bgr, _ = resize_to_max_dim(img2_bgr, MAX_SIFT_DIM)
    
    img1_gray_cv = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
    img2_gray_cv = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)
    
    # --- 1. OpenCV SIFT Baseline ---
    start_time_opencv = time.time()
    sift_opencv = cv2.SIFT_create()
    kp1_opencv, des1_opencv = sift_opencv.detectAndCompute(img1_gray_cv, None)
    kp2_opencv, des2_opencv = sift_opencv.detectAndCompute(img2_gray_cv, None)
    opencv_time = time.time() - start_time_opencv
    
    good_matches_opencv = []
    inliers_opencv = 0
    img_matches_opencv = None
    
    if des1_opencv is not None and des2_opencv is not None and des1_opencv.shape[0] > 0 and des2_opencv.shape[0] > 0:
        bf = cv2.BFMatcher()
        matches_opencv = bf.knnMatch(des1_opencv, des2_opencv, k=2) 
        
        for m, n in matches_opencv:
            if m.distance < 0.75 * n.distance:
                good_matches_opencv.append(m)
        
        if len(good_matches_opencv) >= 4:
            src_pts_opencv = np.float32([kp1_opencv[m.queryIdx].pt for m in good_matches_opencv]).reshape(-1, 1, 2)
            dst_pts_opencv = np.float32([kp2_opencv[m.trainIdx].pt for m in good_matches_opencv]).reshape(-1, 1, 2)
            
            H_opencv, mask_opencv = cv2.findHomography(src_pts_opencv, dst_pts_opencv, cv2.RANSAC, 5.0)
            if mask_opencv is not None:
                inliers_opencv = int(np.sum(mask_opencv))
                
                inlier_matches_opencv = [m for m, mask_val in zip(good_matches_opencv, mask_opencv.flatten()) if mask_val == 1][:50]
                if len(inlier_matches_opencv) > 0:
                    img_matches_opencv = cv2.drawMatches(img1_gray_cv, kp1_opencv, img2_gray_cv, kp2_opencv, 
                                                        inlier_matches_opencv, 
                                                        None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # --- 2. SIFT From Scratch Implementation---
    start_time_scratch = time.time()
    kp1_scratch, des1_scratch = compute_sift(img1_bgr) 
    kp2_scratch, des2_scratch = compute_sift(img2_bgr)
    
    matches_scratch_indices = []
    inliers_scratch = 0
    img_matches_scratch = None
    
    if des1_scratch.size > 0 and des2_scratch.size > 0:
        matches_scratch_indices = match_descriptors(des1_scratch, des2_scratch, ratio=0.75) 
        
    if len(matches_scratch_indices) >= 4:
        H_scratch, inliers_scratch_list, mask_scratch_cv = ransac_homography(
            kp1_scratch, kp2_scratch, matches_scratch_indices, threshold=5.0
        )
        
        if mask_scratch_cv is not None:
            inliers_scratch = len(inliers_scratch_list)
            mask_scratch = mask_scratch_cv.flatten().astype(np.uint8)
            
            kp1_scratch_cv = [cv2.KeyPoint(x=p[0], y=p[1], size=p[2], angle=p[3] * 180 / np.pi) for p in kp1_scratch]
            kp2_scratch_cv = [cv2.KeyPoint(x=p[0], y=p[1], size=p[2], angle=p[3] * 180 / np.pi) for p in kp2_scratch]
            all_scratch_dmatches = [cv2.DMatch(m[0], m[1], 0) for m in matches_scratch_indices]
            
            img_matches_scratch = cv2.drawMatches(img1_gray_cv, kp1_scratch_cv, img2_gray_cv, kp2_scratch_cv, 
                                                  all_scratch_dmatches, 
                                                  None, 
                                                  matchesMask=mask_scratch.tolist(), 
                                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    scratch_time_total = time.time() - start_time_scratch
    
    return {
        'metrics': {
            'kp1_opencv': len(kp1_opencv) if kp1_opencv is not None else 0,
            'kp2_opencv': len(kp2_opencv) if kp2_opencv is not None else 0,
            'matches_opencv': len(good_matches_opencv),
            'inliers_opencv': inliers_opencv,
            'time_opencv': round(opencv_time, 3),
            'kp1_scratch': len(kp1_scratch),
            'kp2_scratch': len(kp2_scratch),
            'matches_scratch': len(matches_scratch_indices),
            'inliers_scratch': inliers_scratch,
            'time_scratch': round(scratch_time_total, 3),
        },
        'images': {
            'opencv_matches': img_matches_opencv,
            'scratch_matches': img_matches_scratch,
        },
    }
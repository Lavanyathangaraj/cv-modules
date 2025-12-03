import os
import numpy as np
import cv2
from skimage import color, transform, filters
from skimage.util import img_as_float
from typing import List, Tuple, Optional, Any


def resize_to_max_dim(image: np.ndarray, max_dim: int) -> Tuple[np.ndarray, float]:
    """
    Resize image so that its largest dimension equals max_dim.
    Returns the resized image and the scale factor applied relative to the original.
    (NOTE: This is a standalone version for SIFT's internal use.)
    """
    if max_dim <= 0:
        return image.copy(), 1.0
    
    h, w = image.shape[:2]
    max_side = max(h, w)
    if max_side <= max_dim:
        return image.copy(), 1.0
    
    scale = max_dim / float(max_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def gaussian_pyramid(img_gray_float, num_octaves=4, num_scales=3):
    pyramids = []
    current_img = img_gray_float
    current_img = transform.resize(current_img, (current_img.shape[0] * 2, current_img.shape[1] * 2), anti_aliasing=True)
    
    for o in range(num_octaves):
        octave = []
        for s in range(num_scales + 3):
            sigma_base = 1.6
            k = 2 ** (1.0 / num_scales)
            sigma = sigma_base * (k ** s)
            octave.append(filters.gaussian(current_img, sigma=sigma, channel_axis=None, preserve_range=True))
        pyramids.append(octave)
        
        if o < num_octaves - 1:
            current_img = transform.resize(octave[num_scales], (octave[num_scales].shape[0] // 2, octave[num_scales].shape[1] // 2), anti_aliasing=True)
            
    return pyramids

def dog_pyramid(gp):
    dogs = []
    for octave in gp:
        dog_oct = []
        for i in range(1, len(octave)):
            dog_oct.append(octave[i] - octave[i-1])
        dogs.append(dog_oct)
    return dogs

def is_edge(dog, x, y, r=10):
    Dxx = dog[y, x+1] + dog[y, x-1] - 2*dog[y, x]
    Dyy = dog[y+1, x] + dog[y-1, x] - 2*dog[y, x]
    Dxy = (dog[y+1, x+1] - dog[y+1, x-1] - dog[y-1, x+1] + dog[y-1, x-1]) / 4.0
    
    Tr = Dxx + Dyy
    Det = Dxx*Dyy - Dxy*Dxy
    
    if Det <= 0:
        return True 
    
    r_ratio = (Tr**2)/Det
    return r_ratio > ((r+1)**2)/r

def detect_keypoints(dogs, threshold_ratio=0.03, edge_r=10):
    keypoints = []
    for o, octave in enumerate(dogs):
        for s in range(1, len(octave)-1):
            prev_dog = octave[s-1]
            curr_dog = octave[s]
            next_dog = octave[s+1]
            
            th = threshold_ratio * np.max(np.abs(curr_dog))
            rows, cols = curr_dog.shape
            
            for y in range(1, rows-1):
                for x in range(1, cols-1):
                    val = curr_dog[y, x]
                    
                    if abs(val) < th:
                        continue
                        
                    patch = curr_dog[y-1:y+2, x-1:x+2]
                    patch_prev = prev_dog[y-1:y+2, x-1:x+2]
                    patch_next = next_dog[y-1:y+2, x-1:x+2]
                    
                    combined = np.concatenate([patch_prev.flatten(),
                                               patch.flatten(),
                                               patch_next.flatten()])
                                               
                    if val == np.max(combined) or val == np.min(combined):
                        if not is_edge(curr_dog, x, y, r=edge_r):
                            keypoints.append((o, s, x, y))
    return keypoints


def assign_orientations(gp, keypoints, num_bins=36, peak_ratio=0.8):
    oriented = []
    
    for (o, s, x_int, y_int) in keypoints:
        img = gp[o][s]
        
        if x_int < 1 or y_int < 1 or x_int >= img.shape[1]-1 or y_int >= img.shape[0]-1:
            continue
            
        gx = img[y_int, x_int+1] - img[y_int, x_int-1]
        gy = img[y_int+1, x_int] - img[y_int-1, x_int]
        
        magnitude = np.sqrt(gx**2 + gy**2)
        angle = (np.arctan2(gy, gx) + 2*np.pi) % (2*np.pi)
        
        hist = np.zeros(num_bins)
        bin_idx = int(round((angle / (2*np.pi))*(num_bins-1)))
        hist[bin_idx] = magnitude

        max_bin = np.max(hist)
        
        for i, h in enumerate(hist):
            if h >= peak_ratio * max_bin:
                theta = (i + 0.5)*(2*np.pi/num_bins)
                oriented.append((o, s, float(x_int), float(y_int), theta))
                
    return oriented

def compute_descriptor(img_float, kp, window=16, cells=4):
    o, s, x, y, theta = kp
    x_int, y_int = int(round(x)), int(round(y))
    half = window // 2
    
    if x_int-half < 0 or x_int+half >= img_float.shape[1] or \
       y_int-half < 0 or y_int+half >= img_float.shape[0]:
        return None
        
    patch = img_float[y_int-half:y_int+half, x_int-half:x_int+half]
    
    gy, gx = np.gradient(patch)
    mag = np.sqrt(gx**2 + gy**2)
    
    ang_rad = np.arctan2(gy, gx) 
    ang = (ang_rad - theta) % (2*np.pi) 

    coords = np.meshgrid(np.arange(window), np.arange(window))
    cx, cy = window/2, window/2
    sigma_desc = window / 2.0 
    gauss = np.exp(-((coords[0]-cx)**2 + (coords[1]-cy)**2)/(2*sigma_desc**2))
    mag *= gauss

    desc = []
    cell = window // cells
    
    for cy_i in range(cells):
        for cx_i in range(cells):
            block_mag = mag[cy_i*cell:(cy_i+1)*cell, cx_i*cell:(cx_i+1)*cell]
            block_ang = ang[cy_i*cell:(cy_i+1)*cell, cx_i*cell:(cx_i+1)*cell]
            
            hist, _ = np.histogram(block_ang, bins=8, range=(0, 2*np.pi), weights=block_mag)
            desc.extend(hist)
            
    desc = np.array(desc, dtype=np.float32)
    
    norm = np.linalg.norm(desc)
    if norm > 0:
        desc /= norm
    
    desc = np.clip(desc, 0, 0.2)
    
    norm = np.linalg.norm(desc)
    if norm > 0:
        desc /= norm
        
    return desc

def compute_sift(img_bgr_or_gray: np.ndarray) -> Tuple[List[Tuple[float, float, float, float]], np.ndarray]:
    """
    Main SIFT function that runs the entire pipeline for one image.
    Returns: List of (x, y, size, angle_radians) tuples, and Descriptors (N x 128)
    """
    
    if len(img_bgr_or_gray.shape) == 3:
        img_gray = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_bgr_or_gray
        
    img_float = img_as_float(img_gray)
    
    gp = gaussian_pyramid(img_float)
    dp = dog_pyramid(gp)
    
    kps_octave = detect_keypoints(dp, threshold_ratio=0.03)
    oriented_octave = assign_orientations(gp, kps_octave)

    descriptors = []
    keypoints_out = []
    
    for kp in oriented_octave:
        o, s, x_octave, y_octave, theta = kp
        
        scale_factor = 2 ** o
        x_full = x_octave * scale_factor / 2.0
        y_full = y_octave * scale_factor / 2.0
        
        kp_full = (o, s, x_full, y_full, theta) 
        
        d = compute_descriptor(img_float, kp_full)
        
        if d is not None:
            size_approx = 1.6 * scale_factor 
            keypoints_out.append((x_full, y_full, size_approx, theta))
            descriptors.append(d)

    return keypoints_out, np.array(descriptors, dtype=np.float32)


def match_descriptors(desc1: np.ndarray, desc2: np.ndarray, ratio: float = 0.75) -> List[Tuple[int, int]]:
    """Matches descriptors using Lowe's ratio test."""
    matches = []
    if desc1.size == 0 or desc2.size == 0 or len(desc1) == 0 or len(desc2) == 0:
        return []
        
    for i, d1 in enumerate(desc1):
        dists = np.linalg.norm(desc2 - d1, axis=1)
        if len(dists) < 2:
            continue
        
        idx = np.argsort(dists)[:2]
        
        if dists[idx[0]] < ratio * dists[idx[1]]:
            matches.append((i, idx[0]))
            
    return matches

def ransac_homography(kp1: List[Tuple[float, float, float, float]], 
                      kp2: List[Tuple[float, float, float, float]], 
                      matches: List[Tuple[int, int]], 
                      threshold: float = 5.0) -> Tuple[Optional[np.ndarray], List[Tuple[int, int]], Optional[np.ndarray]]:
    """
    Filters matches and computes homography using OpenCV's RANSAC.
    Returns H, inliers list, and inlier mask array.
    """
    if len(matches) < 4:
        return None, [], None
        
    pts1 = np.float32([[kp1[i][0], kp1[i][1]] for i, _ in matches])
    pts2 = np.float32([[kp2[j][0], kp2[j][1]] for _, j in matches])
    
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, threshold)
    
    if mask is None:
        return H, [], None
        
    inliers = [matches[i] for i in range(len(matches)) if mask[i].ravel()[0] == 1]
    
    return H, inliers, mask
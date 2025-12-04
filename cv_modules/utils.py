import cv2
import numpy as np
import base64
from werkzeug.datastructures import FileStorage
from typing import List, Dict, Any, Tuple

# --- Helper Functions from integrated_server.py ---

def image_to_base64(img: np.ndarray) -> str:
    """Encodes a NumPy array image to a Base64 JPEG string."""
    if img is None:
        return ""
        
    if len(img.shape) == 2:
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 3 and img.shape[2] == 3 and img.dtype != np.uint8:
        img = img.astype(np.uint8)

    _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return base64.b64encode(buffer).decode('utf-8')

def load_image_from_file(file: FileStorage) -> np.ndarray:
    """Loads a single image from Flask FileStorage into OpenCV/NumPy format."""
    filestr = file.read()
    npimg = np.frombuffer(filestr, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR) 
    if img is None:
        raise ValueError(f"Failed to decode image: {file.filename}")
    return img

def load_images_from_files_stitch(files: List[FileStorage]) -> List[np.ndarray]:
    """Loads multiple images from Flask FileStorage objects."""
    images = []
    for file in files:
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode image: {file.filename}")
        images.append(img)
    return images

def resize_for_web(img: np.ndarray, max_dim: int = 400) -> np.ndarray:
    """Resizes an image to fit a maximum dimension while maintaining aspect ratio."""
    h, w = img.shape[:2]
    max_side = max(h, w)
    if max_side <= max_dim:
        return img
    
    scale = max_dim / float(max_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized

def resize_to_max_dim(img: np.ndarray, max_dim: int = 400) -> Tuple[np.ndarray, float]:
    """Resizes an image and returns the scale factor (used by Module 4 SIFT)."""
    h, w = img.shape[:2]
    max_side = max(h, w)
    if max_side <= max_dim:
        return img, 1.0
    
    scale = max_dim / float(max_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale

def clamp_roi_to_frame(roi: Tuple[int, int, int, int], frame_width: int, frame_height: int, min_size: int = 20) -> Tuple[int, int, int, int]:
    """Clamp ROI to be fully inside the frame and enforce a sensible minimum size."""
    x, y, w, h = roi
    try:
        x = int(round(x))
        y = int(round(y))
        w = int(round(w))
        h = int(round(h))
    except Exception:
        x, y, w, h = 0, 0, max(min_size, frame_width // 4), max(min_size, frame_height // 4)

    x = max(0, min(x, max(0, frame_width - 1)))
    y = max(0, min(y, max(0, frame_height - 1)))
    w = max(min_size, min(w, frame_width - x))
    h = max(min_size, min(h, frame_height - y))
    return (x, y, w, h)
    
def rotate_image(image, angle):
    """Rotates an image around its center by a specified angle."""
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    
    return cv2.warpAffine(image, M, (nW, nH), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
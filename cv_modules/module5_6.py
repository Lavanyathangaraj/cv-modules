import cv2
import numpy as np
import time
import os
from typing import Dict, Any, Tuple, Generator
from .utils import image_to_base64, clamp_roi_to_frame

# --------------------------
# GLOBAL TRACKING STATE
# --------------------------
tracker = None             
qr_tracker = None              
initial_roi = (150, 100, 340, 300)  
current_roi = initial_roi
tracker_initialized = False    
tracking_mode = 'marker-free'  
WORKING_CAMERA_ID = 0          

NPZ_FOLDER = "npz"             
NPZ_FILE_NAME = "sam2_data.npz"

INIT_DELAY_SECONDS = 5        
init_countdown_state = 0     
start_time_s = 0               


# --------------------------
# QR CODE TRACKER
# --------------------------
class QRTracker:
    """
    Handles detection, prediction, and drawing of QR codes.
    Uses velocity-based prediction if QR is temporarily lost.
    """
    
    def __init__(self):
        self.qr_detector = cv2.QRCodeDetector()
        self.last_center = None
        self.last_corners = None
        self.last_data = None
        self.lost_frames = 0
        self.max_lost_frames = 5
        self.velocity = None
        self.last_position = None
    
    def detect(self, frame):
        """
        Detect QR codes in the frame.
        If temporarily lost, predict location based on previous velocity.
        Returns: detected flag, center, corners, data
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected, center, corners, data = False, None, None, None
        
        # Try detecting QR codes
        try:
            retval, decoded_info, points, _ = self.qr_detector.detectAndDecodeMulti(gray)
            if retval and len(points) > 0 and decoded_info[0]:
                qr_corners = points[0]
                center = tuple(np.mean(qr_corners, axis=0).astype(int))
                detected = True
                corners = points
                data = decoded_info[0]
        except:
            pass

        # Update tracking or predict if lost
        if detected:
            if self.last_position is not None:
                dx = center[0] - self.last_position[0]
                dy = center[1] - self.last_position[1]
                self.velocity = (dx, dy)
            self.last_position = center
            self.last_center = center
            self.last_corners = corners
            self.last_data = data
            self.lost_frames = 0
        elif self.lost_frames < self.max_lost_frames and self.last_center is not None:
            self.lost_frames += 1
            if self.velocity is not None:
                center = (int(self.last_center[0] + self.velocity[0]),
                          int(self.last_center[1] + self.velocity[1]))
            else:
                center = self.last_center
            corners = self.last_corners
            data = self.last_data
            detected = True
        
        return detected, center, corners, data
    
    def draw(self, frame, center, corners=None, data=None):
        """
        Draw QR detection on frame:
        - Circle at center
        - Bounding polygon if corners available
        - QR data text
        - 'PREDICTED' label if using predicted location
        """
        if center is None:
            return frame
        
        cv2.circle(frame, center, 10, (0, 255, 0), -1)
        if corners is not None and corners[0] is not None:
            cv2.polylines(frame, [corners[0].astype(int)], True, (0, 255, 0), 2)
        if data is not None:
            display_data = data[:27] + "..." if len(data) > 30 else data
            cv2.putText(frame, f'QR: {display_data}', (center[0]+30, center[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if self.lost_frames > 0:
            cv2.putText(frame, 'PREDICTED', (center[0]-10, center[1]+40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        return frame


# --------------------------
# SAM2 INITIAL ROI LOADING
# --------------------------
def get_sam2_initial_bbox(npz_file_path, frame_shape):
    """
    Load initial bounding box from SAM2 NPZ file.
    If missing or invalid, return default ROI relative to frame size.
    """
    h, w = frame_shape[:2]
    try:
        if not os.path.exists(npz_file_path):
            raise FileNotFoundError()
        data = np.load(npz_file_path, allow_pickle=True)
        boxes = data.get('boxes', None)
        if boxes is None or boxes.size == 0:
            raise ValueError()
        init_box = boxes[0].astype(int)
        x, y, w_box, h_box = init_box[0], init_box[1], init_box[2]-init_box[0], init_box[3]-init_box[1]
        return (x, y, w_box, h_box)
    except Exception:
        # Fallback ROI
        return (int(w*0.3), int(h*0.3), int(w*0.4), int(h*0.4))


# --------------------------
# TRACKER INITIALIZATION
# --------------------------
def init_tracking_state(camera_id: int):
    """Initialize global tracker objects and set camera ID."""
    global tracker, qr_tracker, WORKING_CAMERA_ID
    WORKING_CAMERA_ID = camera_id
    qr_tracker = QRTracker()
    try:
        tracker = cv2.TrackerCSRT_create()
    except AttributeError:
        print("Warning: TrackerCSRT unavailable.")


def set_tracking_mode(new_mode: str) -> None:
    """Set tracking mode and reset tracker/ROI state."""
    global tracking_mode, tracker_initialized, current_roi, initial_roi, init_countdown_state, start_time_s
    if new_mode not in ['marker-free', 'marker', 'sam']:
        raise ValueError("Invalid mode")
    tracking_mode = new_mode
    tracker_initialized = False
    init_countdown_state = 0
    start_time_s = 0
    current_roi = initial_roi


def set_new_roi(x: int, y: int, w: int, h: int) -> None:
    """
    Initialize tracker with a new ROI using a fresh camera frame.
    Ensures ROI stays within frame bounds.
    """
    global tracker, current_roi, tracker_initialized, WORKING_CAMERA_ID
    new_roi = (x, y, w, h)
    temp_cap = None
    try:
        temp_cap = cv2.VideoCapture(WORKING_CAMERA_ID, cv2.CAP_DSHOW)
        if not temp_cap.isOpened():
            temp_cap = cv2.VideoCapture(WORKING_CAMERA_ID, cv2.CAP_FFMPEG)
        if not temp_cap.isOpened():
            temp_cap = cv2.VideoCapture(WORKING_CAMERA_ID)
        if not temp_cap.isOpened():
            raise IOError(f"Camera {WORKING_CAMERA_ID} inaccessible.")
        temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        success, frame = temp_cap.read()
        if not success or tracker is None:
            raise IOError("Failed to read frame or tracker not initialized.")
        fh, fw = frame.shape[:2]
        safe_new_roi = clamp_roi_to_frame(new_roi, fw, fh)
        tracker.init(frame, safe_new_roi)
        current_roi = safe_new_roi
        tracker_initialized = True
    finally:
        if temp_cap is not None:
            temp_cap.release()


# --------------------------
# FRAME GENERATOR
# --------------------------
def gen_frames() -> Generator[bytes, None, None]:
    """
    Capture webcam frames and apply tracking based on mode.
    Yields frames in MJPEG format suitable for web streaming.
    """
    global tracker, qr_tracker, tracker_initialized, current_roi, tracking_mode, WORKING_CAMERA_ID, init_countdown_state, start_time_s
    cap = cv2.VideoCapture(WORKING_CAMERA_ID, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(WORKING_CAMERA_ID, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(WORKING_CAMERA_ID)
    if not cap.isOpened():
        print(f"Camera {WORKING_CAMERA_ID} inaccessible.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Countdown before tracker initialization (marker-free / SAM)
    if (tracking_mode in ['marker-free', 'sam']) and not tracker_initialized:
        if init_countdown_state == 0:
            init_countdown_state = 1
            start_time_s = time.time()
        while cap.isOpened() and init_countdown_state == 1:
            success, frame = cap.read()
            if not success: break
            current_time_s = time.time()
            time_remaining = max(0, INIT_DELAY_SECONDS - int(current_time_s - start_time_s))
            guide_roi = get_sam2_initial_bbox(os.path.join(NPZ_FOLDER, NPZ_FILE_NAME), frame.shape) if tracking_mode=='sam' else clamp_roi_to_frame(initial_roi, *frame.shape[:2])
            p1 = (guide_roi[0], guide_roi[1])
            p2 = (guide_roi[0]+guide_roi[2], guide_roi[1]+guide_roi[3])
            cv2.rectangle(frame, p1, p2, (0,255,255),2)
            msg = f"TRACKING IN {time_remaining}s" if time_remaining>0 else "INITIALIZING..."
            cv2.putText(frame, msg, (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),2)
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            if time_remaining <= 0 and (current_time_s - start_time_s > INIT_DELAY_SECONDS+0.5):
                init_countdown_state = -1
                break

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        # Marker-free using CSRT
        if tracking_mode in ['marker-free', 'sam'] and tracker is not None and tracker_initialized:
            ok, new_box = tracker.update(frame)
            tracking_status = "Tracking Lost"
            color = (0,0,255)
            if ok:
                x, y, w, h = new_box
                cx, cy = x+w/2, y+h/2
                fw, fh = frame.shape[1], frame.shape[0]
                if 0.07*fw < cx < 0.93*fw and 0.07*fh < cy < 0.93*fh:
                    tracking_status = "CSRT Tracking" if tracking_mode=='marker-free' else "SAM2 Initialized"
                    color = (0,255,0) if tracking_mode=='marker-free' else (255,100,0)
                    current_roi = new_box
                p1 = (int(x), int(y))
                p2 = (int(x+w), int(y+h))
                cv2.rectangle(frame, p1, p2, color, 2)
            cv2.putText(frame, tracking_status, (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)

        # Marker-based tracking using QR
        elif tracking_mode=='marker' and qr_tracker is not None:
            detected, center, corners, data = qr_tracker.detect(frame)
            if detected:
                frame = qr_tracker.draw(frame, center, corners, data)
                cv2.putText(frame,'QR Code Found',(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            else:
                cv2.putText(frame,'NO QR CODE',(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    if cap is not None:
        cap.release()
        print(f"Camera {WORKING_CAMERA_ID} released.")


# --------------------------
# CAMERA UTILS
# --------------------------
def try_open_camera(id, backend):
    """Try opening camera with a specific backend. Return cap object if successful."""
    try:
        temp_cap = cv2.VideoCapture(id, backend)
        time.sleep(0.1)
        if temp_cap.isOpened():
            ret, _ = temp_cap.read()
            if ret: return temp_cap
            temp_cap.release()
    except: pass
    return None

def find_working_camera_id():
    """Scan camera IDs 0-2 and return the first working one."""
    for i in range(3):
        cap_test = try_open_camera(i, cv2.CAP_DSHOW)
        if cap_test is None:
            cap_test = try_open_camera(i, cv2.CAP_FFMPEG)
        if cap_test is not None:
            cap_test.release()
            return i
    return 0

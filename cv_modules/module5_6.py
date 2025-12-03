import cv2
import numpy as np
import time
import os
from typing import Dict, Any, Tuple, Generator
from .utils import image_to_base64, clamp_roi_to_frame # Import necessary helpers

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

class QRTracker:
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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected, center, corners, data = False, None, None, None
        
        try:
            retval, decoded_info, points, straight_qrcode = self.qr_detector.detectAndDecodeMulti(gray)
            if retval and len(points) > 0 and decoded_info[0]:
                qr_corners = points[0]
                center = np.mean(qr_corners, axis=0).astype(int)
                center = tuple(center)
                detected = True
                corners = points
                data = decoded_info[0]
        except:
            pass 

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
                center = (int(self.last_center[0] + self.velocity[0]), int(self.last_center[1] + self.velocity[1]))
            else:
                center = self.last_center
            corners = self.last_corners
            data = self.last_data
            detected = True 
        
        return detected, center, corners, data
    
    def draw(self, frame, center, corners=None, data=None):
        if center is None:
            return frame
        
        cv2.circle(frame, center, 10, (0, 255, 0), -1)
        if corners is not None and corners[0] is not None:
            qr_corners = corners[0].astype(int)
            cv2.polylines(frame, [qr_corners], True, (0, 255, 0), 2)
        
        if data is not None:
            display_data = data[:27] + "..." if len(data) > 30 else data
            cv2.putText(frame, f'QR: {display_data}', (center[0] + 30, center[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if self.lost_frames > 0 and self.lost_frames <= self.max_lost_frames:
            cv2.putText(frame, 'PREDICTED', (center[0] - 10, center[1] + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return frame

def get_sam2_initial_bbox(npz_file_path, frame_shape):
    h, w = frame_shape[:2]
    
    try:
        if not os.path.exists(npz_file_path):
            raise FileNotFoundError("NPZ file not found, using dummy box.")
            
        data = np.load(npz_file_path, allow_pickle=True)
        boxes = data.get('boxes', None)
        
        if boxes is None or boxes.size == 0:
            raise ValueError("NPZ file is empty or missing 'boxes'.")
            
        init_box_raw = boxes[0].astype(int)
        
        x = init_box_raw[0]
        y = init_box_raw[1]
        w_box = init_box_raw[2] - x
        h_box = init_box_raw[3] - y
        
        return (x, y, w_box, h_box)
        
    except Exception as e:
        print(f"Error loading SAM2 NPZ: {e}. Falling back to default ROI.")
        return (int(w*0.3), int(h*0.3), int(w*0.4), int(h*0.4))



def init_tracking_state(camera_id: int):
    """Initializes global tracker objects and sets the camera ID."""
    global tracker, qr_tracker, WORKING_CAMERA_ID
    
    WORKING_CAMERA_ID = camera_id
    
    qr_tracker = QRTracker() 
    try:
        tracker = cv2.TrackerCSRT_create()
    except AttributeError:
        print("Warning: Could not create cv2.TrackerCSRT_create(). Tracking module may fail.")


def set_tracking_mode(new_mode: str) -> None:
    """Sets the global tracking mode and resets initialization state."""
    global tracking_mode, tracker_initialized, current_roi, initial_roi, init_countdown_state, start_time_s
    
    if new_mode not in ['marker-free', 'marker', 'sam']:
        raise ValueError("Invalid tracking mode.")

    tracking_mode = new_mode
    tracker_initialized = False
    init_countdown_state = 0 
    start_time_s = 0 
    
    if new_mode == 'sam':
        current_roi = initial_roi 
    else:
        current_roi = initial_roi


def set_new_roi(x: int, y: int, w: int, h: int) -> None:
    """
    Sets a new ROI for the CSRT tracker using a fresh camera frame for initialization.
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
            raise IOError(f"Failed to read frame: Camera ID {WORKING_CAMERA_ID} inaccessible.")

        temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        success, frame = temp_cap.read()
        
        if not success:
            raise IOError("Failed to read frame for ROI initialization.")
            
        if tracker is None:
             raise Exception("Tracker object not initialized.")
             
        fh, fw = frame.shape[:2]
        safe_new_roi = clamp_roi_to_frame(new_roi, fw, fh)
        
        tracker.init(frame, safe_new_roi)
        current_roi = safe_new_roi
        tracker_initialized = True
        
    finally:
        if temp_cap is not None:
            temp_cap.release()
            


def gen_frames() -> Generator[bytes, None, None]:
    """
    Generator that captures frames from the webcam and applies the selected tracking logic.
    """
    global tracker, qr_tracker, tracker_initialized, current_roi, initial_roi, tracking_mode, WORKING_CAMERA_ID, init_countdown_state, start_time_s
    
    cap = cv2.VideoCapture(WORKING_CAMERA_ID, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(WORKING_CAMERA_ID, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(WORKING_CAMERA_ID)
    
    if not cap.isOpened():
        print(f"FATAL STREAM ERROR: Camera ID {WORKING_CAMERA_ID} locked or inaccessible.")
        return 
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    
    if (tracking_mode == 'marker-free' or tracking_mode == 'sam') and not tracker_initialized:
        
        if init_countdown_state == 0:
            init_countdown_state = 1
            start_time_s = time.time()

        
        while cap.isOpened() and init_countdown_state == 1:
            success, frame = cap.read()
            if not success: break
            
            current_time_s = time.time()
            time_elapsed = current_time_s - start_time_s
            
            time_remaining = max(0, INIT_DELAY_SECONDS - int(time_elapsed))
            
            if tracking_mode == 'sam':
                guide_roi = get_sam2_initial_bbox(os.path.join(NPZ_FOLDER, NPZ_FILE_NAME), frame.shape)
            else: 
                 fh, fw = frame.shape[:2]
                 guide_roi = clamp_roi_to_frame(initial_roi, fw, fh)
                 
            p1 = (guide_roi[0], guide_roi[1])
            p2 = (guide_roi[0] + guide_roi[2], guide_roi[1] + guide_roi[3])
            cv2.rectangle(frame, p1, p2, (0, 255, 255), 2)

            if time_remaining > 0:
                 cv2.putText(frame, f"PLACE OBJECT: TRACKING IN {time_remaining}s", (100, 50), 
                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                 cv2.putText(frame, "INITIALIZING...", (100, 50), 
                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                 
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            if time_remaining <= 0 and (current_time_s - start_time_s > INIT_DELAY_SECONDS + 0.5):
                init_countdown_state = -1 
                break
        
        if init_countdown_state == -1:
            success, init_frame = cap.read()
            
            if success and init_frame is not None and tracker is not None:
                fh, fw = init_frame.shape[:2]
                
                if tracking_mode == 'sam':
                    npz_file_path = os.path.join(NPZ_FOLDER, NPZ_FILE_NAME)
                    current_roi = get_sam2_initial_bbox(npz_file_path, init_frame.shape)
                
                safe_roi = clamp_roi_to_frame(current_roi, fw, fh)
                try:

                    tracker.init(init_frame, safe_roi)
                    tracker_initialized = True
                    init_countdown_state = 0 
                    print("Tracker initialized successfully after delay.")
                except cv2.error as e:
                    print(f"CSRT Tracker initialization failed: {e}. Proceeding without tracking.")
                    init_countdown_state = 0
            else:
                 init_countdown_state = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        if (tracking_mode == 'marker-free' or tracking_mode == 'sam') and tracker is not None and tracker_initialized:
            
            ok, new_box = tracker.update(frame)

            
            tracking_status = "Tracking Lost"
            color = (0, 0, 255)
            
            if ok:
                x, y, w, h = new_box
                center_x = x + w / 2
                center_y = y + h / 2
                
                frame_width = frame.shape[1]
                frame_height = frame.shape[0]
                
                if (center_x > 0.07 * frame_width and center_x < 0.93 * frame_width and
                    center_y > 0.07 * frame_height and center_y < 0.93 * frame_height):
                    
                    tracking_status = "CSRT Tracking" if tracking_mode == 'marker-free' else "SAM2 Initialized"
                    color = (0, 255, 0) if tracking_mode == 'marker-free' else (255, 100, 0) 
                    
                    current_roi = new_box
            
            if tracking_status != "Tracking Lost":
                p1 = (int(new_box[0]), int(new_box[1]))
                p2 = (int(new_box[0] + new_box[2]), int(new_box[1] + new_box[3]))
                cv2.rectangle(frame, p1, p2, color, 2)
            
            cv2.putText(frame, tracking_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            
        elif tracking_mode == 'marker' and qr_tracker is not None:
            # QR Code Tracking
            detected, center, corners, data = qr_tracker.detect(frame)
            if detected:
                frame = qr_tracker.draw(frame, center, corners, data)
                cv2.putText(frame, 'QR Code Found', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'NO QR CODE', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               
    if cap is not None:
        cap.release()
        print(f"INFO: Camera ID {WORKING_CAMERA_ID} released after stream end.")



def try_open_camera(id, backend):
    """Attempt to open camera with specified ID and backend, returning cap object if successful."""
    try:
        temp_cap = cv2.VideoCapture(id, backend)
        time.sleep(0.1) 
        if temp_cap.isOpened():
            ret, frame = temp_cap.read()
            if ret:
                return temp_cap
            temp_cap.release()
    except Exception:
        pass
    return None

def find_working_camera_id():
    """Tries IDs 0, 1, 2 for reliable startup."""
    for i in range(3): 
        cap_test = try_open_camera(i, cv2.CAP_DSHOW)
        if cap_test is None:
            cap_test = try_open_camera(i, cv2.CAP_FFMPEG)
        
        if cap_test is not None:
            cap_test.release()
            return i
    
    return 0
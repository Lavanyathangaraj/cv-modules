import cv2
import numpy as np
import csv
import mediapipe as mp
import time
from typing import Dict, Any, Generator

# --- Camera Intrinsics (Fixed) ---
FX_STEREO = 3081.84216 #
FY_STEREO = 3077.33506 #

# --- Global Constants for Pose/Hand CSV ---
CSV_FILE_PATH = "pose_landmarks_m7.csv"
CSV_COLUMNS = ['frame', 'body_landmark', 'x', 'y', 'z', 'visibility', 'hand_label']
WORKING_CAMERA_ID = 0 # Camera ID set on app startup


# --- Module 7: Calibrated Stereo Logic ---

def compute_stereo_dimension(data: Dict[str, Any]) -> Dict[str, float]:
    """
    Computes depth (Z) and real-world dimensions (dX, dY) from pixel coordinates 
    in a stereo pair and a known baseline (B).
    
    Formula: Z = (F * B) / disparity
    """
    try:
        B = float(data['B']) 
    
        # Pixel points
        L1 = data['L1']
        R1 = data['R1']
        L2 = data['L2']
        R2 = data['R2']
        
        # Ground truth for error calculation (received in meters)
        W_true_m = float(data.get('W_true', 0)) 
        H_true_m = float(data.get('H_true', 0)) 
    
        # Disparities (L - R)
        d1 = (L1['x'] - R1['x'])
        d2 = (L2['x'] - R2['x'])
        
        if abs(d1) < 1e-6 or abs(d2) < 1e-6:
             raise ValueError("Disparity is too close to zero (check point alignment or depth).")

        # Compute depth for each point 
        Z1 = (FX_STEREO * B) / d1
        Z2 = (FX_STEREO * B) / d2
    
        # Average Z
        Z = (Z1 + Z2) / 2
    
        # Size in pixels (Left image)
        dx = abs(L2['x'] - L1['x'])
        dy = abs(L2['y'] - L1['y'])
    
        # Size in real world (m)
        dX = (dx * Z) / FX_STEREO
        dY = (dy * Z) / FY_STEREO
        
        # Error calculation 
        error_W = 0.0
        error_H = 0.0
        
        if W_true_m > 1e-6:
            error_W = abs(dX - W_true_m) # Error for Estimated Width (dX)
        if H_true_m > 1e-6:
            error_H = abs(dY - H_true_m) # Error for Estimated Height (dY) - Corrected to use H_true_m
        
        return {
            "disparity_1": round(d1, 4),
            "disparity_2": round(d2, 4),
            "Z1": round(Z1, 4),
            "Z2": round(Z2, 4),
            "Z_avg": round(Z, 4),
            "dX_estimated": round(dX, 4),
            "dY_estimated": round(dY, 4),
            "error_W": round(error_W, 4),
            "error_H": round(error_H, 4)
        }
    except Exception as e:
        raise ValueError(f"Stereo computation failed: {str(e)}")


# --- Module 7: Pose & Hand Tracking Stream (from integrated_server.py) ---

def gen_frames_pose_hand(camera_id: int) -> Generator[bytes, None, None]:
    """
    Generator for MediaPipe Pose and Hand Tracking. Saves landmarks to CSV and streams annotated frames.
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2) as hands:
        
            cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(camera_id, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap = cv2.VideoCapture(camera_id)
                
            if not cap.isOpened():
                print(f"FATAL STREAM ERROR: Camera ID {camera_id} locked for pose/hand stream.")
                return 

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            frame_count = 0
            
            # Open CSV file for writing (overwrite previous data)
            with open(CSV_FILE_PATH, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(CSV_COLUMNS)
            
                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        break
                    frame_count += 1
                    
                    image = cv2.flip(frame, 1)
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    pose_results = pose.process(image)
                    hand_results = hands.process(image)
                    
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
                    
                    # --- 1. Draw Pose Landmarks & Save to CSV ---
                    if pose_results.pose_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            pose_results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                        )
                        
                        # Save Pose Landmarks to CSV
                        for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
                            csv_writer.writerow([
                                frame_count, 
                                mp_pose.PoseLandmark(idx).name,
                                round(landmark.x, 6), 
                                round(landmark.y, 6), 
                                round(landmark.z, 6), 
                                round(landmark.visibility, 6),
                                'N/A' 
                            ])
                        
                    # --- 2. Draw Hand Landmarks & Save to CSV ---
                    if hand_results.multi_hand_landmarks:
                        for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                            mp_drawing.draw_landmarks(
                                image,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                            )

                            hand_label = hand_results.multi_handedness[hand_idx].classification[0].label

                            # Save Hand Landmarks to CSV
                            for idx, landmark in enumerate(hand_landmarks.landmark):
                                csv_writer.writerow([
                                    frame_count, 
                                    f'HAND_{idx}', 
                                    round(landmark.x, 6), 
                                    round(landmark.y, 6), 
                                    round(landmark.z, 6), 
                                    round(landmark.visibility, 6),
                                    hand_label
                                ])

                    ret, buffer = cv2.imencode('.jpg', image)
                    frame_bytes = buffer.tobytes()
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                           
            if cap is not None:
                cap.release()
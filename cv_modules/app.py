import os
import time
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, Response
from werkzeug.datastructures import FileStorage
from typing import List, Dict, Union, Any, Tuple

# --- Import Core Logic from Separate Files ---
from . import utils 
from . import module1
from . import module2
from . import module3
from . import module4
from . import module5_6
from . import module7

app = Flask(__name__)

# --- Global Initialization ---
WORKING_CAMERA_ID = module5_6.find_working_camera_id()
module5_6.init_tracking_state(WORKING_CAMERA_ID)
module7.WORKING_CAMERA_ID = WORKING_CAMERA_ID 

# Ensure NPZ folder exists for SAM2 tracking (M5/6)
if not os.path.exists(module5_6.NPZ_FOLDER):
    os.makedirs(module5_6.NPZ_FOLDER)

# ============================================================================
# --- STATIC FILE ROUTES ---
# ============================================================================

@app.route('/')
def index():
    """Serves the main index.html file."""
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'index.html') 

@app.route('/<path:filename>')
def serve_static(filename):
    """Serves all static files (HTML, CSS, JS, PDFs)."""
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), filename)


# ============================================================================
# --- MODULE 1: 3D DIMENSION MEASUREMENT (/compute) ---
# ============================================================================

@app.route('/compute', methods=['POST'])
def compute():
    try:
        data = request.json
        results = module1.calculate_dimension_m1(data)
        return jsonify(results)
    except Exception as e:
        print(f"Error in /compute (M1): {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# --- MODULE 2: FILTER & OBJECT DETECTION (/filter, /detect_template) ---
# ============================================================================

@app.route('/filter', methods=['POST'])
def handle_filter():
    try:
        image_file = request.files.get('image')
        sigma = float(request.form.get('sigma', 3.0)) 
        K = float(request.form.get('K', 0.1)) 

        if not image_file:
            return jsonify({'error': 'No image file uploaded.'}), 400

        img_bgr = utils.load_image_from_file(image_file)
        
        images_dict = module2.process_image_filter(img_bgr, sigma, K)
        
        return jsonify({
            'original_data': utils.image_to_base64(images_dict['original_data']),
            'blurred_data': utils.image_to_base64(images_dict['blurred_data']),
            'recovered_data': utils.image_to_base64(images_dict['recovered_data']),
            'sigma': sigma,
            'K': K
        })
    except Exception as e:
        print(f"Error in /filter (M2): {e}")
        return jsonify({'error': f'Image filtering failed: {str(e)}'}), 500

@app.route('/detect_template', methods=['POST'])
def detect_template():
    try:
        scene_file = request.files.get('scene_image')
        if not scene_file:
            return jsonify({'error': 'No scene image provided.'}), 400
        
        img_bgr = utils.load_image_from_file(scene_file)
        
        results = module2.process_template_matching(img_bgr)

        return jsonify({
            'original_data': utils.image_to_base64(results['original_data']),
            'detected_data': utils.image_to_base64(results['detected_data']),
            'match_summary': results['match_summary'],
            'status': 'Multi-scale and multi-orientation search complete.'
        })

    except Exception as e:
        print(f"Error in /detect_template (M2): {e}")
        return jsonify({'error': f'Template matching failed: {str(e)}'}), 500


# ============================================================================
# --- MODULE 3: EDGE & CORNER DETECTION (/detect_*, /segment_*) ---
# ============================================================================

@app.route('/detect_gradient', methods=['POST'])
def detect_gradient():
    try:
        file = request.files.get('image')
        if not file: return jsonify({'error': 'No image provided'}), 400
        img_bgr = utils.load_image_from_file(file)
        
        images_dict = module3.process_gradient_log(img_bgr)
        
        return jsonify({
            'images': {
                'original': utils.image_to_base64(images_dict['original']),
                'gradient_magnitude': utils.image_to_base64(images_dict['gradient_magnitude']),
                'gradient_angle': utils.image_to_base64(images_dict['gradient_angle']),
                'log': utils.image_to_base64(images_dict['log'])
            }
        })
    except Exception as e:
        print(f"Error in /detect_gradient (M3): {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/detect_edge_corner', methods=['POST'])
def detect_edge_corner():
    try:
        file = request.files.get('image')
        if not file: return jsonify({'error': 'No image provided'}), 400
        img_bgr = utils.load_image_from_file(file)
        
        images_dict = module3.process_edge_corner(img_bgr)
        
        return jsonify({
            'images': {
                'original': utils.image_to_base64(images_dict['original']),
                'edges': utils.image_to_base64(images_dict['edges']),
                'corners': utils.image_to_base64(images_dict['corners']),
                'final': utils.image_to_base64(images_dict['final'])
            }
        })
    except Exception as e:
        print(f"Error in /detect_edge_corner (M3): {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/detect_boundary', methods=['POST'])
def detect_boundary():
    try:
        file = request.files.get('image')
        if not file: return jsonify({'error': 'No image provided'}), 400
        img_bgr = utils.load_image_from_file(file)
        
        output_img = module3.process_boundary(img_bgr)
        
        return jsonify({
            'images': {
                'output': utils.image_to_base64(output_img),
            }
        })
    except Exception as e:
        print(f"Error in /detect_boundary (M3): {e}")
        return jsonify({'error': f'Image detection failed: {str(e)}'}), 500

@app.route('/segment_aruco', methods=['POST'])
def segment_aruco():
    try:
        file = request.files.get('image')
        if not file: return jsonify({'error': 'No image provided'}), 400
        img_bgr = utils.load_image_from_file(file)
        
        segmented_img = module3.process_aruco_segmentation(img_bgr)
        
        return jsonify({
            'images': {
                'segmented': utils.image_to_base64(segmented_img),
            }
        })
    except Exception as e:
        print(f"Error in /segment_aruco (M3): {e}")
        return jsonify({'error': f'ArUco Segmentation Failed: {str(e)}'}), 500

@app.route('/segment_comparison', methods=['POST'])
def segment_comparison():
    try:
        original_file = request.files.get('original_file')
        aruco_segmented_file = request.files.get('aruco_segmented_file')

        if not original_file or not aruco_segmented_file:
            return jsonify({'error': 'Both original image and ArUco segmented image are required.'}), 400

        img_original = utils.load_image_from_file(original_file)
        img_aruco = utils.load_image_from_file(aruco_segmented_file)
        
        images_dict = module3.process_segmentation_comparison(img_original, img_aruco)
        
        return jsonify({
            'images': {
                'original': utils.image_to_base64(images_dict['original']),
                'aruco_mask': utils.image_to_base64(images_dict['aruco_mask']), 
                'sam2_mask': utils.image_to_base64(images_dict['sam2_mask'])   
            }
        })
    except Exception as e:
        print(f"Error in /segment_comparison (M3): {e}")
        return jsonify({'error': f'Segmentation Comparison Failed: {str(e)}'}), 500


# ============================================================================
# --- MODULE 4: STITCHING & SIFT (/stitch_compare, /sift_compare) ---
# ============================================================================

@app.route('/stitch_compare', methods=['POST'])
def handle_stitch_compare():
    try:
        sequential_files = request.files.getlist('sequential_files')
        reference_file = request.files.get('reference_file')

        if len(sequential_files) < 2:
            return jsonify({'error': 'Please upload at least 2 sequential images.'}), 400
        if not reference_file:
            return jsonify({'error': 'Please upload the mobile reference panorama.'}), 400
        
        sequential_images = utils.load_images_from_files_stitch(sequential_files)
        reference_pano = utils.load_image_from_file(reference_file)
        
        results = module4.process_stitch_compare(sequential_images, reference_pano)

        return jsonify({
            'comparison_image_data': utils.image_to_base64(results['comparison_image']),
            'message': results['message']
        })

    except Exception as e:
        print(f"Error in /stitch_compare (M4): {e}")
        return jsonify({'error': f'Stitching and comparison failed: {str(e)}'}), 500

@app.route('/sift_compare', methods=['POST'])
def handle_sift_compare():
    try:
        files = request.files.getlist('files')
        if len(files) != 2:
            return jsonify({'error': 'Please upload exactly two images for SIFT comparison.'}), 400
        
        img1_bgr = utils.load_image_from_file(files[0])
        img2_bgr = utils.load_image_from_file(files[1])
        
        results = module4.process_sift_compare(img1_bgr, img2_bgr)
        
        results['images']['opencv_matches'] = utils.image_to_base64(results['images']['opencv_matches'])
        results['images']['scratch_matches'] = utils.image_to_base64(results['images']['scratch_matches'])
        
        results['status'] = 'Comparison successful.'
        
        return jsonify(results)

    except Exception as e:
        print(f"Error in /sift_compare (M4): {e}")
        return jsonify({'error': f'SIFT comparison failed: {str(e)}'}), 500


# ============================================================================
# --- MODULE 5-6: TRACKING ROUTES (VIDEO STREAMS) ---
# ============================================================================

@app.route('/start_tracking_mode', methods=['POST'])
def start_tracking_mode():
    try:
        data = request.json
        new_mode = data.get('mode', 'marker-free')
        
        module5_6.set_tracking_mode(new_mode)

        return jsonify({'status': 'Success', 'message': f'Mode set to {new_mode}. Stream starting.'})
        
    except Exception as e:
        print(f"Error initializing tracking mode: {e}")
        return jsonify({'status': 'Error', 'message': f"Initialization failed: {str(e)}"}), 500


@app.route('/video_feed_marker')
@app.route('/video_feed_tracking')
@app.route('/video_feed_sam2')
def video_feed_tracking_universal():
    """Universal route for all M5/6 video feeds, using the global mode state."""
    try:
        return Response(module5_6.gen_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Tracking error: {e}")
        return Response(f"Error: {e}", status=500, mimetype='text/plain')

@app.route('/get_snapshot')
def get_snapshot():
    try:
        base64_image = module5_6.get_snapshot(module5_6.WORKING_CAMERA_ID)
        
        return jsonify({
            'status': 'Success',
            'image_data': base64_image
        })

    except Exception as e:
        print(f"Error in /get_snapshot: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/set_roi', methods=['POST'])
def set_roi():
    try:
        data = request.json
        x, y, w, h = data['x'], data['y'], data['w'], data['h']
        
        module5_6.set_new_roi(x, y, w, h)
        
        return jsonify({'status': 'Success', 'message': 'Tracker re-initialized with new ROI.'})
        
    except Exception as e:
        print(f"Error in /set_roi: {e}")
        module5_6.tracker_initialized = False 
        return jsonify({'error': f'Failed to set ROI: {str(e)}'}), 500


# ============================================================================
# --- MODULE 7: CALIBRATION & STEREO VISION (/compute_stereo, /video_feed_pose_hand) ---
# ============================================================================

@app.route('/compute_stereo', methods=['POST'])
def compute_stereo():
    try:
        data = request.json
        results = module7.compute_stereo_dimension(data)

        return jsonify(results)
    except Exception as e:
        print(f"Error in /compute_stereo (M7): {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/video_feed_pose_hand')
def video_feed_pose_hand():
    """Serves the MediaPipe Pose and Hand Tracking video stream."""
    try:
        return Response(module7.gen_frames_pose_hand(module7.WORKING_CAMERA_ID),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Pose/Hand Tracking error: {e}")
        return Response(f"Error: {e}", status=500, mimetype='text/plain')


# --- LOCAL STARTUP BLOCK ---
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
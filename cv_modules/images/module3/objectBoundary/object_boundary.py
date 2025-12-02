import cv2
import numpy as np
import os

# --- Path setup ---
input_path = 'obj/box.jpg'  # change to your image
output_path = 'results/boundary_output.jpg'
os.makedirs('results', exist_ok=True)

# --- Step 1: Read the image ---
img = cv2.imread(input_path)
if img is None:
    print("Error: Could not read image.")
    exit()

# --- Step 2: Convert to grayscale ---
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- Step 3: Apply Gaussian Blur to remove noise ---
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# --- Step 4: Edge detection (Canny) ---
edges = cv2.Canny(blur, threshold1=100, threshold2=200)

# --- Step 5: Find contours ---
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# --- Step 6: Draw contours on the original image ---
output = img.copy()
cv2.drawContours(output, contours, -1, (0, 255, 0), 6)

# --- Step 7: Save result only ---
cv2.imwrite(output_path, output)
print(f"Boundary detection complete. Saved result to: {output_path}")

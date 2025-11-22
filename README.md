# Defective Product Detection System  
This project detects defective products using **Thresholding + Quick Sort Algorithm** and **OpenCV**.  
It is designed for low-cost quality control in small industries.

## Features
- Threshold-based defect detection  
- Automatic defect area calculation  
- Sorting algorithm for grading  
- Classification: Good / Minor Defect / Defective  
- Real-time or image-based input  

## Requirements
- Python 3.x
- OpenCV (`pip install opencv-python`)

## Running the Project
# --------------------------------------------------------
# DEFECTIVE PRODUCT DETECTION USING THRESHOLDING + SORTING
# OpenCV + Quick Sort Algorithm
# --------------------------------------------------------

import cv2

# ---------------- IMAGE ACQUISITION ----------------
image_path = "product.jpg"       # <-- Put your image in same folder
img = cv2.imread(image_path)

if img is None:
    raise Exception("Error: Image not found! Place product.jpg in folder.")

# ---------------- PREPROCESSING ----------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)

# ---------------- THRESHOLDING (DEFECT DETECTION) ----------------
_, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV)

# ---------------- FEATURE EXTRACTION ----------------
defect_area = cv2.countNonZero(thresh)
print("Detected Defect Area:", defect_area)

# ---------------- QUICK SORT IMPLEMENTATION ----------------
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr)//2]
    left = [x for x in arr if x < pivot]
    mid  = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + mid + quick_sort(right)

# Example batch defect values (add more if needed)
defect_areas = [defect_area, 120, 40, 300, 80]

sorted_defects = quick_sort(defect_areas)
print("Sorted Defect Values:", sorted_defects)

# ---------------- CLASSIFICATION LOGIC ----------------
def classify(area):
    if area < 50:
        return "GOOD PRODUCT"
    elif 50 <= area <= 200:
        return "MINOR DEFECT"
    else:
        return "DEFECTIVE PRODUCT"

status = classify(defect_area)
print("Final Result:", status)

# ---------------- RESULT VISUALIZATION ----------------
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
marked = img.copy()

for cnt in contours:
    if cv2.contourArea(cnt) > 10:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(marked, (x, y), (x+w, y+h), (0, 0, 255), 2)

cv2.putText(marked, f"Status: {status}", (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

# Show output
cv2.imshow("Original Image", img)
cv2.imshow("Processed Image", marked)
cv2.imshow("Threshold Image", thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Save output image
cv2.imwrite("output_result.jpg", marked)
print("Output saved as: output_result.jpg")

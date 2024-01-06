# PAN Card Tampering Detection

This project aims to detect tampering in PAN (Permanent Account Number) cards using structural similarity comparison.

## How it Works

1. **Image Processing:**
   - The uploaded image is resized to (250, 160) pixels and saved.
   - The original image is loaded, resized to the same dimensions, and saved.

2. **Image Comparison:**
   - Both images are read as arrays and converted to grayscale.
   - Structural similarity (SSIM) index and a difference map are calculated.

3. **Tampering Detection:**
   - The difference map is visualized and thresholded to create a binary image.
   - Contours are found in the binary image.
   - Rectangles are drawn around the detected contours on both the original and uploaded images.

## Usage

- Upload a PAN card image using the provided form.
- The system will compare the uploaded image with the original and highlight any detected tampering.

## Dependencies

- Python 3.x
- scikit-image
- OpenCV
- imutils

## Setup

1. Install the required dependencies: `pip install scikit-image opencv-python imutils`.
2. Run the script: `python pan_card_tampering_detection.py`.
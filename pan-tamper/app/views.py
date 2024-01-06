# Important imports
from app import app
from flask import request, render_template
import os
from skimage.metrics import structural_similarity
import imutils
import cv2
from PIL import Image

# Adding path to config
app.config['INITIAL_FILE_UPLOADS'] = '/Users/manoj/Documents/Udemy/Machine Learning Projects/pan-tamper/app/static/uploads'
app.config['EXISTNG_FILE'] = '/Users/manoj/Documents/Udemy/Machine Learning Projects/pan-tamper/app/static/original'
app.config['GENERATED_FILE'] = '/Users/manoj/Documents/Udemy/Machine Learning Projects/pan-tamper/app/static/generated'

# Route to home page
@app.route("/", methods=["GET", "POST"])
def index():
    # Execute if request is get
    if request.method == "GET":
        return render_template("index.html")
    
    # Execute if reuqest is post
    if request.method == "POST":
        # Get uploaded image
        file_upload = request.files['file_upload']
        filename = file_upload.filename
        
        # Resize and save the uploaded image
        uploaded_image = Image.open(file_upload).resize((250,160))
        uploaded_image.save(os.path.join(app.config['INITIAL_FILE_UPLOADS'], 'image.png'))

        # Resize and save the original image to ensure both uploaded and original matches in size
        original_image = Image.open(os.path.join(app.config['EXISTNG_FILE'], 'image.png')).resize((250,160))
        original_image.save(os.path.join(app.config['EXISTNG_FILE'], 'image.png'))

        # Read uploaded and original image as array
        original_image = cv2.imread(os.path.join(app.config['EXISTNG_FILE'], 'image.png'))
        uploaded_image = cv2.imread(os.path.join(app.config['INITIAL_FILE_UPLOADS'], 'image.png'))

        # Convert image into grayscale
        original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        uploaded_gray = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2GRAY)

        # Calculate structural similarity
        (score, diff) = structural_similarity(original_gray, uploaded_gray, full=True)
        diff = (diff * 255).astype("uint8")

        # Calculate threshold and contours
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        # Draw contours on image
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(uploaded_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Save all output images (if required)
        # cv2.imwrite(os.path.join(app.config['GENERATED_FILE'], 'image_original.png'), original_image)
        # cv2.imwrite(os.path.join(app.config['GENERATED_FILE'], 'image_uploaded.png'), uploaded_image)
        # cv2.imwrite(os.path.join(app.config['GENERATED_FILE'], 'image_diff.png'), diff)
        # cv2.imwrite(os.path.join(app.config['GENERATED_FILE'], 'image_thresh.png'), thresh)
        return render_template('index.html',pred=str(round(score*100,2)) + '%' + ' match with original')
       
# Main function
if __name__ == '__main__':
    app.run(debug=True)

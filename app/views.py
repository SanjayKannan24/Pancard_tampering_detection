from app import app
from flask import request, render_template
import os
from skimage.metrics import structural_similarity
import imutils
import cv2
from PIL import Image

# Adding path to config
app.config["FILE_UPLOADS"] = "app/static/uploads"
app.config["ORIGINAL_FILE"] = "app/static/original"
app.config["GENERATED_FILE"] = "app/static/generated"

# Route to home page
@app.route("/", methods = ["GET", "POST"])
def index():

    # Execute if request is GET
    if request.method == "GET":
        return render_template("index.html")

    # Execute if request is POST
    if request.method == "POST":
        # Get input image
        input_file = request.files["input_file"]
        filename = input_file.filename

        # Resize and save an input image
        input_image = Image.open(input_file).resize((250, 160))
        input_image.save(os.path.join(app.config["FILE_UPLOADS"], "image.jpg"))

        # Resize and save the original image
        original_image = Image.open(os.path.join(app.config["ORIGINAL_FILE"], "image.jpg")).resize((250, 160))
        original_image.save(os.path.join(app.config["ORIGINAL_FILE"], "image.jpg"))

        # Read input and original images
        original_image = cv2.imread(os.path.join(app.config["ORIGINAL_FILE"], "image.jpg"))
        input_image = cv2.imread(os.path.join(app.config["FILE_UPLOADS"], "image.jpg"))

        # Convert image into grayscale inorder to calculate structural similarity
        original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        input_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        # Calculate structural similarity
        (score, diff) = structural_similarity(original_gray, input_gray, full=True)
        diff = (diff * 255).astype("uint8")

        # Calculate threshold and contours
        th = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # Draw contours on image
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(input_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Save all the outputs
        cv2.imwrite(os.path.join(app.config["GENERATED_FILE"], "image_original.jpg"), original_image)
        cv2.imwrite(os.path.join(app.config["GENERATED_FILE"], "image_input.jpg"), input_image)
        cv2.imwrite(os.path.join(app.config["GENERATED_FILE"], "diff.jpg"), diff)
        cv2.imwrite(os.path.join(app.config["GENERATED_FILE"], "threshold.jpg"), th)

        if (round(score * 100, 2) < 70):
            output = "% Matched Hence, the card is tampered."
        else:
            output = "% Matched Hence, the card is real."


        return render_template("index.html", pred=str(round(score * 100, 2)) + "% correct")

# Main Function
if __name__ == "__main__":
    app.run(debug=True)








from flask import Flask, request, render_template, send_from_directory
from ultralytics import YOLO
import os
import cv2

app = Flask(__name__, template_folder="templates")

model = YOLO("yolov8n.pt")

upload_folder = "uploads"
result_folder = "results"

@app.route("/", methods=["GET", "POST"])
def index():
    result_image = None

    if request.method == "POST":

        file = request.files["image"]

        if file:
            upload_path = os.path.join(
                upload_folder,
                file.filename
            )

            file.save(upload_path)

            result = model(upload_path)

            result_with_box = result[0].plot()

            output_path = os.path.join(
                result_folder,
                file.filename
            )

            cv2.imwrite(output_path, result_with_box)

            result_image = file.filename

    return render_template(
        "index.html",
        result_image = result_image
    )


@app.route("/result/<filename>")
def result_image(filename):
    return send_from_directory(
        result_folder,
        filename
    )



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5400, debug=True)
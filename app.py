import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from PIL import Image, ImageEnhance, ImageFilter

app = Flask(__name__)

# Cấu hình thư mục
UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Định dạng file cho phép
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return redirect(request.url)
    
    file = request.files["file"]
    if file.filename == "":
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        
        # Tạo thumbnail để hiển thị
        img = Image.open(file_path)
        img.thumbnail((800, 800))
        thumbnail_path = os.path.join(app.config["UPLOAD_FOLDER"], "thumbnail_" + filename)
        img.save(thumbnail_path)
        
        return render_template("index.html", 
                             uploaded_img=file_path, 
                             thumbnail_img=thumbnail_path)
    
    return redirect(url_for("index"))

@app.route("/process", methods=["POST"])
def process_image():
    # Lấy thông số từ form
    filter_type = request.form.get("filter", "none")
    brightness = float(request.form.get("brightness", 1.0))
    contrast = float(request.form.get("contrast", 1.0))
    edge_detection = request.form.get("edge", "none")
    face_detection = request.form.get("face_detect", "off")
    rotate_angle = request.form.get("rotate", "0")
    flip_direction = request.form.get("flip", "none")
    output_format = request.form.get("format", "original")

    uploaded_img = request.form.get("uploaded_img")
    if not uploaded_img:
        return redirect(url_for("index"))

    # Mở ảnh
    img = Image.open(uploaded_img)
    img_cv = np.array(img)

    # Chuyển đổi định dạng ảnh
    if len(img_cv.shape) == 2:  # Grayscale
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
    elif img_cv.shape[2] == 4:   # RGBA
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2RGB)
    
    img_cv = img_cv.astype(np.uint8)

    # Áp dụng bộ lọc
    if filter_type == "grayscale":
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    elif filter_type == "blur":
        img_cv = cv2.GaussianBlur(img_cv, (15, 15), 0)
    elif filter_type == "enhance":
        img_cv = cv2.detailEnhance(img_cv, sigma_s=10, sigma_r=0.15)
    elif filter_type == "sharpen":
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img_cv = cv2.filter2D(img_cv, -1, kernel)

    # Điều chỉnh độ sáng và tương phản
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    img_pil = ImageEnhance.Brightness(img_pil).enhance(brightness)
    img_pil = ImageEnhance.Contrast(img_pil).enhance(contrast)
    img_cv = np.array(img_pil)

    # Phát hiện cạnh
    if edge_detection == "canny":
        img_cv = cv2.Canny(img_cv, 100, 200)
    elif edge_detection == "sobel":
        img_cv = cv2.Sobel(img_cv, cv2.CV_64F, 1, 0, ksize=5)
        img_cv = np.uint8(np.absolute(img_cv))

    # Nhận diện khuôn mặt
    if face_detection == "on":
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(img_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Xoay ảnh
    if rotate_angle != "0":
        angle = int(rotate_angle)
        if angle == 90:
            img_cv = cv2.rotate(img_cv, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            img_cv = cv2.rotate(img_cv, cv2.ROTATE_180)
        elif angle == 270:
            img_cv = cv2.rotate(img_cv, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Lật ảnh
    if flip_direction != "none":
        if flip_direction == "horizontal":
            img_cv = cv2.flip(img_cv, 1)
        elif flip_direction == "vertical":
            img_cv = cv2.flip(img_cv, 0)

    # Lưu ảnh đã xử lý
    processed_path = os.path.join(app.config["PROCESSED_FOLDER"], "processed.png")
    
    # Chuyển đổi định dạng nếu cần
    if output_format != "original":
        img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        if output_format == "jpg":
            processed_path = os.path.join(app.config["PROCESSED_FOLDER"], "processed.jpg")
            img_pil = img_pil.convert("RGB")
        elif output_format == "png":
            processed_path = os.path.join(app.config["PROCESSED_FOLDER"], "processed.png")
        img_pil.save(processed_path)
    else:
        cv2.imwrite(processed_path, img_cv)

    return render_template("index.html", 
                         uploaded_img=uploaded_img,
                         thumbnail_img=request.form.get("thumbnail_img"),
                         processed_img=processed_path)

@app.route("/crop", methods=["POST"])
def crop_image():
    # Lấy tọa độ crop từ form
    x = int(request.form.get("x"))
    y = int(request.form.get("y"))
    width = int(request.form.get("width"))
    height = int(request.form.get("height"))
    uploaded_img = request.form.get("uploaded_img")

    # Mở ảnh và crop
    img = Image.open(uploaded_img)
    cropped_img = img.crop((x, y, x + width, y + height))

    # Lưu ảnh đã crop
    cropped_path = os.path.join(app.config["UPLOAD_FOLDER"], "cropped_" + os.path.basename(uploaded_img))
    cropped_img.save(cropped_path)

    # Tạo thumbnail mới
    cropped_img.thumbnail((800, 800))
    thumbnail_path = os.path.join(app.config["UPLOAD_FOLDER"], "thumbnail_" + os.path.basename(cropped_path))
    cropped_img.save(thumbnail_path)

    return render_template("index.html", 
                         uploaded_img=cropped_path,
                         thumbnail_img=thumbnail_path)

@app.route("/download")
def download():
    processed_img = os.path.join(app.config["PROCESSED_FOLDER"], "processed.png")
    if not os.path.exists(processed_img):
        processed_img = os.path.join(app.config["PROCESSED_FOLDER"], "processed.jpg")
    return send_file(processed_img, as_attachment=True)

# if __name__ == "__main__":
    # port = int(os.environ.get("PORT", 5000))  # Lấy PORT từ biến môi trường
    # app.run(host="0.0.0.0", port=port)
if __name__ == "__main__":
    app.run(debug=True)

import os
import torch
import cv2
import numpy as np
from flask import Flask, request, render_template, send_from_directory, url_for, flash
from werkzeug.utils import secure_filename
import uuid
import traceback

from Unet.UNet_3Plus import UNet_3Plus
from processing import process_single_image 

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
MODEL_PATH = './model/epoch_1_MIoU_0.7826_Dice_0.7629.pt' 

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(APP_ROOT, UPLOAD_FOLDER)
app.config['PROCESSED_FOLDER'] = os.path.join(APP_ROOT, PROCESSED_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
app.secret_key = 'change_this_to_a_real_secret_key_39$dk' 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = None

try:
    print("Loading model...")
    model_load_path = os.path.join(APP_ROOT, MODEL_PATH)
    model = UNet_3Plus(in_channels=3)
    model.load_state_dict(torch.load(model_load_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded successfully from {model_load_path}")
except FileNotFoundError:
    print(f"ERROR: Model weights not found at {model_load_path}")
    traceback.print_exc()
except Exception as e:
    print(f"ERROR: Failed to load model: {e}")
    traceback.print_exc()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    if model is None:
         flash("ERROR: Model could not be loaded. Processing is disabled. Check server logs.", "error") # English
         return render_template('index.html', error="Model loading failed")

    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part in request', 'error')
            return render_template('index.html')

        file = request.files['image']

        if file.filename == '':
            flash('No file selected', 'error') 
            return render_template('index.html')

        if file and allowed_file(file.filename):
            original_filename_secure = secure_filename(file.filename)
            filename_base, file_ext = os.path.splitext(original_filename_secure)
            unique_suffix = uuid.uuid4().hex[:8]
            unique_original_filename = f"{filename_base}_{unique_suffix}{file_ext}"
            processed_filename = f"processed_{filename_base}_{unique_suffix}.png"

            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_original_filename)
            processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)

            try:
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

                file.save(upload_path)
                print(f"Image uploaded to: {upload_path}")

                success = process_single_image(model, upload_path, processed_path)

                if success:
                    print("Image processing successful, rendering results.")
                    flash("Image processing successful!", "info") # English
                    return render_template('index.html',
                                           original_image_filename=unique_original_filename,
                                           processed_image_filename=processed_filename)
                else:
                    flash("Image processing failed. Check server logs for details.", "error") # English
                    return render_template('index.html', original_image_filename=unique_original_filename, error="Image processing failed")

            except Exception as e:
                print(f"ERROR during file handling or processing call: {e}")
                traceback.print_exc()
                flash(f"Internal server error during processing.", "error")
                return render_template('index.html', error=f"Internal server error")
        else:
            flash(f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}", 'error') 
            return render_template('index.html')

    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=False)
    except Exception as e:
        print(f"ERROR sending uploaded file {filename}: {e}")
        return "File not found", 404


@app.route('/processed/<filename>')
def processed_file(filename):
    try:
        return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=False)
    except Exception as e:
        print(f"ERROR sending processed file {filename}: {e}")
        return "File not found", 404


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
    print("Starting Flask development server on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
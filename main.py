from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import numpy as np
import os

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model
model = load_model('models/pneumonia_model.h5')

# Prediction function
def predict_pneumonia(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        return "PNEUMONIA DETECTED"
    else:
        return "NORMAL"

# Home page
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    image_path = None

    if request.method == 'POST':

        if 'file' not in request.files:
            return render_template('index.html', result="No file selected")

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', result="No file selected")

        # Safe filename
        filename = secure_filename(file.filename)

        # Save file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Predict
        result = predict_pneumonia(filepath)

        image_path = filepath

    return render_template(
        'index.html',
        result=result,
        image_path=image_path
    )

if __name__ == '__main__':
    app.run(debug=True)
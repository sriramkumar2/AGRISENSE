import os
import numpy as np
import pickle
import tensorflow as tf
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

app = Flask(__name__)

# Upload folder setup
UPLOAD_FOLDER = 'static/distatic/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Crop Recommendation Models
with open('models/crop_model.pkl', 'rb') as f:
    crop_model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load Soil Classification Model
soil_model = tf.keras.models.load_model("models/soil_classification_model.h5")

# Load Crop Disease Prediction Model
disease_model = tf.keras.models.load_model("models/cnn_crop_disease.h5")

# Soil Class Info
SOIL_CLASSES = {
    "Alluvial Soil": {
        "Description": "Highly fertile and found near rivers.",
        "Suitable Crops": "Rice, Wheat, Sugarcane, Maize",
        "Regions Found": "Indo-Gangetic plains, Coastal regions"
    },
    "Black Soil": {
        "Description": "Rich in clay and retains moisture well.",
        "Suitable Crops": "Cotton, Soybean, Millets, Sunflower",
        "Regions Found": "Deccan Plateau - Maharashtra, Madhya Pradesh"
    },
    "Clay Soil": {
        "Description": "Heavy, retains water but drains slowly.",
        "Suitable Crops": "Paddy, Legumes, Vegetables",
        "Regions Found": "Plains and valley regions"
    },
    "Red Soil": {
        "Description": "Low in nutrients but improves with fertilizers.",
        "Suitable Crops": "Groundnut, Millet, Pulses",
        "Regions Found": "Eastern and Southern India"
    }
}

# Disease Class Labels
CLASS_LABELS = sorted(os.listdir("agridis/dataset/train"))

# -------------------- ROUTES --------------------

# Home Page
@app.route('/')
def index():
    return render_template('index.html')

# About Us Page
@app.route('/about')
def about():
    return render_template('about.html')

# Soil Analysis
@app.route('/soil')
def soil_index():
    return render_template('soilindex.html')

@app.route('/soil/predict', methods=['POST'])
def soil_predict():
    try:
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        img = Image.open(file_path).convert('RGB').resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        predictions = soil_model.predict(img_array)
        predicted_class = np.argmax(predictions)
        prediction_label = list(SOIL_CLASSES.keys())[predicted_class]
        soil_info = SOIL_CLASSES[prediction_label]

        return render_template('soilresults.html', prediction=prediction_label, soil_info=soil_info)

    except Exception as e:
        return f"Error: {str(e)}", 500

# Crop Recommendation
@app.route('/crop')
def crop_index():
    return render_template('cropindex.html')

@app.route('/crop/predict', methods=['POST'])
def crop_predict():
    try:
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        input_scaled = scaler.transform(input_data)
        prediction = crop_model.predict(input_scaled)
        predicted_crop = label_encoder.inverse_transform(prediction)[0]

    
        crop_image_path = f"cropstatic/{predicted_crop.lower()}.jpeg"

        # Debug print

        return render_template('cropresult.html', crop=predicted_crop, crop_image=crop_image_path)

    except Exception as e:
        return f"Error: {str(e)}", 500

# Crop Disease Prediction
@app.route('/disease')
def dis_index():
    return render_template('disindex.html')

@app.route('/disease/predict', methods=['POST'])
def disease_predict():
    try:
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        img = load_img(file_path, target_size=(224, 224))
        img_array = np.expand_dims(img_to_array(img) / 255.0, axis=0)

        predictions = disease_model.predict(img_array)
        predicted_label = np.argmax(predictions)
        prediction = CLASS_LABELS[predicted_label]

        return render_template('disresults.html', image_filename=filename, prediction=prediction)

    except Exception as e:
        return f"Error: {str(e)}", 500

# -------------------- RUN APP --------------------
if __name__ == '__main__':
    app.run(debug=True)

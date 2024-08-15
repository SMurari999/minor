from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
from PIL import Image
from pathlib import Path

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

model = load_model('minvgg.h5')
model2 = load_model('mininc.h5')
#model3 = load_model('minor_vgg16.h5')

class_labels = ['Food waste', 'Leaf Waste', 'Paper Waste', 'Wood Waste',
                'Ewaste', 'Metal cans', 'Plastic bags', 'Plastic bottles']

def predict_image(image_path):
    if not Path(image_path).exists():
        raise FileNotFoundError(f"The file at {image_path} does not exist.")
    
    img = load_img(image_path, target_size=(180, 180))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  

    prediction = model.predict(img)
    predicted_class = class_labels[np.argmax(prediction)]
    prediction = model2.predict(img)
    predicted_class2 = class_labels[np.argmax(prediction)]
   
    return [predicted_class,predicted_class2]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            img = Image.open(file_path)
            img = img.resize((180, 180))
            img.save(file_path)

            predicted_class,predicted_class2,predicted_class3 = predict_image(file_path)
            return render_template('result.html', image_path=file_path, prediction1=predicted_class,prediction2=predicted_class2)
    return render_template('index.html')

@app.route('/result', methods=['GET'])
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)

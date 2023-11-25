from flask import Flask, render_template, request
import tensorflow as tf
import cv2
from werkzeug.utils import secure_filename


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


model50 = tf.keras.models.load_model('model50.h5')
model100 = tf.keras.models.load_model('model100.h5')
model150 = tf.keras.models.load_model('model150.h5')
vgg16 = tf.keras.models.load_model('vgg16.h5')

@app.route('/classify', methods=['POST'])
def classify():
    selected_model = request.form['model']
    image_file = request.files['image']
    image_file.save('static/image.jpg')

    
    result = process_image(selected_model, 'static/image.jpg')

    if result[0] == 1:
        result = "Dog"
    else:
        result = "Cat"

    return render_template('index.html', text_result=result, image_url='static/image.jpg')

def process_image(model, image):
    img = cv2.imread(image)
    
    if model == "50":
        img = cv2.resize(img,(50,50))
        img = img.reshape((1,50,50,3))
        res = model50.predict(img)
    elif model == "100":
        img = cv2.resize(img,(100,100))
        img = img.reshape((1,100,100,3))
        res = model100.predict(img)
    elif model == "150":
        img = cv2.resize(img,(150,150))
        img = img.reshape((1,150,150,3))
        res = model150.predict(img)
    else:
        img = cv2.resize(img,(224,224))
        img = img.reshape((1,224,224,3))
        res = vgg16.predict(img)

    return res

if __name__ == '__main__':
    app.run(debug=True)

#app.py
from flask import Flask, flash, request, redirect, url_for, render_template
import os
import numpy as np
import cv2
import pickle as pk 
from joblib import load
pca = pk.load(open("pca.pkl",'rb'))
sc = pk.load(open('scaler.pkl','rb'))
sv=load('std_scaler.bin')

def predict(image_path):
    img = cv2.imread(image_path,0)
    img = cv2.resize(img, (200,200))
    pre=[img]
    pre = np.array(pre)/255.0
    pre_updated = pre.reshape(len(pre),-1 )
    pre_updated_new = sc.transform(pre_updated)
    pca_new = pca.transform(pre_updated_new)
    p= sv.predict(pca_new)
    return p

app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #print('upload_image filename: ' + filename)
        print(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        details=predict(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        #flash(a)
        return render_template('index.html', filename=filename,details=details[0])
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
 
if __name__ == "__main__":
    app.run()

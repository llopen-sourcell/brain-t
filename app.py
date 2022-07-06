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

meningioma='''
Brain tumor is considered as one of the aggressive diseases, among children and adults. Brain tumors account for 85 to 90 percent of all primary Central Nervous System(CNS) tumors. Every year, around 11,700 people are diagnosed with a brain tumor. The 5-year survival rate for people with a cancerous brain or CNS tumor is approximately 34 percent for men and 36 percent for women. BrainTumors are classiﬁed as: Benign Tumor, Malignant Tumor, Pituitary Tumor, etc. Proper treatment planning and accurate diagnostics should be implemented to improve life expectancy of the patients. The best technique to detect brain tumor is Magnetic Resonance Imaging (MRI).
'''
glioma='''
Brainove life expectancy of the patients. The best technique to detect brain tumor is Magnetic Resonance Imaging (MRI).
'''
pituitary='''

'''
notumor='''
'''

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
gliomaSymptoms=["Nausea or vomiting","Muscle Weakness or weakness in one side","Double vision or vison disorder"]
meningiomaSymptoms={}
pituitarySymptoms={}
notumor={}

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
        if details[0] == "meningioma":
            return render_template('tumordetails.html', filename=filename,details=details[0].capitalize(),tumor=meningioma,symptoms=meningiomaSymptoms,len=len(meningiomaSymptoms))
        if details[0] == "glioma":
            return render_template('tumordetails.html', filename=filename,details=details[0].capitalize(),tumor=glioma,symptoms=gliomaSymptoms,len=len(gliomaSymptoms))

        if details[0] == "pituitary":
            return render_template('tumordetails.html', filename=filename,details=details[0].capitalize(),tumor=pituitary,symptoms=pituitarySymptoms,len=len(pituitarySymptoms))
        
        if details[0] == "notumor":
            return render_template('tumordetails.html', flag=1, filename=filename,details="No Tumor",len=0,tumor=notumor)
        
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
 
if __name__ == "__main__":
    app.run(debug=True)

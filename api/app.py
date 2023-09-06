from distutils.log import debug
from fileinput import filename
import os
from flask import Flask, flash, request, redirect, url_for,jsonify
from flask import send_file
from flask_cors import CORS
import base64

from werkzeug.utils import secure_filename
import sys
import time

# sys.path.append('CSAIL_Segment')
# from test import load_model,infer
from Transformer_Segment.model import load_model,infer
from config import TOKEN
import uuid
import warnings
warnings.simplefilter("ignore")


WALL_UPLOAD_FOLDER = os.path.join(os.getcwd(),'wallimages')
DESIGN_UPLOAD_FOLDER = os.path.join(os.getcwd(),'designimages')
OUTPUT_IMAGE_FOLDER = os.path.join(os.getcwd(),'outputimages')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


app = Flask(__name__)
CORS(app)

app.config['WALL_UPLOAD_FOLDER'] = WALL_UPLOAD_FOLDER
app.config['DESIGN_UPLOAD_FOLDER'] = DESIGN_UPLOAD_FOLDER
app.config['OUTPUT_IMAGE_FOLDER'] = OUTPUT_IMAGE_FOLDER

load_model()



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS




@app.route('/api/v1')
def main():
    return "API for Detecting Walls and Floors in an Image"

@app.route('/api/v1/get_prediction')
def get_image():
    filename =  request.args.get('imgname')
    filepath = os.path.join(app.config['OUTPUT_IMAGE_FOLDER'], filename)
    return send_file(filepath, mimetype='image/jpg')

@app.route('/api/v1/infer', methods = ['POST'])
def success():
    st = 0
    et = 0
    datasend = None
    detectiomodemappings = {"walls":0,"floors":3,"carpet":28}
    acceptabledetectionmode = detectiomodemappings.keys()
    access_token = request.headers.get('auth-token')
    if(access_token != TOKEN):
        return jsonify({"data":"Invalid auth-token in Header"}),401
    if request.method == 'POST':
        requestjson = request.json
        wallimgbase64 = requestjson.get('wallimg')
        designimgbase64 = requestjson.get('designimg')
        detectionmode = requestjson.get('detectionmode')
        if wallimgbase64 is None or designimgbase64 is None or detectionmode is None:
            return jsonify({"data":"No file/mode Sent"}),400

        if detectionmode not in acceptabledetectionmode:
            return jsonify({"data":"Improper detection mode Selected(Choose between walls/floors)"}),400

        # Converting and Saving Wall Images
        unique_id_wallimg = uuid.uuid4()
        wallimgfilepath = os.path.join(app.config['WALL_UPLOAD_FOLDER'], str(unique_id_wallimg)+".jpg")
        with open(wallimgfilepath, "wb") as fh:
            try:
                wallimgbase64_data = wallimgbase64.split(',')[1]
                fh.write(base64.b64decode(wallimgbase64_data))
                #fh.write(base64.urlsafe_b64decode(wallimgbase64))
            except Exception as e:
                return jsonify({"data":str(e),"image_name":"Wall Image"}),400

        # Converting and Saving Design Images
        unique_id_designimg = uuid.uuid4()
        designimgfilepath = os.path.join(app.config['DESIGN_UPLOAD_FOLDER'], str(unique_id_designimg)+".jpg")
        with open(designimgfilepath, "wb") as fh:
            try:
                designimgbase64_data = designimgbase64.split(',')[1]
                fh.write(base64.b64decode(designimgbase64_data))
                # fh.write(base64.urlsafe_b64decode(designimgbase64))
            except Exception as e:
                return jsonify({"data":str(e),"image_name":"Design Image"}),400

        
        # Getting the Model Code for Detection Mode Selected
        detectionmodevalue  = detectiomodemappings[detectionmode]
        
        # Generating unique name for output Images
        unique_id = uuid.uuid4()
        outputimgfilepath = os.path.join(app.config['OUTPUT_IMAGE_FOLDER'], str(unique_id)+".jpg")

        st = time.time()
        modelinferresp = infer(wallimgfilepath,designimgfilepath,outputimgfilepath,mode = detectionmodevalue)
        et = time.time()

        if (modelinferresp == 0):
            # If Model Infered Correctly as the selected mode was not found in the image  
            datasend = {"data":"Requested Feature not detected in the image"}
            return jsonify(responsedata),200
        else:
            # If Model Infered Correctly Sending Base64 of Output Image
            imgencode = ""
            with open(outputimgfilepath, "rb") as img_file:
                imgencode = base64.b64encode(img_file.read())
            return imgencode

if __name__ == '__main__':
	app.run(debug=True)

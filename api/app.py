from distutils.log import debug
from fileinput import filename
import os
from flask import Flask, flash, request, redirect, url_for,jsonify
from flask import send_file
from werkzeug.utils import secure_filename
import sys
import time

# sys.path.append('CSAIL_Segment')
# from test import load_model,infer
from Transformer_Segment.model import load_model,infer
from config import TOKEN
import uuid



WALL_UPLOAD_FOLDER = os.path.join(os.getcwd(),'wallimages')
DESIGN_UPLOAD_FOLDER = os.path.join(os.getcwd(),'designimages')
OUTPUT_IMAGE_FOLDER = os.path.join(os.getcwd(),'outputimages')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


app = Flask(__name__)
app.config['WALL_UPLOAD_FOLDER'] = WALL_UPLOAD_FOLDER
app.config['DESIGN_UPLOAD_FOLDER'] = DESIGN_UPLOAD_FOLDER
app.config['OUTPUT_IMAGE_FOLDER'] = OUTPUT_IMAGE_FOLDER

load_model()



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS




@app.route('/api/v1')
def main():
    return "Hello"

@app.route('/api/v1/get_prediction')
def get_image():
    filename =  request.args.get('imgname')
    filepath = os.path.join(app.config['OUTPUT_IMAGE_FOLDER'], filename)
    return send_file(filepath, mimetype='image/jpg')

@app.route('/api/v1/infer', methods = ['POST'])
def success():
    st = 0
    et = 0
    access_token = request.headers.get('auth-token')
    if(access_token != TOKEN):
        return jsonify({"data":"Invalid auth-token in Header"}),401
    if request.method == 'POST':
        if 'wallimg' not in request.files or 'designimg' not in request.files:
            return jsonify({"data":"No file Selected"}),400
        wallimg = request.files['wallimg']
        designimg = request.files['designimg']
        if designimg.filename == '' or wallimg.filename == '':
            return jsonify({"data":"No file Selected"}),400
        if wallimg and allowed_file(wallimg.filename) and designimg and allowed_file(designimg.filename):
            wallimgfilename = secure_filename(wallimg.filename)
            wallimgfilepath = os.path.join(app.config['WALL_UPLOAD_FOLDER'], wallimgfilename)
            designimgfilename = secure_filename(designimg.filename)
            designimgfilepath = os.path.join(app.config['WALL_UPLOAD_FOLDER'], designimgfilename)

            wallimg.save(wallimgfilepath)
            designimg.save(designimgfilepath)
            unique_id = uuid.uuid4()
            outputimgfilepath = os.path.join(app.config['OUTPUT_IMAGE_FOLDER'], str(unique_id)+".jpg")
            st = time.time()
            infer(wallimgfilepath,designimgfilepath,outputimgfilepath)
            et = time.time()
        else:
            return jsonify({"data":"Wall Image / Desgn Image  File Format not supported"}),400
        responsedata = {
            "inference_time":(et - st),
            "data":str(unique_id)+".jpg"
        }
        return jsonify(responsedata),200

if __name__ == '__main__':
	app.run(debug=True)
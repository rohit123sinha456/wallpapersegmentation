from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation
from transformers import AutoImageProcessor, MaskFormerModel
from PIL import Image
import requests
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from numba import njit, prange

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

feature_extractor = None
model = None
device = torch.device('cpu')

@njit(parallel=True)
def create_wall_overlay(mask,dsgn,woverlay):
    w,h,_ = woverlay.shape
    dw,dh,_ = dsgn.shape
    for i in prange(0,w):
        for j in prange(0,h):
            if(mask[i][j][0] == 255):
                p = dsgn[i%dw][j%dh]
                woverlay[i][j]= p
    return woverlay

@njit(parallel=True)
def create_output_image(imagearray,walloverlayarray):
    h,w,_ =  walloverlayarray.shape
    for i in prange(0,h):
        for j in prange(0,w):
            if(walloverlayarray[i][j].sum() > 0 ):
                imagearray[i][j] =  walloverlayarray[i][j]
    return imagearray.astype(np.uint8)

@njit(parallel=True)
def create_image_with_shadow(img_gray,hsv_image,walloverlayarray):
  h,w,_ =  hsv_image.shape
  hsvmin = np.min(hsv_image[:,:,2])
  hsvmax = np.max(hsv_image[:,:,2])
  for i in prange(0,h):
    for j in prange(0,w):
      if(walloverlayarray[i][j].sum() > 0 ):
        # hsv_image[i][j][2] = hsv_image[i][j][2] - (img_gray[i][j]/2)
        hsv_image[i][j][2] = abs(hsv_image[i][j][2] - (((img_gray[i][j]/1)-hsvmin)/(hsvmax-hsvmin))*100)
  print(hsv_image.shape)
  return hsv_image.astype(np.uint8)

def create_tile_perspective(design,image,factor):
  iw,ih = image.size
  INITIAL_SIZE = 2*iw
  opencv_img = np.array(design)
  tile = np.tile(opencv_img,(int(iw/design.size[0]),1,1))
  src = tile
  r = INITIAL_SIZE / src.shape[1]
  dim = (INITIAL_SIZE, int(src.shape[0] * r))
  src = cv2.resize(src, dim, interpolation=cv2.INTER_AREA)

  h,w,_ = src.shape
  print(h,w)
  srcs = np.array([[0,0],[w,0],[w,h],[0,h]],np.float32)
  dst = np.array([[((1*w/4)-150),0],[((3*w/4)+150),0],[w,h],[0,h]],np.float32)

  # Get the homographic transform
  M1 = cv2.getPerspectiveTransform(srcs,dst)

  # Warp the image
  dst = cv2.warpPerspective(src, M1, (src.shape[1], src.shape[0]),flags = cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue = [0, 0, 0, 0])
  # dst = cv2.rotate(dst, cv2.ROTATE_90_CLOCKWISE)
  cropped_design = dst[:,int((1*w/4)-150):int((3*w/4)+150)]#dst[:,100:400]

  design_pil = Image.fromarray(cropped_design)
  width, height = design_pil.size
  resize_ratio = iw / width#min(iw / width, ih / (height*factor))
  new_design = design_pil.resize((int(iw), int(ih * 0.5)))
  return new_design

def load_model():
    global feature_extractor,model,device
    # load MaskFormer fine-tuned on COCO panoptic segmentation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    feature_extractor = MaskFormerFeatureExtractor.from_pretrained("facebook/maskformer-swin-base-ade")
    model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-ade")
    # model.to(device)
    print("Model Successfully Loaded")
    # image_processor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-base-ade")
    # model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-ade")
def infer(imagepath,designimgpath,outputpath,mode = 0):
    #mode 0 for walls
    #model 3 for floors
    #model 28 for carpet
    FLOOR_MODE = 3
    FLOOR_FACTOR = 0.4
    global feature_extractor,model,device
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.to(device)
    image = Image.open(imagepath).convert('RGB')
    inputs = feature_extractor(images=image, return_tensors="pt")
    # inputs = feature_extractor(images=image, return_tensors="pt")
    inputs.to(device)
    outputs = model(**inputs)
    # model predicts class_queries_logits of shape `(batch_size, num_queries)`
    # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
    # class_queries_logits = outputs.class_queries_logits
    # masks_queries_logits = outputs.masks_queries_logits

    # you can pass them to feature_extractor for postprocessing
    result = feature_extractor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    # we refer to the demo notebooks for visualization (see "Resources" section in the MaskFormer docs)
    predicted_panoptic_map = result["segmentation"].cpu()

    # Checking if the requested feature is in the image 
    if (mode not in [info['label_id'] for info in result['segments_info']]):
        return 0

    # Finding the id of the wall from the segment predictions
    # facebook/maskformer-swin-base-coco" -> 131
    # facebook/maskformer-swin-base-ade => 0
    wallitem = next(item for item in result['segments_info'] if item["label_id"] == mode)
    wallitemid = wallitem['id']

    #creating empty panoptic map
    color_predicted_panoptic_map = np.zeros((predicted_panoptic_map.shape[0], predicted_panoptic_map.shape[1], 3), dtype=np.uint8) # height, width, 3
    color_predicted_panoptic_map[predicted_panoptic_map == wallitemid ] = (255,0,0)

    design = Image.open(designimgpath).convert('RGB')
    dw,dh = design.size

    w,h = image.size
    walloverlay = Image.new("RGB", (w, h))

    # Creating Image of floor perspective 
    if(mode == FLOOR_MODE):
       design = create_tile_perspective(design,image,FLOOR_FACTOR)
    # Copying the design pixels into the wall overlap images
    walloverlayarray = create_wall_overlay(color_predicted_panoptic_map,np.array(design),np.array(walloverlay))

    # Creating output image and HSV of output image
    imagearray = np.array(image)
    imagearray = create_output_image(imagearray,walloverlayarray)
    hsv_image = cv2.cvtColor(imagearray, cv2.COLOR_RGB2HSV)

    # Getting Shadows of the original Image
    testgray = cv2.imread(imagepath)
    blurred_image = cv2.GaussianBlur(testgray, (5, 5), 0)
    ret, thresholded_image = cv2.threshold(blurred_image, 100, 255, cv2.THRESH_BINARY)
    shadow = thresholded_image - blurred_image
    img_gray = cv2.cvtColor(shadow, cv2.COLOR_RGB2GRAY)

    # Creating Image with Shadow
    with_shadow_image = create_image_with_shadow(img_gray,hsv_image,walloverlayarray)
    imgarray_with_shadow = cv2.cvtColor(with_shadow_image, cv2.COLOR_HSV2RGB)

    plt.imsave(outputpath,imgarray_with_shadow)
    print('Inference done!')
    if torch.cuda.is_available():
        model.to('cpu')
        del inputs,outputs,result
        torch.cuda.empty_cache()
    print(torch.cuda.memory_allocated())
    return 1

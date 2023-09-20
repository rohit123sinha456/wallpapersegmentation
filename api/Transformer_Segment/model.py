from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation
from transformers import AutoImageProcessor, MaskFormerModel
from PIL import Image
import requests
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
feature_extractor = None
model = None
device = torch.device('cpu')
def load_model():
    global feature_extractor,model,device
    # load MaskFormer fine-tuned on COCO panoptic segmentation
    device = 
    if torch.cuda.is_available():
        device = torch.device("cuda")
    feature_extractor = MaskFormerFeatureExtractor.from_pretrained("facebook/maskformer-swin-base-ade")
    model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-ade")
    model.to(device)
    # image_processor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-base-ade")
    # model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-ade")
def infer(imagepath,designimgpath,outputpath,mode = 0):
    #mode 0 for walls
    #model 3 for floors
    #model 28 for carpet
    global feature_extractor,model
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)
    image = Image.open(imagepath).convert('RGB')
    inputs = feature_extractor(images=image, return_tensors="pt")
    inputs = feature_extractor(images=image, return_tensors="pt")
    inputs.to(device)
    outputs = model(**inputs)
    # model predicts class_queries_logits of shape `(batch_size, num_queries)`
    # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
    class_queries_logits = outputs.class_queries_logits
    masks_queries_logits = outputs.masks_queries_logits

    # you can pass them to feature_extractor for postprocessing
    result = feature_extractor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    # we refer to the demo notebooks for visualization (see "Resources" section in the MaskFormer docs)
    predicted_panoptic_map = result["segmentation"]

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

    color_predicted_panoptic_map_img = Image.fromarray(color_predicted_panoptic_map)
    design = Image.open(designimgpath).convert('RGB')
    dw,dh = design.size

    w,h = image.size
    walloverlay = Image.new("RGB", (w, h))
    #Copying the design pixels into the wall overlap images
    for i in range(0,w):
        for j in range(0,h):
            if(color_predicted_panoptic_map_img.getpixel((i,j))[0] == 255):
                p = design.getpixel((i%dw,j%dh))
                walloverlay.putpixel((i,j),p)
    walloverlayarray = np.array(walloverlay)
    # walloverlayarray = transfer_tones(np.array(image), walloverlayarray)
    # walloverlayarray = np.array(walloverlayarray) * np.int_(np.array(walloverlay) > 0)

    imagearray = np.array(image)
    h,w,_ =  walloverlayarray.shape
    for i in range(0,h):
        for j in range(0,w):
            if(walloverlayarray[i][j].sum() > 0 ):
                imagearray[i][j] =  walloverlayarray[i][j]
    imagearray = imagearray.astype(np.uint8)
    plt.imsave(outputpath,imagearray)
    print('Inference done!')
    return 1
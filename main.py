import argparse
import cv2
import os
import numpy as np
import onnxruntime as ort
#import torch
from PIL import Image, ImageDraw

from Utils.patch_unpatch_code import *

from ultralytics.utils import yaml_load
from ultralytics.utils.checks import check_requirements, check_yaml


PATCH_IMAGES_DIR = os.path.join(os.getcwd(), "Intermediate/patch_normal_images")
INFERENCED_IMAGES_DIR = os.path.join(os.getcwd(), "Intermediate/patch_inferenced_images")
#OUTPUT_DIR = os.path.join(os.getcwd(), "Output")

onnx_model = "Models/best.onnx"
input_image = "SampleData/Input/Image.tif"
image_path = os.path.abspath(input_image)
file_name = os.path.splitext(os.path.basename(image_path))[0]
patch_size = (640, 640)
patch_images_dir = PATCH_IMAGES_DIR
os.makedirs(PATCH_IMAGES_DIR, exist_ok=True)
os.makedirs(INFERENCED_IMAGES_DIR, exist_ok=True)
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# os.makedirs(PADDED_DIR, exist_ok=True)
confidence_thres = 0.3
iou_thres = 0.7
yolo_classes = ["track"]

polygon_updated = []


def intersection(box1, box2):
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[:4]
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[:4]
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)
    return (x2 - x1) * (y2 - y1)


def union(box1, box2):
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[:4]
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[:4]
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    return box1_area + box2_area - intersection(box1, box2)


def iou(box1, box2):
    return intersection(box1, box2) / union(box1, box2)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# parse segmentation mask
def get_mask(row, box, img_width, img_height):
    # convert mask to image (matrix of pixels)
    mask = row.reshape(160, 160)
    mask = sigmoid(mask)
    mask = (mask > 0.5).astype("uint8") * 255
    # crop the object defined by "box" from mask
    x1, y1, x2, y2 = box
    mask_x1 = round(x1 / img_width * 160)
    mask_y1 = round(y1 / img_height * 160)
    mask_x2 = round(x2 / img_width * 160)
    mask_y2 = round(y2 / img_height * 160)
    mask = mask[mask_y1:mask_y2, mask_x1:mask_x2]
    # resize the cropped mask to the size of object
    img_mask = Image.fromarray(mask, "L")
    img_mask = img_mask.resize((round(x2 - x1), round(y2 - y1)))
    mask = np.array(img_mask)

    return mask


def get_polygon(mask):
    contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    polygon = [[contour[0][0], contour[0][1]] for contour in contours[0][0]]
    # print("polygon",polygon)
    return polygon



def preprocess():
    """
    Preprocesses the input image before performing inference.

    Returns:
        image_data: Preprocessed image data ready for inference.
    """

    old_shape, new_shape = patch_(
        input_image, patch_size, patch_images_dir
    )
    return old_shape, new_shape

def process(input_image, outputs, file_name):
    img_width, img_height = patch_size
    output0 = outputs[0]
    output1 = outputs[1]
    output0 = output0[0].transpose()
    output1 = output1[0]
    boxes = output0[:, 0:5]
    masks = output0[:, 5:]
    output1 = output1.reshape(32, 160 * 160)
    masks = masks @ output1
    boxes = np.hstack([boxes, masks])
    # print("length of boxes",len(boxes))
    objects = []
    for row in boxes:
        xc, yc, w, h = row[:4]
        x1 = (xc - w / 2) / 640 * img_width
        y1 = (yc - h / 2) / 640 * img_height
        x2 = (xc + w / 2) / 640 * img_width
        y2 = (yc + h / 2) / 640 * img_height
        prob = row[4]
        if prob < confidence_thres:
            continue
        mask = get_mask(row[5:], (x1, y1, x2, y2), img_width, img_height)
        contains_non_zero = np.any(mask != 0)
        if contains_non_zero == True:
            polygon = get_polygon(mask)
            objects.append([x1, y1, x2, y2, prob, mask, polygon])

    # apply non-maximum suppression
    objects.sort(key=lambda x: x[4], reverse=True)
    result = []
    while len(objects) > 0:
        result.append(objects[0])
        new_objects = []
        for obj in objects:
            if iou(obj, objects[0]) < iou_thres:
                new_objects.append(obj)
        objects = new_objects

    # print("legth of results", len(result))
    img_pil = Image.fromarray(input_image)
    draw = ImageDraw.Draw(img_pil, "RGBA")
    for object in result:
        [x1, y1, x2, y2, prob, mask, polygon] = object
        polygon = [(int(x1 + point[0]), int(y1 + point[1])) for point in polygon]
        draw.polygon(polygon, fill=(0, 255, 0, 125))
        draw.polygon(polygon, fill=(0, 255, 0, 125))
    output_file_name = f"{INFERENCED_IMAGES_DIR}/{file_name}"
    img_pil.save(output_file_name, format="TIFF")

    
session = ort.InferenceSession(
            onnx_model,
            providers=["CUDAExecutionProvider"],  # ,"CPUExecutionProvider"
        )

        # Get the model inputs
model_inputs = session.get_inputs()

outputs = session.get_outputs()
# print("LENGTH OF OUTPUT", len(outputs))

input_width = 640  # input_shape[2]
input_height = 640  # input_shape[3]

# Preprocess the image data
old_shape, new_shape = preprocess()

for filename in os.listdir(patch_images_dir):
    file_path = os.path.join(patch_images_dir, filename)
    img = read_tif_file(file_path)
    image_data = np.array(img) / 255.0
    image_data = np.transpose(image_data, (2, 0, 1))
    image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
    outputs = session.run(None, {model_inputs[0].name: image_data})
    process(img, outputs, filename)

unpatch_(
    f"Runtime/{file_name}-out.tif",
    old_shape,
    new_shape,
    patch_size,
    INFERENCED_IMAGES_DIR,
)


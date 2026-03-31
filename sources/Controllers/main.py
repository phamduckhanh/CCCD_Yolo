import logging
import os

import numpy as np
import torch

# PyTorch 2.6+ defaults weights_only=True in torch.load, which breaks
# loading YOLOv5 custom .pt weights. Patch to restore old behavior.
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import yolov5
from fastapi import File, Form, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from PIL import Image
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

import sources.Controllers.config as cfg
from sources import app, templates
from sources.Controllers import utils

logger = logging.getLogger(__name__)

# ---- Model Initialization ----

CORNER_MODEL = yolov5.load(cfg.CORNER_MODEL_PATH)
CONTENT_MODEL = yolov5.load(cfg.CONTENT_MODEL_PATH)
FACE_MODEL = yolov5.load(cfg.FACE_MODEL_PATH)

CONTENT_MODEL.conf = cfg.CONF_CONTENT_THRESHOLD
CONTENT_MODEL.iou = cfg.IOU_CONTENT_THRESHOLD
CORNER_MODEL.conf = cfg.CONF_CORNER_THRESHOLD
CORNER_MODEL.iou = cfg.IOU_CORNER_THRESHOLD

UPLOAD_FOLDER = cfg.UPLOAD_FOLDER
SAVE_DIR = cfg.SAVE_DIR
FACE_CROP_DIR = cfg.FACE_DIR

# ---- OCR Initialization ----

config = Cfg.load_config_from_name("vgg_seq2seq")
config["cnn"]["pretrained"] = False
config["device"] = cfg.DEVICE
config["predictor"]["beamsearch"] = False
detector = Predictor(config)


# ---- Helper Functions ----

def _clear_directory(directory):
    """Remove all files in a directory, create it if needed."""
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
    else:
        for f in os.listdir(directory):
            filepath = os.path.join(directory, f)
            if os.path.isfile(filepath):
                os.remove(filepath)


# ---- Page Routes ----

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse(request, "home.html")


@app.get("/home")
async def home(request: Request):
    return templates.TemplateResponse(request, "home.html")


@app.get("/id_card")
async def id_extract_page(request: Request):
    return templates.TemplateResponse(request, "idcard.html")


@app.get("/ekyc")
async def ekyc_page(request: Request):
    return templates.TemplateResponse(request, "ekyc.html")


# ---- ID Card Extraction ----

@app.post("/uploader")
async def upload(file: UploadFile = File(...)):
    _clear_directory(UPLOAD_FOLDER)

    file_location = os.path.join(UPLOAD_FOLDER, file.filename)
    contents = await file.read()
    with open(file_location, "wb") as f:
        f.write(contents)

    INPUT_FILE = os.listdir(UPLOAD_FOLDER)[0]
    if INPUT_FILE == "NULL":
        os.remove(os.path.join(UPLOAD_FOLDER, INPUT_FILE))
        return JSONResponse(status_code=400, content={"message": "No file selected!"})
    elif INPUT_FILE == "WRONG_EXTS":
        os.remove(os.path.join(UPLOAD_FOLDER, INPUT_FILE))
        return JSONResponse(status_code=400, content={"message": "This file is not supported!"})

    return await extract_info()


@app.post("/extract")
async def extract_info(ekyc=False, path_id=None):
    """Main extraction pipeline: Corner -> Align -> Content -> OCR"""
    os.makedirs(cfg.UPLOAD_FOLDER, exist_ok=True)

    INPUT_IMG = os.listdir(UPLOAD_FOLDER)
    if not INPUT_IMG:
        return JSONResponse(status_code=400, content={"message": "No image uploaded!"})

    img = path_id if ekyc else os.path.join(UPLOAD_FOLDER, INPUT_IMG[0])

    # Step 1: Corner detection & perspective transform
    CORNER = CORNER_MODEL(img)
    predictions = CORNER.pred[0]
    categories = predictions[:, 5].tolist()
    IMG = Image.open(img)

    if len(categories) == 4:
        boxes = utils.class_Order(predictions[:, :4].tolist(), categories)
        center_points = list(map(utils.get_center_point, boxes))
        c2, c3 = center_points[2], center_points[3]
        center_points = [
            center_points[0], center_points[1],
            (c2[0], c2[1] + 30), (c3[0], c3[1] + 30)
        ]
        center_points = np.asarray(center_points)
        aligned = Image.fromarray(utils.four_point_transform(IMG, center_points))
    else:
        logger.warning(f"Corner detection found {len(categories)}/4 corners, skipping perspective transform")
        aligned = IMG

    # Step 2: Content detection
    CONTENT = CONTENT_MODEL(aligned)
    predictions = CONTENT.pred[0]
    categories = predictions[:, 5].tolist()
    logger.info(f"Content detection: {len(categories)} fields, categories={sorted(set(int(c) for c in categories))}")

    if len(categories) < 6:
        return JSONResponse(status_code=422, content={"message": "Missing fields! Detecting content failed!"})

    boxes = predictions[:, :4].tolist()

    # Step 3: NMS & crop fields
    boxes, categories = utils.non_max_suppression_fast(np.array(boxes), categories, 0.7)
    boxes = utils.class_Order(boxes, categories)

    _clear_directory(SAVE_DIR)

    for index, box in enumerate(boxes):
        left, top, right, bottom = box
        if 5 < index < 9:
            right = right + 100
        cropped_image = aligned.crop((left, top, right, bottom))
        cropped_image.save(os.path.join(SAVE_DIR, f"{index}.jpg"))

    # Step 4: OCR each cropped field
    FIELDS_DETECTED = []
    for idx, img_crop in enumerate(sorted(os.listdir(SAVE_DIR))):
        if idx > 0:  # Skip index 0 (face photo)
            img_ = Image.open(os.path.join(SAVE_DIR, img_crop))
            s = detector.predict(img_)
            FIELDS_DETECTED.append(s)

    # Merge 2-line address if category 7 present and enough fields detected
    if 7 in categories and len(FIELDS_DETECTED) >= 9:
        FIELDS_DETECTED = (
            FIELDS_DETECTED[:6]
            + [FIELDS_DETECTED[6] + ", " + FIELDS_DETECTED[7]]
            + [FIELDS_DETECTED[8]]
        )

    return JSONResponse(content=jsonable_encoder({"data": FIELDS_DETECTED}))


@app.post("/download")
async def download(file: str = Form(...)):
    if file != "undefined":
        return JSONResponse(status_code=201, content={"message": "Download file successfully!"})
    else:
        return JSONResponse(status_code=400, content={"message": "No file to download!"})


# ---- eKYC ----

@app.post("/ekyc/uploader")
async def get_id_card(id: UploadFile = File(...), img: UploadFile = File(...)):
    _clear_directory(UPLOAD_FOLDER)

    id_location = os.path.join(UPLOAD_FOLDER, id.filename)
    with open(id_location, "wb") as f:
        f.write(await id.read())

    img_location = os.path.join(UPLOAD_FOLDER, img.filename)
    with open(img_location, "wb") as f_:
        f_.write(await img.read())

    INPUT_FILE = os.listdir(UPLOAD_FOLDER)
    if "NULL_1" in INPUT_FILE and "NULL_2" not in INPUT_FILE:
        _clear_directory(UPLOAD_FOLDER)
        return JSONResponse(status_code=400, content={"message": "Missing ID card image!"})
    elif "NULL_2" in INPUT_FILE and "NULL_1" not in INPUT_FILE:
        _clear_directory(UPLOAD_FOLDER)
        return JSONResponse(status_code=400, content={"message": "Missing person image!"})
    elif "NULL_1" in INPUT_FILE and "NULL_2" in INPUT_FILE:
        _clear_directory(UPLOAD_FOLDER)
        return JSONResponse(status_code=400, content={"message": "Missing ID card and person images!"})

    id_ext = id.filename.rsplit(".", 1)[-1]
    new_id_name = os.path.join(UPLOAD_FOLDER, f"id.{id_ext}")
    os.rename(id_location, new_id_name)

    img_ext = img.filename.rsplit(".", 1)[-1]
    new_img_name = os.path.join(UPLOAD_FOLDER, f"person.{img_ext}")
    os.rename(img_location, new_img_name)

    # Face detection
    FACE = FACE_MODEL(new_img_name)
    predictions = FACE.pred[0]
    categories = predictions[:, 5].tolist()

    if 0 not in categories:
        return JSONResponse(status_code=422, content={"message": "No face detected!"})
    elif categories.count(0) > 1:
        return JSONResponse(status_code=422, content={"message": "Multiple faces detected!"})

    boxes = predictions[:, :4].tolist()

    _clear_directory(FACE_CROP_DIR)

    FACE_IMG = Image.open(new_img_name)
    cropped_image = FACE_IMG.crop((boxes[0]))
    cropped_image.save(os.path.join(FACE_CROP_DIR, "face_crop.jpg"))

    return await extract_info(ekyc=True, path_id=new_id_name)

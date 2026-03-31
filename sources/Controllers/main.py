import logging
import os
import uuid

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
config["predictor"]["beamsearch"] = True   # beam search giúp tăng độ chính xác
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

async def _run_pipeline(img_path: str):
    """Core extraction pipeline: Corner -> Align -> Content -> OCR.

    Returns:
        list[str] on success, or None if content detection failed.
    """
    # Step 1: Corner detection & perspective transform
    CORNER = CORNER_MODEL(img_path)
    predictions = CORNER.pred[0]
    categories = predictions[:, 5].tolist()
    IMG = Image.open(img_path)

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
        return None

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
            img_ = utils.preprocess_for_ocr(img_)      # upscale + sharpen
            s = detector.predict(img_)
            # s = utils.post_process_field(s)             # sửa lỗi số/ngày tháng
            FIELDS_DETECTED.append(s)

    # Merge 2-line address if category 7 present and enough fields detected
    if 7 in categories and len(FIELDS_DETECTED) >= 9:
        FIELDS_DETECTED = (
            FIELDS_DETECTED[:6]
            + [FIELDS_DETECTED[6] + ", " + FIELDS_DETECTED[7]]
            + [FIELDS_DETECTED[8]]
        )

    return FIELDS_DETECTED


@app.post("/uploader")
async def upload(file: UploadFile = File(...)):
    """Dùng bởi web UI — lưu file vào UPLOAD_FOLDER rồi trả {"data": [...]}."""
    _clear_directory(UPLOAD_FOLDER)

    contents = await file.read()
    file_location = os.path.join(UPLOAD_FOLDER, file.filename or "upload.jpg")
    with open(file_location, "wb") as f:
        f.write(contents)

    fields = await _run_pipeline(file_location)
    if fields is None:
        return JSONResponse(status_code=422, content={"message": "Missing fields! Detecting content failed!"})

    return JSONResponse(content=jsonable_encoder({"data": fields}))


@app.post("/extract")
async def extract_info(file: UploadFile = File(...)):
    """Upload ảnh CCCD, trả về JSON có cấu trúc cố định.

    Request: multipart/form-data, field name = file
    Response:
        {
          "documentNumber": "string",
          "fullName": "string",
          "dayOfBirth": "string",
          "sex": "string",
          "nationality": "string",
          "placeOfOrigin": "string",
          "placeOfResident": "string",
          "faceMatching": number
        }
    """
    contents = await file.read()
    if not contents:
        return JSONResponse(status_code=400, content={"message": "Empty file!"})

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    ext = (file.filename or "upload.jpg").rsplit(".", 1)[-1].lower()
    tmp_path = os.path.join(UPLOAD_FOLDER, f"_api_{uuid.uuid4().hex}.{ext}")
    try:
        with open(tmp_path, "wb") as f:
            f.write(contents)
        fields = await _run_pipeline(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if fields is None:
        return JSONResponse(
            status_code=422,
            content={"message": "Missing fields! Detecting content failed!"},
        )

    def _get(i: int) -> str:
        return fields[i].strip() if len(fields) > i else ""

    return JSONResponse(content={
        "documentNumber":  _get(0),
        "fullName":        _get(1),
        "dayOfBirth":      _get(2),
        "sex":             _get(3),
        "nationality":     _get(4),
        "placeOfOrigin":   _get(5),
        "placeOfResident": _get(6),
        "faceMatching":    0,
    })


@app.post("/download")
async def download(file: str = Form(...)):
    if file != "undefined":
        return JSONResponse(status_code=201, content={"message": "Download file successfully!"})
    else:
        return JSONResponse(status_code=400, content={"message": "No file to download!"})


# ---- eKYC ----

@app.post("/ekyc/extract")
async def ekyc_extract(id_card: UploadFile = File(...), person: UploadFile = File(...)):
    """eKYC: Upload ảnh CCCD + ảnh chụp người, trả về thông tin + độ khớp khuôn mặt.

    Request: multipart/form-data
        - id_card: ảnh CCCD
        - person:  ảnh chụp người thật

    Response:
        {
          "documentNumber": "string",
          "fullName": "string",
          "dayOfBirth": "string",
          "sex": "string",
          "nationality": "string",
          "placeOfOrigin": "string",
          "placeOfResident": "string",
          "faceMatching": number   // 0.0 – 1.0, > 0.6 là cùng người
        }
    """
    id_bytes = await id_card.read()
    person_bytes = await person.read()

    if not id_bytes:
        return JSONResponse(status_code=400, content={"message": "Empty ID card image!"})
    if not person_bytes:
        return JSONResponse(status_code=400, content={"message": "Empty person image!"})

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    uid = uuid.uuid4().hex

    id_ext = (id_card.filename or "id.jpg").rsplit(".", 1)[-1].lower()
    person_ext = (person.filename or "person.jpg").rsplit(".", 1)[-1].lower()
    id_path = os.path.join(UPLOAD_FOLDER, f"_ekyc_id_{uid}.{id_ext}")
    person_path = os.path.join(UPLOAD_FOLDER, f"_ekyc_person_{uid}.{person_ext}")

    try:
        with open(id_path, "wb") as f:
            f.write(id_bytes)
        with open(person_path, "wb") as f:
            f.write(person_bytes)

        # --- 1. OCR pipeline trên ảnh CCCD ---
        fields = await _run_pipeline(id_path)
        if fields is None:
            return JSONResponse(
                status_code=422,
                content={"message": "Missing fields! Detecting content failed!"},
            )

        # --- 2. Lấy ảnh khuôn mặt từ CCCD (file 0.jpg trong SAVE_DIR) ---
        face_cccd_path = os.path.join(SAVE_DIR, "0.jpg")
        if not os.path.exists(face_cccd_path):
            return JSONResponse(
                status_code=422,
                content={"message": "Cannot extract face from ID card!"},
            )
        face_cccd = Image.open(face_cccd_path)

        # --- 3. Detect khuôn mặt trong ảnh người ---
        FACE = FACE_MODEL(person_path)
        predictions = FACE.pred[0]
        categories = predictions[:, 5].tolist()

        if 0 not in categories:
            return JSONResponse(status_code=422, content={"message": "No face detected in person image!"})

        left, top, right, bottom = predictions[categories.index(0), :4].tolist()
        # Thêm 20% padding quanh box để alignment có đủ context
        person_img = Image.open(person_path)
        pw, ph = person_img.size
        pad_x = (right - left) * 0.2
        pad_y = (bottom - top) * 0.2
        face_person = person_img.crop((
            max(0, left - pad_x), max(0, top - pad_y),
            min(pw, right + pad_x), min(ph, bottom + pad_y),
        ))

        # --- 4. So khớp khuôn mặt ---
        similarity = utils.compare_faces(face_cccd, face_person)

    finally:
        for p in (id_path, person_path):
            if os.path.exists(p):
                os.remove(p)

    def _get(i: int) -> str:
        return fields[i].strip() if len(fields) > i else ""

    return JSONResponse(content={
        "documentNumber":  _get(0),
        "fullName":        _get(1),
        "dayOfBirth":      _get(2),
        "sex":             _get(3),
        "nationality":     _get(4),
        "placeOfOrigin":   _get(5),
        "placeOfResident": _get(6),
        "faceMatching":    similarity,
    })

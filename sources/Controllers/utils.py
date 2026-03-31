import logging
import os
import re
import urllib.request
import zipfile

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ArcFace (MobileFaceNet) — face verification model
# ---------------------------------------------------------------------------
_FACE_RECOG_PATH = os.path.join("sources", "Database", "OCR", "weights", "w600k_mbf.onnx")
_BUFFALO_SC_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip"
_face_session = None

# Haar cascade eye detector — bundled in opencv, không cần file ngoài
_cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
_EYE_CASCADE = cv2.CascadeClassifier(_cascade_path)
if _EYE_CASCADE.empty():
    logger.warning(f"Haar eye cascade not found at {_cascade_path!r}, face alignment will use fallback resize.")
    _EYE_CASCADE = None

# Tọa độ mắt chuẩn trong không gian 112×112 của ArcFace
_LEFT_EYE_TARGET  = np.float32([38.29, 51.70])
_RIGHT_EYE_TARGET = np.float32([73.53, 51.70])


def _ensure_face_model():
    """Đảm bảo ArcFace ONNX model đã được tải về. Chỉ download lần đầu."""
    global _face_session
    if _face_session is not None:
        return _face_session

    if not os.path.exists(_FACE_RECOG_PATH):
        logger.info("Downloading ArcFace face recognition model (buffalo_sc) ...")
        os.makedirs(os.path.dirname(_FACE_RECOG_PATH), exist_ok=True)
        zip_path = _FACE_RECOG_PATH + ".zip"
        urllib.request.urlretrieve(_BUFFALO_SC_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if name.endswith("w600k_mbf.onnx"):
                    with zf.open(name) as src, open(_FACE_RECOG_PATH, "wb") as dst:
                        dst.write(src.read())
                    break
        os.remove(zip_path)
        logger.info(f"ArcFace model saved to {_FACE_RECOG_PATH}")

    _face_session = ort.InferenceSession(
        _FACE_RECOG_PATH,
        providers=["CPUExecutionProvider"],
    )
    return _face_session


def _align_and_prepare(img: Image.Image) -> np.ndarray:
    """Căn chỉnh khuôn mặt về pose chuẩn ArcFace rồi trả về blob 1×3×112×112.

    Các bước:
    1. Thêm padding để Haar cascade có đủ context
    2. CLAHE để chuẩn hoá ánh sáng
    3. Detect mắt → tính Affine transform về tọa độ chuẩn
    4. Fallback: resize thẳng nếu không detect được mắt
    5. Normalize về [-1, 1] theo RGB (đúng với training ArcFace)
    """
    img = img.convert("RGB")
    w, h = img.size

    # Thêm 30% padding xung quanh để cascade detect tốt hơn
    pad = max(int(max(w, h) * 0.3), 20)
    padded = Image.new("RGB", (w + 2 * pad, h + 2 * pad), (127, 127, 127))
    padded.paste(img, (pad, pad))

    bgr = cv2.cvtColor(np.array(padded), cv2.COLOR_RGB2BGR)
    ph, pw = bgr.shape[:2]

    # CLAHE để giảm ảnh hưởng ánh sáng
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray_eq = clahe.apply(gray)

    # Detect mắt trong ảnh đã padding
    aligned_bgr = None
    if _EYE_CASCADE is not None:
        min_eye = max(8, int(min(ph, pw) * 0.04))
        eyes = _EYE_CASCADE.detectMultiScale(gray_eq, 1.1, 4, minSize=(min_eye, min_eye))

        if len(eyes) >= 2:
            # Lấy 2 mắt ngoài cùng, sort theo x → [mắt trái, mắt phải]
            eyes = sorted(eyes, key=lambda e: e[0])[:2]
            lc = np.float32([eyes[0][0] + eyes[0][2] / 2, eyes[0][1] + eyes[0][3] / 2])
            rc = np.float32([eyes[1][0] + eyes[1][2] / 2, eyes[1][1] + eyes[1][3] / 2])
            M, _ = cv2.estimateAffinePartial2D(
                np.stack([lc, rc]),
                np.stack([_LEFT_EYE_TARGET, _RIGHT_EYE_TARGET]),
            )
            if M is not None:
                aligned_bgr = cv2.warpAffine(
                    bgr, M, (112, 112), borderMode=cv2.BORDER_REFLECT
                )

    if aligned_bgr is None:
        # Fallback: bỏ padding, resize vùng mặt gốc
        aligned_bgr = cv2.resize(bgr[pad: pad + h, pad: pad + w], (112, 112))

    # Đổi sang RGB và normalize [-1, 1] (đúng convention training ArcFace)
    rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = (rgb - 127.5) / 128.0   # scale giống InsightFace inference
    return np.expand_dims(rgb.transpose(2, 0, 1), axis=0)  # 1×3×112×112


def class_Order(boxes, categories):
    Z = []
    # Z = [x for _,x in sorted(zip(categories, boxes))]
    cate = np.argsort(categories)
    for index in cate:
        Z.append(boxes[index])

    return Z


def non_max_suppression_fast(boxes, labels, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

        # if the bounding boxes integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

        # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 1]
    y1 = boxes[:, 0]
    x2 = boxes[:, 3]
    y2 = boxes[:, 2]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(
            idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
        )

    # return only the bounding boxes that were picked using the
    # integer data type
    final_labels = [labels[idx] for idx in pick]
    final_boxes = boxes[pick].astype("int")
    return final_boxes, final_labels


def get_center_point(box):
    left, top, right, bottom = box
    return left + ((right - left) // 2), top + (
        (bottom - top) // 2
    )  # (x_c, y_c) # Need to fix bottom_left and bottom_right


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    image = np.asarray(image)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------

# Bảng thay thế ký tự dễ nhầm trong nhận dạng số
_DIGIT_TRANS = str.maketrans("OoIlSsBbGgZz", "001155886622")


def _normalize_digits(text: str) -> str:
    """Thay thế các ký tự dễ nhầm với chữ số (O→0, l→1, S→5 …)."""
    return text.translate(_DIGIT_TRANS)


def preprocess_for_ocr(img: Image.Image) -> Image.Image:
    """Upscale và tăng độ nét ảnh crop trước khi đưa vào OCR.

    VietOCR hoạt động tốt nhất với ảnh có chiều cao ≥ 64 px và rõ nét.
    """
    w, h = img.size

    # Upscale nếu ảnh quá nhỏ
    if h < 64:
        scale = max(2.0, 64 / h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    elif h < 128:
        img = img.resize((w * 2, h * 2), Image.LANCZOS)

    # Tăng độ nét
    img = img.filter(ImageFilter.SHARPEN)

    # Tăng nhẹ độ tương phản
    img = ImageEnhance.Contrast(img).enhance(1.2)

    return img


def _fix_id_number(text: str) -> str:
    """Chuẩn hoá chuỗi số CCCD (12 chữ số).

    Xử lý trường hợp OCR nhầm O→0, l→1, S→5 …
    """
    candidate = re.sub(r"\s", "", _normalize_digits(text))
    digits = re.sub(r"\D", "", candidate)
    # Chỉ áp dụng nếu chuỗi gốc gần như đều là chữ số và đủ 12 ký tự
    if len(digits) == 12 and len(candidate) <= 13:
        return digits
    return text


def _fix_date(text: str) -> str:
    """Chuẩn hoá chuỗi ngày tháng về dạng DD/MM/YYYY.

    Xử lý các lỗi phổ biến:
    - thiếu dấu / : "17122028" → "17/12/2028"
    - dấu / bị mất một bên: "17/122028" hay "17122/028" …
    - ký tự nhiễu: "17l12/2028" …
    """
    # Trường hợp đã đúng định dạng
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", text)
    if m:
        d, mo, y = m.groups()
        return f"{int(d):02d}/{int(mo):02d}/{y}"

    # Chuẩn hoá chữ số, bỏ ký tự không phải số
    digits = re.sub(r"\D", "", _normalize_digits(text))
    if len(digits) == 8:
        d, mo, y = digits[:2], digits[2:4], digits[4:]
        if 1 <= int(d) <= 31 and 1 <= int(mo) <= 12:
            return f"{d}/{mo}/{y}"

    return text


def post_process_field(text: str) -> str:
    """Áp dụng hậu xử lý dựa theo pattern để sửa lỗi OCR phổ biến."""
    text = text.strip()

    # Thử fix số CCCD (12 chữ số)
    fixed = _fix_id_number(text)
    if fixed != text:
        return fixed

    # Thử fix ngày tháng
    fixed = _fix_date(text)
    return fixed


def compare_faces(img1: Image.Image, img2: Image.Image) -> float:
    """So sánh 2 ảnh khuôn mặt bằng ArcFace có căn chỉnh.

    Trả về cosine similarity (0.0 – 1.0).
    Thông thường: cùng người > 0.4, khác người < 0.3.
    """
    session = _ensure_face_model()
    input_name = session.get_inputs()[0].name

    t1 = _align_and_prepare(img1)
    t2 = _align_and_prepare(img2)

    emb1 = session.run(None, {input_name: t1})[0][0]
    emb2 = session.run(None, {input_name: t2})[0][0]

    # L2 normalize để đảm bảo cosine similarity đúng
    emb1 = emb1 / (np.linalg.norm(emb1) + 1e-8)
    emb2 = emb2 / (np.linalg.norm(emb2) + 1e-8)

    sim = float(np.dot(emb1, emb2))
    return round(max(0.0, min(1.0, sim)), 4)


# def getMissingCorner(categories, boxes): # boxes: top_left, top_right, bottom_left, bottom_right
# 	if 0 not in categories: # Missing top_left
# 		delta_vertical = boxes[3][2] - boxes[1][2]
# 		delta_horizon = boxes[3][3] - boxes[2][3]
# 		x_miss =

# Vietnamese ID Card Extractor v2.0

Ứng dụng web trích xuất thông tin từ **Căn cước công dân (CCCD) gắn chip** của Việt Nam sử dụng Deep Learning. Hỗ trợ 2 tính năng chính:

1. **ID Card Extractor** — Upload ảnh CCCD → Trích xuất thông tin (số ID, họ tên, ngày sinh, giới tính, quốc tịch, quê quán, nơi thường trú, ngày hết hạn)
2. **eKYC** — Upload ảnh CCCD + ảnh chân dung → Trích xuất thông tin + crop khuôn mặt từ ảnh chân dung để đối chiếu

---

## Kiến trúc hệ thống

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────┐
│  Browser/UI │────▶│  FastAPI      │────▶│ YOLOv5 Models│────▶│ VietOCR  │
│  (HTML+JS)  │◀────│  (Uvicorn)    │◀────│ (Detection)  │◀────│ (OCR)    │
└─────────────┘     └──────────────┘     └──────────────┘     └──────────┘
```

### Thành phần chính

| Thành phần | Mô tả |
|---|---|
| `run.py` | Entry point — khởi động Uvicorn server |
| `sources/__init__.py` | Khởi tạo FastAPI app, mount static files, global exception handler |
| `sources/Controllers/main.py` | Toàn bộ API routes và logic xử lý |
| `sources/Controllers/config.py` | Cấu hình threshold, đường dẫn model, port |
| `sources/Controllers/utils.py` | Hàm tiện ích: NMS, perspective transform, sắp xếp boxes |
| `sources/Database/OCR/weights/` | Chứa 3 file model YOLOv5 (.pt) + 1 file OCR (.pth) |
| `sources/Views/templates/` | Jinja2 HTML templates |
| `sources/static/` | CSS, JS, assets |

---

## Luồng xử lý (Pipeline)

### 1. ID Card Extraction (`POST /uploader`)

```
Upload ảnh → Lưu file → Gọi extract_info()
```

### 2. extract_info() — Pipeline chính

```
Ảnh CCCD
    │
    ▼
┌──────────────────────────────────────┐
│ Step 1: Corner Detection (corner.pt) │
│ - Detect 4 góc của thẻ CCCD         │
│ - Nếu đủ 4 góc → Perspective        │
│   Transform (deskew/align ảnh)       │
│ - Nếu < 4 góc → Dùng ảnh gốc       │
│   (fallback)                         │
└──────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│ Step 2: Content Detection            │
│ (content.pt)                         │
│ - Detect các vùng chứa thông tin     │
│ - Mỗi vùng = 1 field (ID, tên,      │
│   ngày sinh, giới tính, v.v.)        │
│ - Yêu cầu tối thiểu 6 fields        │
└──────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│ Step 3: Non-Maximum Suppression      │
│ - Loại bỏ bounding boxes trùng lặp  │
│ - Sắp xếp theo thứ tự class         │
└──────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│ Step 4: OCR (VietOCR - vgg_seq2seq)  │
│ - Crop từng field → nhận dạng chữ   │
│ - Trả về text cho mỗi field         │
│ - Merge 2 dòng địa chỉ nếu cần     │
└──────────────────────────────────────┘
    │
    ▼
JSON Response: { "data": [id, name, dob, sex, nationality, hometown, address, expiry] }
```

### 3. eKYC (`POST /ekyc/uploader`)

```
Upload ảnh CCCD + ảnh chân dung
    │
    ├─▶ Face Detection (face.pt) trên ảnh chân dung
    │   → Crop khuôn mặt → Lưu face_crop.jpg
    │
    └─▶ Gọi extract_info(ekyc=True) → Trích xuất thông tin CCCD
```

---

## Các model AI sử dụng

| Model | File | Mục đích |
|---|---|---|
| Corner Detection | `corner.pt` | YOLOv5 — Detect 4 góc thẻ CCCD |
| Content Detection | `content.pt` | YOLOv5 — Detect các vùng chứa văn bản |
| Face Detection | `face.pt` | YOLOv5 — Detect khuôn mặt trong ảnh |
| OCR | `vgg_seq2seq.pth` | VietOCR — Nhận dạng chữ tiếng Việt |

---

## Cài đặt & Chạy

### Yêu cầu
- Python 3.10+
- CUDA (tùy chọn, để tăng tốc GPU)

### Cài đặt local

```bash
# Tạo virtual environment
python -m venv .venv

# Kích hoạt (Windows)
.\.venv\Scripts\Activate.ps1

# Cài dependencies
pip install -r requirements.txt
pip install vietocr --no-deps

# Chạy
python run.py
```

Server sẽ chạy tại: `http://localhost:8080`

### Chạy bằng Docker

```bash
# Build và chạy
docker compose up --build -d

# Xem logs
docker compose logs -f
```

---

## Cấu hình

File `sources/Controllers/config.py`:

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `PORT` | `8080` | Port cho web server |
| `CONF_CONTENT_THRESHOLD` | `0.4` | Ngưỡng confidence cho content detection |
| `IOU_CONTENT_THRESHOLD` | `0.5` | Ngưỡng IOU cho content NMS |
| `CONF_CORNER_THRESHOLD` | `0.25` | Ngưỡng confidence cho corner detection |
| `IOU_CORNER_THRESHOLD` | `0.5` | Ngưỡng IOU cho corner NMS |
| `DEVICE` | `cpu` | Device cho OCR (`cpu` hoặc `cuda:0`) |

---

## API Endpoints

| Method | Path | Mô tả |
|---|---|---|
| `GET` | `/` | Trang chủ |
| `GET` | `/home` | Trang chủ |
| `GET` | `/id_card` | Trang ID Card Extractor |
| `GET` | `/ekyc` | Trang eKYC |
| `POST` | `/uploader` | Upload ảnh CCCD và trích xuất |
| `POST` | `/extract` | API trích xuất (internal) |
| `POST` | `/download` | Xác nhận download CSV |
| `POST` | `/ekyc/uploader` | Upload ảnh CCCD + chân dung cho eKYC |

---

## Lỗi thường gặp & Cách xử lý

### 1. "Detecting corner failed!" (đã fix — fallback tự động)
- **Nguyên nhân**: Model không detect đủ 4 góc CCCD
- **Xử lý hiện tại**: Tự động bỏ qua, dùng ảnh gốc làm input cho content detection
- **Khuyến nghị**: Upload ảnh chụp thẳng, đủ sáng

### 2. "Missing fields! Detecting content failed!" (HTTP 422)
- **Nguyên nhân**: Content model detect < 6 fields
- **Giải pháp**:
  - Giảm `CONF_CONTENT_THRESHOLD` trong config (thử `0.3`)
  - Đảm bảo ảnh rõ ràng, không bị mờ/che
  - Ảnh bị cắt quá sát hoặc bị xoay sẽ khiến model không nhận dạng được

### 3. "No face detected!" / "Multiple faces detected!" (HTTP 422)
- **Nguyên nhân**: eKYC — ảnh chân dung không thấy khuôn mặt, hoặc có nhiều người
- **Giải pháp**: Upload ảnh chân dung rõ mặt, chỉ 1 người

### 4. "Internal server error" (HTTP 500)
- **Nguyên nhân**: Exception không xử lý được (đã có global exception handler)
- **Giải pháp**: Kiểm tra log server để xem chi tiết lỗi

### 5. `ModuleNotFoundError: No module named 'vietocr'`
- **Giải pháp**: `pip install vietocr --no-deps`

### 6. `TypeError: unhashable type: 'dict'` (TemplateResponse)
- **Nguyên nhân**: Starlette phiên bản mới thay đổi API
- **Giải pháp**: Đã fix — sử dụng `TemplateResponse(request, "template.html")`

### 7. PyTorch `weights_only=True` error
- **Nguyên nhân**: PyTorch 2.6+ mặc định `weights_only=True` khi load model
- **Giải pháp**: Đã patch tự động trong `main.py`

---

## Bảo mật (Production)

- Docker image sử dụng **multi-stage build** để giảm kích thước
- Source code được **compile thành .pyc** và xóa file .py gốc (chống dịch ngược)
- Container chạy với **non-root user**
- `docker-compose.yml` bật `read_only`, `no-new-privileges`
- Global exception handler không leak stack trace ra client
- `reload=False` trong production (không hot-reload)

---

## Cấu trúc thư mục

```
├── Dockerfile              # Multi-stage Docker build
├── docker-compose.yml      # Docker Compose production config
├── .dockerignore           # Loại trừ file không cần cho Docker
├── requirements.txt        # Python dependencies
├── run.py                  # Entry point
└── sources/
    ├── __init__.py          # FastAPI app + global exception handler
    ├── Controllers/
    │   ├── config.py        # Cấu hình
    │   ├── main.py          # API routes + logic xử lý
    │   └── utils.py         # Hàm tiện ích (NMS, transform, ...)
    ├── Database/
    │   ├── OCR/weights/     # Model weights (.pt, .pth)
    │   └── uploads/         # Thư mục upload tạm
    ├── Models/              # (Legacy — không còn sử dụng)
    ├── Views/templates/     # HTML templates
    └── static/              # CSS, JS, hình ảnh
```

---

## Ghi chú

- Ứng dụng **chỉ xử lý mặt trước** của CCCD gắn chip
- Độ chính xác phụ thuộc vào chất lượng ảnh đầu vào
- Nên sử dụng ảnh có độ phân giải tối thiểu 640x400px
- Các ảnh bị nghiêng, mờ, hoặc bị che một phần sẽ giảm độ chính xác

import base64
import io
import os
import re
from typing import Optional
import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pydantic import BaseModel
import requests

app = FastAPI(title="Onboarding Image Creator API")

# 1. Bật CORS để trình duyệt web (index.html/Apps Script) gọi API không bị chặn
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class OnboardRequest(BaseModel):
    image_url: str
    frame_url: str
    name: str
    position: str
    squad: str


def load_image_from_bytes(data: bytes) -> Image.Image:
    """Đọc bytes ảnh, tự động xoay theo EXIF và chuyển về RGBA"""
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)  # Xử lý ảnh bị xoay ngược từ điện thoại
    return img.convert("RGBA")


def download_image(url: str) -> Image.Image:
    try:
        # 1. Bắt ID nếu là link Google Drive
        drive_match = re.search(r"(?:id=|\/d\/)([a-zA-Z0-9_-]+)", url)
        if "drive.google.com" in url and drive_match:
            file_id = drive_match.group(1)

            # Tạo Session để duy trì Cookie xác nhận tải file từ Drive
            session = requests.Session()
            drive_url = "https://docs.google.com/uc?export=download"

            # Request lần 1 để lấy Cookie
            response = session.get(
                drive_url, params={"id": file_id}, timeout=15
            )

            # Tìm token xác nhận nếu file bị dính trang cảnh báo virus/file lớn
            token = None
            for key, value in response.cookies.items():
                if key.startswith("download_warning"):
                    token = value
                    break

            # Request lần 2 với token xác nhận để tải đúng file ảnh Binary
            if token:
                response = session.get(
                    drive_url,
                    params={"id": file_id, "confirm": token},
                    timeout=15,
                )

            return load_image_from_bytes(response.content)

        # 2. Nếu là link URL ảnh thông thường trên mạng
        else:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            return load_image_from_bytes(response.content)

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Không tải được ảnh từ URL/Drive: {str(e)}",
        )

def get_round_avatar(photo: Image.Image, size: int) -> Image.Image:
    photo_np = np.array(photo.convert("RGB"))
    gray = cv2.cvtColor(photo_np, cv2.COLOR_RGB2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
    )

    width, height = photo.size

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        crop_top = max(0, y - int(h * 0.65))
        crop_bottom = min(height, y + h + int(h * 0.85))
        crop_left = max(0, x - int(w * 0.55))
        crop_right = min(width, x + w + int(w * 0.55))
        photo_cropped = photo.crop(
            (crop_left, crop_top, crop_right, crop_bottom)
        )
    else:
        min_dim = min(width, height)
        photo_cropped = photo.crop(
            (
                (width - min_dim) // 2,
                (height - min_dim) // 2,
                (width + min_dim) // 2,
                (height + min_dim) // 2,
            )
        )

    photo_square = ImageOps.fit(
        photo_cropped, (size, size), Image.Resampling.LANCZOS
    )

    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size, size), fill=255)

    round_img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    round_img.paste(photo_square, (0, 0), mask)

    return round_img

def process_welcome_card(
    photo: Image.Image,
    template: Image.Image,
    name: str,
    position: str,
    squad: str,
) -> str:
    # 1. Resize khung về chuẩn 1000x1000
    template = template.resize((1000, 1000), Image.Resampling.LANCZOS)

    # 2. Cắt & dán Avatar tròn (Kích thước 460px, đặt tại X: 270, Y: 180)
    avatar_size = 460
    round_avatar = get_round_avatar(photo, avatar_size)
    template.paste(round_avatar, (270, 180), round_avatar)

    draw = ImageDraw.Draw(template)

    # 3. CHE CHỮ MẪU TRÊN KHUNG GỐC:
    # - Che chữ "NAME" bằng cách vẽ lại dải bo góc màu đen mới (X: 200->800, Y: 615->705)
    draw.rounded_rectangle(
        [200, 615, 800, 705], radius=45, fill="#2b2b2b"
    )  # Màu đen đè đè chữ NAME

    # - Che chữ "Position" & "Squad" mẫu bằng hình chữ nhật màu đỏ của nền
    # Màu đỏ nền gốc là #b9121b (hoặc RGB 185, 18, 27)
    bg_color = template.getpixel((100, 750))  # Lấy màu đỏ tự động từ nền
    draw.rectangle([350, 725, 650, 845], fill=bg_color)  # Xóa sạch chữ cũ

    # 4. LOAD FONT & VẼ CHỮ THẬT
    font_path = os.path.join("fonts", "Montserrat-Bold.ttf")
    try:
        font_name = ImageFont.truetype(font_path, 34)
        font_sub = ImageFont.truetype(font_path, 25)
    except IOError:
        font_name = ImageFont.load_default()
        font_sub = ImageFont.load_default()

    # - Vẽ HỌ TÊN (Nằm giữa dải đen)
    draw.text(
        (500, 660), name.upper(), font=font_name, fill="white", anchor="mm"
    )

    # - Vẽ CHỨC DANH (Position)
    draw.text((500, 755), position, font=font_sub, fill="white", anchor="mm")

    # - Vẽ SQUAD
    draw.text((500, 818), squad, font=font_sub, fill="white", anchor="mm")

    buffered = io.BytesIO()
    template.convert("RGB").save(buffered, format="JPEG", quality=95)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Onboarding Image Creator Service"}


# Endpoint 1: Nhận JSON payload
@app.post("/generate-welcome-card")
async def generate_welcome_card(req: OnboardRequest):
    photo = download_image(req.image_url)
    template = download_image(req.frame_url)
    img_str = process_welcome_card(
        photo, template, req.name, req.position, req.squad
    )
    return {"success": True, "image_base64": img_str}


# Endpoint 2: Linh hoạt nhận File Upload HOẶC URL/Drive Link
@app.post("/generate-welcome-card-upload")
async def generate_welcome_card_upload(
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    frame_file: Optional[UploadFile] = File(None),
    frame_url: Optional[str] = Form(None),
    name: str = Form(...),
    position: str = Form(...),
    squad: str = Form(...),
):
    try:
        # 1. Lấy Ảnh Ứng Viên
        if file and file.filename:
            contents = await file.read()
            photo = load_image_from_bytes(contents)
        elif image_url and image_url.strip():
            photo = download_image(image_url.strip())
        else:
            raise HTTPException(
                status_code=400,
                detail="Cần truyền ảnh ứng viên (file upload hoặc image_url)",
            )

        # 2. Lấy Khung Template
        if frame_file and frame_file.filename:
            frame_contents = await frame_file.read()
            template = load_image_from_bytes(frame_contents)
        elif frame_url and frame_url.strip():
            template = download_image(frame_url.strip())
        else:
            raise HTTPException(
                status_code=400,
                detail="Cần truyền khung template (frame_file hoặc frame_url)",
            )

        # 3. Xử lý ghép ảnh
        img_str = process_welcome_card(photo, template, name, position, squad)
        return {"success": True, "image_base64": img_str}

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Lỗi xử lý ảnh: {str(e)}"
        )
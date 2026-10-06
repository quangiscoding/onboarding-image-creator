import base64
import io
import os
import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pydantic import BaseModel
import requests

app = FastAPI(title="Onboarding Image Creator API")


class OnboardRequest(BaseModel):
    image_url: str
    frame_url: str
    name: str
    position: str
    squad: str


def download_image(url: str) -> Image.Image:
    try:
        response = requests.get(url, timeout=15)
        return Image.open(io.BytesIO(response.content)).convert("RGBA")
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Không tải được ảnh từ URL: {str(e)}"
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
    template = template.resize((1000, 1000), Image.Resampling.LANCZOS)

    avatar_size = 470
    round_avatar = get_round_avatar(photo, avatar_size)
    template.paste(round_avatar, (265, 225), round_avatar)

    draw = ImageDraw.Draw(template)
    font_path = os.path.join("fonts", "Montserrat-Bold.ttf")

    try:
        font_name_text = ImageFont.truetype(font_path, 36)
        font_sub_text = ImageFont.truetype(font_path, 28)
    except IOError:
        font_name_text = ImageFont.load_default()
        font_sub_text = ImageFont.load_default()

    draw.text(
        (500, 645), name.upper(), font=font_name_text, fill="white", anchor="mm"
    )
    draw.text(
        (500, 740), position, font=font_sub_text, fill="white", anchor="mm"
    )
    draw.text((500, 805), squad, font=font_sub_text, fill="white", anchor="mm")

    buffered = io.BytesIO()
    template.convert("RGB").save(buffered, format="JPEG", quality=95)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Onboarding Image Creator Service"}


@app.post("/generate-welcome-card")
async def generate_welcome_card(req: OnboardRequest):
    photo = download_image(req.image_url)
    template = download_image(req.frame_url)
    img_str = process_welcome_card(
        photo, template, req.name, req.position, req.squad
    )
    return {"success": True, "image_base64": img_str}


@app.post("/generate-welcome-card-upload")
async def generate_welcome_card_upload(
    file: UploadFile = File(...),
    frame_url: str = Form(...),
    name: str = Form(...),
    position: str = Form(...),
    squad: str = Form(...),
):
    try:
        contents = await file.read()
        photo = Image.open(io.BytesIO(contents)).convert("RGBA")
        template = download_image(frame_url)

        img_str = process_welcome_card(
            photo, template, name, position, squad
        )
        return {"success": True, "image_base64": img_str}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Lỗi xử lý ảnh: {str(e)}"
        )
import base64
import io
import os
from fastapi import FastAPI, HTTPException
from PIL import Image, ImageDraw, ImageFont, ImageOps
import face_recognition
from pydantic import BaseModel
import requests

app = FastAPI(title="Onboarding Image Creator API")


class OnboardRequest(BaseModel):
    image_url: str  # Link ảnh ứng viên
    frame_url: str  # Link template kyanon (image_3.png)
    name: str  # Tên ứng viên (NGAN NGUYEN)
    position: str  # Position
    squad: str  # Squad / Department


def download_image(url: str) -> Image.Image:
    try:
        response = requests.get(url, timeout=15)
        return Image.open(io.BytesIO(response.content)).convert("RGBA")
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Không tải được ảnh từ URL: {str(e)}"
        )


def get_round_avatar(photo: Image.Image, size: int) -> Image.Image:
    photo_rgb = photo.convert("RGB")
    photo_bytes = io.BytesIO()
    photo_rgb.save(photo_bytes, format="JPEG")
    photo_bytes.seek(0)

    image_np = face_recognition.load_image_file(photo_bytes)
    face_locations = face_recognition.face_locations(image_np)

    width, height = photo.size

    if face_locations:
        top, right, bottom, left = face_locations[0]
        face_w, face_h = right - left, bottom - top

        crop_top = max(0, top - int(face_h * 0.65))
        crop_bottom = min(height, bottom + int(face_h * 0.95))
        crop_left = max(0, left - int(face_w * 0.55))
        crop_right = min(width, right + int(face_w * 0.55))

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


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Onboarding Image Creator Service"}


@app.post("/generate-welcome-card")
async def generate_welcome_card(req: OnboardRequest):
    photo = download_image(req.image_url)
    template = download_image(req.frame_url)

    # Chuẩn hóa kích thước template 1000x1000
    template = template.resize((1000, 1000), Image.Resampling.LANCZOS)

    # 1. Cắt tròn avatar và dán vào vị trí
    avatar_size = 470
    round_avatar = get_round_avatar(photo, avatar_size)
    template.paste(round_avatar, (265, 225), round_avatar)

    # 2. Vẽ Text
    draw = ImageDraw.Draw(template)

    font_path = os.path.join("fonts", "Montserrat-Bold.ttf")
    if not os.path.exists(font_path):
        font_path = "arialbd.ttf"  # Fallback nếu không thấy font

    try:
        font_name = ImageFont.truetype(font_path, 36)
        font_text = ImageFont.truetype(font_path, 28)
    except IOError:
        font_name = ImageFont.load_default()
        font_text = ImageFont.load_default()

    # In hoa tên, căn giữa các ô[cite: 3, 5]
    draw.text(
        (500, 645),
        req.name.upper(),
        font=font_name,
        fill="white",
        anchor="mm",
    )
    draw.text(
        (500, 740), req.position, font=font_text, fill="white", anchor="mm"
    )
    draw.text((500, 805), req.squad, font=font_text, fill="white", anchor="mm")

    # 3. Xuất kết quả Base64
    buffered = io.BytesIO()
    template.convert("RGB").save(buffered, format="JPEG", quality=95)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {"success": True, "image_base64": img_str}
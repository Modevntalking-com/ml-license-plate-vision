import cv2
import numpy as np
from PIL import ImageOps, ImageEnhance, Image


def enhance(crop: Image.Image) -> Image.Image:
    image = ImageOps.grayscale(crop)
    image = ImageEnhance.Contrast(image).enhance(5)
    image = ImageEnhance.Sharpness(image).enhance(5)
    image = ImageOps.scale(image, 3.5)

    image = _remove_dark_frame(image)
    image = _remove_shades(image)
    return image


def _remove_dark_frame(img: Image.Image) -> Image.Image:
    # Convert PIL Image to OpenCV format
    open_cv_image = np.array(img.convert('L'))

    # Blur the image to reduce noise
    blurred = cv2.GaussianBlur(open_cv_image, (5, 5), 0)

    # Apply a binary threshold after blurring
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours from the binary image
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Assume the largest contour is the frame
    c = max(cnts, key=cv2.contourArea)

    # Find the bounding box coordinates from the contour
    x, y, w, h = cv2.boundingRect(c)

    # Crop the original image using the bounding box coordinates
    cropped = open_cv_image[y:y+h, x:x+w]

    # Convert back to PIL format
    cropped_pil = Image.fromarray(cropped)
    return cropped_pil

def _remove_shades(image: Image.Image, threshold: int = 64) -> Image.Image:
    # Apply the threshold
    # All pixels value > threshold will be set to 255 (white)
    # All pixels value <= threshold will be set to 0 (black)
    return image.point(lambda x: 255 if x > threshold else 0, '1')
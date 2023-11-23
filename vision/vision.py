import yolov5
from PIL import Image
from pytesseract import pytesseract
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def detect_license_plate(image_path: str, save_crops=False, save_dir="/tmp/crops") -> Image.Image:
    """
    Detect license plate from image_path
    :param image_path: image_path path
    :param save_crops: save crops to filesystem
    :param save_dir: directory to save crops
    :rtype: :py:class:`~PIL.Image.Image`
    :returns: An :py:class:`~PIL.Image.Image` object.
    """
    # load model
    model = yolov5.load('keremberke/yolov5n-license-plate')

    # set model parameters
    model.conf = 0.25  # NMS confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    model.agnostic = False  # NMS class-agnostic
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = 1  # maximum number of detections per image

    # perform inference
    results = model(image_path, size=640)
    crops = results.crop(save=save_crops, save_dir=save_dir)

    return Image.fromarray(crops[0]['im'][..., ::-1])


def read_license_plate_tesseract(image_path) -> str:
    return pytesseract.image_to_string(image_path,
                                       config='--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')


def read_license_plate_ml(image_path: str) -> str:
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
    pixel_values = processor(images=Image.open(image_path).convert("RGB"), return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
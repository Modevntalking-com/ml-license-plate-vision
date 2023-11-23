import getopt
import sys

from vision.enhancer import enhance
from vision.vision import detect_license_plate, read_license_plate_tesseract, \
    read_license_plate_ml

if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "hi:", ["input="])
    input_url = ""
    for opt, arg in opts:
        if opt == '-h':
            print('detect.py -i input_image_url')
            sys.exit()
        elif opt in ("-i", "--input"):
            input_url = arg

    print("detecting license plate from image: " + input_url)

    crop = detect_license_plate(input_url)
    enhanced_path = "./enhanced.png"

    with enhance(crop) as enhanced:
        enhanced.save(enhanced_path)

    print("license plate trocr: " + read_license_plate_ml(enhanced_path))
    print("license plate tesseract: " + read_license_plate_tesseract(enhanced_path))


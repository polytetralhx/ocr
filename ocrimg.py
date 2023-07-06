import pytesseract
import imutils
from ocrutils import pil_as_array, transform_image
from textutils import extract_json

class OCRImg:
    def __init__(self, img):
        self.img = img
        
        # for quick debugging, use get_text()
        self.text = pytesseract.image_to_string(transform_image(imutils.resize(pil_as_array(img), height = 500)))
    
    def get_text(self):
        return self.text
    
    def get_json(self):
        return extract_json(self.img, "eng")
import streamlit as st
import numpy as np
from ocrimg import OCRImg
from ocrutils import transform_image, pil_as_array, DOC_HEIGHT, DOC_WIDTH
from PIL import Image
import cv2
import imutils

def main():
    test_path = "./Images/20230704_151610.jpg"
    pic = Image.open(test_path)
    original = pil_as_array(pic)
    #original = cv2.imread(test_path)
    cv2.imshow("original", original)
    
    transformed = transform_image(original)
    cv2.imshow("transformed", transformed)
    
    cv2.namedWindow("original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("transformed", cv2.WINDOW_NORMAL)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    ocr_img = OCRImg(original)
    print(ocr_img.get_text())
    print(ocr_img.get_json())
    
if __name__ == '__main__':
    main()

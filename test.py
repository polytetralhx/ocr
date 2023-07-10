import os
import streamlit as st
import numpy as np
from ocrimg import OCRImg
from ocrutils import transform_image
from PIL import Image
import cv2
import imutils

def main():
    # Create the full path of the new folder
    new_folder_path = "./processed_images"

    # Create the new folder
    os.makedirs(new_folder_path, exist_ok = True)
    
    for image in os.listdir('./Images'):
        print(image)
        original = imutils.resize(cv2.imread('./Images/' + image), width = 500)
        transformed = transform_image(original)
        
        cv2.imwrite(f"{new_folder_path}/{image}.png", transformed)
        
        cv2.imshow("original", original)
        cv2.imshow("transformed", transformed)
        
        cv2.namedWindow("original", cv2.WINDOW_NORMAL)
        cv2.namedWindow("transformed", cv2.WINDOW_NORMAL)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()

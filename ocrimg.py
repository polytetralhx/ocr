import cv2
import pytesseract
import os
import json
import regex as re
from PIL import Image

class OCRImg:
    def __init__(self, img):
        self.img = img
        self.text = pytesseract.image_to_string(img)
    
    def get_text(self):
        return self.text
    
    def get_json(self):
        name_pattern = r'(?i)(?:^|\b)([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)*)(?:\b|$)'
        company_pattern = r'(?i)(?:^|\b)([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)*)(?:\b|$)'
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        contact_pattern = r'\b\d+\b'
        
        name = re.search(name_pattern, self.text)
        company = re.search(company_pattern, self.text)    
        email = re.search(email_pattern, self.text)
        contact = re.findall(contact_pattern, self.text)
        full_contact = ''.join(contact)
        
        res_dict = {
            "name": name.group(),
            "email": email.group(),
            "contact": full_contact,
            "company": company.group()
        }
        res_json = json.dumps(res_dict)
        
        return res_json
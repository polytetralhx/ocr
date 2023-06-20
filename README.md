# CardR: An OCR Application for Business Cards
### Deployed using Streamlit on the following website: https://polytetralhx-ocr-dashboard-first-cf9cnk.streamlit.app/

### Requirements: 

Included in requirements.txt. You can download the dependencies into your own environment by using `pip install -r requirements.txt`.

### API: 

`ocrimg.py` is provided as an API that wraps over Pytesseract, JSON and Regex libraries to easily generate JSON objects that extract the Name, Contact Number, Email Address and the Company Name the individual belongs to.

To make use of `ocrimg.py`, you can initialize an object of the `OCRImg` class :

```python
# import library used to open images
from PIL import Image

# initialize image
sample_img = Image.open("./path/to/sample/img.png")

img_to_ocr = OCRImg(sample_img)
```

The following functions can be used to extract both the resulting text and parsed JSON file with the specified keys from OCR processing of the image:

```python
res_text = img_to_ocr.get_text() #directly returns the text that was identified by the OCR engine
res_json = img_to_ocr.get_json() #takes the printed text identified and performs regex to obtain the name, contact, email and company in the scanned image
```

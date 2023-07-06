import pytesseract
import regex as re
import json

def extract_text(img, lang):
    return pytesseract.image_to_string(img, lang = lang)

def extract_name(ocr_result):
    # Define pattern to extract company
    company_pattern = r'(?i)(?:^|\b)([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)*)(?:\b|$)'

    # Extract company using regular expression pattern
    company_match = re.search(company_pattern, ocr_result)
    if company_match:
        return company_match.group(1)
    else:
        return None

def extract_email(text):
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.,-]+\.[A-Z|a-z]{2,}(?:,[A-Za-z]+(?:\.[A-Za-z]+)?)*\b"
    emails = re.findall(pattern, text)
    if emails:
        return " - ".join(emails)
    return ""

def extract_company(ocr_text):
    patterns = [
        r"(?i)(Pte Ltd|PTE LTD|\(Pte.\) Ltd\.|Co, Ltd|Co\., Ltd|CO\. PTE LTD|Ltd|Ltd\.) Trading",
        r"(?i)(Pte Ltd|PTE LTD|\(Pte.\) Ltd\.|Co, Ltd|Co\., Ltd|CO\. PTE LTD|Ltd|Ltd\.)",
        r"[\w\s]+"
    ]

    company_names = []

    for pattern in patterns:
        matches = re.findall(pattern, ocr_text)
        for match in matches:
            stripped_match = match.strip()
            if not any(char.isdigit() for char in stripped_match) and stripped_match != "":
                ignore_keywords = ["com", "www", "website"]
                if not any(keyword.lower() in stripped_match.lower() for keyword in ignore_keywords):
                    company_names.append(stripped_match)

    return " - ".join(company_names)

def extract_contact(ocr_result):
    # Extract contact number using regular expression pattern
    patterns = [
        r"\d{8}",  # 00000000
        r"\d{4} \d{4}",  # 0000 0000
        r"\d{2} \d{4} \d{4}",  # 65 0000 0000
        r"\(\d{2}\) \d{4} \d{4}",  # (65) 0000 0000
        r"\+\d{2} \d{4} \d{4}",  # +65 0000 0000
        r"\+\d{2} \d{4}-\d{4}",  # +65 0000-0000
        r"\+\d{2} \d{8}",  # +65 00000000
        r"\+\(\d{2}\) \d{4} \d{4}",  # +(65) 0000 0000
        r"\+\(\d{3}\) \d{3} \d{3} \d{3}",  # +(000) 000 000 000
        r"\+\d{2}-\d{2}-\d{4}-\d{4}",  # +00-00-0000-0000
        r"\+\d{2}-\d{3}-\d{2}-\d{4}",  # +00-000-00-0000
        r"\+\d{2} \d-\d{4}-\d{4}"  # +00 0-0000-0000
    ]

    contact_numbers = []

    for pattern in patterns:
        matches = re.findall(pattern, ocr_result)
        if matches:
            contact_numbers.extend(matches)

    return " - ".join(contact_numbers)
    

# function to produce the json
def extract_json(img, lang):
    res_dict = {}
    
    extracted_txt = extract_text(img, lang = lang)
    res_dict["name"], res_dict["email"], res_dict["contact"], res_dict["company"] = extract_name(extracted_txt), extract_email(extracted_txt), extract_contact(extracted_txt), extract_company(extracted_txt)
    
    res_json = json.dumps(res_dict)
    return res_json
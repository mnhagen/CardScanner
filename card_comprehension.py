import pytesseract
import cv2
import re

def parse_set_info(ocr_text):
    lines = ocr_text.strip().splitlines()
    set_number = None
    set_code = None
    rarity = None

    #line one: set_number/set_number + rarity
    for line in lines:
        match = re.search(r'(\d{1,3}/\d{1,3})\s*([LTCURM])', line)
        if match:
            set_number = match.group(1)
            rarity = match.group(2)
            break

    #line two: set_code + language
    for line in lines:
        match = re.search(r'([A-Z0-9]{3,}\.[A-Z]{2})', line, re.IGNORECASE)
        if match:
            set_code = match.group(1)
            break

    return {
        "set number": set_number,
        "rarity": rarity,
        "set code": set_code 
    }






path = "cropped_test_image.png"
config = '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789./ '
original = cv2.imread(path)
img = original.copy()
scale = 3 #might cause issues for non-integer scales; fix by using int() on scaling operations, but check that it is applied correctly everywhere
img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
#cv2.imshow("img",img)



gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#card name
name_bar = gray[0:90, 0:500]
name_bar = gray[0:90*scale, 0:500*scale] #rescaled
card_name = pytesseract.image_to_string(name_bar, config='--psm 7')
print("Card Name:", card_name.strip())

#set info
info_box = gray[800:889, 10:200]
info_box = gray[800*scale:889*scale, 10*scale:200*scale] #rescaled
set_info = pytesseract.image_to_string(info_box, config=config)
print("Set info:", set_info.strip())

cv2.imshow("test", name_bar)
cv2.imshow("test2", info_box)

cv2.waitKey(0)
from functions import *

path = "test_image_3.jpg"
original = cv2.imread(path)
img = original.copy()

# === Main logic ===
M, corners = get_transform_matrix(img)
if M is None:
    print("Card not found.")
    exit()

M_inv = np.linalg.inv(M)

# Define regions in flattened image space
name_bar_rect = (30, 30, 500, 90)        # x1, y1, x2, y2
set_info_rect = (10, 800, 200, 889)

# Extract from original image
name_bar_img = extract_region(original, M_inv, name_bar_rect, scale=1)
set_info_img = extract_region(original, M_inv, set_info_rect, scale=1)

# Optional: Resize for OCR
name_bar_img = cv2.resize(name_bar_img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
set_info_img = cv2.resize(set_info_img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

# Grayscale for OCR
name_bar_gray = cv2.cvtColor(name_bar_img, cv2.COLOR_BGR2GRAY)
set_info_gray = cv2.cvtColor(set_info_img, cv2.COLOR_BGR2GRAY)

# OCR config
config = '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789./ '

# OCR
card_name = pytesseract.image_to_string(name_bar_gray, config='--psm 7').strip()
set_info = pytesseract.image_to_string(set_info_gray, config=config)
set_number = set_info[:3]

rarity = set_info[7:8]
set_code = set_info[9:12]
language = set_info[13:15]

print(card_name)
print(set_number)
print(rarity)
print(set_code)
print(language)

result = search_card(
    name = card_name,
    set_code = set_code,
    number = set_number,
    rarity = rarity,
    lang = "not english"
    )

print()
print(result)

cv2.waitKey(0)

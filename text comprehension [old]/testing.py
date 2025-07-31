import numpy as np
import cv2
import pytesseract

path = "test_image_3.jpg"
original = cv2.imread(path)
img = original.copy()

width = 635
height = 889

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[2] = pts[np.argmax(s)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def get_transform_matrix(img, threshold = 0.2):
    magic_aspect = float(width) / height
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best_candidate = None
    best_area = 0

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            area = cv2.contourArea(approx)

            if 1 - threshold <= aspect_ratio / magic_aspect <= 1 + threshold and area > best_area:
                pts = approx.reshape(4, 2).astype(np.float32)
                ordered = order_points(pts)
                best_candidate = ordered
                best_area = area

    if best_candidate is not None:  
        dest = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(best_candidate, dest)
        return M, best_candidate

    return None, None


def extract_region(original_img, M_inv, rect_coords, scale=3):
    """
    rect_coords: coordinates in flat image (x1, y1, x2, y2)
    """
    # Scale region
    x1, y1, x2, y2 = [c * scale for c in rect_coords]

    # Create a grid of destination points in the warped image
    grid_x, grid_y = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))
    flat_coords = np.stack([grid_x.ravel(), grid_y.ravel(), np.ones_like(grid_x).ravel()])

    # Map them back to original image using inverse perspective
    mapped = M_inv @ flat_coords
    mapped /= mapped[2]  # Normalize by third coord

    # Sample pixels from original image using mapped coords
    map_x = mapped[0].reshape((y2 - y1, x2 - x1)).astype(np.float32)
    map_y = mapped[1].reshape((y2 - y1, x2 - x1)).astype(np.float32)

    remapped = cv2.remap(original_img, map_x, map_y, interpolation=cv2.INTER_CUBIC)
    return remapped

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
set_number = set_info[:7]

rarity = set_info[7:8]
set_code = set_info[9:12]
language = set_info[13:15]

print(card_name)
print(set_number)
print(rarity)
print(set_code)
print(language)

cv2.waitKey(0)

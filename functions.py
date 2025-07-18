import numpy as np
import cv2
import requests
import pytesseract

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

def get_transform_matrix(img, threshold=0.2, min_area_fraction=0.2, return_img=False):
    height_img, width_img = img.shape[:2]
    magic_aspect = float(width) / height
    min_area = min_area_fraction * (width_img * height_img)

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

            if (
                1 - threshold <= aspect_ratio / magic_aspect <= 1 + threshold
                and area > best_area
                and area > min_area
            ):
                pts = approx.reshape(4, 2).astype(np.float32)
                ordered = order_points(pts)
                best_candidate = ordered
                best_area = area

    if best_candidate is not None:
        dest = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(best_candidate, dest)
        if return_img:
            img_out = img.copy()
            cv2.drawContours(img_out, [np.int32(best_candidate)], -1, (255, 255, 0), 2)
            return M, best_candidate, img_out
        else:
            return M, best_candidate

    return None, None, img if return_img else (None, None)



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


def search_card(name=None, set_code=None, number=None, rarity=None, lang=None):
    query_parts = []

    if name:
        query_parts.append(f'name:"{name}"')  # Only quote the value
    if set_code:
        query_parts.append(f"set:{set_code.lower()}")
    if number:
        query_parts.append(f"number:{number.lower()}")
    if rarity:
        query_parts.append(f"rarity:{rarity.lower()}")
    if lang:
        query_parts.append(f"lang:{lang.lower()}")

    query = " ".join(query_parts)
    url = "https://api.scryfall.com/cards/search"
    params = {"q": query}

    response = requests.get(url, params=params)
    return response.json()

def is_found(response):
    return response.get("object") != "error" and response.get("total_cards", 0) > 0
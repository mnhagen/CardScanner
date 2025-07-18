from functions import *
import cv2
import numpy as np
import pytesseract

name_bar_rect = (30, 30, 500, 90)
set_info_rect = (10, 800, 200, 889)

path = "test_image_3.jpg"
original = cv2.imread(path)
img = original.copy()

# === Main logic ===
capture = cv2.VideoCapture(0)
frame_count = 0
process_every_n = 15  # Only run OCR every N frames

while True:
    ret, frame = capture.read()
    if not ret:
        break

    frame_count += 1

    M, corners, img_out = get_transform_matrix(frame, return_img=True)

    if M is None:
        cv2.imshow("card scanner", frame)
    else:
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            cv2.imshow("card scanner", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        if frame_count % process_every_n == 0:
            # Extract from original image
            name_bar_img = extract_region(frame, M_inv, name_bar_rect, scale=1)
            set_info_img = extract_region(frame, M_inv, set_info_rect, scale=1)

            # Resize for OCR
            name_bar_img = cv2.resize(name_bar_img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            set_info_img = cv2.resize(set_info_img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

            # Grayscale for OCR
            name_bar_gray = cv2.cvtColor(name_bar_img, cv2.COLOR_BGR2GRAY)
            set_info_gray = cv2.cvtColor(set_info_img, cv2.COLOR_BGR2GRAY)

            # OCR config
            config1 = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789./ '
            config2 = '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789./ '

            # OCR
            card_name = pytesseract.image_to_string(name_bar_gray, config=config2).strip()
            set_info = pytesseract.image_to_string(set_info_gray, config=config2).strip()

            if len(set_info) >= 15:
                set_number = set_info[:5].strip()
                rarity = set_info[7:8].strip()
                set_code = set_info[9:12].strip()
                language = set_info[13:15].strip()
            else:
                set_number = rarity = set_code = language = ""

        # Draw info on img_out if card_name is available
        if 'card_name' in locals() and card_name:
            cv2.rectangle(img_out, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(img_out, f"Detected: {set_number}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Always show frame (processed or raw)
    cv2.imshow("card scanner", img_out if M is not None else frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

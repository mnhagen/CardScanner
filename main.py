# main.py
import os
import json
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
from skimage.metrics import structural_similarity as ssim

# ------------------------
# Load model + class mapping
# ------------------------
model_name = "scanner_EOE+SLD"
MODEL_PATH = f"models/{model_name}.pth"
IDX_PATH   = f"models/idx_{model_name}.json"

with open(IDX_PATH, "r") as f:
    idx_to_class = json.load(f)

num_classes = len(idx_to_class)
model = models.resnet18(weights = None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# ------------------------
# Transform (same as training)
# ------------------------
val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ------------------------
# Utility functions
# ------------------------
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def predict_card(cv2_img):
    img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    img_tensor = val_transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        pred_class = idx_to_class[str(pred_idx)]
        confidence = probs[0, pred_idx].item()
    return pred_class, confidence

def draw_and_warp(img, threshold_1=50, threshold_2=150, tolerance=0.2, min_area_frac=0.2):
    img_height, img_width = img.shape[:2]
    magic_aspect = 6.35 / 8.89  # MTG card aspect ratio
    min_area = min_area_frac * img_height * img_width

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold_1, threshold_2)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    warped = None
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            rect = order_points(pts)
            (tl, tr, br, bl) = rect
            w = np.linalg.norm(br - bl)
            h = np.linalg.norm(tr - br)
            aspect_ratio = w / h if h != 0 else 0
            area = cv2.contourArea(rect)
            if magic_aspect - tolerance <= aspect_ratio <= magic_aspect + tolerance and area > min_area:
                cv2.drawContours(img, [approx], -1, (255, 255, 0), 2)
                dst_height, dst_width = 680, 488
                dst = np.array([[0, 0], [dst_width - 1, 0],
                                [dst_width - 1, dst_height - 1], [0, dst_height - 1]], dtype="float32")
                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(img, M, (dst_width, dst_height))
                break
    return img, warped

# ------------------------
# Main webcam loop
# ------------------------
def main():
    capture = cv2.VideoCapture(0)

    last_warped_gray = None
    pred_class_old = None
    SIMILARITY_THRESHOLD = 0.9  # higher = stricter
    CONF_THRESHOLD = 0.5

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        contoured, warped = draw_and_warp(frame)
        cv2.imshow("card scanner", contoured)

        if warped is not None:
            cv2.imshow("warped", warped)
            gray_current = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

            if last_warped_gray is None:
                # First detection → always predict
                pred_class, confidence = predict_card(warped)
                if pred_class != pred_class_old and confidence > CONF_THRESHOLD:
                    print(f"Predicted: {pred_class} | conf: {confidence:.2f}")
                    pred_class_old = pred_class
                    last_warped_gray = gray_current

            else:
                # Compare to last frame using SSIM
                score = ssim(gray_current, last_warped_gray)
                if score < SIMILARITY_THRESHOLD:  # card changed significantly
                    pred_class, confidence = predict_card(warped)
                    if pred_class != pred_class_old and confidence > CONF_THRESHOLD:
                        print(f"Predicted: {pred_class} | conf: {confidence:.4f}")
                        pred_class_old = pred_class
                        last_warped_gray = gray_current
        else:
            last_warped_gray = None  # reset when no card detected

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

# Run only if executed directly
if __name__ == "__main__":
    main()

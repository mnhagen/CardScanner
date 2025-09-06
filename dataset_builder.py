# dataset_builder.py
import os
import re
import json
import requests
import urllib.request
from PIL import Image

#set code
set_code = "TDM" #current model supports EOE and SLD

# ------------------------
# Helpers
# ------------------------
def build_global_classes(root_dirs):
    """
    Scan multiple dataset roots and build a consistent class_to_idx mapping.

    Args:
        root_dirs (list[str]): List of dataset root folders
                               e.g. ["data/EOE/dataset", "data/SLD/dataset"]

    Returns:
        class_to_idx (dict): mapping class_name -> idx
        idx_to_class (dict): reverse mapping idx -> class_name
    """
    all_classes = set()

    # collect all subfolder names across roots
    for root in root_dirs:
        if not os.path.isdir(root):
            continue
        classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        all_classes.update(classes)

    # assign consistent indices
    all_classes = sorted(all_classes)  # sort so it's stable across runs
    class_to_idx = {cls_name: i for i, cls_name in enumerate(all_classes)}
    idx_to_class = {i: cls_name for cls_name, i in class_to_idx.items()}

    return class_to_idx, idx_to_class





def safe_filename(name: str) -> str:
    """
    Replace any character that's not alphanumeric, space, dot, or underscore.
    Used to avoid filesystem issues with card names.
    """
    return re.sub(r'[^a-zA-Z0-9 _.-]', '_', name)


def fetch_set_cards(set_code: str):
    """
    Fetch all cards in a set from the Scryfall API.
    Returns the list of card objects.
    """
    url = f"https://api.scryfall.com/cards/search?q=set:{set_code}&unique=prints"
    cards = []
    has_more = True
    while has_more:
        response = requests.get(url)
        data = response.json()
        cards.extend(data["data"])
        has_more = data.get("has_more", False)
        if has_more:
            url = data["next_page"]
    return cards


def save_metadata(cards, set_code: str):
    """Save card metadata to JSON file in data/<set_code>/meta/"""
    meta_dir = f"data/{set_code}/meta"
    os.makedirs(meta_dir, exist_ok=True)
    with open(f"{meta_dir}/cards.json", "w") as f:
        json.dump(cards, f, indent=2)


def download_images(cards, set_code: str):
    """
    Download card images into data/<set_code>/dataset/.
    Each card has its own folder with one or more face images.
    """
    dataset_dir = f"data/{set_code}/dataset"
    os.makedirs(dataset_dir, exist_ok=True)

    for card in cards:
        name = safe_filename(card["name"])
        set_name = card["set"].upper()
        collector = card.get("collector_number", "NA")
        filename_base = f"{name}_{set_name}_{collector}"

        dst_dir = os.path.join(dataset_dir, filename_base)
        os.makedirs(dst_dir, exist_ok=True)

        if "image_uris" in card:
            out_path = os.path.join(dst_dir, f"{filename_base}.jpg")
            if not os.path.exists(out_path):
                img_url = card["image_uris"]["normal"]
                urllib.request.urlretrieve(img_url, out_path)

                #resize
                with Image.open(out_path) as img:
                    img = img.resize((224,224), Image.LANCZOS)
                    img.save(out_path)

        elif "card_faces" in card:
            for i, face in enumerate(card["card_faces"]):
                if "image_uris" in face:
                    out_path = os.path.join(dst_dir, f"{filename_base}_face{i}.jpg")
                    if not os.path.exists(out_path):
                        img_url = face["image_uris"]["normal"]
                        urllib.request.urlretrieve(img_url, out_path)

                        with Image.open(out_path) as img:
                            img = img.resize((224, 224), Image.LANCZOS)
                            img.save(out_path)

    print(f"Finished building dataset at {dataset_dir}")


# ------------------------
# Main runnable script
# ------------------------
if __name__ == "__main__":
    print(f"Fetching cards for set {set_code}...")
    cards = fetch_set_cards(set_code)

    print(f"Found {len(cards)} printings.")
    save_metadata(cards, set_code)
    download_images(cards, set_code)
    print("Dataset build complete.")

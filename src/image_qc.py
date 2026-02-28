import csv
import hashlib
import os
import re
import sys
import urllib.request
from urllib.parse import urlparse

import pandas as pd
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_CSV = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "processed", "master_dataset.csv"))
IMAGES_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "raw", "images"))
QC_REPORT = os.path.normpath(os.path.join(BASE_DIR, "..", "results", "metrics", "image_qc_report.csv"))
VALID_LIST = os.path.normpath(os.path.join(BASE_DIR, "..", "results", "metrics", "valid_images.csv"))

URL_PATTERN = re.compile(r"^https?://", re.IGNORECASE)


def safe_filename(url, fallback_ext=".jpg"):
    parsed = urlparse(url)
    name = os.path.basename(parsed.path)
    if not name or "." not in name:
        name_hash = hashlib.md5(url.encode("utf-8")).hexdigest()
        return f"{name_hash}{fallback_ext}"
    return name


def download_file(url, dest_path, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest_path, "wb") as f:
        f.write(resp.read())


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(QC_REPORT), exist_ok=True)

    df = pd.read_csv(MASTER_CSV)
    if "Image URL" not in df.columns:
        print("Missing 'Image URL' column.")
        sys.exit(1)

    rows = []
    valid_rows = []

    for idx, url in enumerate(df["Image URL"].astype(str)):
        url = url.strip()
        status = "ok"
        reason = ""
        image_path = ""
        rgb_path = ""
        width = ""
        height = ""
        img_format = ""

        if not URL_PATTERN.match(url):
            status = "bad_url"
            reason = "Not a valid URL"
        else:
            filename = safe_filename(url)
            image_path = os.path.join(IMAGES_DIR, filename)
            rgb_path = os.path.join(IMAGES_DIR, f"rgb_{os.path.splitext(filename)[0]}.jpg")

            try:
                if not os.path.exists(image_path):
                    download_file(url, image_path)
            except Exception as e:
                status = "download_failed"
                reason = str(e)

            if status == "ok":
                try:
                    with Image.open(image_path) as img:
                        img_format = img.format or ""
                        width, height = img.size
                        rgb_img = img.convert("RGB")
                        rgb_img.save(rgb_path, format="JPEG")
                except Exception as e:
                    status = "corrupt_or_unreadable"
                    reason = str(e)

        rows.append(
            {
                "row_index": idx,
                "url": url,
                "status": status,
                "reason": reason,
                "download_path": image_path,
                "rgb_path": rgb_path,
                "width": width,
                "height": height,
                "format": img_format,
            }
        )

        if status == "ok":
            valid_rows.append(
                {
                    "row_index": idx,
                    "url": url,
                    "rgb_path": rgb_path,
                    "width": width,
                    "height": height,
                    "format": img_format,
                }
            )

    pd.DataFrame(rows).to_csv(QC_REPORT, index=False)
    pd.DataFrame(valid_rows).to_csv(VALID_LIST, index=False)
    print(f"QC report saved: {QC_REPORT}")
    print(f"Valid images list saved: {VALID_LIST}")


if __name__ == "__main__":
    main()

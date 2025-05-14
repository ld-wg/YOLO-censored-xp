import cv2
import json
import os
import argparse
import sys
from pathlib import Path
from typing import Dict, Literal, Tuple
from tqdm import tqdm

# constant for optimal face anonymization while keeping context
BLUR_KERNEL_SIZE = 45  

def load_annotations(annotation_path: str) -> Dict:
    # load annotations from .odgt file
    annotations = {}
    print(f"Loading annotations from: {annotation_path}")
    try:
        with open(annotation_path, 'r') as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    if 'ID' in data:
                        annotations[data['ID']] = data
                    else:
                        print(f"Warning: Missing 'ID' in line {i+1}", file=sys.stderr)
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON in line {i+1}", file=sys.stderr)
    except FileNotFoundError:
        print(f"Error: Annotation file not found at {annotation_path}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded annotations for {len(annotations)} images.")
    return annotations

def censor_image(img: cv2.Mat, annotation_data: Dict, method: Literal['blur', 'bbox']) -> Tuple[cv2.Mat, int]:
    # apply censoring to head regions in image
    h_img, w_img = img.shape[:2]
    censor_count = 0

    if 'gtboxes' not in annotation_data:
        print(f"Warning: No 'gtboxes' found for image ID {annotation_data.get('ID', 'N/A')}", file=sys.stderr)
        return img, censor_count

    for gtbox in annotation_data['gtboxes']:
        if gtbox.get('tag') == 'person':
            # check if head is ignored
            head_attr = gtbox.get('head_attr', {})
            if head_attr.get('ignore', 0) == 1:
                continue

            # check if head box exists
            if 'hbox' not in gtbox:
                continue

            hbox = gtbox['hbox']
            try:
                x, y, w, h = map(int, hbox)
            except (ValueError, TypeError):
                print(f"Warning: Invalid hbox format {hbox} for image ID {annotation_data.get('ID', 'N/A')}", file=sys.stderr)
                continue

            # clamp coordinates to image boundaries
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w_img, x + w)
            y2 = min(h_img, y + h)

            # ensure roi is valid
            if x1 >= x2 or y1 >= y2:
                continue

            # apply censoring
            if method == 'blur':
                head_roi = img[y1:y2, x1:x2]
                # apply gaussian blur if roi is not empty
                if head_roi.size > 0:
                    blurred_roi = cv2.GaussianBlur(head_roi, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0)
                    img[y1:y2, x1:x2] = blurred_roi
                    censor_count += 1
            elif method == 'bbox':
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)  # -1 fills rectangle
                censor_count += 1

    return img, censor_count

def process_dataset(
    image_dir: str,
    annotation_file: str,
    output_dir: str,
    method: Literal['blur', 'bbox']
) -> Tuple[int, int]:
    # create output directory
    os.makedirs(output_dir, exist_ok=True)

    # load annotations
    annotations = load_annotations(annotation_file)

    processed_count = 0
    skipped_count = 0

    # process images with progress bar
    for image_id, annotation_data in tqdm(annotations.items(), desc=f"Processing images ({method} method)"):
        # try both jpg and jpeg extensions
        for ext in ['.jpg', '.jpeg']:
            image_filename = f"{image_id}{ext}"
            image_path = os.path.join(image_dir, image_filename)
            if os.path.exists(image_path):
                break
        else:
            print(f"Warning: Image file not found, skipping: {image_id}", file=sys.stderr)
            skipped_count += 1
            continue

        output_path = os.path.join(output_dir, image_filename)

        # load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Failed to load image, skipping: {image_path}", file=sys.stderr)
            skipped_count += 1
            continue

        # censor image
        censored_img, num_censored = censor_image(img, annotation_data, method)

        # save censored image
        if cv2.imwrite(output_path, censored_img):
            processed_count += 1
        else:
            print(f"Error: Failed to write censored image: {output_path}", file=sys.stderr)
            skipped_count += 1

    return processed_count, skipped_count

def main():
    parser = argparse.ArgumentParser(description="Censor head regions in CrowdHuman dataset images using both blur and bbox methods.")
    parser.add_argument('--dataset-path', required=True, help="Path to the dataset root directory (e.g., ./crowdhuman)")

    args = parser.parse_args()

    # setup paths relative to dataset root
    dataset_path = Path(args.dataset_path)
    image_dir = dataset_path / "uncensored"
    annotation_file = dataset_path / "annotation.odgt"
    output_dir_blur = dataset_path / "censored-blur"
    output_dir_bbox = dataset_path / "censored-bbox"

    # validate dataset structure
    if not image_dir.exists():
        print(f"Error: Image directory not found at {image_dir}", file=sys.stderr)
        sys.exit(1)
    if not annotation_file.exists():
        print(f"Error: Annotation file not found at {annotation_file}", file=sys.stderr)
        sys.exit(1)

    # process dataset with both methods
    total_processed = 0
    total_skipped = 0

    # process blur method
    print("\nProcessing blur censoring...")
    processed, skipped = process_dataset(
        str(image_dir),
        str(annotation_file),
        str(output_dir_blur),
        'blur'
    )
    total_processed += processed
    total_skipped += skipped
    print(f"\nBlur censoring complete:")
    print(f" Successfully processed: {processed}")
    print(f" Skipped (not found/error): {skipped}")
    print(f" Censored images saved to: '{output_dir_blur}'")

    # process bbox method
    print("\nProcessing bbox censoring...")
    processed, skipped = process_dataset(
        str(image_dir),
        str(annotation_file),
        str(output_dir_bbox),
        'bbox'
    )
    total_processed += processed
    total_skipped += skipped
    print(f"\nBbox censoring complete:")
    print(f" Successfully processed: {processed}")
    print(f" Skipped (not found/error): {skipped}")
    print(f" Censored images saved to: '{output_dir_bbox}'")

    print("\nOverall Processing Summary:")
    print(f" Total images processed: {total_processed}")
    print(f" Total images skipped: {total_skipped}")

if __name__ == "__main__":
    main() 
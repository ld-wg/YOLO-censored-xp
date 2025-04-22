import cv2
import json
import os
import argparse
import sys

# example usage:
# python censor_crowdface.py --image-dir /crowdface/train_uncensored --annotation-file /crowdface/annotation_train.odgt --output-dir /crowdface/train_censored --method blur

def load_annotations(annotation_path):
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

def censor_image(img, annotation_data, method='blur', blur_kernel_size=51):
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
                # ensure kernel size is odd
                k = blur_kernel_size if blur_kernel_size % 2 != 0 else blur_kernel_size + 1
                head_roi = img[y1:y2, x1:x2]
                # apply gaussian blur if roi is not empty
                if head_roi.size > 0:
                    blurred_roi = cv2.GaussianBlur(head_roi, (k, k), 0)
                    img[y1:y2, x1:x2] = blurred_roi
                    censor_count += 1
            elif method == 'blackbox':
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1) # -1 fills rectangle
                censor_count += 1
            else:
                 print(f"Warning: Unknown censoring method '{method}'", file=sys.stderr)
                 # skip censoring if method unknown
                 pass


    return img, censor_count


def main():
    parser = argparse.ArgumentParser(description="Censor head regions in CrowdHuman dataset images.")
    parser.add_argument('--image-dir', required=True, help="Directory containing the original images.")
    parser.add_argument('--annotation-file', required=True, help="Path to the .odgt annotation file (e.g., annotation_train.odgt).")
    parser.add_argument('--output-dir', required=True, help="Directory to save the censored images.")
    parser.add_argument('--method', choices=['blur', 'blackbox'], default='blur', help="Censoring method: 'blur' or 'blackbox'.")
    parser.add_argument('--blur-kernel-size', type=int, default=51, help="Odd integer kernel size for Gaussian blur.")

    args = parser.parse_args()

    # validate blur kernel size
    if args.method == 'blur' and (args.blur_kernel_size <= 0 or args.blur_kernel_size % 2 == 0):
        print("Error: --blur-kernel-size must be a positive odd integer.", file=sys.stderr)
        sys.exit(1)


    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # load annotations
    annotations = load_annotations(args.annotation_file)

    processed_count = 0
    skipped_count = 0
    # process images
    for image_id, annotation_data in annotations.items():
        # assume jpg images
        image_filename = f"{image_id}.jpg"
        image_path = os.path.join(args.image_dir, image_filename)
        output_path = os.path.join(args.output_dir, image_filename)

        if not os.path.exists(image_path):
            print(f"Warning: Image file not found, skipping: {image_path}", file=sys.stderr)
            skipped_count += 1
            continue

        # load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Failed to load image, skipping: {image_path}", file=sys.stderr)
            skipped_count += 1
            continue

        # censor image
        censored_img, num_censored = censor_image(img, annotation_data, args.method, args.blur_kernel_size)

        # save censored image
        if cv2.imwrite(output_path, censored_img):
             print(f"Processed '{image_filename}': Censored {num_censored} heads. Saved to '{output_path}'")
             processed_count += 1
        else:
             print(f"Error: Failed to write censored image: {output_path}", file=sys.stderr)
             skipped_count += 1


    print("Processing Summary:")
    print(f" Total images in annotation: {len(annotations)}")
    print(f" Successfully processed:     {processed_count}")
    print(f" Skipped (not found/error): {skipped_count}")
    print(f" Censored images saved to:  '{args.output_dir}'")

if __name__ == "__main__":
    main() 
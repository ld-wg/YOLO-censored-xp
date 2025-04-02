import os
import shutil
import kagglehub
import zipfile

# Kaggle dataset identifier
KAGGLE_DATASET_ID = "techzizou/labeled-mask-dataset-yolo-darknet"

# Destination directory for Kaggle dataset processing
KAGGLE_DEST_DIR = os.path.join("data", "kaggle", "person")
KAGGLE_CENSORED_IMAGES = os.path.join(KAGGLE_DEST_DIR, "censored", "images")
KAGGLE_CENSORED_LABELS = os.path.join(KAGGLE_DEST_DIR, "censored", "labels")
KAGGLE_UNCENSORED_IMAGES = os.path.join(KAGGLE_DEST_DIR, "uncensored", "images")
KAGGLE_UNCENSORED_LABELS = os.path.join(KAGGLE_DEST_DIR, "uncensored", "labels")

# Constants for other datasets (placeholders)
CROWDHUMAN_DIR = 'datasets/crowdhuman'
OPENIMAGES_DIR = 'datasets/openimages'

def setup_kaggle_directory_structure():
    """
    Create the directory structure for the Kaggle dataset.
    Creates two folders: censored and uncensored, each with images and labels subfolders.
    """
    os.makedirs(KAGGLE_CENSORED_IMAGES, exist_ok=True)
    os.makedirs(KAGGLE_CENSORED_LABELS, exist_ok=True)
    os.makedirs(KAGGLE_UNCENSORED_IMAGES, exist_ok=True)
    os.makedirs(KAGGLE_UNCENSORED_LABELS, exist_ok=True)
    print(f"Created Kaggle directories:\n{KAGGLE_CENSORED_IMAGES}\n{KAGGLE_CENSORED_LABELS}\n{KAGGLE_UNCENSORED_IMAGES}\n{KAGGLE_UNCENSORED_LABELS}")

def process_kaggle_dataset():
    """
    Download and process the Kaggle Labeled Mask Dataset.
    The dataset is structured with files like '131-with-mask.jpg' and '131-with-mask.txt' or 
    'masked (131).jpg' and 'masked (131).txt' for censored images, and files like '132.jpg' and '132.txt' 
    for uncensored images.
    
    Files will be copied into:
      - data/kaggle/person/censored/images and /labels for files containing "mask" (except those with "unmasked")
      - data/kaggle/person/uncensored/images and /labels for files containing "unmasked" or not containing "mask".
    """
    print("Downloading Kaggle dataset...")
    kaggle_path = kagglehub.dataset_download(KAGGLE_DATASET_ID)
    print("Files in downloaded dataset:", os.listdir(kaggle_path))
    print("Path to dataset files:", kaggle_path)

    # Use subdirectory if exists (e.g. 'obj')
    contents = os.listdir(kaggle_path)
    if len(contents) == 1 and os.path.isdir(os.path.join(kaggle_path, contents[0])):
        source_dir = os.path.join(kaggle_path, contents[0])
        print(f"Using source directory: {source_dir}")
    else:
        source_dir = kaggle_path

    # Extract any zip files if present
    for file in os.listdir(source_dir):
        if file.endswith('.zip'):
            zip_file = os.path.join(source_dir, file)
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(source_dir)
            print(f"Extracted {zip_file}")

    # Process downloaded files: copy images and labels to censored or uncensored folders based on filename
    for file in os.listdir(source_dir):
        file_lower = file.lower()
        src_file = os.path.join(source_dir, file)
        if file_lower.endswith(('.jpg', '.jpeg', '.png')):
            if "unmasked" in file_lower:
                shutil.copy(src_file, KAGGLE_UNCENSORED_IMAGES)
                print(f"Copied uncensored image: {file}")
            elif "mask" in file_lower:
                shutil.copy(src_file, KAGGLE_CENSORED_IMAGES)
                print(f"Copied censored image: {file}")
            else:
                shutil.copy(src_file, KAGGLE_UNCENSORED_IMAGES)
                print(f"Copied uncensored image: {file}")
        elif file_lower.endswith('.txt'):
            if "unmasked" in file_lower:
                shutil.copy(src_file, KAGGLE_UNCENSORED_LABELS)
                print(f"Copied uncensored label: {file}")
            elif "mask" in file_lower:
                shutil.copy(src_file, KAGGLE_CENSORED_LABELS)
                print(f"Copied censored label: {file}")
            else:
                shutil.copy(src_file, KAGGLE_UNCENSORED_LABELS)
                print(f"Copied uncensored label: {file}")
    print("Kaggle dataset processed.")

def process_crowdhuman_dataset():
    """
    Process the CrowdHuman dataset.
    Conversion from the native format to YOLO format must be implemented.
    """
    print("Processing CrowdHuman dataset (conversion logic not implemented).")

def process_openimages_dataset():
    """
    Process the Open Images dataset.
    Conversion from CSV/XML annotation format to YOLO format must be implemented.
    """
    print("Processing Open Images dataset (conversion logic not implemented).")

def rename_image_label_pairs(images_dir, labels_dir):
    """
    Rename image-label pairs in the given directories. It assumes that for each image file in images_dir,
    there is a corresponding label file in labels_dir with the same base name and a .txt extension.
    The function renames them to consecutive numbers (starting from 1) with the original image extension and .txt for labels.
    """
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    for idx, image_file in enumerate(image_files, start=1):
        base, ext = os.path.splitext(image_file)
        old_image_path = os.path.join(images_dir, image_file)
        new_image_name = f"{idx}{ext}"
        new_image_path = os.path.join(images_dir, new_image_name)
        os.rename(old_image_path, new_image_path)
        print(f"Renamed image {image_file} to {new_image_name}")
        
        # Rename the corresponding label file if it exists
        old_label_file = base + ".txt"
        old_label_path = os.path.join(labels_dir, old_label_file)
        if os.path.exists(old_label_path):
            new_label_name = f"{idx}.txt"
            new_label_path = os.path.join(labels_dir, new_label_name)
            os.rename(old_label_path, new_label_path)
            print(f"Renamed label {old_label_file} to {new_label_name}")
        else:
            print(f"Warning: No corresponding label for image {image_file}")

def main():
    setup_kaggle_directory_structure(KAGGLE_DEST_DIR)
    print("Processing Kaggle dataset...")
    process_kaggle_dataset()
    print("Processing CrowdHuman dataset...")
    process_crowdhuman_dataset()
    print("Processing Open Images dataset...")
    process_openimages_dataset()
    
    # Rename files in each Kaggle folder (censored and uncensored)
    print("Renaming files in censored folder...")
    rename_image_label_pairs(KAGGLE_CENSORED_IMAGES, KAGGLE_CENSORED_LABELS)
    print("Renaming files in uncensored folder...")
    rename_image_label_pairs(KAGGLE_UNCENSORED_IMAGES, KAGGLE_UNCENSORED_LABELS)
    
    print("Unified dataset creation complete.")

if __name__ == '__main__':
    main()
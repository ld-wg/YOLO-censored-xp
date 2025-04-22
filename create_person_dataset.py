import os
import shutil
import kagglehub
import zipfile
import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub
from huggingface_hub import snapshot_download

# kaggle dataset identifier
KAGGLE_DATASET_ID = "techzizou/labeled-mask-dataset-yolo-darknet"

# destination directory for kaggle dataset processing
KAGGLE_DEST_DIR = os.path.join("data", "kaggle", "person")
KAGGLE_CENSORED_IMAGES = os.path.join(KAGGLE_DEST_DIR, "censored", "images")
KAGGLE_CENSORED_LABELS = os.path.join(KAGGLE_DEST_DIR, "censored", "labels")
KAGGLE_UNCENSORED_IMAGES = os.path.join(KAGGLE_DEST_DIR, "uncensored", "images")
KAGGLE_UNCENSORED_LABELS = os.path.join(KAGGLE_DEST_DIR, "uncensored", "labels")

# constants for other datasets
CROWDHUMAN_DIR = 'datasets/crowdhuman'
OPENIMAGES_DIR = 'datasets/openimages'

def setup_kaggle_directory_structure():
    # create directory structure for kaggle dataset
    os.makedirs(KAGGLE_CENSORED_IMAGES, exist_ok=True)
    os.makedirs(KAGGLE_CENSORED_LABELS, exist_ok=True)
    os.makedirs(KAGGLE_UNCENSORED_IMAGES, exist_ok=True)
    os.makedirs(KAGGLE_UNCENSORED_LABELS, exist_ok=True)
    print(f"Created Kaggle directories:\n{KAGGLE_CENSORED_IMAGES}\n{KAGGLE_CENSORED_LABELS}\n{KAGGLE_UNCENSORED_IMAGES}\n{KAGGLE_UNCENSORED_LABELS}")

def process_kaggle_dataset():
    # download and process kaggle mask dataset
    print("Downloading Kaggle dataset...")
    kaggle_path = kagglehub.dataset_download(KAGGLE_DATASET_ID)
    print("Files in downloaded dataset:", os.listdir(kaggle_path))
    print("Path to dataset files:", kaggle_path)

    # use subdirectory if exists
    contents = os.listdir(kaggle_path)
    if len(contents) == 1 and os.path.isdir(os.path.join(kaggle_path, contents[0])):
        source_dir = os.path.join(kaggle_path, contents[0])
        print(f"Using source directory: {source_dir}")
    else:
        source_dir = kaggle_path

    # extract zip files if present
    for file in os.listdir(source_dir):
        if file.endswith('.zip'):
            zip_file = os.path.join(source_dir, file)
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(source_dir)
            print(f"Extracted {zip_file}")

    # process files: copy to censored or uncensored folders based on filename
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
    # process crowdhuman dataset
    print("Downloading CrowdHuman dataset using Snapshot Download...")
    # specify target directory
    dataset_path = snapshot_download(
        "sshao0516/CrowdHuman",
        token="****",
        local_dir="crowdface",
        repo_type="dataset"
    )
    print("Dataset downloaded to:", dataset_path)
    

def process_openimages_dataset():
    # process open images dataset
    print("Processing Open Images dataset (conversion logic not implemented).")

def rename_image_label_pairs(images_dir, labels_dir):
    # rename image-label pairs to consecutive numbers
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    for idx, image_file in enumerate(image_files, start=1):
        base, ext = os.path.splitext(image_file)
        old_image_path = os.path.join(images_dir, image_file)
        new_image_name = f"{idx}{ext}"
        new_image_path = os.path.join(images_dir, new_image_name)
        os.rename(old_image_path, new_image_path)
        print(f"Renamed image {image_file} to {new_image_name}")
        
        # rename corresponding label file if it exists
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
    
    # rename files in each kaggle folder
    print("Renaming files in censored folder...")
    rename_image_label_pairs(KAGGLE_CENSORED_IMAGES, KAGGLE_CENSORED_LABELS)
    print("Renaming files in uncensored folder...")
    rename_image_label_pairs(KAGGLE_UNCENSORED_IMAGES, KAGGLE_UNCENSORED_LABELS)
    
    print("Unified dataset creation complete.")

if __name__ == '__main__':
    main()
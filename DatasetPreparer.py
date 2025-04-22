#!/usr/bin/env python3
# DatasetPreparer module for CrowdHuman Dataset Trainer
# Handles dataset preparation, annotation conversion and symlink creation

import os
import json
import yaml
import random
import shutil
import logging
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from Config import Config

logger = logging.getLogger('crowdhuman_trainer')

class DatasetPreparer:
    # dataset preparation, including annotation conversion and symlink creation.
    
    def __init__(self, config: Config) -> None:
        self.config: Config = config
        self.image_extensions: Set[str] = {'.jpg', '.jpeg', '.png'}
        
    def clean_dataset_directory(self) -> None:
        # clean existing dataset directory to ensure a clean state.
        if os.path.exists(self.config.yolo_dataset_dir):
            logger.info(f"Removing existing dataset directory: {self.config.yolo_dataset_dir}")
            try:
                shutil.rmtree(self.config.yolo_dataset_dir)
                logger.info("Previous dataset directory removed successfully")
            except Exception as e:
                logger.warning(f"Could not remove directory: {e}")
                
        # create necessary directories
        dirs_to_create = [
            self.config.uncensored_images_dir, self.config.uncensored_labels_dir,
            self.config.censored_images_dir, self.config.censored_labels_dir,
            self.config.val_images_dir, self.config.val_labels_dir
        ]
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
            logger.debug(f"Created directory: {dir_path}")
    
    def validate_dataset_pairings(self, image_dir: Path, label_dir: Path) -> bool:
        valid = True
        
        # Check images -> labels
        for img_file in os.listdir(image_dir):
            if any(img_file.lower().endswith(ext) for ext in self.image_extensions):
                base_name = os.path.splitext(img_file)[0]
                label_file = os.path.join(label_dir, f"{base_name}.txt")
                if not os.path.exists(label_file):
                    logger.warning(f"Image {img_file} has no corresponding label file")
                    valid = False
        
        # Check labels -> images
        for label_file in os.listdir(label_dir):
            if label_file.endswith('.txt'):
                base_name = os.path.splitext(label_file)[0]
                img_exists = any(
                    os.path.exists(os.path.join(image_dir, f"{base_name}{ext}")) 
                    for ext in self.image_extensions
                )
                if not img_exists:
                    logger.warning(f"Label {label_file} has no corresponding image file (checked extensions: {self.image_extensions})")
                    valid = False
        
        return valid
    
    def process_dataset(self) -> Dict[str, int]:
        # process the entire dataset, creating labels and symlinks.
        logger.info("Starting dataset preparation")
        
        # clean the dataset directory first
        self.clean_dataset_directory()
        
        # create labels
        uncensored_count = self.create_labels(
            self.config.annotation_train, 
            self.config.image_uncensored,
            self.config.uncensored_labels_dir, 
            apply_fraction=True
        )
        
        censored_count = self.create_labels(
            self.config.annotation_train, 
            self.config.image_censored,
            self.config.censored_labels_dir, 
            apply_fraction=True
        )
        
        val_count = self.create_labels(
            self.config.annotation_val, 
            self.config.image_val,
            self.config.val_labels_dir, 
            apply_fraction=True
        )
        
        # create symlinks for images that have labels
        self.create_image_symlinks(self.config.image_uncensored, self.config.uncensored_images_dir, self.config.uncensored_labels_dir)
        self.create_image_symlinks(self.config.image_censored, self.config.censored_images_dir, self.config.censored_labels_dir)
        self.create_image_symlinks(self.config.image_val, self.config.val_images_dir, self.config.val_labels_dir)

        # validate dataset pairings AFTER symlinks are created
        logger.info("Validating final dataset pairings in prepared directories...")
        valid = True
        valid &= self.validate_dataset_pairings(self.config.uncensored_images_dir, self.config.uncensored_labels_dir)
        valid &= self.validate_dataset_pairings(self.config.censored_images_dir, self.config.censored_labels_dir)
        valid &= self.validate_dataset_pairings(self.config.val_images_dir, self.config.val_labels_dir)
        
        if not valid:
            logger.error("Dataset validation failed - mismatches between images and labels found in prepared directories")
            return {
                'uncensored_count': 0,
                'censored_count': 0,
                'val_count': 0
            }
        
        # log summary of processed labels
        logger.info("Label Processing Summary:")
        logger.info(f"  Uncensored Training: {uncensored_count} labels created")
        logger.info(f"  Censored Training:   {censored_count} labels created")
        logger.info(f"  Validation:          {val_count} labels created")
        
        return {
            'uncensored_count': uncensored_count,
            'censored_count': censored_count,
            'val_count': val_count
        }
    
    def create_labels(self, annotation_file: Path, image_dir: Path, label_dir: Path, apply_fraction: bool = False) -> int:
        # convert ODGT annotations to YOLO format and save as label files.
        logger.info(f"Creating labels: {annotation_file} -> {label_dir}")
        
        # ensure the label directory exists
        os.makedirs(label_dir, exist_ok=True)
        
        # check if the image directory exists
        if not os.path.exists(image_dir):
            logger.error(f"Image directory not found: {image_dir}")
            return 0
        
        # map (image ids --> file paths)
        image_id_to_path: Dict[str, str] = {}
        for file in os.listdir(image_dir):
            file_base, ext = os.path.splitext(file)
            if ext.lower() in self.image_extensions:
                image_id_to_path[file_base] = os.path.join(image_dir, file)
        
        # load annotations
        try:
            with open(annotation_file, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            logger.error(f"Annotation file not found: {annotation_file}")
            return 0
        
        logger.info(f"Total annotations in file: {len(lines)}")
        
        # apply sampling fraction if requested (regardless of train/val)
        if apply_fraction and self.config.dataset_fraction < 1.0:
            # use fixed seed for consistent sampling across datasets
            random.seed(42)
            random.shuffle(lines)
            sample_size = max(1, int(len(lines) * self.config.dataset_fraction))
            logger.info(f"Sampling {sample_size} annotations ({self.config.dataset_fraction:.2%} of {len(lines)})")
            lines = lines[:sample_size]  # take the first N after shuffling
        
        # process annotations
        processed_count = 0
        skipped_image_read = 0
        skipped_no_boxes = 0
        
        for line in tqdm(lines, desc=f"Processing {os.path.basename(annotation_file)}"):
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
                
            image_id = data.get('ID')
            if not image_id:
                continue
            
            # find the image file
            image_path = None
            if image_id in image_id_to_path:
                image_path = image_id_to_path[image_id]
            else:
                # try with different extensions
                for ext in self.image_extensions:
                    possible_path = os.path.join(image_dir, f"{image_id}{ext}")
                    if os.path.exists(possible_path):
                        image_path = possible_path
                        image_id_to_path[image_id] = possible_path
                        break
            
            if not image_path or not os.path.exists(image_path):
                continue
                
            label_path = os.path.join(label_dir, f"{image_id}.txt")
            
            # get image dimensions for normalization
            img_size = self.get_image_size(image_path)
            if img_size is None:
                skipped_image_read += 1
                continue
            img_w, img_h = img_size
            
            # process bounding boxes
            yolo_labels = []
            gtboxes = data.get('gtboxes', [])
            
            if not gtboxes:
                skipped_no_boxes += 1
                with open(label_path, 'w') as lf:
                    pass  # write empty file for images with no boxes
                continue
            
            for gtbox in gtboxes:
                tag = gtbox.get('tag')
                # only include 'person' tags for processing vbox and hbox
                if tag != 'person':
                    continue

                # skip ignored boxes
                extra = gtbox.get('extra', {})
                if extra.get('ignore', 0) == 1:
                    continue

                # process visible box (vbox) for class 'person' (ID 0)
                vbox = gtbox.get('vbox')
                if vbox and len(vbox) == 4:
                    try:
                        vbox_float = [float(c) for c in vbox]
                        if vbox_float[2] > 0 and vbox_float[3] > 0: # Check width/height > 0
                            # clamp to image bounds
                            x_v, y_v, w_v, h_v = vbox_float
                            x1_v = max(0.0, x_v)
                            y1_v = max(0.0, y_v)
                            x2_v = min(float(img_w), x_v + w_v)
                            y2_v = min(float(img_h), y_v + h_v)
                            clamped_w_v = x2_v - x1_v
                            clamped_h_v = y2_v - y1_v

                            if clamped_w_v > 0 and clamped_h_v > 0:
                                clamped_vbox = [x1_v, y1_v, clamped_w_v, clamped_h_v]
                                yolo_vbox = self.convert_to_yolo(clamped_vbox, img_w, img_h)
                                person_class_id = self.config.classes.get('person')
                                if person_class_id is not None:
                                    yolo_labels.append(f"{person_class_id} {' '.join(map(str, yolo_vbox))}")

                    except (ValueError, TypeError) as e:
                         logger.debug(f"Skipping vbox for image {image_id} due to processing error: {e}")

                # process head box (hbox) for class 'hbox' (ID 1)
                hbox = gtbox.get('hbox')
                if hbox and len(hbox) == 4:
                    try:
                        hbox_float = [float(c) for c in hbox]
                        if hbox_float[2] > 0 and hbox_float[3] > 0: # Check width/height > 0
                            # clamp to image bounds
                            x_h, y_h, w_h, h_h = hbox_float
                            x1_h = max(0.0, x_h)
                            y1_h = max(0.0, y_h)
                            x2_h = min(float(img_w), x_h + w_h)
                            y2_h = min(float(img_h), y_h + h_h)
                            clamped_w_h = x2_h - x1_h
                            clamped_h_h = y2_h - y1_h

                            if clamped_w_h > 0 and clamped_h_h > 0:
                                clamped_hbox = [x1_h, y1_h, clamped_w_h, clamped_h_h]
                                yolo_hbox = self.convert_to_yolo(clamped_hbox, img_w, img_h)
                                hbox_class_id = self.config.classes.get('hbox')
                                if hbox_class_id is not None:
                                     yolo_labels.append(f"{hbox_class_id} {' '.join(map(str, yolo_hbox))}")

                    except (ValueError, TypeError) as e:
                         logger.debug(f"Skipping hbox for image {image_id} due to processing error: {e}")

            # write labels if any were generated for this image
            if yolo_labels:
                # Only count as processed if we actually wrote non-empty labels
                with open(label_path, 'w') as lf:
                    lf.write("\n".join(yolo_labels))
                processed_count += 1
            elif os.path.exists(image_path): # If image exists but had no valid boxes
                 # Write empty file if no valid boxes found for this image ID,
                 # but only if the image file actually exists.
                 # This prevents creating labels for missing images.
                 with open(label_path, 'w') as lf:
                     pass # write empty file
                 skipped_no_boxes += 1 # Increment no boxes count
        
        logger.info(f"Finished processing {os.path.basename(annotation_file)}")
        logger.info(f"  Labels written: {processed_count}")
        logger.info(f"  Skipped (image read error): {skipped_image_read}")
        logger.info(f"  Skipped (no valid boxes): {skipped_no_boxes}")
        
        return processed_count
    
    def create_image_symlinks(self, src_dir: Path, dest_dir: Path, label_dir: Path) -> bool:
        # create symlinks from source to destination directory only for images with labels.
        logger.info(f"Creating symlinks: {src_dir} -> {dest_dir} (based on labels in {label_dir})")
        
        if not os.path.exists(src_dir) or not os.path.exists(label_dir):
            logger.error(f"Source or label directory not found")
            return False
        
        # get list of image IDs that have labels
        label_files = [os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith('.txt')]
        logger.info(f"Found {len(label_files)} label files to create symlinks for")
        
        if not label_files:
            logger.warning(f"No label files found in {label_dir}")
            return False
        
        # create destination directory
        os.makedirs(dest_dir, exist_ok=True)
        
        # clean existing files
        for item in os.listdir(dest_dir):
            item_path = os.path.join(dest_dir, item)
            if os.path.islink(item_path) or os.path.isfile(item_path):
                try:
                    if os.path.islink(item_path):
                        os.unlink(item_path)
                    else:
                        os.remove(item_path)
                except Exception as e:
                    logger.warning(f"Could not remove {item_path}: {e}")
        
        # create symlinks
        count = 0
        for image_id in label_files:
            found = False
            for ext in self.image_extensions:
                src_path = os.path.join(src_dir, f"{image_id}{ext}")
                if os.path.exists(src_path):
                    dest_path = os.path.join(dest_dir, f"{image_id}{ext}")
                    
                    try:
                        # try relative symlink first, then absolute
                        rel_path = os.path.relpath(src_path, os.path.dirname(dest_path))
                        os.symlink(rel_path, dest_path)
                        count += 1
                        found = True
                        break
                    except (OSError, ValueError):
                        try:
                            os.symlink(os.path.abspath(src_path), dest_path)
                            count += 1
                            found = True
                            break
                        except OSError as e:
                            logger.warning(f"Failed to create symlink for {image_id}: {e}")
            
            if not found:
                logger.debug(f"Could not find image for label {image_id}")
        
        logger.info(f"Created {count} symlinks out of {len(label_files)} labels")
        return count > 0
    
    def get_image_size(self, image_path: str) -> Optional[Tuple[int, int]]:
        # get image dimensions using OpenCV.
        try:
            import cv2
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            return img.shape[1], img.shape[0]  # width, height
        except Exception:
            return None
    
    def convert_to_yolo(self, box: List[float], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
        # convert [x, y, w, h] to YOLO format [x_center_norm, y_center_norm, width_norm, height_norm].
        x, y, w, h = box
        x_center = x + w / 2
        y_center = y + h / 2
        
        x_center_norm = x_center / img_w
        y_center_norm = y_center / img_h
        width_norm = w / img_w
        height_norm = h / img_h
        
        return x_center_norm, y_center_norm, width_norm, height_norm
    
    def create_yaml(self, yaml_path: Path, run_type: str) -> Dict[str, Any]:
        # create YAML configuration file for training.
        logger.info(f"Creating YAML configuration for {run_type}")
        
        if run_type == 'uncensored':
            train_dir = self.config.uncensored_images_dir
        elif run_type == 'censored':
            train_dir = self.config.censored_images_dir
        else:
            raise ValueError(f"Invalid run_type: {run_type}")
        
        # get absolute paths
        train_images_path = os.path.abspath(train_dir)
        val_images_path = os.path.abspath(self.config.val_images_dir)
        
        # verify paths
        logger.debug(f"Train images path: {train_images_path} (exists: {os.path.exists(train_images_path)})")
        logger.debug(f"Val images path: {val_images_path} (exists: {os.path.exists(val_images_path)})")
        
        # create configuration
        data_config = {
            'train': train_images_path,
            'val': val_images_path,
            'nc': len(self.config.classes),
            'names': {i: name for i, name in enumerate(self.config.classes.keys())}
        }
        
        # write configuration
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"YAML configuration written to {yaml_path}")
        return data_config

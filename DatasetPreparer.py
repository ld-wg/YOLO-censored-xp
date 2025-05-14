#!/usr/bin/env python3
# DatasetPreparer module for CrowdHuman Dataset Trainer
# Handles dataset preparation, annotation conversion and symlink creation

import os
import json
import yaml
import random
import shutil
import logging
import math
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from Config import Config

logger = logging.getLogger('crowdhuman_trainer')

class DatasetPreparer:
    # dataset preparation, including annotation conversion and symlink creation
    
    def __init__(self, config: Config) -> None:
        self.config: Config = config
        self.image_extensions: Set[str] = {'.jpg', '.jpeg', '.png'}
        
    def clean_dataset_directory(self) -> None:
        # clean existing dataset directory to ensure a clean state
        if os.path.exists(self.config.yolo_dataset_dir):
            logger.info(f"Removing existing dataset directory: {self.config.yolo_dataset_dir}")
            try:
                shutil.rmtree(self.config.yolo_dataset_dir)
                logger.info("Previous dataset directory removed successfully")
            except Exception as e:
                logger.warning(f"Could not remove directory: {e}")
                
        # create necessary directories for all data variants
        dirs_to_create = [
            # Uncensored
            self.config.uncensored_train_images, self.config.uncensored_train_labels,
            self.config.uncensored_val_images, self.config.uncensored_val_labels,
            self.config.uncensored_test_images, self.config.uncensored_test_labels,
            # Censored-blur
            self.config.censored_blur_train_images, self.config.censored_blur_train_labels,
            self.config.censored_blur_val_images, self.config.censored_blur_val_labels,
            # Censored-bbox
            self.config.censored_bbox_train_images, self.config.censored_bbox_train_labels,
            self.config.censored_bbox_val_images, self.config.censored_bbox_val_labels
        ]
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
            logger.debug(f"Created directory: {dir_path}")
            
        # Verify that source image directories exist
        for img_dir in [self.config.image_uncensored, self.config.image_censored_blur, self.config.image_censored_bbox]:
            if not os.path.exists(img_dir):
                logger.error(f"Source image directory does not exist: {img_dir}")
                logger.error(f"Make sure that the base_data_path '{self.config.base_data_path}' contains subdirectories:")
                logger.error(f"  - uncensored/")
                logger.error(f"  - censored-blur/")
                logger.error(f"  - censored-bbox/")
    
    def validate_dataset_pairings(self, image_dir: Path, label_dir: Path) -> bool:
        # validate image-label pairings
        valid = True
        missing_labels = []
        missing_images = []
        
        # Check images -> labels
        for img_file in os.listdir(image_dir):
            if any(img_file.lower().endswith(ext) for ext in self.image_extensions):
                base_name = os.path.splitext(img_file)[0]
                label_file = os.path.join(label_dir, f"{base_name}.txt")
                if not os.path.exists(label_file):
                    missing_labels.append(img_file)
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
                    missing_images.append(label_file)
                    valid = False
        
        # Log summary instead of individual warnings
        if missing_labels:
            logger.warning(f"Found {len(missing_labels)} images without labels in {image_dir}")
            logger.debug(f"Images without labels: {', '.join(missing_labels)}")
            
        if missing_images:
            logger.warning(f"Found {len(missing_images)} labels without images in {label_dir}")
            logger.debug(f"Labels without images: {', '.join(missing_images)}")
            
        if valid:
            logger.info(f"All images in {image_dir} have corresponding labels")
            
        return valid
    
    def process_dataset(self) -> Dict[str, int]:
        # process the entire dataset, creating labels and symlinks
        logger.info("Starting dataset preparation")
        
        # clean the dataset directory first
        self.clean_dataset_directory()
        
        # read and sample annotations
        try:
            with open(self.config.annotation_file, 'r') as f:
                all_lines = f.readlines()
            
            total_annotations = len(all_lines)
            logger.info(f"Total annotations in source file: {total_annotations}")
            
            if self.config.dataset_fraction < 1.0:
                random.seed(42)  # ensure consistent sampling
                random.shuffle(all_lines)
                sample_size = max(1, int(total_annotations * self.config.dataset_fraction))
                all_lines = all_lines[:sample_size]
                logger.info(f"Sampled {len(all_lines)} annotations ({self.config.dataset_fraction:.2%})")
            
            # split data into train/val/test
            random.seed(42)  # ensure consistent splits
            random.shuffle(all_lines)
            
            train_size = int(len(all_lines) * self.config.train_ratio)
            val_size = int(len(all_lines) * self.config.val_ratio)
            
            train_lines = all_lines[:train_size]
            val_lines = all_lines[train_size:train_size + val_size]
            test_lines = all_lines[train_size + val_size:]
            
            logger.info(f"Split sizes: train={len(train_lines)}, val={len(val_lines)}, test={len(test_lines)}")
            
            # process each split
            train_count = self._create_labels_from_lines(
                train_lines,
                self.config.image_uncensored,
                self.config.uncensored_train_labels,
                "train"
            )
            
            val_count = self._create_labels_from_lines(
                val_lines,
                self.config.image_uncensored,
                self.config.uncensored_val_labels,
                "validation"
            )
            
            test_count = self._create_labels_from_lines(
                test_lines,
                self.config.image_uncensored,
                self.config.uncensored_test_labels,
                "test"
            )
            
            # create symlinks for all variants
            variants = [
                (self.config.image_uncensored, 'uncensored'),
                (self.config.image_censored_blur, 'censored-blur'),
                (self.config.image_censored_bbox, 'censored-bbox')
            ]
            
            for src_dir, variant_name in variants:
                # Skip if source directory doesn't exist
                if not os.path.exists(src_dir):
                    logger.warning(f"Source directory not found: {src_dir}")
                    continue
                    
                # Create symlinks for train and val
                for split in ['train', 'val']:
                    target_dir = getattr(self.config, f"{variant_name.replace('-', '_')}_{split}_images")
                    label_dir = getattr(self.config, f"uncensored_{split}_labels")  # always use uncensored labels
                    
                    success = self.create_image_symlinks(src_dir, target_dir, label_dir)
                    if not success:
                        logger.error(f"Failed to create symlinks for {variant_name} {split} set")
                        return {'train_count': 0, 'val_count': 0, 'test_count': 0}
                
                # Only create test symlinks for uncensored variant
                if variant_name == 'uncensored':
                    success = self.create_image_symlinks(
                        src_dir,
                        self.config.uncensored_test_images,
                        self.config.uncensored_test_labels
                    )
                    if not success:
                        logger.error("Failed to create symlinks for test set")
                        return {'train_count': 0, 'val_count': 0, 'test_count': 0}
            
            # validate dataset pairings
            logger.info("Validating final dataset pairings...")
            valid = True
            
            # Validate all splits for each variant
            for variant_name in ['uncensored', 'censored_blur', 'censored_bbox']:
                for split in ['train', 'val']:
                    img_dir = getattr(self.config, f"{variant_name}_{split}_images")
                    label_dir = getattr(self.config, f"uncensored_{split}_labels")  # always use uncensored labels
                    valid &= self.validate_dataset_pairings(img_dir, label_dir)
                    
            # Validate test set (uncensored only)
            valid &= self.validate_dataset_pairings(
                self.config.uncensored_test_images,
                self.config.uncensored_test_labels
            )
            
            if not valid:
                logger.error("Dataset validation failed - mismatches between images and labels found")
                return {'train_count': 0, 'val_count': 0, 'test_count': 0}
            
            logger.info("Dataset Processing Summary:")
            logger.info(f"  Training samples: {train_count}")
            logger.info(f"  Validation samples: {val_count}")
            logger.info(f"  Test samples: {test_count}")
            
            return {
                'train_count': train_count,
                'val_count': val_count,
                'test_count': test_count
            }
            
        except FileNotFoundError:
            logger.error(f"Annotation file not found: {self.config.annotation_file}")
            return {'train_count': 0, 'val_count': 0, 'test_count': 0}
    
    def _create_labels_from_lines(self, annotation_lines: List[str], image_dir: Path, label_dir: Path, split_name: str) -> int:
        # convert ODGT annotations to YOLO format and save as label files
        logger.info(f"Creating labels for {split_name}: {len(annotation_lines)} annotations -> {label_dir}")
        
        os.makedirs(label_dir, exist_ok=True)
        if not os.path.exists(image_dir):
            logger.error(f"Image directory not found: {image_dir}")
            return 0
        
        image_id_to_path: Dict[str, str] = {}
        for file in os.listdir(image_dir):
            file_base, ext = os.path.splitext(file)
            if ext.lower() in self.image_extensions:
                image_id_to_path[file_base] = os.path.join(image_dir, file)
        
        processed_count = 0
        skipped_image_read = 0
        skipped_no_boxes = 0
        min_pixel_dim = 2
        
        for line in tqdm(annotation_lines, desc=f"Processing {split_name} annotations"):
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON line in {split_name}")
                continue
                
            image_id = data.get('ID')
            if not image_id:
                logger.warning(f"Skipping annotation line with no ID in {split_name}")
                continue
            
            image_path_str = None
            if image_id in image_id_to_path:
                image_path_str = image_id_to_path[image_id]
            else:
                for ext_ in self.image_extensions:
                    possible_path = os.path.join(image_dir, f"{image_id}{ext_}")
                    if os.path.exists(possible_path):
                        image_path_str = str(possible_path)
                        image_id_to_path[image_id] = image_path_str
                        break
            
            if not image_path_str or not os.path.exists(image_path_str):
                logger.debug(f"Image ID {image_id} not found in {image_dir}")
                continue
                
            label_path = os.path.join(label_dir, f"{image_id}.txt")
            
            img_size = self.get_image_size(image_path_str)
            if img_size is None:
                logger.warning(f"Could not read image size for {image_path_str}")
                skipped_image_read += 1
                continue
            img_w, img_h = img_size
            
            yolo_labels = []
            gtboxes = data.get('gtboxes', [])
            
            if not gtboxes:
                logger.debug(f"Image {image_id} had no gtboxes in annotation")
                skipped_no_boxes += 1
                continue

            valid_boxes_for_image = False
            for gtbox in gtboxes:
                tag = gtbox.get('tag')
                if tag != 'person': 
                    continue

                extra = gtbox.get('extra', {})
                if extra.get('ignore', 0) == 1:
                    continue
                
                # process visible box (vbox) - person detection
                vbox = gtbox.get('vbox')
                if vbox and len(vbox) == 4:
                    try:
                        vbox_float = [float(c) for c in vbox]
                        if vbox_float[2] > 0 and vbox_float[3] > 0:
                            x_v, y_v, w_v, h_v = vbox_float
                            x1_v = max(0.0, x_v)
                            y1_v = max(0.0, y_v)
                            x2_v = min(float(img_w), x_v + w_v)
                            y2_v = min(float(img_h), y_v + h_v)
                            clamped_w_v = x2_v - x1_v
                            clamped_h_v = y2_v - y1_v
                            
                            if clamped_w_v >= min_pixel_dim and clamped_h_v >= min_pixel_dim:
                                clamped_vbox = [x1_v, y1_v, clamped_w_v, clamped_h_v]
                                yolo_vbox = self.convert_to_yolo(clamped_vbox, img_w, img_h)
                                
                                if any(math.isnan(c) or math.isinf(c) for c in yolo_vbox):
                                    logger.debug(f"Skipping vbox for {image_id} due to NaN/Inf")
                                else:
                                    person_class_id = self.config.classes.get('person')
                                    if person_class_id is not None:
                                        yolo_labels.append(f"{person_class_id} {' '.join(map(str, yolo_vbox))}")
                                        valid_boxes_for_image = True
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Skipping vbox for {image_id} due to: {e}")
                
                # process head box (hbox) - head detection
                hbox = gtbox.get('hbox')
                if hbox and len(hbox) == 4:
                    try:
                        hbox_float = [float(c) for c in hbox]
                        if hbox_float[2] > 0 and hbox_float[3] > 0:
                            x_h, y_h, w_h, h_h = hbox_float
                            x1_h = max(0.0, x_h)
                            y1_h = max(0.0, y_h)
                            x2_h = min(float(img_w), x_h + w_h)
                            y2_h = min(float(img_h), y_h + h_h)
                            clamped_w_h = x2_h - x1_h
                            clamped_h_h = y2_h - y1_h
                            
                            if clamped_w_h >= min_pixel_dim and clamped_h_h >= min_pixel_dim:
                                clamped_hbox = [x1_h, y1_h, clamped_w_h, clamped_h_h]
                                yolo_hbox = self.convert_to_yolo(clamped_hbox, img_w, img_h)
                                
                                if any(math.isnan(c) or math.isinf(c) for c in yolo_hbox):
                                    logger.debug(f"Skipping hbox for {image_id} due to NaN/Inf")
                                else:
                                    hbox_class_id = self.config.classes.get('hbox')
                                    if hbox_class_id is not None:
                                        yolo_labels.append(f"{hbox_class_id} {' '.join(map(str, yolo_hbox))}")
                                        valid_boxes_for_image = True
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Skipping hbox for {image_id} due to: {e}")
            
            if yolo_labels and valid_boxes_for_image:
                with open(label_path, 'w') as lf:
                    lf.write("\n".join(yolo_labels))
                processed_count += 1
            else: 
                if os.path.exists(image_path_str):
                    if not gtboxes: 
                        pass 
                    elif gtboxes and not valid_boxes_for_image: 
                        logger.debug(f"Image {image_id} had gtboxes, but none were valid after processing")
                        skipped_no_boxes += 1 

        logger.info(f"Finished processing {split_name} annotations:")
        logger.info(f"  Labels written: {processed_count}")
        logger.info(f"  Skipped (image read error): {skipped_image_read}")
        logger.info(f"  Skipped (no valid boxes): {skipped_no_boxes}")
        return processed_count

    def create_image_symlinks(self, src_dir: Path, dest_dir: Path, label_dir: Path) -> bool:
        # create symlinks from source to destination directory only for images with labels
        logger.info(f"Creating symlinks: {src_dir} -> {dest_dir} (based on labels in {label_dir})")
        
        if not os.path.exists(src_dir):
            logger.error(f"Source directory not found: {src_dir}")
            return False
        
        if not os.path.exists(label_dir):
            logger.error(f"Label directory not found: {label_dir}")
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
        missing_files = []
        
        for image_id in label_files:
            found = False
            for ext in self.image_extensions:
                src_path = os.path.join(src_dir, f"{image_id}{ext}")
                if os.path.exists(src_path):
                    dest_path = os.path.join(dest_dir, f"{image_id}{ext}")
                    
                    try:
                        # try relative symlink first
                        rel_path = os.path.relpath(src_path, os.path.dirname(dest_path))
                        os.symlink(rel_path, dest_path)
                        count += 1
                        found = True
                        break
                    except (OSError, ValueError) as e:
                        try:
                            # try absolute path if relative fails
                            os.symlink(os.path.abspath(src_path), dest_path)
                            count += 1
                            found = True
                            break
                        except OSError as e:
                            logger.warning(f"Failed to create symlink for {image_id}: {e}")
            
            if not found:
                missing_files.append(image_id)
        
        if missing_files:
            logger.warning(f"Could not find {len(missing_files)} images in source directory")
            if len(missing_files) <= 10:  # Only show a few to avoid log spam
                logger.warning(f"Missing image IDs: {', '.join(missing_files[:10])}")
            else:
                logger.warning(f"First 10 missing image IDs: {', '.join(missing_files[:10])}...")
        
        success_rate = count / len(label_files) if label_files else 0
        logger.info(f"Created {count} symlinks out of {len(label_files)} labels ({success_rate:.1%})")
        
        # Also copy the label files to ensure they're available in the corresponding directory
        # This ensures censored variants have their own copies of the label files
        dest_label_dir = Path(str(dest_dir).replace('/images/', '/labels/'))
        os.makedirs(dest_label_dir, exist_ok=True)
        
        # Only copy if source and destination are different
        if str(label_dir) != str(dest_label_dir):
            logger.info(f"Copying label files to {dest_label_dir}")
            for label_file in os.listdir(label_dir):
                if label_file.endswith('.txt'):
                    src_label = os.path.join(label_dir, label_file)
                    dest_label = os.path.join(dest_label_dir, label_file)
                    try:
                        shutil.copy2(src_label, dest_label)
                    except Exception as e:
                        logger.warning(f"Failed to copy label file {label_file}: {e}")
        
        return count > 0
    
    def get_image_size(self, image_path: str) -> Optional[Tuple[int, int]]:
        # get image dimensions using OpenCV
        try:
            import cv2
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            return img.shape[1], img.shape[0]  # width, height
        except Exception:
            return None
    
    def convert_to_yolo(self, box: List[float], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
        # convert [x, y, w, h] to YOLO format [x_center_norm, y_center_norm, width_norm, height_norm]
        x, y, w, h = box
        x_center = x + w / 2
        y_center = y + h / 2
        
        x_center_norm = x_center / img_w
        y_center_norm = y_center / img_h
        width_norm = w / img_w
        height_norm = h / img_h
        
        return x_center_norm, y_center_norm, width_norm, height_norm

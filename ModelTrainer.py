#!/usr/bin/env python3
# ModelTrainer module for CrowdHuman Dataset Trainer
# Handles model training for uncensored and censored datasets

import os
import logging
import traceback
import torch
from pathlib import Path
from ultralytics import YOLO
from typing import Dict, Any, Literal

from Config import Config

logger = logging.getLogger('crowdhuman_trainer')

class ModelTrainer:
    # model training for uncensored and censored datasets
    
    def __init__(self, config: Config) -> None:
        # initialize trainer with configuration
        self.config: Config = config
        self.device: str = self.get_device()
    
    def get_device(self) -> str:
        # get the best available device for training
        if torch.backends.mps.is_available():
            logger.info("MPS (Apple Silicon GPU) is available! Using GPU acceleration.")
            torch.mps.empty_cache()
            return "mps"
        elif torch.cuda.is_available():
            logger.info("CUDA GPU is available! Using GPU acceleration.")
            torch.cuda.empty_cache()
            return "cuda"
        else:
            logger.info("No GPU available. Using CPU.")
            return "cpu"
    
    def train_model(
        self, 
        train_data: Literal['uncensored', 'censored-bbox', 'censored-blur'],
        val_data: Literal['uncensored', 'censored-bbox', 'censored-blur'],
        test_data: Literal['uncensored'] = 'uncensored'
    ) -> Dict[str, Any]:
        """Train a single model with specified training, validation and test data."""
        logger.info(f"Starting model training on device: {self.device}")
        logger.info(f"Training data: {train_data}")
        logger.info(f"Validation data: {val_data}")
        logger.info(f"Test data: {test_data}")
        
        # create YAML configuration
        yaml_path = Path(f"data_{train_data}.yaml").resolve()
        self.create_yaml(yaml_path, train_data, val_data, test_data)
        
        # initialize model
        experiment_name = self.config.get_experiment_name(train_data)
        logger.info(f"Training model: {experiment_name}")
        
        model = YOLO(self.config.model_name)
        
        try:
            # Clear memory before training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
                
            # Train model
            train_results = model.train(
                data=str(yaml_path),
                epochs=self.config.epochs,
                imgsz=self.config.img_size,
                patience=self.config.patience,
                batch=self.config.batch_size,
                workers=0,
                project="runs/train",
                name=experiment_name,
                exist_ok=True,
                device=self.device,
                amp=False,
                mosaic=0.0,  # disable mosaic augmentation
                mixup=0.0,   # disable mixup augmentation
                copy_paste=0.0,  # disable copy-paste augmentation
                fliplr=0.0,  # disable horizontal flips
                flipud=0.0,  # disable vertical flips
                degrees=0.0, # disable rotation
                translate=0.0,  # disable translation
                scale=0.0,   # disable scaling
                shear=0.0,   # disable shearing
                perspective=0.0,  # disable perspective transforms
                rect=False   # disable rectangular training
            )
            
            model_path = Path("runs/train") / experiment_name / "weights/best.pt"
            test_map50 = 0.0
            
            if os.path.exists(model_path):
                logger.info(f"Model saved at: {model_path}")
                # Evaluate on the test set using uncensored data
                logger.info(f"Evaluating model on uncensored test set...")
                try:
                    # Create test YAML with uncensored test data
                    test_yaml_path = Path(f"data_test_{train_data}.yaml").resolve()
                    self.create_yaml(test_yaml_path, train_data, val_data, test_data, test_only=True)
                    
                    test_results = model.val(
                        data=str(test_yaml_path),
                        split='test',
                        project="runs/test",
                        name=f"{experiment_name}_test"
                    )
                    test_map50 = test_results.box.map50
                    logger.info(f"Test set evaluation complete. mAP50: {test_map50:.3f}")
                except Exception as e_test:
                    logger.error(f"Error during test set evaluation: {e_test}")
                    traceback.print_exc()
            else:
                logger.warning(f"Expected model file not found at {model_path}")
                
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            traceback.print_exc()
            return {'model_path': None, 'test_map50': 0.0}
        
        return {
            'model_path': str(model_path),
            'test_map50': test_map50
        }
    
    def create_yaml(
        self, 
        yaml_path: Path,
        train_data: str,
        val_data: str,
        test_data: str = 'uncensored',
        test_only: bool = False
    ) -> None:
        """Create YAML configuration file for training or testing."""
        import yaml
        
        def get_image_path(data_type: str, split: str) -> str:
            if data_type == 'uncensored':
                if split == 'train':
                    return str(self.config.uncensored_train_images)
                elif split == 'val':
                    return str(self.config.uncensored_val_images)
                else:  # test
                    return str(self.config.uncensored_test_images)
            elif data_type == 'censored-bbox':
                if split == 'train':
                    return str(self.config.censored_bbox_train_images)
                elif split == 'val':
                    return str(self.config.censored_bbox_val_images)
                else:  # test - use uncensored test set
                    return str(self.config.uncensored_test_images)
            else:  # censored-blur
                if split == 'train':
                    return str(self.config.censored_blur_train_images)
                elif split == 'val':
                    return str(self.config.censored_blur_val_images)
                else:  # test - use uncensored test set
                    return str(self.config.uncensored_test_images)
        
        # create configuration - YOLOv8 requires train and val keys even for test-only configs
        data_config = {
            'train': get_image_path(train_data, 'train'),
            'val': get_image_path(val_data, 'val'),
            'test': get_image_path(test_data, 'test'),
            'nc': len(self.config.classes),
            'names': {i: name for i, name in enumerate(self.config.classes.keys())}
        }
        
        # write configuration
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"YAML configuration written to {yaml_path}") 
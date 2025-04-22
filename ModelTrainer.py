#!/usr/bin/env python3
# ModelTrainer module for CrowdHuman Dataset Trainer
# Handles model training for both censored and uncensored datasets

import os
import logging
import traceback
import torch
from pathlib import Path
from ultralytics import YOLO
from typing import Dict, Any

from Config import Config
from DatasetPreparer import DatasetPreparer

logger = logging.getLogger('crowdhuman_trainer')

class ModelTrainer:
    # model training for both censored and uncensored datasets
    
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
    
    def train_models(self, dataset_preparer: DatasetPreparer) -> Dict[str, str]:
        # train both censored and uncensored models
        logger.info(f"Starting model training on device: {self.device}")
        
        # create YAML configurations
        yaml_uncensored = Path("data_uncensored.yaml").resolve()
        yaml_censored = Path("data_censored.yaml").resolve()
        
        dataset_preparer.create_yaml(yaml_uncensored, 'uncensored')
        dataset_preparer.create_yaml(yaml_censored, 'censored')
        
        # train uncensored model
        uncensored_experiment = self.config.get_experiment_name('uncensored')
        logger.info(f"Training uncensored model: {uncensored_experiment}")
        
        model_uncensored = YOLO(self.config.model_name)
        
        try:
            # Clear memory before training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
                
            uncensored_results = model_uncensored.train(
                data=str(yaml_uncensored),
                epochs=self.config.epochs,
                imgsz=self.config.img_size,
                patience=self.config.patience,
                batch=self.config.batch_size,
                workers=0,
                project="runs/train",
                name=uncensored_experiment,
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
            
            uncensored_model_path = Path("runs/train") / uncensored_experiment / "weights/best.pt"
            if os.path.exists(uncensored_model_path):
                logger.info(f"Uncensored model saved at: {uncensored_model_path}")
            else:
                logger.warning(f"Expected model file not found at {uncensored_model_path}")
                
        except Exception as e:
            logger.error(f"Error during uncensored model training: {e}")
            traceback.print_exc()
        
        # Clear memory between training runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
            
        # train censored model
        censored_experiment = self.config.get_experiment_name('censored')
        logger.info(f"Training censored model: {censored_experiment}")
        
        model_censored = YOLO(self.config.model_name)
        
        try:
            censored_results = model_censored.train(
                data=str(yaml_censored),
                epochs=self.config.epochs,
                imgsz=self.config.img_size,
                patience=self.config.patience,
                batch=self.config.batch_size,
                workers=0,
                project="runs/train",
                name=censored_experiment,
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
            
            censored_model_path = Path("runs/train") / censored_experiment / "weights/best.pt"
            if os.path.exists(censored_model_path):
                logger.info(f"Censored model saved at: {censored_model_path}")
            else:
                logger.warning(f"Expected model file not found at {censored_model_path}")
                
        except Exception as e:
            logger.error(f"Error during censored model training: {e}")
            traceback.print_exc()
        
        logger.info("Training complete")
        return {
            'uncensored_path': f"runs/train/{uncensored_experiment}",
            'censored_path': f"runs/train/{censored_experiment}"
        } 
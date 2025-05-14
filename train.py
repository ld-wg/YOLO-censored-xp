#!/usr/bin/env python3
"""
CrowdHuman Dataset Trainer for Detection Models

This script trains YOLOv8 models on three variants of the CrowdHuman dataset:
1. Uncensored (original images)
2. Censored with black boxes
3. Censored with Gaussian blur

All models are trained using the same train/val split and evaluated on uncensored test data.
"""

import os
import sys
import logging
from pathlib import Path
from Config import Config, parse_arguments
from DatasetPreparer import DatasetPreparer
from ModelTrainer import ModelTrainer


# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('crowdhuman_trainer')


def main():
    """Main function to run the training pipeline."""
    # Parse arguments
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Initialize configuration
    config = Config(args)
    logger.info(f"Starting CrowdHuman training with configuration:\n{config}")
    
    # Verify source directories exist
    required_dirs = [
        (config.base_data_path, "Base data path"),
        (config.image_uncensored, "Uncensored images"),
        (config.image_censored_blur, "Blur-censored images"),
        (config.image_censored_bbox, "Bbox-censored images"),
        (config.annotation_file, "Annotation file (ODGT)")
    ]
    
    missing = False
    for path, desc in required_dirs:
        if not os.path.exists(path):
            logger.error(f"Required {desc} not found at: {path}")
            missing = True
    
    if missing:
        logger.error("Required directories or files are missing. Please ensure all paths exist.")
        logger.error(f"Expected directory structure under {config.base_data_path}:")
        logger.error("  - uncensored/")
        logger.error("  - censored-blur/")
        logger.error("  - censored-bbox/")
        logger.error("  - annotation.odgt")
        sys.exit(1)
    
    # Prepare dataset
    dataset_preparer = DatasetPreparer(config)
    dataset_stats = dataset_preparer.process_dataset()
    
    # Check if dataset preparation was successful
    if all(count > 0 for count in dataset_stats.values()):
        # Train models
        trainer = ModelTrainer(config)
        
        # Train and evaluate uncensored model
        logger.info("\n=== Training Uncensored Model ===")
        uncensored_results = trainer.train_model(
            train_data='uncensored',
            val_data='uncensored',
            test_data='uncensored'
        )
        
        # Train and evaluate bbox-censored model
        logger.info("\n=== Training Bbox-Censored Model ===")
        bbox_results = trainer.train_model(
            train_data='censored-bbox',
            val_data='censored-bbox',
            test_data='uncensored'  # always test on uncensored
        )
        
        # Train and evaluate blur-censored model
        logger.info("\n=== Training Blur-Censored Model ===")
        blur_results = trainer.train_model(
            train_data='censored-blur',
            val_data='censored-blur',
            test_data='uncensored'  # always test on uncensored
        )
        
        logger.info("\n=== Training Pipeline Complete ===")
        logger.info("Model Paths:")
        logger.info(f"  Uncensored: {uncensored_results.get('model_path', 'Failed')}")
        logger.info(f"  Bbox-Censored: {bbox_results.get('model_path', 'Failed')}")
        logger.info(f"  Blur-Censored: {blur_results.get('model_path', 'Failed')}")
        
        logger.info("\nTest Results (on uncensored data):")
        logger.info(f"  Uncensored Model mAP50: {uncensored_results.get('test_map50', 0.0):.3f}")
        logger.info(f"  Bbox-Censored Model mAP50: {bbox_results.get('test_map50', 0.0):.3f}")
        logger.info(f"  Blur-Censored Model mAP50: {blur_results.get('test_map50', 0.0):.3f}")
        
    else:
        logger.error("Dataset preparation failed. Please check the errors above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
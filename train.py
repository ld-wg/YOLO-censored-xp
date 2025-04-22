#!/usr/bin/env python3
"""
CrowdHuman Dataset Trainer for Detection Models

This script trains YOLOv8 models on censored and uncensored versions of the CrowdHuman dataset.
It handles dataset preparation, annotation conversion, and model training with configurable parameters.
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
    
    # Prepare dataset
    dataset_preparer = DatasetPreparer(config)
    dataset_stats = dataset_preparer.process_dataset()
    
    # Check if dataset preparation was successful
    if all(count > 0 for count in dataset_stats.values()):
        # Train models
        trainer = ModelTrainer(config)
        training_results = trainer.train_models(dataset_preparer)
        
        logger.info("Training pipeline completed successfully")
        logger.info(f"Model paths:\n  Uncensored: {training_results['uncensored_path']}\n  Censored: {training_results['censored_path']}")
    else:
        logger.error("Dataset preparation failed. Please check the errors above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
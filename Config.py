import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any, List, Union, Tuple
import torch

logger = logging.getLogger('crowdhuman_trainer')


# command line arguments.
class Args:
    
    def __init__(self) -> None:
        # initialize empty argument holder
        self.data_path: Optional[str] = None
        self.fraction: Optional[float] = None
        self.epochs: Optional[int] = None
        self.batch_size: Optional[int] = None
        self.img_size: Optional[int] = None
        self.model: Optional[str] = None
        self.workers: Optional[int] = None
        self.device: Optional[str] = None
        self.verbose: Optional[bool] = None


class Config:
    # configuration parameters for the YOLOv8 training pipeline
    # dataset paths, training hyperparameters, and model settings.
    
    def __init__(self, args: Optional[Args] = None) -> None:
        # base paths
        self.base_data_path: Path = Path('./crowdface').resolve()
        
        # annotation files
        self.annotation_train: Path = self.base_data_path / 'annotation_train.odgt'
        self.annotation_val: Path = self.base_data_path / 'annotation_val.odgt'
        
        # image directories
        self.image_uncensored: Path = self.base_data_path / 'train_uncensored'
        self.image_censored: Path = self.base_data_path / 'train_censored'
        self.image_val: Path = self.base_data_path / 'validation'
        self.image_test: Path = self.base_data_path / 'test'
        
        # YOLO dataset directory (for prepared data)
        self.yolo_dataset_dir: Path = Path('./crowdface_yolo').resolve()
        
        # dataset fraction for sampling
        self.dataset_fraction: float = 0.01  # 1/100 of the dataset
        
        # classes
        self.classes: Dict[str, int] = {'person': 0, 'hbox': 1}
        
        # training parameters
        self.epochs: int = 10
        self.img_size: int = 640
        self.batch_size: int = 8
        self.workers: int = 2
        self.patience: int = 4
        self.model_name: str = 'yolov8n.pt'  # smallest YOLOv8 model
        
        # paths for prepared data - will be set in _setup_prepared_paths()
        self.uncensored_images_dir: Path
        self.uncensored_labels_dir: Path
        self.censored_images_dir: Path
        self.censored_labels_dir: Path
        self.val_images_dir: Path
        self.val_labels_dir: Path
        
        # setup the prepared paths
        self._setup_prepared_paths()
        
        # override with command line args if provided
        if args:
            self.update_from_args(args)
            
    # set up paths for prepared data based on configuration
    def _setup_prepared_paths(self) -> None:
        # Uncensored data
        self.uncensored_images_dir = self.yolo_dataset_dir / 'uncensored' / 'images' / 'train'
        self.uncensored_labels_dir = self.yolo_dataset_dir / 'uncensored' / 'labels' / 'train'
        
        # Censored data
        self.censored_images_dir = self.yolo_dataset_dir / 'censored' / 'images' / 'train'
        self.censored_labels_dir = self.yolo_dataset_dir / 'censored' / 'labels' / 'train'
        
        # Validation data (simplified)
        self.val_images_dir = self.yolo_dataset_dir / 'validation' / 'images' / 'val'
        self.val_labels_dir = self.yolo_dataset_dir / 'validation' / 'labels' / 'val'
    
    # update configuration from command line arguments
    def update_from_args(self, args: Args) -> None:
        if args.data_path is not None:
            self.base_data_path = Path(args.data_path).resolve()
            # update annotation files and image directories
            self.annotation_train = self.base_data_path / 'annotation_train.odgt'
            self.annotation_val = self.base_data_path / 'annotation_val.odgt'
            self.image_uncensored = self.base_data_path / 'train_uncensored'
            self.image_censored = self.base_data_path / 'train_censored'
            self.image_val = self.base_data_path / 'validation'
            self.image_test = self.base_data_path / 'test'
        
        if args.fraction is not None:
            self.dataset_fraction = args.fraction
        
        if args.epochs is not None:
            self.epochs = args.epochs
            
        if args.batch_size is not None:
            self.batch_size = args.batch_size
            
        if args.img_size is not None:
            self.img_size = args.img_size
            
        if args.model is not None:
            self.model_name = args.model
            
        if args.workers is not None:
            self.workers = args.workers
        
        # re-setup prepared paths in case base_data_path changed
        self._setup_prepared_paths()
    
    # generate experiment name for the specified run type (censored or uncensored)
    def get_experiment_name(self, run_type: str) -> str:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fraction_str = f"{self.dataset_fraction:.2f}".replace('.', '_')
        return f"{run_type}_frac_{fraction_str}_{self.epochs}ep_{timestamp}"
    
    # string representation of the configuration
    def __str__(self) -> str:
        return (
            f"CrowdHuman Training Configuration:\n"
            f"  Dataset Fraction: {self.dataset_fraction} ({self.dataset_fraction*100:.1f}%)\n"
            f"  Base Path: {self.base_data_path}\n"
            f"  Classes: {list(self.classes.keys())}\n"
            f"  Epochs: {self.epochs}\n"
            f"  Batch Size: {self.batch_size}\n"
            f"  Image Size: {self.img_size}\n"
            f"  Model: {self.model_name}"
        )


# parse command line arguments and return Args object
def parse_arguments() -> Args:
    parser = argparse.ArgumentParser(description='Train YOLOv8 on CrowdHuman dataset with censored/uncensored faces')
    
    # define arguments without defaults - Config will apply defaults
    parser.add_argument('--data-path', type=str, help='Base path to CrowdHuman dataset')
    parser.add_argument('--fraction', type=float, help='Fraction of dataset to use (0-1)')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size for training')
    parser.add_argument('--img-size', type=int, help='Image size for training')
    parser.add_argument('--model', type=str, help='YOLOv8 model to use')
    parser.add_argument('--workers', type=int, help='Number of worker threads')
    parser.add_argument('--device', type=str, help='Device to use (cuda, mps, cpu)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    # parse arguments
    parsed_args = parser.parse_args()
    args = Args()
    
    # transfer provided values from parsed_args to our Args class
    args.data_path = parsed_args.data_path
    args.fraction = parsed_args.fraction
    args.epochs = parsed_args.epochs
    args.batch_size = parsed_args.batch_size
    args.img_size = parsed_args.img_size
    args.model = parsed_args.model
    args.workers = parsed_args.workers
    args.device = parsed_args.device
    args.verbose = parsed_args.verbose
    
    return args 

if torch.backends.mps.is_available():
    torch.mps.empty_cache() 
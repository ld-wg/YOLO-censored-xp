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
    # configuration for YOLOv8 training pipeline with new directory structure
    
    def __init__(self, args: Optional[Args] = None) -> None:
        # base paths
        self.base_data_path: Path = Path('./crowdhuman').resolve()
        
        # single annotation file
        self.annotation_file: Path = self.base_data_path / 'annotation.odgt'
        
        # image directories for different versions
        self.image_uncensored: Path = self.base_data_path / 'uncensored'
        self.image_censored_blur: Path = self.base_data_path / 'censored-blur'
        self.image_censored_bbox: Path = self.base_data_path / 'censored-bbox'
        
        # YOLO dataset directory (for prepared data)
        self.yolo_dataset_dir: Path = Path('./crowdhuman_yolo').resolve()
        
        # dataset fraction for sampling
        self.dataset_fraction: float = 0.01  # 1% of the dataset
        
        # train/val/test split ratios (must sum to 1.0)
        self.train_ratio: float = 0.7
        self.val_ratio: float = 0.15
        self.test_ratio: float = 0.15
        
        # classes - now includes head bounding box
        self.classes: Dict[str, int] = {'person': 0, 'hbox': 1}
        
        # training parameters
        self.epochs: int = 10
        self.img_size: int = 640
        self.batch_size: int = 8
        self.workers: int = 2
        self.patience: int = 7
        self.model_name: str = 'yolov8n.pt'  # nano model
        
        # paths for prepared data - will be set in _setup_prepared_paths()
        self.uncensored_train_images: Path
        self.uncensored_train_labels: Path
        self.uncensored_val_images: Path
        self.uncensored_val_labels: Path
        self.uncensored_test_images: Path
        self.uncensored_test_labels: Path
        
        self.censored_blur_train_images: Path
        self.censored_blur_train_labels: Path
        self.censored_blur_val_images: Path
        self.censored_blur_val_labels: Path
        
        self.censored_bbox_train_images: Path
        self.censored_bbox_train_labels: Path
        self.censored_bbox_val_images: Path
        self.censored_bbox_val_labels: Path
        
        # setup the prepared paths
        self._setup_prepared_paths()
        
        # override with command line args if provided
        if args:
            self.update_from_args(args)
            
    # set up paths for prepared data based on configuration
    def _setup_prepared_paths(self) -> None:
        # Uncensored data paths
        self.uncensored_train_images = self.yolo_dataset_dir / 'uncensored' / 'images' / 'train'
        self.uncensored_train_labels = self.yolo_dataset_dir / 'uncensored' / 'labels' / 'train'
        self.uncensored_val_images = self.yolo_dataset_dir / 'uncensored' / 'images' / 'val'
        self.uncensored_val_labels = self.yolo_dataset_dir / 'uncensored' / 'labels' / 'val'
        self.uncensored_test_images = self.yolo_dataset_dir / 'uncensored' / 'images' / 'test'
        self.uncensored_test_labels = self.yolo_dataset_dir / 'uncensored' / 'labels' / 'test'
        
        # Censored-blur data paths
        self.censored_blur_train_images = self.yolo_dataset_dir / 'censored-blur' / 'images' / 'train'
        self.censored_blur_train_labels = self.yolo_dataset_dir / 'censored-blur' / 'labels' / 'train'
        self.censored_blur_val_images = self.yolo_dataset_dir / 'censored-blur' / 'images' / 'val'
        self.censored_blur_val_labels = self.yolo_dataset_dir / 'censored-blur' / 'labels' / 'val'
        
        # Censored-bbox data paths
        self.censored_bbox_train_images = self.yolo_dataset_dir / 'censored-bbox' / 'images' / 'train'
        self.censored_bbox_train_labels = self.yolo_dataset_dir / 'censored-bbox' / 'labels' / 'train'
        self.censored_bbox_val_images = self.yolo_dataset_dir / 'censored-bbox' / 'images' / 'val'
        self.censored_bbox_val_labels = self.yolo_dataset_dir / 'censored-bbox' / 'labels' / 'val'
    
    # update configuration from command line arguments
    def update_from_args(self, args: Args) -> None:
        if args.data_path is not None:
            self.base_data_path = Path(args.data_path).resolve()
            # update paths
            self.annotation_file = self.base_data_path / 'annotation.odgt'
            self.image_uncensored = self.base_data_path / 'uncensored'
            self.image_censored_blur = self.base_data_path / 'censored-blur'
            self.image_censored_bbox = self.base_data_path / 'censored-bbox'
        
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
            f"  Split Ratios: train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio}\n"
            f"  Epochs: {self.epochs}\n"
            f"  Batch Size: {self.batch_size}\n"
            f"  Image Size: {self.img_size}\n"
            f"  Model: {self.model_name}"
        )


# parse command line arguments and return Args object
def parse_arguments() -> Args:
    parser = argparse.ArgumentParser(description='Train YOLOv8 on CrowdHuman dataset with different censoring methods')
    
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
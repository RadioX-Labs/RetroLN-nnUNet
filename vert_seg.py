#!/usr/bin/env python3
import os
import glob
import subprocess
import argparse
from tqdm import tqdm
import multiprocessing
import nibabel as nib
import numpy as np
import traceback
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("vertebral_segmentation.log")
    ]
)
logger = logging.getLogger("VertebralSegmentation")

def validate_nifti(file_path):
    """Check if the NIFTI file is valid and readable."""
    try:
        img = nib.load(file_path)
        # Test reading a small portion of data
        _ = img.get_fdata()[:5, :5, :5]
        # Check if image has reasonable dimensions
        shape = img.shape
        if any(s < 10 for s in shape):
            logger.warning(f"Image has small dimensions: {shape}")
            return False
        # Check if image has reasonable voxel spacing
        spacing = img.header.get_zooms()
        if any(s > 10 for s in spacing):
            logger.warning(f"Image has unusual voxel spacing: {spacing}")
            return False
        return True
    except Exception as e:
        logger.error(f"File validation failed for {file_path}: {str(e)}")
        return False

def process_image(args):
    img_file, output_dir, vertebrae, options = args
    case_id = os.path.basename(img_file).replace('_0000.nii.gz', '').replace('.nii.gz', '').replace('.nii', '')
    case_output_dir = os.path.join(output_dir, case_id)
    
    try:
        # Check if already processed
        all_files_exist = True
        for vert in vertebrae:
            # Check both .nii and .nii.gz formats
            vert_file_nii = os.path.join(case_output_dir, f"{vert}.nii")
            vert_file_gz = os.path.join(case_output_dir, f"{vert}.nii.gz")
            if not (os.path.exists(vert_file_nii) or os.path.exists(vert_file_gz)):
                all_files_exist = False
                break
        
        if all_files_exist:
            logger.info(f"Skipping {case_id} - already processed")
            return case_id, "skipped"
        
        # Validate the input file
        if not validate_nifti(img_file):
            logger.error(f"Skipping {case_id} - corrupted file")
            return case_id, "corrupted"
        
        # Create output directory
        os.makedirs(case_output_dir, exist_ok=True)
        
        # Build command - CORRECTED ARGUMENTS FOR TOTALSEGMENTATOR 2.4.0
        cmd = [
            "TotalSegmentator", 
            "-i", img_file, 
            "-o", case_output_dir, 
            "-ta", "total", 
            "-rs"  # Changed from --roi_subset to -rs
        ] + vertebrae
        
        # Add optional parameters with corrected flags
        if options.get("use_gpu", True):
            cmd.extend(["-d", "gpu"])  # Changed from 'cuda' to 'gpu'
        else:
            cmd.extend(["-d", "cpu"])
        
        if options.get("use_fast", False):
            cmd.append("-f")  # Changed from --fast to -f
            
        if options.get("use_ml", True):
            cmd.append("-ml")  # This one was correct

        if options.get("verbose", False):
            cmd.append("-v")  # Changed from --verbose to -v
            
        if options.get("task", None):
            cmd.extend(["-ta", options["task"]])  # This was correct
            
        logger.info(f"Processing {case_id} with command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            logger.error(f"Command failed for {case_id}: {result.stderr}")
            return case_id, "failed"
            
        # Verify output
        missing_files = []
        for vert in vertebrae:
            vert_file_nii = os.path.join(case_output_dir, f"{vert}.nii")
            vert_file_gz = os.path.join(case_output_dir, f"{vert}.nii.gz")
            if not (os.path.exists(vert_file_nii) or os.path.exists(vert_file_gz)):
                missing_files.append(vert)
        
        if missing_files:
            logger.warning(f"Missing output files for {case_id}: {missing_files}")
            return case_id, "incomplete"
            
        # Save memory by converting .nii.gz to .nii if disk space isn't critical
        if options.get("save_space", False):
            for vert in vertebrae:
                vert_file_gz = os.path.join(case_output_dir, f"{vert}.nii.gz")
                vert_file_nii = os.path.join(case_output_dir, f"{vert}.nii")
                if os.path.exists(vert_file_gz) and not os.path.exists(vert_file_nii):
                    img = nib.load(vert_file_gz)
                    nib.save(img, vert_file_nii)
                    os.remove(vert_file_gz)
                    
        return case_id, "success"
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error processing {case_id}: {e}")
        return case_id, "failed"
    except Exception as e:
        logger.error(f"Unexpected error for {case_id}: {e}")
        traceback.print_exc()
        return case_id, "failed"

def worker_init():
    """Initialize each worker process."""
    # This helps avoid freezing issues on some systems
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def check_totalsegmentator():
    """Check if TotalSegmentator is installed and working."""
    try:
        result = subprocess.run(["TotalSegmentator", "--version"], 
                                capture_output=True, text=True, check=False)
        if result.returncode == 0:
            logger.info(f"TotalSegmentator version: {result.stdout.strip()}")
            return True
        else:
            logger.error(f"TotalSegmentator check failed: {result.stderr}")
            return False
    except FileNotFoundError:
        logger.error("TotalSegmentator not found. Please install it with: "
                     "pip install TotalSegmentator")
        return False
    except Exception as e:
        logger.error(f"Error checking TotalSegmentator: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process multiple images with TotalSegmentator")
    parser.add_argument("--input_dir", type=str, required=True, 
                       help="Directory containing input images (*_0000.nii.gz)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Base directory for outputs")
    parser.add_argument("--num_processes", type=int, default=2,
                       help="Number of parallel processes (default: 2)")
    parser.add_argument("--no_gpu", action="store_true",
                       help="Disable GPU usage")
    parser.add_argument("--fast", action="store_true",
                       help="Use fast mode (lower quality but faster)")
    parser.add_argument("--save_space", action="store_true",
                       help="Convert .nii.gz to .nii to save disk space")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output from TotalSegmentator")
    parser.add_argument("--file_pattern", type=str, default="*_0000.nii.gz",
                       help="File pattern for input CT files (default: *_0000.nii.gz)")
    parser.add_argument("--batch_size", type=int, default=0,
                       help="Process only a specified number of images (0 = all)")
    parser.add_argument("--task", type=str, 
                       help="Specific TotalSegmentator task (default: spine_vertebrae)")
    args = parser.parse_args()
    
    # Log script start
    logger.info(f"Script started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Arguments: {vars(args)}")
    
    # Check if TotalSegmentator is installed
    if not check_totalsegmentator():
        logger.error("Exiting due to TotalSegmentator check failure.")
        exit(1)
    
    # Find all input images
    input_files = sorted(glob.glob(os.path.join(args.input_dir, args.file_pattern)))
    if not input_files:
        logger.error(f"No image files found in {args.input_dir} with pattern {args.file_pattern}")
        exit(1)
    
    logger.info(f"Found {len(input_files)} images to process")
    
    # Apply batch size limit if specified
    if args.batch_size > 0 and args.batch_size < len(input_files):
        logger.info(f"Limiting to batch size of {args.batch_size}")
        input_files = input_files[:args.batch_size]
    
    # Vertebrae to segment (T12-L5)
    vertebrae = ["vertebrae_T12", "vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4", "vertebrae_L5"]
    
    # Options for TotalSegmentator
    options = {
        "use_gpu": not args.no_gpu,
        "use_fast": args.fast,
        "save_space": args.save_space,
        "verbose": args.verbose,
        "crop": False,  # We'll do our own cropping
        "use_ml": True,
        "task": args.task if args.task else None
    }
    
    # Create job arguments
    job_args = [(img_file, args.output_dir, vertebrae, options) for img_file in input_files]
    
    # Process images (in parallel if requested)
    if args.num_processes > 1:
        with multiprocessing.Pool(args.num_processes, initializer=worker_init) as pool:
            results = list(tqdm(pool.imap(process_image, job_args), total=len(job_args)))
    else:
        results = [process_image(args) for args in tqdm(job_args)]
    
    # Count results by status
    status_count = {"success": 0, "skipped": 0, "failed": 0, "corrupted": 0, "incomplete": 0}
    for case_id, status in results:
        if status in status_count:
            status_count[status] += 1
    
    logger.info("Processing complete!")
    logger.info(f"Total processed: {len(results)}")
    logger.info(f"Success: {status_count['success']}")
    logger.info(f"Skipped: {status_count['skipped']}")
    logger.info(f"Failed: {status_count['failed']}")
    logger.info(f"Corrupted: {status_count['corrupted']}")
    logger.info(f"Incomplete: {status_count['incomplete']}")
    logger.info(f"Script completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#!/usr/bin/env python3
import os
import numpy as np
import nibabel as nib
from scipy import ndimage
import time
import argparse
from datetime import datetime
import logging
import glob
import multiprocessing
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("retroperitoneal_cropping.log")
    ]
)
logger = logging.getLogger("RetroperitonealCropping")

def normalize_intensity(ct_data, min_bound=-1000, max_bound=400, output_range=(0, 1)):
    """Normalize CT intensity values to a specified range"""
    # Create a copy to avoid modifying the original data
    ct_norm = ct_data.copy().astype(float)
    
    # Clip values outside HU range
    ct_norm = np.clip(ct_norm, min_bound, max_bound)
    
    # Normalize to [0,1] range
    ct_norm = (ct_norm - min_bound) / (max_bound - min_bound)
    
    # Scale to output range if needed
    if output_range != (0, 1):
        ct_norm = ct_norm * (output_range[1] - output_range[0]) + output_range[0]
    
    return ct_norm

def find_matching_file(base_file, target_dir, suffix="", prefix=""):
    """Find the matching file in a target directory using multiple strategies"""
    # Extract base name (e.g., '539_0000' from '539_0000.nii.gz')
    base_name = os.path.splitext(base_file)[0]
    if base_file.endswith('.gz'):
        base_name = os.path.splitext(base_name)[0]
    
    # Extract ID part (e.g., '539' from '539_0000')
    id_part = base_name.split('_')[0] if '_' in base_name else base_name
    
    logger.info(f"Looking for match: Base: {base_name}, ID: {id_part}")
    
    # Common patterns to try (prioritized)
    patterns = [
        f"{prefix}{base_name}{suffix}.nii.gz",
        f"{prefix}{base_name}{suffix}.nii",
        f"{prefix}{id_part}{suffix}.nii.gz",
        f"{prefix}{id_part}{suffix}.nii",
    ]
    
    # Add more variations
    if "_" in base_name:
        parts = base_name.split('_')
        for i in range(1, len(parts)):
            subset = "_".join(parts[:i])
            patterns.append(f"{prefix}{subset}{suffix}.nii.gz")
            patterns.append(f"{prefix}{subset}{suffix}.nii")
    
    # Check each pattern
    for pattern in patterns:
        potential_path = os.path.join(target_dir, pattern)
        if os.path.exists(potential_path):
            logger.info(f"Found match: {pattern}")
            return pattern
            
    # If no match found with patterns, look for partial matches
    logger.info("No exact matches found. Scanning for partial matches...")
    files = [f for f in os.listdir(target_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]
    
    for file in files:
        if id_part in file:
            logger.info(f"Found potential partial match: {file}")
            return file
    
    # Still no match
    logger.warning(f"No matching file found for {base_file}")
    return None

def load_vertebral_segmentations(vert_dir, case_id):
    """Load vertebral segmentations from directory"""
    # Look for individual vertebra segmentations in case subdirectory
    case_dir = os.path.join(vert_dir, case_id)
    if os.path.isdir(case_dir):
        logger.info(f"Found case directory: {case_dir}")
        vertebrae = {}
        for vert_name in ["vertebrae_T12", "vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4", "vertebrae_L5"]:
            for ext in [".nii.gz", ".nii"]:
                vert_file = os.path.join(case_dir, f"{vert_name}{ext}")
                if os.path.exists(vert_file):
                    try:
                        nii = nib.load(vert_file)
                        vertebrae[vert_name] = {
                            "data": nii.get_fdata() > 0,
                            "affine": nii.affine,
                            "header": nii.header,
                            "file": vert_file
                        }
                        logger.info(f"Loaded {vert_name}")
                    except Exception as e:
                        logger.error(f"Failed to load {vert_file}: {str(e)}")
        
        if vertebrae:
            return vertebrae
    
    # If no case directory or no vertebrae found, try to find a single segmentation file
    potential_files = [
        os.path.join(vert_dir, f"{case_id}.nii.gz"),
        os.path.join(vert_dir, f"{case_id}.nii"),
        os.path.join(vert_dir, f"{case_id}_seg.nii.gz"),
        os.path.join(vert_dir, f"{case_id}_seg.nii")
    ]
    
    # Add pattern-based file search
    for file in os.listdir(vert_dir):
        if case_id in file and (file.endswith('.nii.gz') or file.endswith('.nii')):
            potential_files.append(os.path.join(vert_dir, file))
    
    for file_path in potential_files:
        if os.path.exists(file_path):
            try:
                nii = nib.load(file_path)
                data = nii.get_fdata()
                
                # Create a combined vertebrae object
                vertebrae = {
                    "combined": {
                        "data": data > 0,
                        "affine": nii.affine,
                        "header": nii.header,
                        "file": file_path
                    },
                    "vertebrae_L2": {  # Create an L2 entry for reference
                        "data": data > 0, 
                        "affine": nii.affine,
                        "header": nii.header,
                        "file": file_path
                    }
                }
                logger.info(f"Loaded combined vertebral segmentation from {file_path}")
                return vertebrae
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {str(e)}")
    
    logger.error(f"No vertebral segmentations found for {case_id}")
    return None

def find_reference_vertebra(ct_path, vertebrae_dict):
    """Find the L2 vertebra and reference points for cropping"""
    if not vertebrae_dict:
        logger.error("No vertebral segmentations provided")
        return None
        
    logger.info(f"Extracting reference points from CT: {os.path.basename(ct_path)}")
    
    # Load the CT volume
    ct_nii = nib.load(ct_path)
    ct_data = ct_nii.get_fdata()
    header = ct_nii.header
    
    # Get voxel dimensions in mm
    voxel_dims = header.get_zooms()
    
    # Get orientation information
    orientation = nib.aff2axcodes(ct_nii.affine)
    
    # Determine which axes correspond to anatomical directions
    ap_axis = None
    ap_direction = None
    si_axis = None
    si_direction = None
    lr_axis = None  # Left-right axis
    
    for i, ornt in enumerate(orientation):
        if ornt == 'A':
            ap_axis = i
            ap_direction = 1  # Increasing indices = anterior
        elif ornt == 'P':
            ap_axis = i
            ap_direction = -1  # Decreasing indices = anterior
        elif ornt == 'S':
            si_axis = i
            si_direction = 1  # Increasing indices = superior
        elif ornt == 'I':
            si_axis = i
            si_direction = -1  # Decreasing indices = superior
        elif ornt == 'R' or ornt == 'L':
            lr_axis = i  # Identify the left-right axis
    
    if ap_axis is None:
        logger.error("Could not determine anterior-posterior axis from image orientation")
        return None
    if si_axis is None:
        logger.error("Could not determine superior-inferior axis from image orientation")
        return None
    if lr_axis is None:
        # If no explicit L/R axis, it's the remaining axis
        lr_axis = [i for i in range(3) if i != ap_axis and i != si_axis][0]
        logger.warning(f"Left-right axis not explicitly identified, using axis {lr_axis}")
    
    # Use L2 vertebra if available, otherwise use combined vertebrae or any available vertebra
    vert_mask = None
    if "vertebrae_L2" in vertebrae_dict:
        logger.info("Using L2 vertebra")
        vert_mask = vertebrae_dict["vertebrae_L2"]["data"]
    elif "combined" in vertebrae_dict:
        logger.info("Using combined vertebral segmentation")
        vert_mask = vertebrae_dict["combined"]["data"]
    else:
        # Use any vertebra available
        for name, vert in vertebrae_dict.items():
            logger.info(f"Using {name} for reference")
            vert_mask = vert["data"]
            break
    
    # Find Superior-Inferior (SI) boundaries from vertebrae
    si_indices = np.where(np.any(vert_mask, axis=tuple(i for i in range(3) if i != si_axis)))[0]
    if len(si_indices) == 0:
        logger.error("No segmentation data found in the volume")
        return None
    
    si_min, si_max = si_indices.min(), si_indices.max()
    
    # Find vertebra anterior margin and center
    anterior_margins = []
    centers = []
    
    # Find center slices - use the middle 50% of the vertebra for more stable reference
    si_middle_start = si_min + int((si_max - si_min) * 0.25)
    si_middle_end = si_min + int((si_max - si_min) * 0.75)
    
    # Process each slice in the middle section
    for idx in range(si_middle_start, si_middle_end + 1):
        # Create slice indices for this position
        slice_idx = [slice(None)] * 3
        slice_idx[si_axis] = idx
        
        # Get the vertebra mask for this slice
        slice_mask = vert_mask[tuple(slice_idx)]
        
        if np.any(slice_mask):
            # Find center of mass
            center = ndimage.center_of_mass(slice_mask)
            
            # Extract center coordinate based on axis arrangement
            lr_idx = 0 if lr_axis == 0 else (1 if lr_axis == 1 else 2)
            lr_center = int(center[lr_idx])
            centers.append(lr_center)
            
            # Find the anterior margin
            # Create a projection along AP axis
            if ap_axis == 0:
                ap_projection = np.any(slice_mask, axis=1)
            elif ap_axis == 1:
                ap_projection = np.any(slice_mask, axis=0)
            else:  # ap_axis == 2
                ap_projection = np.any(slice_mask, axis=0)
                
            ap_indices = np.where(ap_projection)[0]
            
            if len(ap_indices) > 0:
                if ap_direction == 1:  # Increasing indices = anterior
                    anterior_margins.append(int(ap_indices.max()))
                else:  # Decreasing indices = anterior
                    anterior_margins.append(int(ap_indices.min()))
    
    # Calculate average anterior margin and center
    if not anterior_margins or not centers:
        logger.error("Could not calculate vertebra anterior margin and center")
        return None
        
    anterior_margin = int(np.median(anterior_margins))
    lr_center = int(np.median(centers))
    
    # Calculate cropping parameters
    # 180mm crop size with 30mm backward offset from vertebrae
    crop_size_voxels = int(180 / voxel_dims[ap_axis])  # 180mm crop size
    offset_voxels = int(30 / voxel_dims[ap_axis])      # 30mm backward offset
    
    # Calculate the fixed crop boundaries based on vertebra with offset
    if ap_direction == 1:  # Increasing indices = anterior
        # Subtract offset to move backward from anterior margin
        ap_start = max(0, anterior_margin - offset_voxels)
        ap_end = min(ap_start + crop_size_voxels, ct_data.shape[ap_axis])
    else:  # Decreasing indices = anterior
        # Add offset to move backward from anterior margin
        ap_end = min(anterior_margin + offset_voxels, ct_data.shape[ap_axis])
        ap_start = max(ap_end - crop_size_voxels, 0)
    
    # Adjust LR boundaries centered on vertebra
    lr_width_mm = 150  # 150mm width for left-right dimension
    lr_width_voxels = int(lr_width_mm / voxel_dims[lr_axis])
    lr_start = max(0, lr_center - lr_width_voxels // 2)
    lr_end = min(ct_data.shape[lr_axis], lr_center + lr_width_voxels // 2)
    
    # Return reference information
    return {
        "si_min": si_min,
        "si_max": si_max,
        "ap_start": ap_start,
        "ap_end": ap_end,
        "lr_start": lr_start,
        "lr_end": lr_end,
        "ap_axis": ap_axis,
        "si_axis": si_axis,
        "lr_axis": lr_axis,
        "ap_direction": ap_direction,
        "si_direction": si_direction,
        "voxel_dims": voxel_dims,
        "orientation": orientation,
        "shape": ct_data.shape,
        "l2_center": lr_center,
        "l2_anterior": anterior_margin,
    }

def process_volume(args):
    """Process volume (CT or mask) using reference data for cropping"""
    input_path, reference_data, output_path, options = args
    
    try:
        logger.info(f"Processing: {os.path.basename(input_path)}")
        start_time = time.time()
        
        # Load the input volume
        nii = nib.load(input_path)
        data = nii.get_fdata()
        is_mask = options.get("is_mask", False)
        
        # Extract reference data
        si_min = reference_data["si_min"]
        si_max = reference_data["si_max"]
        ap_start = reference_data["ap_start"]
        ap_end = reference_data["ap_end"]
        lr_start = reference_data["lr_start"]
        lr_end = reference_data["lr_end"]
        ap_axis = reference_data["ap_axis"]
        si_axis = reference_data["si_axis"]
        lr_axis = reference_data["lr_axis"]
        
        # Create cropping slices for each dimension
        crop_slices = [slice(None), slice(None), slice(None)]
        crop_slices[si_axis] = slice(si_min, si_max + 1)
        crop_slices[ap_axis] = slice(ap_start, ap_end)
        crop_slices[lr_axis] = slice(lr_start, lr_end)
        
        # Crop the volume
        cropped_data = data[tuple(crop_slices)]
        
        # Normalize intensity for CT (not for masks)
        if not is_mask and options.get("normalize", True):
            cropped_data = normalize_intensity(
                cropped_data,
                min_bound=options.get("min_hu", -1000),
                max_bound=options.get("max_hu", 400),
                output_range=options.get("output_range", (0, 1))
            )
            
        # Create new NIfTI image with updated affine for cropping
        new_affine = nii.affine.copy()
        
        # Update the origin in the affine matrix to account for cropping
        offset_vector = np.zeros(3)
        offset_vector[si_axis] = si_min * reference_data["voxel_dims"][si_axis]
        offset_vector[ap_axis] = ap_start * reference_data["voxel_dims"][ap_axis]
        offset_vector[lr_axis] = lr_start * reference_data["voxel_dims"][lr_axis]
        
        # Apply the offset to the last column of the affine matrix
        new_affine[:3, 3] += np.dot(new_affine[:3, :3], offset_vector)
        
        # Create the processed NIfTI image
        processed_nii = nib.Nifti1Image(cropped_data, new_affine, nii.header)
        
        # Save the processed image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        nib.save(processed_nii, output_path)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Processed {os.path.basename(input_path)} in {elapsed_time:.2f}s")
        
        return {
            "file": os.path.basename(input_path),
            "output": os.path.basename(output_path),
            "status": "success",
            "shape": cropped_data.shape,
            "time": elapsed_time
        }
        
    except Exception as e:
        logger.error(f"Error processing {os.path.basename(input_path)}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "file": os.path.basename(input_path),
            "status": "failed",
            "error": str(e)
        }

def validate_region_coverage(ct_path, mask_path, reference_data):
    """Validate that the cropped region contains the expected anatomy"""
    try:
        ct_nii = nib.load(ct_path)
        ct_data = ct_nii.get_fdata()
        
        mask_nii = nib.load(mask_path)
        mask_data = mask_nii.get_fdata()
        
        # Check if dimensions match
        if ct_data.shape != mask_data.shape:
            return {
                "valid": False,
                "reason": f"Dimension mismatch: CT {ct_data.shape}, Mask {mask_data.shape}"
            }
        
        # Check if mask contains segmentations
        if not np.any(mask_data > 0):
            return {
                "valid": False,
                "reason": "No segmentation data found in mask"
            }
        
        # Calculate statistics for CT values in the cropped region
        ct_mean = np.mean(ct_data)
        ct_std = np.std(ct_data)
        
        # Basic range check for CT values
        if ct_mean < -500 or ct_mean > 500:
            return {
                "valid": False,
                "reason": f"Unusual CT mean value: {ct_mean:.1f} HU"
            }
        
        return {
            "valid": True,
            "ct_mean": ct_mean,
            "ct_std": ct_std
        }
        
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return {
            "valid": False,
            "reason": f"Error during validation: {str(e)}"
        }

def worker_init():
    """Initialize each worker process."""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def parallel_process_volumes(ct_dir, mask_dir, vert_seg_dir, output_dir, options):
    """Process CT volumes and masks in parallel"""
    os.makedirs(os.path.join(output_dir, "ct"), exist_ok=True)
    
    # Check if mask directory exists and has mask files
    has_masks = False
    if mask_dir and os.path.exists(mask_dir):
        mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]
        has_masks = len(mask_files) > 0
    
    if has_masks:
        os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
        logger.info(f"Mask directory found: {mask_dir}")
    else:
        logger.info("No mask directory provided or no mask files found - only CT volumes will be processed")
    
    logger.info(f"Parallel processing with {options.get('num_processes', 1)} processes")
    
    # Get all CT files
    ct_files = sorted([f for f in os.listdir(ct_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
    logger.info(f"Found {len(ct_files)} CT files")
    
    # Process each CT file first to extract reference data
    reference_data_dict = {}
    for ct_file in tqdm(ct_files, desc="Extracting reference data"):
        try:
            # Extract base name for finding matching files
            base_name = os.path.splitext(ct_file)[0]
            if ct_file.endswith('.gz'):
                base_name = os.path.splitext(base_name)[0]
                
            # Extract ID part
            id_part = base_name.split('_')[0] if '_' in base_name else base_name
            
            # Load vertebral segmentations
            vertebrae_dict = load_vertebral_segmentations(vert_seg_dir, id_part)
            if not vertebrae_dict:
                logger.warning(f"No vertebral segmentations found for {id_part}")
                continue
            
            # Extract reference data for cropping
            reference_data = find_reference_vertebra(
                os.path.join(ct_dir, ct_file),
                vertebrae_dict
            )
            
            if reference_data:
                reference_data_dict[id_part] = reference_data
                logger.info(f"Extracted reference data for {id_part}")
            
        except Exception as e:
            logger.error(f"Error extracting reference data for {ct_file}: {str(e)}")
    
    logger.info(f"Extracted reference data for {len(reference_data_dict)} cases")
    
    if not reference_data_dict:
        logger.error("No reference data extracted, aborting")
        return
    
    # Create job lists for CTs and masks
    ct_jobs = []
    mask_jobs = []
    
    # Add CT jobs - PRESERVE ORIGINAL FILENAMES
    for ct_file in ct_files:
        base_name = os.path.splitext(ct_file)[0]
        if ct_file.endswith('.gz'):
            base_name = os.path.splitext(base_name)[0]
            
        id_part = base_name.split('_')[0] if '_' in base_name else base_name
        
        if id_part in reference_data_dict:
            # Preserve original filename
            ct_output_path = os.path.join(output_dir, "ct", ct_file)
            ct_jobs.append((
                os.path.join(ct_dir, ct_file), 
                reference_data_dict[id_part], 
                ct_output_path,
                {"is_mask": False, "normalize": options.get("normalize_ct", True)}
            ))
    
    # Add mask jobs if masks exist - PRESERVE ORIGINAL FILENAMES
    if has_masks:
        mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]
        for mask_file in mask_files:
            base_name = os.path.splitext(mask_file)[0]
            if mask_file.endswith('.gz'):
                base_name = os.path.splitext(base_name)[0]
                
            id_parts = [part for part in reference_data_dict.keys() if part in mask_file]
            
            if id_parts:
                id_part = id_parts[0]  # Use the first matching ID
                # Preserve original filename
                mask_output_path = os.path.join(output_dir, "masks", mask_file)
                mask_jobs.append((
                    os.path.join(mask_dir, mask_file),
                    reference_data_dict[id_part],
                    mask_output_path,
                    {"is_mask": True, "normalize": False}
                ))
    
    # Process jobs in parallel
    all_jobs = ct_jobs + mask_jobs
    logger.info(f"Prepared {len(ct_jobs)} CT jobs and {len(mask_jobs)} mask jobs")
    
    with multiprocessing.Pool(options.get('num_processes', 1), initializer=worker_init) as pool:
        results = list(tqdm(pool.imap(process_volume, all_jobs), total=len(all_jobs), desc="Processing volumes"))
    
    # Count results
    success_ct = sum(1 for r in results[:len(ct_jobs)] if r["status"] == "success")
    failed_ct = len(ct_jobs) - success_ct
    
    success_mask = 0
    failed_mask = 0
    if has_masks:
        success_mask = sum(1 for r in results[len(ct_jobs):] if r["status"] == "success")
        failed_mask = len(mask_jobs) - success_mask
    
    logger.info("\nProcessing Summary:")
    logger.info(f"CT processing: {success_ct} succeeded, {failed_ct} failed")
    if has_masks:
        logger.info(f"Mask processing: {success_mask} succeeded, {failed_mask} failed")
    else:
        logger.info("No masks processed (masks not available)")
    
    # Save summary to file
    summary = {
        "total_ct": len(ct_jobs),
        "total_masks": len(mask_jobs),
        "processed_ct": success_ct,
        "processed_masks": success_mask,
        "failed_ct": failed_ct,
        "failed_masks": failed_mask,
        "masks_available": has_masks
    }
    
    summary_file = os.path.join(output_dir, "processing_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("Processing Summary\n")
        f.write(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"User: {os.environ.get('USER', 'Pankajg959')}\n")
        f.write("=================================================================\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

def batch_process_volumes(ct_dir, mask_dir, vert_seg_dir, output_dir, options):
    """Process CT volumes and masks using reference vertebral segmentations"""
    os.makedirs(os.path.join(output_dir, "ct"), exist_ok=True)
    
    # Check if mask directory exists and has mask files
    has_masks = False
    if mask_dir and os.path.exists(mask_dir):
        mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]
        has_masks = len(mask_files) > 0
    
    if has_masks:
        os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "validation"), exist_ok=True)
        logger.info(f"Mask directory found: {mask_dir}")
    else:
        logger.info("No mask directory provided or no mask files found - only CT volumes will be processed")
    
    logger.info(f"Using vertebral segmentations from: {vert_seg_dir}")
    logger.info(f"Saving results to: {output_dir}")
    
    # Get all CT files
    ct_files = sorted([f for f in os.listdir(ct_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
    logger.info(f"Found {len(ct_files)} CT files")
    
    # Track processing results
    results = {
        "total_ct": len(ct_files),
        "total_masks": 0,
        "processed_ct": 0,
        "processed_masks": 0,
        "failed_ct": 0,
        "failed_masks": 0,
        "skipped": 0,
        "validation_passed": 0,
        "validation_failed": 0,
        "masks_available": has_masks
    }
    
    if has_masks:
        mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]
        results["total_masks"] = len(mask_files)
    
    # Process each CT file and its associated masks
    for ct_file in ct_files:
        try:
            # Extract base name for finding matching files
            base_name = os.path.splitext(ct_file)[0]
            if ct_file.endswith('.gz'):
                base_name = os.path.splitext(base_name)[0]
                
            # Extract ID part
            id_part = base_name.split('_')[0] if '_' in base_name else base_name
            
            logger.info(f"\n=== Processing case {id_part} ===")
            
            # Step 1: Load vertebral segmentations
            vertebrae_dict = load_vertebral_segmentations(vert_seg_dir, id_part)
            if not vertebrae_dict:
                logger.error(f"No vertebral segmentations found for {id_part}, skipping...")
                results["skipped"] += 1
                continue
            
            # Step 2: Extract reference data for cropping
            reference_data = find_reference_vertebra(
                os.path.join(ct_dir, ct_file),
                vertebrae_dict
            )
            
            if not reference_data:
                logger.error(f"Failed to extract reference data for {id_part}, skipping...")
                results["skipped"] += 1
                continue
                
            # Step 3: Process the CT volume
            ct_output_path = os.path.join(output_dir, "ct", ct_file)  # Preserve original filename
            
            ct_result = process_volume(
                (os.path.join(ct_dir, ct_file), reference_data, ct_output_path, 
                 {"is_mask": False, "normalize": options.get("normalize_ct", True)})
            )
            
            if ct_result["status"] == "success":
                results["processed_ct"] += 1
                logger.info(f"Successfully processed CT: {id_part}")
                
                # Process masks if available
                if has_masks:
                    # Find matching masks
                    matching_masks = []
                    for mask_file in [f for f in os.listdir(mask_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]:
                        if id_part in mask_file:
                            matching_masks.append(mask_file)
                    
                    logger.info(f"Found {len(matching_masks)} matching masks for {id_part}")
                    
                    # Process each matching mask
                    for mask_file in matching_masks:
                        mask_output_path = os.path.join(output_dir, "masks", mask_file)  # Preserve original filename
                        
                        mask_result = process_volume(
                            (os.path.join(mask_dir, mask_file), reference_data, mask_output_path,
                            {"is_mask": True, "normalize": False})
                        )
                        
                        if mask_result["status"] == "success":
                            results["processed_masks"] += 1
                            logger.info(f"Successfully processed mask: {mask_file}")
                            
                            # Validate the results
                            validation_result = validate_region_coverage(
                                ct_output_path, mask_output_path, reference_data
                            )
                            
                            # Save validation results
                            validation_file = os.path.join(output_dir, "validation", f"{mask_file.replace('.nii.gz', '').replace('.nii', '')}_validation.txt")
                            with open(validation_file, 'w') as f:
                                for key, value in validation_result.items():
                                    f.write(f"{key}: {value}\n")
                            
                            if validation_result["valid"]:
                                results["validation_passed"] += 1
                            else:
                                results["validation_failed"] += 1
                                logger.warning(f"Validation failed for {mask_file}: {validation_result.get('reason')}")
                        else:
                            results["failed_masks"] += 1
                            logger.error(f"Failed to process mask: {mask_file}")
            else:
                results["failed_ct"] += 1
                logger.error(f"Failed to process CT: {ct_file}")
                
        except Exception as e:
            logger.error(f"Error processing {ct_file}: {str(e)}")
            results["failed_ct"] += 1
            import traceback
            traceback.print_exc()
    
    # Summary
    logger.info("\nProcessing Summary:")
    logger.info(f"Total CT files: {results['total_ct']}")
    if has_masks:
        logger.info(f"Total mask files: {results['total_masks']}")
    logger.info(f"Processed CT: {results['processed_ct']}")
    logger.info(f"Failed CT: {results['failed_ct']}")
    if has_masks:
        logger.info(f"Processed masks: {results['processed_masks']}")
        logger.info(f"Failed masks: {results['failed_masks']}")
        logger.info(f"Validation passed: {results['validation_passed']}")
        logger.info(f"Validation failed: {results['validation_failed']}")
    else:
        logger.info("No masks processed (masks not available)")
    logger.info(f"Skipped cases: {results['skipped']}")
    
    # Save summary to file
    summary_file = os.path.join(output_dir, "processing_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("Processing Summary\n")
        f.write(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"User: {os.environ.get('USER', 'Pankajg959')}\n")
        f.write("=================================================================\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process CT volumes and masks using vertebral segmentations')
    parser.add_argument('--ct_dir', required=True, help='Directory containing CT volumes')
    parser.add_argument('--mask_dir', help='Directory containing masks (optional)')
    parser.add_argument('--vert_seg_dir', required=True, help='Directory containing vertebral segmentations')
    parser.add_argument('--output_dir', required=True, help='Directory to save processed volumes')
    parser.add_argument('--parallel', action='store_true', help='Process in parallel')
    parser.add_argument('--num_processes', type=int, default=2, help='Number of parallel processes')
    parser.add_argument('--no_normalize', action='store_true', help='Disable CT intensity normalization')
    parser.add_argument('--min_hu', type=int, default=-1000, help='Minimum HU value for normalization')
    parser.add_argument('--max_hu', type=int, default=400, help='Maximum HU value for normalization')
    parser.add_argument('--single', action='store_true', help='Process a single CT and mask')
    parser.add_argument('--ct_file', help='Single CT file path (when --single is used)')
    parser.add_argument('--mask_file', help='Single mask file path (when --single is used)')
    parser.add_argument('--vert_seg_file', help='Single vertebral segmentation file path (when --single is used)')
    parser.add_argument('--ct_output', help='Single CT output path (when --single is used)')
    parser.add_argument('--mask_output', help='Single mask output path (when --single is used)')
    
    args = parser.parse_args()
    
    logger.info("Retroperitoneal Lymph Node Region Cropping Tool")
    logger.info("=================================================================")
    logger.info(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"User: {os.environ.get('USER', 'Pankajg959')}")
    logger.info("=================================================================")
    
    # Process options
    options = {
        "normalize_ct": not args.no_normalize,
        "min_hu": args.min_hu,
        "max_hu": args.max_hu,
        "output_range": (0, 1),
        "num_processes": args.num_processes
    }
    
    # Log current settings
    logger.info(f"CT directory: {args.ct_dir}")
    logger.info(f"Mask directory: {args.mask_dir if args.mask_dir else 'Not specified - CT only processing'}")
    logger.info(f"Vertebral segmentation directory: {args.vert_seg_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Normalize CT: {not args.no_normalize}")
    logger.info(f"HU range for normalization: [{args.min_hu}, {args.max_hu}]")
    logger.info(f"Parallel processing: {args.parallel} with {args.num_processes} processes")
    
    if args.single:
        if not all([args.ct_file, args.vert_seg_file]):
            logger.error("CT file and vertebral segmentation file are required with --single option")
        else:
            # Create output directory
            if args.ct_output:
                os.makedirs(os.path.dirname(args.ct_output), exist_ok=True)
            if args.mask_output and args.mask_file:
                os.makedirs(os.path.dirname(args.mask_output), exist_ok=True)
                
            # Load vertebral segmentations
            vertebrae_dict = {}
            try:
                # Load individual vertebrae if multiple files
                if os.path.isdir(args.vert_seg_file):
                    for vert_name in ["vertebrae_T12", "vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4", "vertebrae_L5"]:
                        for ext in [".nii.gz", ".nii"]:
                            path = os.path.join(args.vert_seg_file, f"{vert_name}{ext}")
                            if os.path.exists(path):
                                nii = nib.load(path)
                                vertebrae_dict[vert_name] = {
                                    "data": nii.get_fdata() > 0,
                                    "affine": nii.affine,
                                    "header": nii.header,
                                    "file": path
                                }
                # Or single vertebral segmentation file
                else:
                    nii = nib.load(args.vert_seg_file)
                    vertebrae_dict["vertebrae_L2"] = {
                        "data": nii.get_fdata() > 0,
                        "affine": nii.affine,
                        "header": nii.header,
                        "file": args.vert_seg_file
                    }
            except Exception as e:
                logger.error(f"Failed to load vertebral segmentation: {str(e)}")
                vertebrae_dict = None
                
            if not vertebrae_dict:
                logger.error("Could not load vertebral segmentations")
                exit(1)
                
            # Extract reference data
            reference_data = find_reference_vertebra(args.ct_file, vertebrae_dict)
            
            if not reference_data:
                logger.error("Failed to extract reference data")
                exit(1)
                
            # Process CT
            if args.ct_output:
                ct_result = process_volume(
                    (args.ct_file, reference_data, args.ct_output, 
                     {"is_mask": False, "normalize": options.get("normalize_ct", True)})
                )
                logger.info(f"CT processing result: {ct_result['status']}")
                
            # Process mask if provided
            if args.mask_file and args.mask_output:
                mask_result = process_volume(
                    (args.mask_file, reference_data, args.mask_output,
                     {"is_mask": True, "normalize": False})
                )
                logger.info(f"Mask processing result: {mask_result['status']}")
                
                # Validate if both CT and mask were processed
                if args.ct_output and ct_result["status"] == "success" and mask_result["status"] == "success":
                    validation_result = validate_region_coverage(
                        args.ct_output, args.mask_output, reference_data
                    )
                    logger.info(f"Validation result: {validation_result}")
    else:
        # Batch processing
        if args.parallel:
            parallel_process_volumes(
                args.ct_dir,
                args.mask_dir,  # May be None, function will handle it
                args.vert_seg_dir,
                args.output_dir,
                options
            )
        else:
            batch_process_volumes(
                args.ct_dir,
                args.mask_dir,  # May be None, function will handle it
                args.vert_seg_dir,
                args.output_dir,
                options
            )
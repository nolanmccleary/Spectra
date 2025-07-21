import argparse
import json
import os
from pathlib import Path
from spectra.validation.Image_Validator import directory_compare
from spectra.lpips import ALEX_IMPORT


def validate(device, input_dir, output_dir, verbose="off"):
    """Validate experiment results by comparing input and output images"""
    
    # Validate input paths
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    if not output_path.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    
    # Initialize LPIPS model
    print(f"Initializing LPIPS model on device: {device}")
    LPIPS_MODEL = ALEX_IMPORT(device)
    F_LPIPS = LPIPS_MODEL.get_lpips
    
    # Run validation
    print(f"Running validation...")
    # Look for images in the images/ subdirectory
    images_dir = output_path / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    post_validation = directory_compare(input_dir, str(images_dir), F_LPIPS, device, verbose)
    
    # Create results directory if it doesn't exist
    results_dir = output_path / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Save validation results
    json_filename = results_dir / "validated_results.json"
    print(f"Saving validation results to: {json_filename}")
    
    with open(json_filename, 'w') as f:
        json.dump(post_validation, f, indent=4)
    
    # Print summary
    print(f"Validation complete!")
    print(f"Results saved to: {json_filename}")
    
    # Print summary statistics if available
    if post_validation:
        print(f"\nValidation Summary:")
        for prefix, entries in post_validation.items():
            if "average" in entries:
                avg = entries["average"]
                print(f"  {prefix}:")
                print(f"    Average LPIPS: {avg.get('lpips', 'N/A')}")
                print(f"    Average L2: {avg.get('l2', 'N/A')}")
                print(f"    Average AHash Hamming ImageHash: {avg.get('ahash_hamming_imagehash', 'N/A')}")
                print(f"    Average DHash Hamming ImageHash: {avg.get('dhash_hamming_imagehash', 'N/A')}")
                print(f"    Average PHash Hamming ImageHash: {avg.get('phash_hamming_imagehash', 'N/A')}")
                print(f"    Average PDQ Hamming ImageHash: {avg.get('pdq_hamming_imagehash', 'N/A')}")
                print(f"    Average AHash Hamming Torch: {avg.get('ahash_hamming_torch', 'N/A')}")
                print(f"    Average DHash Hamming Torch: {avg.get('dhash_hamming_torch', 'N/A')}")
                print(f"    Average PHash Hamming Torch: {avg.get('phash_hamming_torch', 'N/A')}")
                print(f"    Average PDQ Hamming Torch: {avg.get('pdq_hamming_torch', 'N/A')}")
    
    return post_validation




if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="Validate experiment results by comparing input and output images")
    parser.add_argument("-d", "--device", type=str, default="cpu", help="Device to use for validation (cpu, cuda, mps)")
    parser.add_argument("-i", "--input_dir", type=str, default="sample_images", help="Input directory containing original images")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Output directory containing attacked images")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output during validation")
    args = parser.parse_args()
    
    try:
        verbose = "on" if args.verbose else "off"
        validate(args.device, args.input_dir, args.output_dir, verbose)
    except Exception as e:
        print(f"Validation failed: {e}")
        exit(1)
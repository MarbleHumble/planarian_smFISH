#!/usr/bin/env python3
"""
Interactive script to collect manual validation results
"""

import json
from pathlib import Path
from datetime import datetime

def collect_validation_results():
    """Interactively collect validation results"""
    
    print("="*60)
    print("Manual Threshold Validation Results Collection")
    print("="*60)
    
    validator_name = input("\nYour name: ").strip()
    general_notes = input("\nGeneral observations about the dataset: ").strip()
    
    # Get list of images from output directory
    output_dir = Path("/Users/eliasguan/Desktop/DL_210_data_analysis/ML_data_set_trainning/Test thresholding")
    
    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        return None
    
    image_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])
    
    if len(image_dirs) == 0:
        print(f"Error: No image directories found in {output_dir}")
        return None
    
    print(f"\nFound {len(image_dirs)} images to validate")
    print("\nFor each image, review the overlays in ImageJ and provide:")
    print("  - Best threshold (0-16)")
    print("  - Alternative thresholds (comma-separated, or 'none')")
    print("  - Spot count range (min, max)")
    print("  - Quality rating (poor/fair/good/excellent)")
    print("  - Observations")
    print("  - Issues (if any)")
    print("  - Recommended for training (yes/no)")
    
    results = {
        "validation_date": datetime.now().strftime("%Y-%m-%d"),
        "validator_name": validator_name,
        "notes": general_notes,
        "images": []
    }
    
    for i, img_dir in enumerate(image_dirs):
        image_name = img_dir.name
        print(f"\n{'='*60}")
        print(f"Image {i+1}/{len(image_dirs)}: {image_name}")
        print(f"{'='*60}")
        print(f"\nPlease review overlays in:")
        print(f"  2D overlays: {img_dir}")
        print(f"  3D overlays: {img_dir / '3D_overlays'}")
        print(f"\nFiles to check:")
        print(f"  - {image_name}_threshold_00_overlay.tif through {image_name}_threshold_16_overlay.tif")
        print(f"  - 3D_overlays/{image_name}_threshold_00_3d_overlay.tif through {image_name}_threshold_16_3d_overlay.tif")
        
        # Load existing results summary if available
        summary_file = img_dir / f"{image_name}_565_results_summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                    print(f"\nSpot counts from detection:")
                    for thresh in sorted([int(k) for k in summary.get('results', {}).keys()])[:5]:
                        n_spots = summary['results'].get(str(thresh), {}).get('n_spots', 0)
                        print(f"  Threshold {thresh}: {n_spots} spots")
            except:
                pass
        
        best_thresh = input("\nBest threshold (0-16): ").strip()
        if not best_thresh:
            print("Skipping this image...")
            continue
        
        alt_thresh = input("Alternative thresholds (comma-separated, or 'none'): ").strip()
        if alt_thresh.lower() == 'none' or not alt_thresh:
            alt_thresh = []
        else:
            try:
                alt_thresh = [int(x.strip()) for x in alt_thresh.split(',') if x.strip()]
            except:
                alt_thresh = []
        
        min_spots = input("Minimum reasonable spot count: ").strip()
        max_spots = input("Maximum reasonable spot count: ").strip()
        
        print("\nQuality rating:")
        print("  1. Poor")
        print("  2. Fair")
        print("  3. Good")
        print("  4. Excellent")
        quality_choice = input("Choice (1-4): ").strip()
        quality_map = {"1": "poor", "2": "fair", "3": "good", "4": "excellent"}
        quality = quality_map.get(quality_choice, "unknown")
        
        observations = input("\nObservations: ").strip()
        issues = input("Issues (or 'none'): ").strip()
        if issues.lower() == 'none':
            issues = ""
        
        recommended = input("Recommended for training? (yes/no): ").strip().lower()
        
        try:
            results["images"].append({
                "image_name": image_name,
                "best_threshold": int(best_thresh),
                "alternative_thresholds": alt_thresh,
                "spot_count_range": {
                    "min_reasonable": int(min_spots) if min_spots else 0,
                    "max_reasonable": int(max_spots) if max_spots else 0
                },
                "quality_rating": quality,
                "observations": observations,
                "issues": issues,
                "recommended_for_training": recommended == "yes"
            })
        except ValueError as e:
            print(f"Error parsing input: {e}")
            print("Skipping this image...")
            continue
        
        if i < len(image_dirs) - 1:
            continue_choice = input("\nContinue to next image? (yes/no): ").strip().lower()
            if continue_choice != "yes":
                break
    
    # Save results
    output_file = Path("manual_validation_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"Validated {len(results['images'])} images")
    print(f"{'='*60}\n")
    
    # Print summary
    print("\nSUMMARY:")
    print(f"  Images validated: {len(results['images'])}")
    recommended_count = sum(1 for img in results['images'] if img.get('recommended_for_training', False))
    print(f"  Recommended for training: {recommended_count}")
    
    if results['images']:
        print("\nBest thresholds by image:")
        for img in results['images']:
            print(f"  {img['image_name']}: {img['best_threshold']} (quality: {img['quality_rating']})")
    
    return results

if __name__ == "__main__":
    try:
        collect_validation_results()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Partial results may have been saved.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


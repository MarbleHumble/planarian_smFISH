#!/usr/bin/env python3
"""
Run comparison tests: Big-FISH CPU vs GPU Improved detection on 3 images.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from test_thresholds_mps_local import test_thresholds_on_region
from test_thresholds_improved import test_thresholds_improved

# Test images
test_images = [
    "/Volumes/Backup Plus/Experiment_results/306_analysis_results/Experiment/12hr_Amputation/Image1/565/12hr_Amputation_Image1_565.tif",
    "/Volumes/Backup Plus/Experiment_results/306_analysis_results/Experiment/12hr_Amputation/Image2/565/12hr_Amputation_Image2_565.tif",
    "/Volumes/Backup Plus/Experiment_results/306_analysis_results/Experiment/12hr_Incision/Image1/565/12hr_Incision_Image1_565.tif",
]

base_output = Path("/Users/eliasguan/Desktop/DL_210_data_analysis/ML_data_set_trainning")

print("="*80)
print("Running comparison tests: Big-FISH CPU vs GPU Improved Detection")
print("="*80)

for i, image_path in enumerate(test_images, 1):
    if not Path(image_path).exists():
        print(f"\nWARNING: Image {i} not found: {image_path}")
        continue
    
    image_name = Path(image_path).stem
    print(f"\n{'#'*80}")
    print(f"Image {i}/3: {image_name}")
    print(f"{'#'*80}")
    
    # Big-FISH CPU test
    print(f"\n[1/2] Running Big-FISH CPU detection...")
    bigfish_output = base_output / "BigFISH_CPU_test" / image_name
    try:
        result_bigfish = test_thresholds_on_region(
            image_path,
            bigfish_output,
            region_size=1024,
            thresholds=list(range(17)),
            config_path="config.yaml"
        )
        print(f"✓ Big-FISH completed: {result_bigfish['total_time']:.2f}s")
    except Exception as e:
        print(f"✗ Big-FISH failed: {e}")
        import traceback
        traceback.print_exc()
    
    # GPU Improved test
    print(f"\n[2/2] Running GPU Improved detection...")
    gpu_output = base_output / "GPU_Improved_test" / image_name
    try:
        result_gpu = test_thresholds_improved(
            image_path,
            gpu_output,
            region_size=1024,
            thresholds=list(range(17)),
            config_path="config.yaml",
            min_contrast=0,  # Disable contrast filter for now
            use_gpu=True,
            device="cpu"  # Use CPU for now (can change to cuda/mps)
        )
        print(f"✓ GPU Improved completed: {result_gpu['total_time']:.2f}s")
    except Exception as e:
        print(f"✗ GPU Improved failed: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*80}")
print("All tests completed!")
print(f"{'='*80}")


"""
Rectification script - applies the correction to the distorted image
Requires: CUDA GPU and numba with CUDA support
"""

import numpy as np
import argparse
import os
from PIL import Image
import matplotlib.pyplot as plt

try:
    from resample.resampling import rectification
    CUDA_AVAILABLE = True
except Exception as e:
    print(f"Warning: CUDA resampling not available: {e}")
    CUDA_AVAILABLE = False

def rectify_image(img_path, flow_path, output_dir='results'):
    """Apply rectification to distorted image using flow"""
    
    if not CUDA_AVAILABLE:
        print("\nERROR: CUDA resampling not available!")
        print("This requires a CUDA-capable GPU and numba with CUDA support.")
        print("Install with: conda install numba cudatoolkit")
        return
    
    # Load image and flow
    distorted_img = np.array(Image.open(img_path))
    flow = np.load(flow_path)
    
    print(f"Image shape: {distorted_img.shape}")
    print(f"Flow shape: {flow.shape}")
    
    # Apply rectification
    print("\nApplying rectification (this may take a moment)...")
    result_img, result_mask = rectification(distorted_img, flow)
    
    # Save results
    base_name = os.path.splitext(os.path.basename(img_path))[0].replace('_resized', '')
    
    result_path = os.path.join(output_dir, f'{base_name}_corrected.png')
    mask_path = os.path.join(output_dir, f'{base_name}_mask.png')
    
    Image.fromarray(result_img).save(result_path)
    Image.fromarray(result_mask).save(mask_path)
    
    print(f"✓ Corrected image saved to: {result_path}")
    print(f"✓ Mask saved to: {mask_path}")
    
    # Create comparison
    create_comparison(img_path, result_path, output_dir, base_name)
    
    return result_img, result_mask

def create_comparison(distorted_path, corrected_path, output_dir, base_name):
    """Create before/after comparison"""
    
    distorted = Image.open(distorted_path)
    corrected = Image.open(corrected_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(distorted)
    axes[0].set_title('Distorted Input')
    axes[0].axis('off')
    
    axes[1].imshow(corrected)
    axes[1].set_title('Corrected Output')
    axes[1].axis('off')
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, f'{base_name}_comparison.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comparison saved to: {comparison_path}")

def main():
    parser = argparse.ArgumentParser(description='Rectify distorted image using flow')
    parser.add_argument('--img_path', type=str, required=True, help='Path to distorted image')
    parser.add_argument('--flow_path', type=str, required=True, help='Path to flow .npy file')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("GeoProj - Image Rectification")
    print(f"{'='*60}")
    print(f"Input image: {args.img_path}")
    print(f"Flow file: {args.flow_path}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")
    
    rectify_image(args.img_path, args.flow_path, args.output_dir)
    
    print(f"\n{'='*60}")
    print("✓ Rectification complete!")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()


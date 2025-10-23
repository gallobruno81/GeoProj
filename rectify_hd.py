"""
HD Rectification script - applies correction at original resolution
Works with CPU (no CUDA required) using OpenCV
"""

import numpy as np
import argparse
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2

def rectify_image_hd(img_path, flow_path, output_dir='results'):
    """Apply rectification at original resolution using OpenCV"""
    
    print(f"\n{'='*60}")
    print("HD Rectification (CPU-based with OpenCV)")
    print(f"{'='*60}\n")
    
    # Load image and flow
    print("Loading image and flow...")
    distorted_img = np.array(Image.open(img_path))
    flow = np.load(flow_path)
    
    h, w = distorted_img.shape[:2]
    
    print(f"Image shape: {distorted_img.shape} ({w}x{h})")
    print(f"Flow shape: {flow.shape}")
    
    if flow.shape[1] != h or flow.shape[2] != w:
        print(f"\n⚠️ Warning: Flow size mismatch!")
        print(f"Expected flow shape: (2, {h}, {w})")
        print(f"Got: {flow.shape}")
    
    # Create coordinate grids
    print("\nCreating coordinate grids...")
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    
    # Apply flow to get source coordinates
    # Flow is in image coordinates (x, y)
    src_x = x_coords + flow[0]  # Add horizontal flow
    src_y = y_coords + flow[1]  # Add vertical flow
    
    # Rectify using OpenCV remap (much faster than manual interpolation)
    print("Applying rectification...")
    result_img = cv2.remap(distorted_img, src_x, src_y, 
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(255, 255, 255))
    
    # Create validity mask
    print("Creating validity mask...")
    mask = ((src_x >= 0) & (src_x < w) & 
            (src_y >= 0) & (src_y < h)).astype(np.uint8) * 255
    
    # Save results
    base_name = os.path.splitext(os.path.basename(img_path))[0].replace('_original', '')
    
    result_path = os.path.join(output_dir, f'{base_name}_corrected_hd.png')
    mask_path = os.path.join(output_dir, f'{base_name}_mask_hd.png')
    
    Image.fromarray(result_img).save(result_path, quality=95)
    Image.fromarray(mask).save(mask_path)
    
    print(f"\n✓ Corrected image saved: {result_path}")
    print(f"✓ Mask saved: {mask_path}")
    
    # Create comparison
    create_comparison_hd(img_path, result_path, mask_path, output_dir, base_name)
    
    return result_img, mask

def create_comparison_hd(distorted_path, corrected_path, mask_path, output_dir, base_name):
    """Create before/after comparison with mask"""
    
    print("\nCreating comparison visualization...")
    
    distorted = Image.open(distorted_path)
    corrected = Image.open(corrected_path)
    mask = Image.open(mask_path)
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 6))
    
    # Distorted
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(distorted)
    ax1.set_title('Distorted Input', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Corrected
    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(corrected)
    ax2.set_title('Corrected Output', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Mask
    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(mask, cmap='gray')
    ax3.set_title('Validity Mask', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, f'{base_name}_comparison_hd.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comparison saved: {comparison_path}")
    
    # Also create side-by-side comparison (2 images only)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    axes[0].imshow(distorted)
    axes[0].set_title('Distorted Input', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(corrected)
    axes[1].set_title('Corrected Output', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    comparison_simple_path = os.path.join(output_dir, f'{base_name}_comparison_simple_hd.png')
    plt.savefig(comparison_simple_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Simple comparison saved: {comparison_simple_path}")

def main():
    parser = argparse.ArgumentParser(description='HD Rectify distorted image using flow')
    parser.add_argument('--img_path', type=str, required=True, help='Path to distorted image')
    parser.add_argument('--flow_path', type=str, required=True, help='Path to flow .npy file')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("GeoProj HD - Image Rectification")
    print(f"{'='*60}")
    print(f"Input image: {args.img_path}")
    print(f"Flow file: {args.flow_path}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")
    
    rectify_image_hd(args.img_path, args.flow_path, args.output_dir)
    
    print(f"\n{'='*60}")
    print("✓ HD Rectification complete!")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()


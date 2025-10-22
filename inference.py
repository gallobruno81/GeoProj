"""
Simple inference script for GeoProj - Geometric Distortion Correction
Usage: python inference.py --input_image path/to/image.jpg --output_dir results/
"""

import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import argparse
import os
from PIL import Image
import matplotlib.pyplot as plt

from modelNetM import EncoderNet, DecoderNet, ClassNet

# Configuration
DISTORTION_TYPES = ['barrel', 'pincushion', 'rotation', 'shear', 'projective', 'wave']

def load_models(model_dir='models', use_gpu=True):
    """Load the pre-trained models"""
    
    # Initialize models
    model_en = EncoderNet([1,1,1,1,2])
    model_de = DecoderNet([1,1,1,1,2])
    model_class = ClassNet()
    
    # Load weights
    model_en_path = os.path.join(model_dir, 'model_en.pkl')
    model_de_path = os.path.join(model_dir, 'model_de.pkl')
    model_class_path = os.path.join(model_dir, 'model_class.pkl')
    
    if not all([os.path.exists(p) for p in [model_en_path, model_de_path, model_class_path]]):
        print("\nERROR: Model files not found!")
        print("Please download the pre-trained models from:")
        print("https://drive.google.com/open?id=1Tdi92IMA-rrX2ozdUMvfiN0jCZY7wIp_")
        print(f"\nPlace the .pkl files in the '{model_dir}/' directory")
        exit(1)
    
    # Load state dicts
    if use_gpu and torch.cuda.is_available():
        state_dict_en = torch.load(model_en_path)
        state_dict_de = torch.load(model_de_path)
        state_dict_class = torch.load(model_class_path)
    else:
        state_dict_en = torch.load(model_en_path, map_location='cpu')
        state_dict_de = torch.load(model_de_path, map_location='cpu')
        state_dict_class = torch.load(model_class_path, map_location='cpu')
    
    # Remove 'module.' prefix if present (from DataParallel training)
    def remove_module_prefix(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k  # remove 'module.' prefix
            new_state_dict[name] = v
        return new_state_dict
    
    state_dict_en = remove_module_prefix(state_dict_en)
    state_dict_de = remove_module_prefix(state_dict_de)
    state_dict_class = remove_module_prefix(state_dict_class)
    
    # Load cleaned state dicts
    model_en.load_state_dict(state_dict_en)
    model_de.load_state_dict(state_dict_de)
    model_class.load_state_dict(state_dict_class)
    
    # Move to GPU if available
    if use_gpu and torch.cuda.is_available():
        model_en = model_en.cuda()
        model_de = model_de.cuda()
        model_class = model_class.cuda()
    
    model_en.eval()
    model_de.eval()
    model_class.eval()
    
    print("✓ Models loaded successfully!")
    return model_en, model_de, model_class

def process_image(image_path, model_en, model_de, model_class, output_dir='results', use_gpu=True):
    """Process a single image"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    
    # Resize to 256x256 (model requirement)
    original_size = img.size
    img_resized = img.resize((256, 256), Image.BILINEAR)
    
    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    img_tensor = transform(img_resized).unsqueeze(0)
    
    if use_gpu and torch.cuda.is_available():
        img_tensor = img_tensor.cuda()
    
    # Inference
    with torch.no_grad():
        middle = model_en(img_tensor)
        flow_output = model_de(middle)
        distortion_class = model_class(middle)
        
        # Get predicted distortion type
        _, predicted_class = torch.max(distortion_class.data, 1)
        predicted_type = DISTORTION_TYPES[predicted_class.cpu().numpy()[0]]
        
        # Get flow
        flow = flow_output.cpu().numpy()[0]  # Shape: (2, 256, 256)
    
    print(f"✓ Detected distortion type: {predicted_type}")
    
    # Save flow
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    flow_path = os.path.join(output_dir, f'{base_name}_flow.npy')
    np.save(flow_path, flow)
    print(f"✓ Flow saved to: {flow_path}")
    
    # Visualize flow
    visualize_flow(flow, output_dir, base_name)
    
    # Save resized input image for resampling
    resized_path = os.path.join(output_dir, f'{base_name}_resized.jpg')
    img_resized.save(resized_path)
    print(f"✓ Resized image saved to: {resized_path}")
    
    return flow, predicted_type, resized_path, flow_path

def visualize_flow(flow, output_dir, base_name):
    """Visualize the optical flow"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # U component
    im0 = axes[0].imshow(flow[0], cmap='RdBu')
    axes[0].set_title('Flow U (horizontal)')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0])
    
    # V component
    im1 = axes[1].imshow(flow[1], cmap='RdBu')
    axes[1].set_title('Flow V (vertical)')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])
    
    # Flow magnitude
    magnitude = np.sqrt(flow[0]**2 + flow[1]**2)
    im2 = axes[2].imshow(magnitude, cmap='hot')
    axes[2].set_title('Flow Magnitude')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    flow_viz_path = os.path.join(output_dir, f'{base_name}_flow_viz.png')
    plt.savefig(flow_viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Flow visualization saved to: {flow_viz_path}")

def main():
    parser = argparse.ArgumentParser(description='GeoProj - Geometric Distortion Correction')
    parser.add_argument('--input_image', type=str, required=True, help='Path to input distorted image')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory containing model weights')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    
    args = parser.parse_args()
    
    use_gpu = not args.cpu and torch.cuda.is_available()
    
    print(f"\n{'='*60}")
    print("GeoProj - Blind Geometric Distortion Correction")
    print(f"{'='*60}")
    print(f"Device: {'GPU (CUDA)' if use_gpu else 'CPU'}")
    print(f"Input: {args.input_image}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")
    
    # Load models
    print("Loading models...")
    model_en, model_de, model_class = load_models(args.model_dir, use_gpu)
    
    # Process image
    print(f"\nProcessing image: {args.input_image}")
    flow, distortion_type, resized_path, flow_path = process_image(
        args.input_image, model_en, model_de, model_class, args.output_dir, use_gpu
    )
    
    print(f"\n{'='*60}")
    print("✓ Processing complete!")
    print(f"{'='*60}")
    print("\nTo apply the correction (rectification), use:")
    print(f"python rectify.py --img_path {resized_path} --flow_path {flow_path}")
    print("\nNote: Rectification requires CUDA GPU with numba-cuda installed")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()


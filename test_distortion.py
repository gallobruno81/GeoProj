"""
Script para generar imágenes de prueba con distorsiones sintéticas
Útil para testing sin necesidad de descargar datasets grandes
"""

import numpy as np
from PIL import Image
import os
import sys

# Añadir el directorio data al path
sys.path.append('data')
from distortion_model import distortionParameter, distortionModel

def create_test_image(size=256):
    """Crear una imagen de prueba simple con patrón de grid"""
    img = np.ones((size, size, 3), dtype=np.uint8) * 255
    
    # Añadir grid
    grid_spacing = 32
    for i in range(0, size, grid_spacing):
        img[i, :] = [0, 0, 0]
        img[:, i] = [0, 0, 0]
    
    # Añadir círculos concéntricos
    center = size // 2
    for radius in range(20, size//2, 30):
        for angle in np.linspace(0, 2*np.pi, 360):
            x = int(center + radius * np.cos(angle))
            y = int(center + radius * np.sin(angle))
            if 0 <= x < size and 0 <= y < size:
                img[y, x] = [255, 0, 0]
    
    return img

def apply_distortion(img, distortion_type):
    """Aplicar distorsión a una imagen"""
    H, W, C = img.shape
    
    # Obtener parámetros de distorsión
    params = distortionParameter(distortion_type)
    
    # Crear imagen distorsionada
    distorted = np.zeros_like(img)
    
    for yd in range(H):
        for xd in range(W):
            # Obtener coordenadas originales
            xu, yu = distortionModel(distortion_type, xd, yd, W, H, params)
            
            # Interpolar si está dentro de límites
            if 0 <= xu < W-1 and 0 <= yu < H-1:
                # Interpolación bilineal simple
                x0, y0 = int(xu), int(yu)
                x1, y1 = x0 + 1, y0 + 1
                
                if x1 < W and y1 < H:
                    dx, dy = xu - x0, yu - y0
                    
                    for c in range(C):
                        distorted[yd, xd, c] = (
                            img[y0, x0, c] * (1-dx) * (1-dy) +
                            img[y0, x1, c] * dx * (1-dy) +
                            img[y1, x0, c] * (1-dx) * dy +
                            img[y1, x1, c] * dx * dy
                        )
    
    return distorted

def generate_test_images(output_dir='test_images', size=256):
    """Generar conjunto de imágenes de prueba con diferentes distorsiones"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Tipos de distorsión
    distortion_types = ['barrel', 'pincushion', 'rotation', 'shear', 'projective', 'wave']
    
    # Crear imagen original
    print("Generando imagen de prueba...")
    original = create_test_image(size)
    original_path = os.path.join(output_dir, 'original.png')
    Image.fromarray(original).save(original_path)
    print(f"✓ Imagen original guardada: {original_path}")
    
    # Generar imagen distorsionada para cada tipo
    print("\nGenerando imágenes distorsionadas...")
    for dist_type in distortion_types:
        print(f"  - Aplicando distorsión {dist_type}...")
        distorted = apply_distortion(original, dist_type)
        
        dist_path = os.path.join(output_dir, f'distorted_{dist_type}.png')
        Image.fromarray(distorted.astype(np.uint8)).save(dist_path)
        print(f"    ✓ Guardada: {dist_path}")
    
    print(f"\n{'='*60}")
    print(f"✓ {len(distortion_types)} imágenes de prueba generadas en '{output_dir}/'")
    print(f"{'='*60}")
    print("\nPuedes usar estas imágenes para probar el modelo:")
    print(f"  python inference.py --input_image {output_dir}/distorted_barrel.png")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generar imágenes de prueba con distorsiones')
    parser.add_argument('--output_dir', type=str, default='test_images', help='Directorio de salida')
    parser.add_argument('--size', type=int, default=256, help='Tamaño de imagen (default: 256)')
    
    args = parser.parse_args()
    
    generate_test_images(args.output_dir, args.size)


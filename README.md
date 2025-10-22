# GeoProj - Blind Geometric Distortion Correction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TU-USUARIO/GeoProj/blob/master/GeoProj_Colab.ipynb)
[![Paper](https://img.shields.io/badge/arXiv-1909.03459-b31b1b.svg)](https://arxiv.org/abs/1909.03459)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Corrección ciega de distorsión geométrica en imágenes usando Deep Learning**

The source code of *Blind Geometric Distortion Correction on Images Through Deep Learning* by Li et al, CVPR 2019.

<img src='imgs/results.jpg' align="center" width=850>

---

## 🚀 Inicio Rápido

### ⚡ Ejecutar en Google Colab (Recomendado)

1. **[Abrir Notebook en Colab](https://colab.research.google.com/github/TU-USUARIO/GeoProj/blob/master/GeoProj_Colab.ipynb)**
2. Descargar [modelos pre-entrenados](https://drive.google.com/open?id=1Tdi92IMA-rrX2ozdUMvfiN0jCZY7wIp_)
3. ¡Seguir las instrucciones del notebook!

### 💻 Instalación Local

```bash
git clone https://github.com/TU-USUARIO/GeoProj.git
cd GeoProj
pip install -r requirements.txt
```

**[Ver Guía de Inicio Rápido Completa →](QUICKSTART.md)**

---

## 📚 Documentación

- **[QUICKSTART.md](QUICKSTART.md)** - Guía de inicio rápido (5 minutos)
- **[README_USAGE.md](README_USAGE.md)** - Documentación completa y ejemplos
- **[INSTRUCCIONES_GITHUB.md](INSTRUCCIONES_GITHUB.md)** - Configurar GitHub y Colab
- **[GeoProj_Colab.ipynb](GeoProj_Colab.ipynb)** - Notebook interactivo para Colab

---

## 🎯 Características

- ✅ **Fácil de usar**: Scripts simplificados para inferencia
- ✅ **Google Colab**: Notebook listo para usar con GPU gratis
- ✅ **6 tipos de distorsión**: Barrel, Pincushion, Rotation, Shear, Projective, Wave
- ✅ **Modelos pre-entrenados**: Descarga y usa inmediatamente
- ✅ **Generador de pruebas**: Crea imágenes distorsionadas para testing

---

## 📦 Requisitos

- Python 3.7+
- PyTorch 1.9+
- CUDA 10.2+ (opcional, para GPU)
- Ver [requirements.txt](requirements.txt) para lista completa

---

## 💡 Uso Básico

### Procesar una Imagen

```bash
# Procesar imagen distorsionada
python inference.py --input_image imagen_distorsionada.jpg

# Aplicar corrección (requiere GPU)
python rectify.py --img_path results/imagen_resized.jpg --flow_path results/imagen_flow.npy
```

### Generar Imágenes de Prueba

```bash
python test_distortion.py
# Genera imágenes de prueba en test_images/
```

---

## 📖 Documentación Original

## Prerequisites
- Linux or Windows
- Python 3.7+
- CPU or NVIDIA GPU + CUDA CuDNN

---

## 🔬 Para Investigadores y Desarrolladores

### Dataset Generation
In order to train the model using the provided code, the data needs to be generated in a certain manner. 

You can use any distortion-free images to generate the dataset. In this paper, we use [Places365-Standard dataset](http://places2.csail.mit.edu/download.html) at the resolution of 512\*512 as the original non-distorted images to generate the 256\*256 dataset.

Run the following command for dataset generation:
```bash
python data/dataset_generate.py [--sourcedir [PATH]] [--datasetdir [PATH]] 
                                [--trainnum [NUMBER]] [--testnum [NUMBER]]

--sourcedir           Path to original non-distorted images
--datasetdir          Path to the generated dataset
--trainnum            Number of generated training samples
--testnum             Number of generated testing samples
```

### Training
Run the following command for help message about optional arguments like learning rate, dataset directory, etc.
```bash
python trainNetS.py --h # if you want to train GeoNetS
python trainNetM.py --h # if you want to train GeoNetM
```

### Use a Pre-trained Model

**Download:** [Pre-trained models](https://drive.google.com/open?id=1Tdi92IMA-rrX2ozdUMvfiN0jCZY7wIp_) (~300 MB)

Place the `.pkl` files in a `models/` directory.

**Usage:**
```bash
python inference.py --input_image your_image.jpg --model_dir models/
```

### Resampling

Import `resample.resampling.rectification` function to resample the distorted image by the forward flow.

```python
from resample.resampling import rectification
import numpy as np
from PIL import Image

# Load distorted image and flow
distorted_img = np.array(Image.open('distorted.jpg'))
flow = np.load('flow.npy')

# Apply rectification
corrected_img, mask = rectification(distorted_img, flow)
```

The distorted image should be a Numpy array with the shape of H\*W\*3 for a color image or H\*W for a greyscale image. The forward flow should be an array with the shape of 2\*H\*W.

The function returns the resulting image and a mask to indicate whether each pixel converged within the maximum iteration.

---

## 🎓 Archivos Incluidos

### Scripts Principales
- `inference.py` - Procesar imágenes (detección + flujo)
- `rectify.py` - Aplicar corrección
- `test_distortion.py` - Generar imágenes de prueba

### Notebooks
- `GeoProj_Colab.ipynb` - Notebook para Google Colab

### Documentación
- `QUICKSTART.md` - Inicio rápido
- `README_USAGE.md` - Documentación completa
- `INSTRUCCIONES_GITHUB.md` - Setup de GitHub/Colab

### Original
- `modelNetM.py` / `modelNetS.py` - Arquitecturas de red
- `trainNetM.py` / `trainNetS.py` - Scripts de entrenamiento
- `eval.py` - Evaluación (original)
- `data/` - Generación de datasets
- `resample/` - Motor de rectificación CUDA

---
## 📄 Citation

Si usas este código en tu investigación, por favor cita:

```bibtex
@inproceedings{li2019blind,
  title={Blind Geometric Distortion Correction on Images Through Deep Learning},
  author={Li, Xiaoyu and Zhang, Bo and Sander, Pedro V and Liao, Jing},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={4855--4864},
  year={2019}
}
```

---

## 🔗 Enlaces

- 📄 **Paper:** https://arxiv.org/abs/1909.03459
- 💾 **Modelos:** https://drive.google.com/open?id=1Tdi92IMA-rrX2ozdUMvfiN0jCZY7wIp_
- 🐙 **Repo Original:** https://github.com/xiaoyu258/GeoProj
- 📓 **Colab Notebook:** [Abrir en Colab](https://colab.research.google.com/github/TU-USUARIO/GeoProj/blob/master/GeoProj_Colab.ipynb)

---

## 📝 Licencia

MIT License - Ver [LICENSE](LICENSE) para detalles

---

## 🤝 Contribuciones

Las contribuciones son bienvenidas! Por favor:

1. Fork el repositorio
2. Crea una rama (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

## ⭐ Agradecimientos

- Autores originales: [Xiaoyu Li](https://github.com/xiaoyu258) et al.
- Paper: CVPR 2019

---

**¿Tienes preguntas?** Abre un [issue](../../issues) en GitHub

**¿Te gusta el proyecto?** Dale una ⭐ en GitHub!

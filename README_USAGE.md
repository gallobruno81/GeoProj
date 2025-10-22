# GeoProj - Guía de Uso Completa

Corrección ciega de distorsión geométrica en imágenes usando Deep Learning.

## 🚀 Inicio Rápido

### Opción 1: Google Colab (Recomendado)

1. Abre el notebook en Colab:
   - Sube `GeoProj_Colab.ipynb` a Google Drive
   - Ábrelo con Google Colab
   - O usa: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](tu-enlace-aqui)

2. Descarga los modelos pre-entrenados:
   - [Descargar modelos](https://drive.google.com/open?id=1Tdi92IMA-rrX2ozdUMvfiN0jCZY7wIp_)
   - Necesitas: `model_en.pkl`, `model_de.pkl`, `model_class.pkl`

3. Sigue las celdas del notebook paso a paso

### Opción 2: Instalación Local

#### Requisitos
- Python 3.7+
- PyTorch 1.9+
- CUDA 10.2+ (para GPU)

#### Instalación

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/GeoProj.git
cd GeoProj

# Instalar dependencias
pip install -r requirements.txt

# Crear carpeta para modelos
mkdir models
```

#### Descargar Modelos

Descarga los modelos pre-entrenados desde [aquí](https://drive.google.com/open?id=1Tdi92IMA-rrX2ozdUMvfiN0jCZY7wIp_) y colócalos en la carpeta `models/`

## 📝 Uso

### Procesar una Imagen

```bash
python inference.py --input_image tu_imagen.jpg --output_dir results
```

Esto generará:
- `*_flow.npy`: Flujo óptico calculado
- `*_flow_viz.png`: Visualización del flujo
- `*_resized.jpg`: Imagen redimensionada a 256x256

### Aplicar Rectificación (Requiere CUDA)

```bash
python rectify.py --img_path results/tu_imagen_resized.jpg --flow_path results/tu_imagen_flow.npy
```

Esto generará:
- `*_corrected.png`: Imagen corregida
- `*_mask.png`: Máscara de convergencia
- `*_comparison.png`: Comparación antes/después

## 🎯 Tipos de Distorsión

El modelo detecta y corrige 6 tipos de distorsión:

1. **Barrel** (Barril) - Común en lentes gran angular
2. **Pincushion** (Cojín) - Común en teleobjetivos
3. **Rotation** (Rotación) - Imagen rotada
4. **Shear** (Cizallamiento) - Deformación por cizalla
5. **Projective** (Proyectiva) - Distorsión de perspectiva
6. **Wave** (Onda) - Distorsión ondulatoria

## 🔧 Opciones Avanzadas

### inference.py

```bash
python inference.py \
    --input_image imagen.jpg \
    --output_dir resultados \
    --model_dir modelos \
    --cpu  # Usar CPU en lugar de GPU
```

### rectify.py

```bash
python rectify.py \
    --img_path results/imagen_resized.jpg \
    --flow_path results/imagen_flow.npy \
    --output_dir resultados
```

## 📊 Generar Dataset de Prueba

Si quieres entrenar o probar con imágenes sintéticas:

```bash
python data/dataset_generate.py \
    --sourcedir /path/to/images \
    --datasetdir /path/to/dataset \
    --trainnum 10000 \
    --testnum 2000
```

## 🐛 Solución de Problemas

### Error: "Model files not found"
- Descarga los modelos pre-entrenados
- Colócalos en la carpeta `models/`
- Verifica los nombres: `model_en.pkl`, `model_de.pkl`, `model_class.pkl`

### Error: "CUDA not available"
- La rectificación requiere GPU con CUDA
- Para CPU: solo puedes generar el flujo (inference.py)
- El flujo puede procesarse después en una máquina con GPU

### Error: "numba cuda not available"
- Instala numba con soporte CUDA:
  ```bash
  conda install numba cudatoolkit
  ```

### Imagen borrosa o artefactos
- El modelo funciona mejor con imágenes de 256x256
- Imágenes muy grandes se redimensionan y pueden perder calidad
- Prueba con imágenes de mejor resolución original

## 📚 Estructura del Proyecto

```
GeoProj/
├── data/
│   ├── dataset_generate.py    # Generador de dataset
│   └── distortion_model.py    # Modelos de distorsión
├── resample/
│   └── resampling.py          # Rectificación con CUDA
├── models/                     # Modelos pre-entrenados
│   ├── model_en.pkl
│   ├── model_de.pkl
│   └── model_class.pkl
├── modelNetM.py               # Arquitectura del modelo
├── modelNetS.py               # Arquitectura simplificada
├── inference.py               # Script de inferencia
├── rectify.py                 # Script de rectificación
├── GeoProj_Colab.ipynb       # Notebook de Colab
└── requirements.txt           # Dependencias
```

## 🎓 Entrenamiento

Para entrenar tu propio modelo:

```bash
# Generar dataset
python data/dataset_generate.py --sourcedir /path/to/images --datasetdir dataset

# Entrenar modelo pequeño
python trainNetS.py --datasetdir dataset --epochs 100

# Entrenar modelo grande
python trainNetM.py --datasetdir dataset --epochs 100
```

## 📖 Citación

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

## 📄 Licencia

MIT License - Ver [LICENSE](LICENSE) para más detalles

## 🔗 Enlaces

- **Paper:** https://arxiv.org/abs/1909.03459
- **Repositorio Original:** https://github.com/xiaoyu258/GeoProj
- **Modelos Pre-entrenados:** https://drive.google.com/open?id=1Tdi92IMA-rrX2ozdUMvfiN0jCZY7wIp_

## 💡 Tips

1. **Calidad de imagen:** Mejores resultados con imágenes claras y bien iluminadas
2. **Tamaño:** El modelo espera 256x256, pero puedes procesar imágenes más grandes (se redimensionan)
3. **GPU:** Usa GPU para mejor velocidad (5-10x más rápido que CPU)
4. **Batch processing:** Procesa múltiples imágenes modificando `inference.py`

## 🤝 Contribuciones

Las contribuciones son bienvenidas! Por favor:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📧 Contacto

Para preguntas o issues, abre un issue en GitHub.

---

**Desarrollado con ❤️ usando PyTorch**


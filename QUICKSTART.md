# 🚀 Inicio Rápido - GeoProj

## ⚡ Opción 1: Google Colab (Más Fácil)

1. **Abre el notebook:**
   - Sube `GeoProj_Colab.ipynb` a Google Drive
   - Ábrelo con Google Colab
   - Activa GPU: `Runtime` → `Change runtime type` → `GPU`

2. **Descarga los modelos:**
   - [Click aquí para descargar](https://drive.google.com/open?id=1Tdi92IMA-rrX2ozdUMvfiN0jCZY7wIp_)
   - Necesitas 3 archivos: `model_en.pkl`, `model_de.pkl`, `model_class.pkl`

3. **Ejecuta las celdas:**
   - Sigue las instrucciones del notebook
   - Sube tus imágenes distorsionadas
   - ¡Obtén resultados en minutos!

---

## 💻 Opción 2: Local (Más Control)

### Paso 1: Instalación
```bash
git clone https://github.com/tu-usuario/GeoProj.git
cd GeoProj
pip install -r requirements.txt
mkdir models
```

### Paso 2: Descargar Modelos
Descarga desde [Google Drive](https://drive.google.com/open?id=1Tdi92IMA-rrX2ozdUMvfiN0jCZY7wIp_) y coloca los archivos `.pkl` en `models/`

### Paso 3: Procesar una Imagen
```bash
# Generar imágenes de prueba (opcional)
python test_distortion.py

# Procesar imagen distorsionada
python inference.py --input_image tu_imagen.jpg

# Aplicar corrección (requiere GPU con CUDA)
python rectify.py --img_path results/tu_imagen_resized.jpg --flow_path results/tu_imagen_flow.npy
```

---

## 📁 ¿Qué Archivos Necesito?

### Esenciales:
- ✅ `inference.py` - Detecta distorsión y calcula flujo
- ✅ `rectify.py` - Aplica corrección
- ✅ `modelNetM.py` - Arquitectura del modelo
- ✅ `resample/resampling.py` - Motor de rectificación
- ✅ `models/` - Carpeta con 3 archivos `.pkl`

### Opcionales:
- 📓 `GeoProj_Colab.ipynb` - Para usar en Colab
- 🧪 `test_distortion.py` - Generar imágenes de prueba
- 📖 `README_USAGE.md` - Documentación completa

---

## 🎯 Ejemplo Completo

```bash
# 1. Clonar
git clone https://github.com/tu-usuario/GeoProj.git
cd GeoProj

# 2. Instalar
pip install -r requirements.txt

# 3. Descargar modelos y colocar en models/

# 4. Generar imagen de prueba
python test_distortion.py

# 5. Procesar
python inference.py --input_image test_images/distorted_barrel.png

# 6. Ver resultados
ls results/
```

---

## ❓ FAQ Rápido

**Q: ¿Necesito GPU?**  
A: Para inferencia no es necesario, pero es mucho más rápido. Para rectificación sí necesitas GPU con CUDA.

**Q: ¿Qué tamaño de imagen?**  
A: El modelo funciona con 256x256. Imágenes más grandes se redimensionan automáticamente.

**Q: ¿Dónde están los modelos?**  
A: [Aquí en Google Drive](https://drive.google.com/open?id=1Tdi92IMA-rrX2ozdUMvfiN0jCZY7wIp_) - Son ~300MB

**Q: Error "CUDA not available"?**  
A: Puedes generar el flujo sin GPU, pero necesitas GPU para la rectificación final.

---

## 🔗 Links Útiles

- 📄 [Paper Original](https://arxiv.org/abs/1909.03459)
- 💾 [Modelos Pre-entrenados](https://drive.google.com/open?id=1Tdi92IMA-rrX2ozdUMvfiN0jCZY7wIp_)
- 📚 [Documentación Completa](README_USAGE.md)
- 🐙 [Repo Original](https://github.com/xiaoyu258/GeoProj)

---

**¿Problemas?** Abre un issue en GitHub 🐛


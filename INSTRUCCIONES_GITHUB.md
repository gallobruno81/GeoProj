# 📤 Instrucciones para Subir a GitHub y Ejecutar en Colab

## 🔄 Paso 1: Configurar tu Repositorio de GitHub

### Opción A: Crear un Nuevo Repositorio

1. Ve a GitHub: https://github.com/new
2. Crea un repositorio llamado `GeoProj`
3. **NO** inicialices con README (ya tienes archivos locales)
4. Copia la URL del repositorio (ej: `https://github.com/tu-usuario/GeoProj.git`)

### Opción B: Cambiar el Remote Actual

Si ya clonaste el repositorio original, cambia el remote:

```bash
# Ver remote actual
git remote -v

# Cambiar remote a tu repositorio
git remote set-url origin https://github.com/TU-USUARIO/GeoProj.git

# Verificar
git remote -v
```

---

## 📤 Paso 2: Subir los Cambios

```bash
# Push a tu GitHub
git push -u origin master
```

Si te pide autenticación, usa tu token de GitHub (no contraseña).

---

## 🚀 Paso 3: Ejecutar en Google Colab

### Método 1: Subir Notebook Manualmente

1. Ve a Google Colab: https://colab.research.google.com
2. `File` → `Upload notebook`
3. Sube `GeoProj_Colab.ipynb`
4. O arrastra el archivo directamente

### Método 2: Abrir desde GitHub (Recomendado)

1. En Colab: `File` → `Open notebook`
2. Selecciona la pestaña `GitHub`
3. Ingresa: `tu-usuario/GeoProj`
4. Selecciona `GeoProj_Colab.ipynb`

### Método 3: Link Directo

Crea un badge en tu README con:

```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TU-USUARIO/GeoProj/blob/master/GeoProj_Colab.ipynb)
```

Reemplaza `TU-USUARIO` con tu usuario de GitHub.

---

## ⚙️ Paso 4: Configurar Colab

1. **Activar GPU:**
   - `Runtime` → `Change runtime type` → `Hardware accelerator: GPU`

2. **Ejecutar Primera Celda:**
   - Verifica que detecte la GPU con `!nvidia-smi`

3. **Seguir las Celdas:**
   - Instala dependencias
   - Clona tu repositorio (cambia la URL en el notebook)
   - Sube los modelos
   - Procesa imágenes

---

## 📥 Paso 5: Descargar los Modelos Pre-entrenados

**IMPORTANTE:** Los modelos no están en GitHub (son muy grandes)

1. Descarga desde: https://drive.google.com/open?id=1Tdi92IMA-rrX2ozdUMvfiN0jCZY7wIp_

2. Deberías obtener 3 archivos:
   - `model_en.pkl` (~100 MB)
   - `model_de.pkl` (~100 MB)
   - `model_class.pkl` (~50 MB)

3. En Colab, usa la celda de upload para subirlos

---

## 🔄 Paso 6: Workflow de Desarrollo

### Flujo Completo:

```bash
# 1. Hacer cambios localmente
code inference.py  # o cualquier archivo

# 2. Probar localmente
python inference.py --input_image test.jpg

# 3. Commit cambios
git add .
git commit -m "Descripción de cambios"

# 4. Push a GitHub
git push origin master

# 5. En Colab, re-clonar o hacer pull
# En una celda de Colab:
!git pull origin master
```

---

## 🧪 Paso 7: Probar Todo el Sistema

### En Local:

```bash
# Generar imágenes de prueba
python test_distortion.py

# Procesar
python inference.py --input_image test_images/distorted_barrel.png

# Ver resultados
ls results/
```

### En Colab:

1. Ejecuta todas las celdas del notebook
2. Sube una imagen de prueba
3. Descarga los resultados al final

---

## 📊 Paso 8: Monitorear Ejecuciones en Colab

### Ver Output:

- Cada celda muestra su output directamente
- Los archivos se guardan en `/content/GeoProj/results/`

### Ver Archivos Generados:

```python
# En una celda de Colab
!ls -lh results/
```

### Descargar Resultados:

```python
# En una celda de Colab
from google.colab import files
files.download('results/mi_imagen_corrected.png')
```

O descarga todo comprimido (ya está en el notebook):

```python
!zip -r results.zip results/
files.download('results.zip')
```

---

## 🔧 Configuraciones Adicionales

### Actualizar URL del Repo en el Notebook:

Edita la celda 5 del notebook:

```python
REPO_URL = "https://github.com/TU-USUARIO/GeoProj.git"  # ← Cambia esto
```

### Crear Badge de Colab para tu README:

Añade esto al inicio de tu `README.md`:

```markdown
# GeoProj

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TU-USUARIO/GeoProj/blob/master/GeoProj_Colab.ipynb)

...resto del README...
```

---

## 🐛 Solución de Problemas

### "Permission denied" al hacer push

```bash
# Configura tu token de GitHub
git config --global credential.helper store
git push
# Ingresa tu username y token (no contraseña)
```

### El notebook no encuentra archivos

```bash
# En Colab, verifica que estás en el directorio correcto
!pwd
!ls

# Si no estás en GeoProj:
%cd GeoProj
```

### Cambios no aparecen en Colab

```bash
# En Colab, después de hacer push:
%cd GeoProj
!git pull origin master
```

### Colab se queda sin memoria

- Usa imágenes más pequeñas
- Reinicia el runtime: `Runtime` → `Restart runtime`
- Limpia outputs: `Edit` → `Clear all outputs`

---

## 📱 Acceder desde Móvil

1. Abre Colab en tu móvil: https://colab.research.google.com
2. Abre tu notebook desde GitHub
3. Ejecuta las celdas (puede ser más lento)

---

## 🎯 Resumen de URLs Importantes

- **Tu GitHub Repo:** `https://github.com/TU-USUARIO/GeoProj`
- **Colab Directo:** `https://colab.research.google.com/github/TU-USUARIO/GeoProj/blob/master/GeoProj_Colab.ipynb`
- **Modelos Pre-entrenados:** `https://drive.google.com/open?id=1Tdi92IMA-rrX2ozdUMvfiN0jCZY7wIp_`
- **Paper Original:** `https://arxiv.org/abs/1909.03459`

---

## ✅ Checklist Final

- [ ] Repositorio creado en GitHub
- [ ] Remote configurado correctamente
- [ ] Código subido con `git push`
- [ ] Notebook abierto en Colab
- [ ] GPU activada en Colab
- [ ] Modelos descargados de Drive
- [ ] Modelos subidos a Colab
- [ ] Primera imagen procesada exitosamente
- [ ] Resultados descargados

---

**¿Listo?** ¡Ahora puedes correr GeoProj en cualquier parte! 🎉

Para más detalles, ve:
- `QUICKSTART.md` - Inicio rápido
- `README_USAGE.md` - Documentación completa


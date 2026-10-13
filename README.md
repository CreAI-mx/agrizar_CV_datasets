# agrizar_cv_datasets

Espacio de investigación de datasets (no es una aplicación) para **Agrizar**.

## Alcance

Esta investigación **solo reúne datos de fuentes públicas distintas**. No entrena el modelo final de Agrizar. Sirve para dos usos:

1. **Preentrenar** modelos de visión por computadora de Agrizar con datos públicos.
2. **Ampliar** los datos que Agrizar aporte por su cuenta, con imágenes y etiquetas de otras fuentes.

Por eso, para cada dataset registramos qué tan utilizable es: licencia, calidad de las etiquetas, duplicados, fugas de datos entre particiones y sesgos de captura. El resultado son dos tipos de producto:

- **Reportes de hallazgos** que dicen qué dataset conviene usar y qué trampas tiene.
- **Datasets normalizados por cultivo**, con un solo vocabulario de clases y etiquetas verificadas, listos para entrenar.

Los datos propios de Agrizar no están en este repositorio. Cuando lleguen, el formato de `normalized/` es el que deben seguir para mezclarse con estas fuentes.

### Qué estudiamos

Dos temas, con énfasis en **frutos rojos** (fresa, arándano, frambuesa), además de **tomate, pimiento y pepino** (ambos temas) y **crecimiento hidropónico** (solo crecimiento):

- **Plaga:** detección de enfermedades, insectos plaga (*plaga*) y deficiencias nutricionales, tanto en general (todos los cultivos) como en frutos rojos. No se descarta un dataset general solo porque cubra poco los frutos rojos.
- **Crecimiento:** etapas de crecimiento y madurez de frutos.

Cada estudio es un notebook (una sección por dataset) y un markdown de hallazgos escrito como entrega al equipo.

## Estudios

| Estudio | Notebook | Hallazgos | Gráficas | Ejemplos de imágenes |
|---|---|---|---|---|
| Enfermedades, plagas y deficiencias (general + frutos rojos) | `plague.ipynb` | [`PLAGUE_DATASET_FINDINGS.md`](PLAGUE_DATASET_FINDINGS.md) | `charts/plague_*.png` | `examples/plague/` |
| Crecimiento / etapas de madurez | `growth.ipynb` | [`GROWTH_DATASET_FINDINGS.md`](GROWTH_DATASET_FINDINGS.md) | `charts/growth_*.png` | `examples/growth/` |

**Empieza por los archivos de hallazgos.** Todas las cifras salen de la salida ejecutada de los notebooks. Los reportes están escritos en inglés.

### Datasets revisados

- **Plaga (12):** PlantVillage, LeafNet2.0 (con acceso restringido), PlantSeg v3, PlantDoc, IP102 (AgML `CT_PestNet`), CCMT, Strawberry_12, AgML detección de enfermedades en hojas de tomate (Imtiaz), AgML detección de enfermedades de tomate (Liu), AgML COLD chile, AgML enfermedades de pepino, Mendeley detección en hojas de pepino.
- **Crecimiento (17):** AgML crecimiento de fresa, Roboflow fresa 5 etapas, AgML fresa/aguacate, AgML madurez de arándano, segmentación de arándano, Roboflow frambuesa, Roboflow crecimiento de chile, GrowliFlower, AgML crecimiento de mango, AgML conteo de hojas, Laboro tomate, AgRob tomate de invernadero, madurez de tomate, madurez de pimiento dulce, lechuga hidropónica AGC 2021, HydroGrowNet, AGC 2018 pepino.

Los datasets descartados y sus razones están en los archivos de hallazgos y en `archive/PLAGUE_DATASET_FINDINGS_v1.md`.

## Datasets normalizados por cultivo

El notebook maestro `normalization/normalize_all.ipynb` (una sección por cultivo: fresa, arándano, frambuesa, tomate, pimiento, pepino; más hidroponía) mezcla las fuentes revisadas en `normalized/<cultivo>/`. El motor compartido está en `tools/cvlib/`:

```
normalized/<cultivo>/
├── images/
├── labels/
│   ├── classification/{condition,maturity}.csv
│   ├── detection/{condition,maturity}/*.txt      # cajas YOLO, normalizadas
│   └── segmentation/{condition,maturity}/*.txt   # polígonos YOLO-seg, normalizados
└── manifest.csv                                  # licencia, group, label_space y dhash por imagen
```

Reglas que hay que respetar al entrenar:

- **Dividir por `group`, nunca por imagen** (clave anti-fuga en `manifest.csv`).
- `condition` (estado sanitario) y `maturity` (madurez) son espacios de etiquetas separados, cada uno con su `classes.txt`: se entrenan como modelos distintos.
- `label_space` lista las clases que la fuente de la imagen sí anota. Las demás se enmascaran, no se tratan como fondo.
- En detección, un archivo vacío es un negativo verificado (solo en `condition/`); sin archivo significa que no fue anotada.
- "Sano" solo viene de una etiqueta explícita, nunca de la ausencia de cajas.
- Las etiquetas que no se pueden confiar ni reconciliar (por ejemplo, casi duplicados que se contradicen) se descartan y se cuentan en el notebook.
- `prelabel/` guarda etiquetas sin verificar y nunca debe mezclarse en `normalized/` antes de revisarse.
- Respeta la licencia de cada imagen (columna `license` del manifiesto). Algunas fuentes son no comerciales, así que hay que revisarla antes de usar los datos en un producto.

Las figuras de verificación (etiquetas leídas de vuelta desde disco) están en `examples/normalized/`.

## Estructura del repositorio

| Ruta | Contenido |
|---|---|
| `plague.ipynb`, `growth.ipynb` | Notebooks de los estudios (generados) |
| `normalization/` | Notebook maestro `normalize_all.ipynb` (generado; hace chdir a la raíz del repo) |
| `tools/cvlib/` | Código compartido: motor de normalización, loaders, hidroponía |
| `Makefile`, `requirements.txt` | Comandos del equipo y dependencias (Python 3.12) |
| `*_DATASET_FINDINGS.md` | Reportes de hallazgos para el equipo |
| `tools/` | Generadores de notebooks y herramientas de etiquetado |
| `charts/`, `examples/` | Gráficas e imágenes de muestra con etiquetas dibujadas |
| `normalized/` | Datasets por cultivo ya mezclados |
| `data/` | Descargas locales (≈8.6 GB), no se comparten |
| `archive/` | `lol.ipynb` original y hallazgos v1. No editar |
| `CLAUDE.md` | Convenciones detalladas y trampas conocidas de los datasets |

## Configuración

- Python 3.12 (entorno pyenv `p312`) con `datasets`, `huggingface_hub`, `pyarrow`, `pandas` 2.3, `matplotlib`, `pillow`, `nbformat`, `jupyter`, `torch` (CUDA), `ultralytics` (solo para el preetiquetado de pimiento) y `tkinter` (etiquetador).
- Se requiere sesión en Hugging Face. LeafNet2.0 tiene acceso restringido (formulario de uso no comercial): cada integrante debe aceptarlo por su cuenta.
- Los datos de Roboflow se obtienen con el conector MCP de Roboflow (workspace `veggies-h2dsa`). Las exportaciones públicas de Universe están rotas, así que la ruta es: fork → versión sin preprocesar → exportación COCO. La cuota gratuita está agotada.
- La mayoría de los datos vive en la caché de Hugging Face (`~/.cache/huggingface`); el resto está en `data/`.

## Regenerar los notebooks

Los notebooks son **generados**: edita el generador en `tools/` (o el código compartido en `tools/cvlib/`), luego reconstruye y ejecuta desde la raíz del repo. Nunca edites un notebook a mano.

```bash
pip install -r requirements.txt   # una vez (Python 3.12)
make setup                        # una vez por clon: git guarda los notebooks sin salidas
make growth                       # crecimiento (~1.5 min con caché caliente)
make plague                       # plaga (~15 min)
make normalize                    # datasets normalizados + hidroponía (~45 min, en segundo plano)
make normalize CROPS="tomato cucumber"   # solo algunos cultivos (hidroponía: `hydroponics`)
make normalize-safe               # un notebook por cultivo, uno a la vez (menos RAM)
```

Los notebooks no se versionan con salidas (filtro de git, `make setup`): tu copia local las conserva, pero el repositorio guarda solo el código. Los hallazgos (`*_FINDINGS.md`), `charts/` y `examples/` son el registro legible.

Notas:

- Ejecutar varios kernels de normalización en paralelo agota la RAM. El notebook maestro ya corre los cultivos en secuencia.
- Cada ejecución borra y reconstruye la carpeta de su cultivo. Si una corrida falla, `images/` queda sin etiquetas: repite ese cultivo.
- Cada notebook normalizado termina con `check()`, que lee todos los archivos desde disco y verifica el formato.
- Tras cualquier cambio, confirma que ninguna celda tenga salida `error` y contrasta el markdown de hallazgos con la salida del notebook. Copia el archivo de hallazgos anterior a `archive/` antes de reemplazarlo por completo.

## Herramientas de etiquetado (pimiento)

- `tools/prelabel_bell_pepper_seg.py`: ajusta yolo11n-seg con las etiquetas de segmentación normalizadas de pimiento y preetiqueta imágenes de pimiento de PlantDoc en `prelabel/bell_pepper/plantdoc_prelabels/` (etiquetas, overlays y `review.csv`). mAP50 de máscara en validación: 0.55, con muchas máscaras falsas y seguras sobre suelo/acolchado.
- `tools/seg_labeler.py`: etiquetador de polígonos con tkinter para archivos YOLO-seg. Escribe solo en `--out`; un archivo ahí significa revisado y un archivo vacío significa revisado sin etiquetas. Admite `--match`, `--review review.csv`, `--export-boxes DIR` y `--selftest`. Ver ejemplos en su docstring.

## Trampas conocidas

La lista completa está en `CLAUDE.md` y en los archivos de hallazgos. Las más frecuentes:

- Los nombres de clase en los parquet de AgML pueden diferir de la ficha del dataset: usa la ficha.
- Los datasets de origen Roboflow contienen copias aumentadas: divide por el nombre de la foto original.
- LeafNet2.0 ⊃ PlantVillage y PlantWild ⊃ PlantSeg: elimina duplicados antes de combinarlos.
- Varios datasets no traen nombres de clase (Strawberry_12, PlantDoc, fresa/aguacate). Los mapeos recuperados están en los generadores.
- pandas 2.3 se cierra con segfault ante cadenas hexadecimales como `1e00…`: los dHash se escriben con prefijo `0x`.
- El dHash de 64 bits da falsos casi duplicados en escenas de cámara fija: confirma con un hash de 256 bits.

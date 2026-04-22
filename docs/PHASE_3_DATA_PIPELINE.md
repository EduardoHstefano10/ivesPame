# Fase 3 — Procesamiento de datos ENDES 2024

Documento tecnico del pipeline que transforma los 13 modulos crudos de la
ENDES 2024 en el dataset analitico usado por el modelo de desnutricion
cronica (`models/desnutricion_model_v1.joblib`) y la app Streamlit.

El pipeline vive en dos scripts complementarios dentro de `scripts/`:

1. `unificar_datos.py` → `data/processed/endes_2024_unificado.csv`
2. `procesar_mvp.py`   → `data/processed/mvp_dataset_fase3_limpio.csv`

Reproducir todo:

```bash
python scripts/unificar_datos.py
python scripts/procesar_mvp.py
```

---

## 1. Entradas (raw)

Los 13 modulos INEI se encuentran en `data/raw/968-Modulo16XX/`:

| Granularidad | Modulos                                                                 | Llave          |
|--------------|-------------------------------------------------------------------------|----------------|
| Niño < 5     | REC44 (antropometria), REC41 (embarazo/parto), REC43 (vacunas), DIT     | `CASEID + NINIO_IDX` |
| Mujer        | REC0111, REC91, RE223132, REC42, RE516171, RE758081, REC84DV, REC93DV   | `CASEID`       |
| Hogar        | RECH0, RECH23, CSALUD01, Programas Sociales x Hogar                     | `HHID`         |
| Programas    | PS_BECA18, PS_COMEDOR, PS_PENSION65, PS_QALIWARMA, PS_TRABAJA, PS_VL, PS_WAWAWASI | `HHID` |

Nota sobre ENDES: `MIDX`, `HIDX` y `HWIDX` refieren al mismo niño de la
historia reciente, por lo que se unifican a `NINIO_IDX`.

---

## 2. Script 1 — `scripts/unificar_datos.py`

**Objetivo:** construir una Analytic Base Table (ABT) a nivel de niño
menor de 5 años.

### 2.1. Estrategia

1. **Carga como `str`** para preservar ceros a la izquierda y evitar
   inferencias erroneas de tipo.
2. **Limpieza consistente**: elimina BOM, recorta espacios y convierte
   codigos missing DHS (`96, 97, 98, 99, 996, 998, 9998, ...`) a `NaN`
   via `to_numeric_clean()`.
3. **Ensamblaje jerarquico** niño → madre → hogar → programas.
4. **Programas sociales** aplanados a flags binarias `programa_*` por
   hogar + `programas_sociales_total`.
5. **Feature engineering** y mapeos a palabras (zona, agua, saneamiento,
   riqueza, educacion_madre, sexo_nino).
6. **Target binario** `Desnutricion_Cronica = 1` si `HAZ < -2` (OMS).

### 2.2. Bloques

#### Bloque niño (`build_child_block`)

- Base: `REC44` con `HWIDX` renombrado a `NINIO_IDX`.
- Merge izquierdo con REC41 (`MIDX`), REC43 (`HIDX`) y DIT (`BIDX`),
  todos realineados a `NINIO_IDX`.

#### Bloque madre (`build_woman_block`)

- Base: `REC0111` (contiene `CASEID` y `HHID`, puente hacia el hogar).
- Merge izquierdo con REC91, RE223132, REC42, RE516171, RE758081,
  REC84DV, REC93DV por `CASEID`.

#### Bloque hogar (`build_household_block`)

- Base: `RECH0` por `HHID`.
- Merge izquierdo con RECH23, PS_HOGAR y CSALUD01.

#### Programas sociales (`build_programs_flags`)

- Por cada CSV de programa: `HHID` → flag `programa_<nombre> = 1`.
- `outer join` de todas las flags; los nulos se rellenan con `0`
  (hogar sin programa). Se agrega `programas_sociales_total`.

### 2.3. Feature engineering

| Columna creada         | Formula / origen                        |
|------------------------|------------------------------------------|
| `haz_score`            | `C44_HW70 / 100`                         |
| `Desnutricion_Cronica` | `1 si haz_score < -2, sino 0`            |
| `edad_meses`           | `C44_HW1`                                |
| `sexo_nino`            | `REC41_B4` → `{1: Hombre, 2: Mujer}`     |
| `peso_nacer_kg`        | `REC41_M19 / 1000`                       |
| `talla_madre_cm`       | `REC42_V438 / 10`                        |
| `peso_madre_kg`        | `REC42_V437 / 10`                        |
| `zona`                 | `H0_HV025` → `{1: Urbano, 2: Rural}`     |
| `agua`                 | `RECH23_HV201` → `MAPA_AGUA`             |
| `saneamiento`          | `RECH23_HV205` → `MAPA_SANEO`            |
| `riqueza`              | `RECH23_HV270` → `MAPA_RIQUEZA`          |
| `educacion_madre`      | `M01_V106` → `MAPA_EDUCACION`            |
| `edad_madre`           | `M01_V012`                               |
| `LLAVE_NINIO`          | `CASEID + "_" + NINIO_IDX`               |

### 2.4. Salida

- Archivo: `data/processed/endes_2024_unificado.csv`
- Dimension: **19 751 filas** (niños) × muchas columnas (todas las
  originales prefijadas por modulo + las derivadas).
- Filtro minimo: `dropna(subset=["CASEID", "NINIO_IDX"])`.

---

## 3. Script 2 — `scripts/procesar_mvp.py`

**Objetivo:** producir un dataset analitico minimo, interpretable y sin
nulos para entrenamiento del MVP.

### 3.1. Columnas finales (13)

```python
COLUMNAS_FINALES_MVP = [
    # Llaves
    "CASEID", "HHID", "LLAVE_NINIO",
    # Target numerico
    "haz_score",
    # Predictores numericos
    "edad_meses", "peso_nacer_kg", "talla_madre_cm",
    # Predictores categoricos (mapeados a palabras)
    "zona", "agua", "saneamiento", "riqueza", "educacion_madre",
    # Target binario
    "Desnutricion_Cronica",
]
```

### 3.2. Logica

1. Carga `endes_2024_unificado.csv`.
2. Si falta `Desnutricion_Cronica`, lo recalcula desde `haz_score`
   (regla OMS: `HAZ < -2`).
3. Selecciona las 13 columnas del MVP.
4. `dropna()` sobre todas ellas (estrategia MVP Fase 3).
5. Castea `Desnutricion_Cronica` a `int8` y guarda el CSV.

### 3.3. Salida

- Archivo: `data/processed/mvp_dataset_fase3_limpio.csv`
- Dimension: **17 698 filas × 13 columnas**.
- Perdida por `dropna`: `2 053` niños (≈ 10.4 %).
- Uso: entrenamiento del modelo (notebook 04) y backend del Streamlit
  (`app_streamlit.py`).

---

## 4. Diagrama del flujo

```
 data/raw/968-Modulo16XX/*.csv   (13 modulos ENDES)
            │
            ▼
 scripts/unificar_datos.py
   - read_csv_clean (BOM, trims, prefijos)
   - to_numeric_clean (codigos missing DHS -> NaN)
   - build_child_block   (CASEID + NINIO_IDX)
   - build_woman_block   (CASEID)
   - build_household_block (HHID)
   - build_programs_flags (HHID -> flags binarias)
   - assemble_final + feature_engineering + reorder_columns
            │
            ▼
 data/processed/endes_2024_unificado.csv   (19 751 filas)
            │
            ▼
 scripts/procesar_mvp.py
   - select(COLUMNAS_FINALES_MVP)
   - dropna()
   - cast Desnutricion_Cronica -> int8
            │
            ▼
 data/processed/mvp_dataset_fase3_limpio.csv   (17 698 x 13)
            │
            ▼
 notebooks/04_modeling.ipynb  +  app_streamlit.py
```

---

## 5. Decisiones de diseño

- **Lectura como string** para todos los CSV: evita que pandas infiera
  tipos erroneos en columnas con ceros significativos o codigos mixtos.
- **Prefijos por modulo** (`C44_`, `REC41_`, `M01_`, `H0_`, `RECH23_`…):
  impide colisiones de nombres al hacer merges, manteniendo trazabilidad
  al origen de cada variable.
- **Codigos missing centralizados** en `MISSING_CODES`: un solo lugar
  para auditar el tratamiento de "no sabe / no responde".
- **Merges `how="left"`** desde el grano niño: el niño es la unidad de
  analisis; no se pierden niños por ausencia de datos en bloques
  superiores.
- **Flags de programas** en lugar de fila-por-programa: mantiene el
  grano niño y simplifica el feature space.
- **Mapeos a palabras** en el unificado: el CSV queda interpretable
  para EDA sin necesidad de diccionarios externos.
- **`dropna` solo en Fase 3**, no en el unificado: preserva el maximo
  de informacion para analisis futuros, y deja la estrategia de
  imputacion / exclusion como una decision explicita del modelado.

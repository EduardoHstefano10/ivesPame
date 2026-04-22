"""
Unificador ENDES 2024 - ivesPame
================================

Fusiona todos los CSV de `data/raw/` en un único dataset limpio y
estandarizado a nivel de niño menor de 5 años, guardado en
`data/processed/endes_2024_unificado.csv`.

Estrategia (nivel de cientifico de datos):

1. Cargar cada modulo como `str` (preserva ceros a la izquierda y evita
   inferencias erroneas de tipo).
2. Limpiar de forma consistente: quitar BOM y espacios, convertir codigos
   "no sabe / no responde" (96/97/98/99, 996/998, 9998, etc.) a NaN, y
   tipar a numerico cuando procede.
3. Ensamblar en un unico ABT (Analytic Base Table) con el niño como grano:
       niño  ->  madre (CASEID)  ->  hogar (HHID)
4. Aplanar programas sociales (PS_*) a flags binarias por hogar.
5. Renombrar las columnas clave mapeandolas a palabras (zona, riqueza,
   educacion_madre, agua, saneamiento), y construir el target binario
   `Desnutricion_Cronica` segun definicion OMS (HAZ < -2).
6. Exportar el CSV unificado ordenado y con columnas clave al inicio.

Diseñado para correr en una sola pasada, con uso moderado de memoria
gracias a `usecols=None` y lectura directa, y documentando cada bloque
para facilitar auditoria.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Rutas y constantes
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
OUT_DIR = BASE_DIR / "data" / "processed"
OUT_PATH = OUT_DIR / "endes_2024_unificado.csv"
OUT_METADATA_PATH = OUT_DIR / "endes_2024_unificado_metadata.json"

# Codigos "missing" tipicos de ENDES/DHS. Se aplican solo a columnas numericas.
MISSING_CODES: tuple[float, ...] = (
    96, 97, 98, 99,
    996, 997, 998, 999,
    9996, 9997, 9998, 9999,
    99.96, 99.97, 99.98, 99.99,
)

# Modulos a nivel de niño (grano: CASEID + indice del niño bajo 5).
# En ENDES MIDX, HIDX y HWIDX se refieren al mismo niño de la historia
# reciente, por lo que se pueden alinear.
CHILD_FILES: dict[str, tuple[Path, str]] = {
    "REC44":  (RAW_DIR / "968-Modulo1638" / "REC44_2024.csv", "HWIDX"),   # antropometria niño
    "REC41":  (RAW_DIR / "968-Modulo1633" / "REC41_2024.csv", "MIDX"),    # embarazo/parto
    "REC43":  (RAW_DIR / "968-Modulo1634" / "REC43_2024.csv", "HIDX"),    # vacunas/enfermedades
    "DIT":    (RAW_DIR / "968-Modulo1634" / "DIT_2024.csv",   "BIDX"),    # desarrollo infantil
    "REC21":  (RAW_DIR / "968-Modulo1632" / "REC21_2024.csv", "BIDX"),    # historia de nacimientos
}

# Modulos a nivel de mujer (grano: CASEID).
WOMAN_FILES: dict[str, Path] = {
    "REC0111":  RAW_DIR / "968-Modulo1631" / "REC0111_2024.csv",
    "REC91":    RAW_DIR / "968-Modulo1631" / "REC91_2024.csv",
    "RE223132": RAW_DIR / "968-Modulo1632" / "RE223132_2024.csv",
    "REC42":    RAW_DIR / "968-Modulo1634" / "REC42_2024.csv",
    "RE516171": RAW_DIR / "968-Modulo1635" / "RE516171_2024.csv",
    "RE758081": RAW_DIR / "968-Modulo1636" / "RE758081_2024.csv",
    "REC82":    RAW_DIR / "968-Modulo1636" / "REC82_2024.csv",
    "REC83":    RAW_DIR / "968-Modulo1637" / "REC83_2024.csv",
    "REC84DV":  RAW_DIR / "968-Modulo1637" / "REC84DV_2024.csv",
    "REC94":    RAW_DIR / "968-Modulo1633" / "REC94_2024.csv",
    "REC95":    RAW_DIR / "968-Modulo1634" / "REC95_2024.csv",
    "REC93DV":  RAW_DIR / "968-Modulo1639" / "REC93DVdisciplina_2024.csv",
}

# Modulos a nivel de hogar (grano: HHID).
HH_FILES: dict[str, Path] = {
    "RECH0":    RAW_DIR / "968-Modulo1629" / "RECH0_2024.csv",
    "RECH1":    RAW_DIR / "968-Modulo1629" / "RECH1_2024.csv",    # miembros (1ra fila/jefe)
    "RECH4":    RAW_DIR / "968-Modulo1629" / "RECH4_2024.csv",    # educacion miembros
    "RECHM":    RAW_DIR / "968-Modulo1629" / "RECHM_2024.csv",    # mortalidad hogar
    "RECH23":   RAW_DIR / "968-Modulo1630" / "RECH23_2024.csv",
    "RECH5":    RAW_DIR / "968-Modulo1638" / "RECH5_2024.csv",    # antropometria mujeres
    "RECH6":    RAW_DIR / "968-Modulo1638" / "RECH6_2024.csv",    # antropometria niños
    "PS_HOGAR": RAW_DIR / "968-Modulo1641" / "Programas Sociales x Hogar_2024.csv",
    "CSALUD01": RAW_DIR / "968-Modulo1640" / "CSALUD01_2024.csv",
    "CSALUD08": RAW_DIR / "968-Modulo1640" / "CSALUD08_2024.csv",
}

# Programas sociales (varias filas por hogar -> flag binaria).
PS_FILES: dict[str, Path] = {
    "PS_BECA18":    RAW_DIR / "968-Modulo1641" / "PS_BECA18_2024.csv",
    "PS_COMEDOR":   RAW_DIR / "968-Modulo1641" / "PS_COMEDOR_2024.csv",
    "PS_PENSION65": RAW_DIR / "968-Modulo1641" / "PS_PENSION65_2024.csv",
    "PS_QALIWARMA": RAW_DIR / "968-Modulo1641" / "PS_QALIWARMA_2024.csv",
    "PS_TRABAJA":   RAW_DIR / "968-Modulo1641" / "PS_TRABAJA_2024.csv",
    "PS_VL":        RAW_DIR / "968-Modulo1641" / "PS_VL_2024.csv",
    "PS_WAWAWASI":  RAW_DIR / "968-Modulo1641" / "PS_WAWAWASI_2024.csv",
}

# Columnas compartidas que NO deben duplicarse al fusionar.
SHARED_DROP = {"ID1", "QHCLUSTER", "QHNUMBER", "QHHOME", "QSNUMERO"}

# Mapas de traduccion (numero -> palabra) para interpretabilidad.
MAPA_ZONA = {1: "Urbano", 2: "Rural"}
MAPA_AGUA = {
    11: "Agua por red pública (dentro)",
    12: "Agua por red pública (fuera)",
    13: "Agua por red pública (pilón)",
    21: "Pozo (dentro de viv.)",
    22: "Pozo (público)",
    41: "Manantial (puquio)",
    43: "Río/Acequia/Laguna",
    51: "Agua de lluvia",
    61: "Camión cisterna",
    71: "Agua embotellada",
    96: "Otro",
}
MAPA_SANEO = {
    11: "Red pública (dentro)",
    12: "Red pública (fuera)",
    21: "Letrina mejorada (ventilada)",
    22: "Pozo séptico",
    23: "Letrina (pozo ciego)",
    24: "Letrina (flotante)",
    31: "Río/Acequia",
    32: "Sin servicio (campo)",
    96: "Otro",
}
MAPA_RIQUEZA = {1: "Más Pobre", 2: "Pobre", 3: "Medio", 4: "Rico", 5: "Más Rico"}
MAPA_EDUCACION = {0: "Sin Educación", 1: "Primaria", 2: "Secundaria", 3: "Superior"}
MAPA_SEXO = {1: "Hombre", 2: "Mujer"}

JSON_DICTIONARIES: dict[str, Path] = {
    "RECH0": RAW_DIR / "968-Modulo1629" / "Diccionario_RECH0.json",
    "RECH23": RAW_DIR / "968-Modulo1630" / "Diccionario_RECH23.json",
    "REC0111": RAW_DIR / "968-Modulo1631" / "Diccionario_REC-0111.json",
}


# ---------------------------------------------------------------------------
# 2. Utilidades de limpieza
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[unificar] {msg}", flush=True)


KEY_COLS = {"CASEID", "HHID", "HWIDX", "MIDX", "HIDX", "BIDX"}


def read_csv_clean(path: Path, prefix: str | None = None) -> pd.DataFrame:
    """Lee un CSV ENDES, quita BOM, recorta espacios y prefija columnas."""
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df.columns = [re.sub(r"^﻿", "", c).strip() for c in df.columns]
    # Recortar espacios en todas las celdas tipo string.
    for col in df.columns:
        df[col] = df[col].str.strip().replace({"": np.nan})
    # Normaliza espacios internos en llaves (CASEID/HHID llegan con padding
    # doble tipo "325503101  2"). Colapsamos a un solo espacio para que el
    # CSV quede legible y los joins sean consistentes.
    for k in KEY_COLS:
        if k in df.columns:
            df[k] = df[k].str.replace(r"\s+", " ", regex=True)
    if prefix:
        rename = {c: f"{prefix}_{c}" for c in df.columns if c not in KEY_COLS and c not in SHARED_DROP}
        df = df.rename(columns=rename)
    return df


def to_numeric_clean(series: pd.Series, missing: Iterable[float] = MISSING_CODES) -> pd.Series:
    """Convierte a numerico aplicando codigos missing de ENDES."""
    s = pd.to_numeric(series, errors="coerce")
    return s.mask(s.isin(list(missing)))


def drop_shared(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in SHARED_DROP if c in df.columns], errors="ignore")


def first_row_per(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Garantiza unicidad del grano (una fila por combinacion de llaves)."""
    return df.drop_duplicates(subset=keys, keep="first")


def load_json_dictionary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def summarize_json_dictionaries() -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for mod, path in JSON_DICTIONARIES.items():
        if not path.exists():
            continue
        raw = load_json_dictionary(path)
        variables = raw.get("variables") or raw.get("diccionario_variables") or []
        metadata = raw.get("metadata", {})
        summary[mod] = {
            "path": str(path.relative_to(BASE_DIR)),
            "metadata": metadata,
            "variables_count": len(variables),
            "sample_variables": [
                {
                    "variable": v.get("variable"),
                    "descripcion": v.get("descripcion"),
                }
                for v in variables[:10]
            ],
        }
    return summary


# ---------------------------------------------------------------------------
# 3. Construccion del bloque Niño (grano fino)
# ---------------------------------------------------------------------------

def build_child_block() -> pd.DataFrame:
    log("Bloque niño: cargando modulos de antropometria/embarazo/vacunas/DIT...")
    # Base: REC44 (antropometria HW*), renombramos HWIDX -> NINIO_IDX.
    rec44 = read_csv_clean(CHILD_FILES["REC44"][0], prefix="C44")
    rec44 = drop_shared(rec44).rename(columns={"HWIDX": "NINIO_IDX"})
    rec44 = first_row_per(rec44, ["CASEID", "NINIO_IDX"])
    log(f"  REC44: {rec44.shape}")

    # REC41 (MIDX), REC43 (HIDX), DIT (BIDX) -> todos al mismo grano.
    for mod, (path, idx_col) in {k: v for k, v in CHILD_FILES.items() if k != "REC44"}.items():
        d = read_csv_clean(path, prefix=mod)
        d = drop_shared(d).rename(columns={idx_col: "NINIO_IDX"})
        d = first_row_per(d, ["CASEID", "NINIO_IDX"])
        log(f"  {mod}: {d.shape}")
        rec44 = rec44.merge(d, on=["CASEID", "NINIO_IDX"], how="left")
    log(f"Bloque niño ensamblado: {rec44.shape}")
    return rec44


# ---------------------------------------------------------------------------
# 4. Construccion del bloque Madre (CASEID)
# ---------------------------------------------------------------------------

def build_woman_block() -> pd.DataFrame:
    log("Bloque madre: uniendo modulos individuales por CASEID...")
    # Base: REC0111 (tiene CASEID + HHID, puente hogar).
    base = read_csv_clean(WOMAN_FILES["REC0111"], prefix="M01")
    base = drop_shared(base)
    base = first_row_per(base, ["CASEID"])
    log(f"  REC0111: {base.shape}")

    for mod, path in WOMAN_FILES.items():
        if mod == "REC0111":
            continue
        d = read_csv_clean(path, prefix=mod)
        d = drop_shared(d)
        d = first_row_per(d, ["CASEID"])
        log(f"  {mod}: {d.shape}")
        base = base.merge(d, on="CASEID", how="left")
    log(f"Bloque madre ensamblado: {base.shape}")
    return base


# ---------------------------------------------------------------------------
# 5. Construccion del bloque Hogar (HHID)
# ---------------------------------------------------------------------------

def build_household_block() -> pd.DataFrame:
    log("Bloque hogar: uniendo modulos de vivienda/servicios/salud...")
    base = read_csv_clean(HH_FILES["RECH0"], prefix="H0")
    base = drop_shared(base)
    base = first_row_per(base, ["HHID"])
    log(f"  RECH0: {base.shape}")

    for mod, path in HH_FILES.items():
        if mod == "RECH0":
            continue
        d = read_csv_clean(path, prefix=mod)
        d = drop_shared(d)
        d = first_row_per(d, ["HHID"])
        log(f"  {mod}: {d.shape}")
        base = base.merge(d, on="HHID", how="left")
    log(f"Bloque hogar ensamblado: {base.shape}")
    return base


# ---------------------------------------------------------------------------
# 6. Programas sociales -> flags por hogar
# ---------------------------------------------------------------------------

def build_programs_flags() -> pd.DataFrame:
    log("Programas sociales: generando flags binarias por hogar...")
    frames: list[pd.DataFrame] = []
    for prog, path in PS_FILES.items():
        d = read_csv_clean(path)  # strip + BOM + celdas recortadas
        if "HHID" not in d.columns:
            log(f"  [warn] {prog} sin HHID, se omite")
            continue
        flag = (
            d[["HHID"]]
            .dropna()
            .assign(**{f"programa_{prog.lower()}": 1})
            .drop_duplicates("HHID")
        )
        frames.append(flag)
        log(f"  {prog}: {flag.shape}")
    if not frames:
        return pd.DataFrame(columns=["HHID"])
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="HHID", how="outer")
    flag_cols = [c for c in out.columns if c.startswith("programa_")]
    out[flag_cols] = out[flag_cols].fillna(0).astype("int8")
    out["programas_sociales_total"] = out[flag_cols].sum(axis=1).astype("int8")
    log(f"Flags ensambladas: {out.shape}")
    return out


# ---------------------------------------------------------------------------
# 7. Ensamblaje final y feature engineering
# ---------------------------------------------------------------------------

def assemble_final(child: pd.DataFrame, woman: pd.DataFrame,
                   household: pd.DataFrame, programs: pd.DataFrame) -> pd.DataFrame:
    """Ensambla con full outer join para preservar TODAS las filas de todos
    los módulos (hogares sin mujer y mujeres sin niño también quedan)."""
    log("Ensamblaje final (outer): hogar + madre + niño + programas...")

    # Puente niño -> hogar usando el HHID que trae el bloque madre (M01_HHID).
    hhid_col = "M01_HHID" if "M01_HHID" in woman.columns else "HHID"
    if hhid_col != "HHID" and "HHID" not in woman.columns:
        woman = woman.rename(columns={hhid_col: "HHID"})
    elif hhid_col != "HHID":
        woman["HHID"] = woman["HHID"].fillna(woman[hhid_col])

    # 1) Madre ⋈ Niño por CASEID (outer): 37 117 mujeres (las que tienen niño
    #    <5 años se enriquecen con las columnas del niño).
    mn = woman.merge(child, on="CASEID", how="outer")
    log(f"  madre ⋈ niño (outer): {mn.shape}")

    # 2) Hogar ⋈ (madre+niño) por HHID (outer): 37 390 hogares, algunos sin
    #    entrevista individual de mujer.
    df = household.merge(mn, on="HHID", how="outer")
    log(f"  hogar ⋈ madre+niño (outer): {df.shape}")

    # 3) Programas sociales (flags por HHID).
    df = df.merge(programs, on="HHID", how="left")
    prog_cols = [c for c in df.columns if c.startswith("programa_")]
    for c in prog_cols:
        df[c] = df[c].fillna(0).astype("int8")
    if "programas_sociales_total" in df.columns:
        df["programas_sociales_total"] = (
            df["programas_sociales_total"].fillna(0).astype("int8")
        )
    log(f"Dataset unificado bruto: {df.shape}")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    log("Feature engineering: target, escalas y mapeos legibles...")

    # --- Target: puntaje Z de talla/edad (HAZ). HW70 en REC44 viene x100. ---
    if "C44_HW70" in df.columns:
        haz = to_numeric_clean(df["C44_HW70"]) / 100.0
        df["haz_score"] = haz
        df["Desnutricion_Cronica"] = np.where(haz < -2, 1, 0).astype("int8")
        df.loc[haz.isna(), "Desnutricion_Cronica"] = np.nan

    # --- Edad del niño en meses ---
    if "C44_HW1" in df.columns:
        df["edad_meses"] = to_numeric_clean(df["C44_HW1"])

    # --- Sexo del niño (B4 via REC41: 1=H, 2=M). ---
    if "REC41_B4" in df.columns:
        df["sexo_nino"] = to_numeric_clean(df["REC41_B4"]).map(MAPA_SEXO)

    # --- Peso al nacer (M19 en gramos -> kg). ---
    if "REC41_M19" in df.columns:
        peso_g = to_numeric_clean(df["REC41_M19"])
        df["peso_nacer_kg"] = peso_g / 1000.0

    # --- Antropometria de la madre (V438 en mm -> cm). ---
    if "REC42_V438" in df.columns:
        df["talla_madre_cm"] = to_numeric_clean(df["REC42_V438"]) / 10.0
    if "REC42_V437" in df.columns:
        df["peso_madre_kg"] = to_numeric_clean(df["REC42_V437"]) / 10.0

    # --- Zona, servicios, riqueza, educacion -> palabras. ---
    if "H0_HV025" in df.columns:
        df["zona"] = to_numeric_clean(df["H0_HV025"]).map(MAPA_ZONA)
    if "RECH23_HV201" in df.columns:
        df["agua"] = to_numeric_clean(df["RECH23_HV201"]).map(MAPA_AGUA)
    if "RECH23_HV205" in df.columns:
        df["saneamiento"] = to_numeric_clean(df["RECH23_HV205"]).map(MAPA_SANEO)
    if "RECH23_HV270" in df.columns:
        df["riqueza"] = to_numeric_clean(df["RECH23_HV270"]).map(MAPA_RIQUEZA)
    if "M01_V106" in df.columns:
        df["educacion_madre"] = to_numeric_clean(df["M01_V106"]).map(MAPA_EDUCACION)
    if "M01_V012" in df.columns:
        df["edad_madre"] = to_numeric_clean(df["M01_V012"])

    # --- Llave unica del niño: "{HHID}-{MADRE_LINE}-{NINIO_IDX}" ---
    # Varias madres pueden compartir HHID; incluimos la linea de la madre
    # (segundo token de CASEID "HHID LINE") para que la llave sea unica.
    hhid_clean = df["HHID"].astype("string").str.strip().str.replace(r"\s+", "", regex=True)
    caseid_clean = df["CASEID"].astype("string").str.strip()
    madre_line = caseid_clean.str.split(" ").str[-1].str.strip()
    idx_clean = df["NINIO_IDX"].astype("string").str.strip()
    child_key_mask = (
        hhid_clean.notna()
        & caseid_clean.notna()
        & idx_clean.notna()
        & (hhid_clean != "")
        & (caseid_clean != "")
        & (idx_clean != "")
        & (idx_clean.str.lower() != "nan")
    )
    df["LLAVE_NINIO"] = pd.Series(pd.NA, index=df.index, dtype="string")
    df.loc[child_key_mask, "LLAVE_NINIO"] = (
        hhid_clean[child_key_mask]
        + "-"
        + madre_line[child_key_mask]
        + "-"
        + idx_clean[child_key_mask]
    )

    woman_key_mask = caseid_clean.notna() & (caseid_clean != "") & (caseid_clean.str.lower() != "nan")
    df["NIVEL_REGISTRO"] = np.select(
        [child_key_mask, woman_key_mask],
        ["nino", "mujer"],
        default="hogar",
    )

    return df


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Deja las columnas clave interpretables primero, el resto ordenadas alfabeticamente."""
    preferred = [
        "CASEID", "HHID", "NINIO_IDX", "LLAVE_NINIO", "NIVEL_REGISTRO",
        "Desnutricion_Cronica", "haz_score",
        "edad_meses", "sexo_nino", "peso_nacer_kg",
        "edad_madre", "talla_madre_cm", "peso_madre_kg", "educacion_madre",
        "zona", "agua", "saneamiento", "riqueza",
        "programas_sociales_total",
    ]
    preferred = [c for c in preferred if c in df.columns]
    rest = sorted(c for c in df.columns if c not in preferred)
    return df[preferred + rest]


# ---------------------------------------------------------------------------
# 8. Orquestador
# ---------------------------------------------------------------------------

def main() -> int:
    t0 = time.perf_counter()
    if not RAW_DIR.exists():
        log(f"ERROR: no existe {RAW_DIR}")
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    child = build_child_block()
    woman = build_woman_block()
    household = build_household_block()
    programs = build_programs_flags()

    df = assemble_final(child, woman, household, programs)
    df = feature_engineering(df)
    df = reorder_columns(df)

    # Requerimos al menos HHID (toda fila viene de algún hogar). No se
    # descartan filas sin niño ni sin madre: así queda TODA la data de todos
    # los módulos (niveles hogar, madre y niño conviven en la misma tabla).
    before = len(df)
    df = df.dropna(subset=["HHID"])
    log(f"Filtro HHID no nulo: {before} -> {len(df)} filas")

    # Deduplicación: la llave natural ahora es (HHID, CASEID, NINIO_IDX).
    before = len(df)
    if "NIVEL_REGISTRO" in df.columns:
        child_rows = df[df["NIVEL_REGISTRO"] == "nino"].drop_duplicates(
            subset=["HHID", "CASEID", "NINIO_IDX"], keep="first"
        )
        woman_rows = df[df["NIVEL_REGISTRO"] == "mujer"].drop_duplicates(
            subset=["HHID", "CASEID"], keep="first"
        )
        household_rows = df[df["NIVEL_REGISTRO"] == "hogar"].drop_duplicates(
            subset=["HHID"], keep="first"
        )
        df = pd.concat([child_rows, woman_rows, household_rows], ignore_index=True, sort=False)
    else:
        df = df.drop_duplicates(subset=["HHID", "CASEID", "NINIO_IDX"], keep="first")
    if before != len(df):
        log(f"drop_duplicates: {before} -> {len(df)} filas")

    dup_child_keys = int(df["LLAVE_NINIO"].dropna().duplicated().sum()) if "LLAVE_NINIO" in df.columns else 0
    metadata = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "source": "scripts/unificar_datos.py",
        "output_csv": str(OUT_PATH.relative_to(BASE_DIR)),
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "row_count_by_level": (
            df["NIVEL_REGISTRO"].value_counts(dropna=False).sort_index().to_dict()
            if "NIVEL_REGISTRO" in df.columns else {}
        ),
        "unique_hhid": int(df["HHID"].nunique(dropna=True)) if "HHID" in df.columns else 0,
        "unique_caseid": int(df["CASEID"].nunique(dropna=True)) if "CASEID" in df.columns else 0,
        "unique_llave_ninio": int(df["LLAVE_NINIO"].nunique(dropna=True)) if "LLAVE_NINIO" in df.columns else 0,
        "duplicate_llave_ninio": dup_child_keys,
        "target_non_null": int(df["Desnutricion_Cronica"].notna().sum()) if "Desnutricion_Cronica" in df.columns else 0,
        "json_dictionaries": summarize_json_dictionaries(),
    }

    log(f"Guardando CSV unificado en {OUT_PATH} ...")
    df.to_csv(OUT_PATH, index=False)

    # Versión comprimida (cabe en GitHub y se lee directo con pd.read_csv).
    gz_path = OUT_PATH.with_suffix(".csv.gz")
    log(f"Guardando versión comprimida en {gz_path} ...")
    df.to_csv(gz_path, index=False, compression="gzip")

    log(f"Guardando metadata en {OUT_METADATA_PATH} ...")
    OUT_METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    log(f"OK. Dimensiones finales: {df.shape}")
    target_nn = df["Desnutricion_Cronica"].notna().sum() if "Desnutricion_Cronica" in df.columns else 0
    log(f"Cobertura target Desnutricion_Cronica: {target_nn} no nulos de {len(df)}")
    log(f"Tiempo total: {time.perf_counter() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

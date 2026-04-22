"""
Procesador ENDES 2024 - ivesPame (MVP analitico)
================================================

Replica el notebook 03_data_preparation.ipynb como un script reproducible.
Genera el dataset analitico final para el modelo de desnutricion cronica
en `data/processed/mvp_dataset_fase3_limpio.csv`.

Entrada : 6 CSVs de `data/raw/`
Salida  : 1 CSV limpio, mapeado y listo para modelado
         (~17.7k ninos x 13 columnas).

Uso:
    python3 scripts/procesar_datos.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Rutas y constantes
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw"
OUT = BASE / "data" / "processed" / "mvp_dataset_fase3_limpio.csv"

FILES = {
    "hogar_zona":  RAW / "968-Modulo1629" / "RECH0_2024.csv",
    "hogar_saneo": RAW / "968-Modulo1630" / "RECH23_2024.csv",
    "madre_edu":   RAW / "968-Modulo1631" / "REC0111_2024.csv",
    "madre_talla": RAW / "968-Modulo1634" / "REC42_2024.csv",
    "nino_peso":   RAW / "968-Modulo1633" / "REC41_2024.csv",
    "nino_target": RAW / "968-Modulo1638" / "REC44_2024.csv",
}

COLS = {
    "hogar_zona":  ["HHID", "HV025"],
    "hogar_saneo": ["HHID", "HV201", "HV205", "HV270"],
    "madre_edu":   ["CASEID", "HHID", "V106"],
    "madre_talla": ["CASEID", "V438"],
    "nino_peso":   ["CASEID", "MIDX", "M19"],
    "nino_target": ["CASEID", "HWIDX", "HW1", "HW70"],
}

MISSING_CODES = [
    9996, 9997, 9998, 9999,
    996, 997, 998, 999,
    96, 97, 98, 99,
    99.96, 99.97, 99.98, 99.99,
]

MAPA_ZONA      = {1: "Urbano", 2: "Rural"}
MAPA_AGUA      = {
    11: "Agua por red pública (dentro)", 12: "Agua por red pública (fuera)",
    13: "Agua por red pública (pilón)",  21: "Pozo (dentro de viv.)",
    22: "Pozo (público)", 41: "Manantial (puquio)", 43: "Río/Acequia/Laguna",
    51: "Agua de lluvia", 61: "Camión cisterna", 71: "Agua embotellada",
    96: "Otro",
}
MAPA_SANEO     = {
    11: "Red pública (dentro)", 12: "Red pública (fuera)",
    21: "Letrina mejorada (ventilada)", 22: "Pozo séptico",
    23: "Letrina (pozo ciego)", 24: "Letrina (flotante)",
    31: "Río/Acequia", 32: "Sin servicio (campo)", 96: "Otro",
}
MAPA_RIQUEZA   = {1: "Más Pobre", 2: "Pobre", 3: "Medio", 4: "Rico", 5: "Más Rico"}
MAPA_EDUCACION = {0: "Sin Educación", 1: "Primaria", 2: "Secundaria", 3: "Superior"}


# ---------------------------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[procesar] {msg}", flush=True)


def clean_numeric(s: pd.Series) -> pd.Series:
    """Quita espacios/BOM, vacios a NaN, aplica codigos missing y tipa a float."""
    s = s.astype(str).str.strip().replace("", np.nan)
    codes = {str(c) for c in MISSING_CODES} | {str(float(c)) for c in MISSING_CODES}
    s = s.replace(list(codes), np.nan)
    return pd.to_numeric(s, errors="coerce")


def load(name: str) -> pd.DataFrame:
    df = pd.read_csv(FILES[name], usecols=COLS[name], dtype=str)
    # Normaliza llaves: quita BOM/espacios extremos y colapsa espacios internos
    # (CASEID llega como "HHID  line" con padding doble -> "HHID line").
    for k in ("HHID", "CASEID", "MIDX", "HWIDX"):
        if k in df.columns:
            df[k] = (
                df[k].astype(str)
                .str.replace("﻿", "", regex=False)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )
    return df


# ---------------------------------------------------------------------------
# 3. ETL
# ---------------------------------------------------------------------------
def build() -> pd.DataFrame:
    log("Bloque 1: Hogar (RECH0 + RECH23)")
    h_zona  = load("hogar_zona")
    h_saneo = load("hogar_saneo")
    h_zona["HV025"]  = clean_numeric(h_zona["HV025"])
    h_saneo["HV201"] = clean_numeric(h_saneo["HV201"])
    h_saneo["HV205"] = clean_numeric(h_saneo["HV205"])
    h_saneo["HV270"] = clean_numeric(h_saneo["HV270"])
    h_zona["zona"]          = h_zona["HV025"].map(MAPA_ZONA)
    h_saneo["agua"]         = h_saneo["HV201"].map(MAPA_AGUA)
    h_saneo["saneamiento"]  = h_saneo["HV205"].map(MAPA_SANEO)
    h_saneo["riqueza"]      = h_saneo["HV270"].map(MAPA_RIQUEZA)
    hogar = h_zona.merge(h_saneo, on="HHID", how="inner")

    log("Bloque 2: Madre (REC0111 + REC42)")
    m_edu   = load("madre_edu")
    m_talla = load("madre_talla")
    m_edu["V106"]   = clean_numeric(m_edu["V106"])
    m_talla["V438"] = clean_numeric(m_talla["V438"])
    m_talla["talla_madre_cm"]   = m_talla["V438"] / 10.0
    m_edu["educacion_madre"]    = m_edu["V106"].map(MAPA_EDUCACION)
    madre = m_edu.merge(m_talla, on="CASEID", how="inner")

    log("Bloque 3: Nino (REC41 + REC44)")
    n_peso   = load("nino_peso")
    n_target = load("nino_target")
    n_peso["M19"]    = clean_numeric(n_peso["M19"])
    n_target["HW1"]  = clean_numeric(n_target["HW1"])
    n_target["HW70"] = clean_numeric(n_target["HW70"])
    n_peso["peso_nacer_kg"] = n_peso["M19"] / 1000.0
    n_target = n_target.rename(columns={"HW1": "edad_meses", "HW70": "haz_score"})
    # MIDX/HWIDX son el indice del niño dentro de la madre (1,2,3...).
    # Lo renombramos a NINIO_IDX para construir luego una LLAVE_NINIO unica.
    n_peso   = n_peso.rename(columns={"MIDX": "NINIO_IDX"})
    n_target = n_target.rename(columns={"HWIDX": "NINIO_IDX"})
    nino = n_peso.merge(n_target, on=["CASEID", "NINIO_IDX"], how="inner")

    log("Ensamblaje: nino + madre + hogar")
    df = nino.merge(madre, on="CASEID", how="inner").merge(hogar, on="HHID", how="inner")

    # Correccion de escala HAZ (HW70 viene x100 en la ENDES).
    df["haz_score"] = df["haz_score"] / 100.0
    return df


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    # LLAVE_NINIO unica por niño: "{HHID}-{MADRE_LINE}-{NINIO_IDX}".
    # Un mismo hogar puede tener varias madres (CASEIDs distintos), por eso
    # usamos la linea de la madre (segundo token de CASEID "HHID LINE") para
    # evitar colisiones entre niños de madres distintas en el mismo hogar.
    caseid = df["CASEID"].astype(str).str.strip()
    madre_line = caseid.str.split(" ").str[-1].str.strip()
    df["LLAVE_NINIO"] = (
        df["HHID"].astype(str).str.strip()
        + "-"
        + madre_line
        + "-"
        + df["NINIO_IDX"].astype(str).str.strip()
    )
    cols = [
        "CASEID", "HHID", "LLAVE_NINIO",
        "haz_score", "edad_meses", "peso_nacer_kg", "talla_madre_cm",
        "zona", "agua", "saneamiento", "riqueza", "educacion_madre",
    ]
    df = df[cols].copy().dropna()
    df = df.drop_duplicates(subset=["CASEID", "LLAVE_NINIO"], keep="first")
    df["Desnutricion_Cronica"] = np.where(df["haz_score"] < -2, 1, 0).astype("int8")
    return df


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = build()
    log(f"Dataset bruto ensamblado: {df.shape}")
    df = finalize(df)
    log(f"Dataset analitico final : {df.shape}")
    log(f"Desnutricion cronica: {int(df['Desnutricion_Cronica'].sum())} "
        f"({df['Desnutricion_Cronica'].mean()*100:.2f}%)")
    df.to_csv(OUT, index=False)
    log(f"Guardado en: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Procesado MVP - ivesPame
========================

Toma el dataset unificado `data/processed/endes_2024_unificado.csv`
(producido por `scripts/unificar_datos.py`) y genera el dataset
analitico final `data/processed/mvp_dataset_fase3_limpio.csv` que
alimenta el modelado (notebook 04) y la app Streamlit.

Pasos:
1. Cargar el CSV unificado.
2. Seleccionar las 13 columnas clave del MVP (llaves + predictores +
   target). Si `Desnutricion_Cronica` no existe (o haz_score),
   se recalcula segun OMS (HAZ < -2 -> 1).
3. Eliminar filas con cualquier nulo (estrategia MVP Fase 3).
4. Guardar el CSV limpio.

Pensado para ejecutarse en una sola pasada, con trazabilidad por
consola de cuantas filas se pierden en cada filtro.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
IN_PATH = BASE_DIR / "data" / "processed" / "endes_2024_unificado.csv"
OUT_PATH = BASE_DIR / "data" / "processed" / "mvp_dataset_fase3_limpio.csv"

COLUMNAS_FINALES_MVP: list[str] = [
    # Llaves (utiles para debug / join)
    "CASEID",
    "HHID",
    "LLAVE_NINIO",
    # Target numerico
    "haz_score",
    # Predictores numericos
    "edad_meses",
    "peso_nacer_kg",
    "talla_madre_cm",
    # Predictores categoricos (mapeados a palabras)
    "zona",
    "agua",
    "saneamiento",
    "riqueza",
    "educacion_madre",
    # Target binario (OMS: HAZ < -2)
    "Desnutricion_Cronica",
]


def log(msg: str) -> None:
    print(f"[mvp] {msg}", flush=True)


def main() -> int:
    if not IN_PATH.exists():
        log(f"ERROR: no existe {IN_PATH}. Corre primero scripts/unificar_datos.py")
        return 1

    log(f"Cargando {IN_PATH.name} ...")
    df = pd.read_csv(IN_PATH, low_memory=False)
    log(f"Dataset unificado: {df.shape}")

    # Asegurar target binario (por si se regenero el unificado sin el).
    if "Desnutricion_Cronica" not in df.columns and "haz_score" in df.columns:
        log("Recalculando Desnutricion_Cronica a partir de haz_score (HAZ < -2).")
        haz = pd.to_numeric(df["haz_score"], errors="coerce")
        df["Desnutricion_Cronica"] = np.where(haz < -2, 1, 0).astype("int8")
        df.loc[haz.isna(), "Desnutricion_Cronica"] = np.nan

    faltantes = [c for c in COLUMNAS_FINALES_MVP if c not in df.columns]
    if faltantes:
        log(f"ERROR: faltan columnas en el unificado: {faltantes}")
        return 2

    df_mvp = df[COLUMNAS_FINALES_MVP].copy()
    antes = len(df_mvp)
    df_mvp = df_mvp.dropna()
    log(f"dropna(): {antes} -> {len(df_mvp)} filas "
        f"(perdida {antes - len(df_mvp)} / {100 * (antes - len(df_mvp)) / max(antes, 1):.1f}%)")

    # Garantia de unicidad a nivel niño en el CSV final.
    antes = len(df_mvp)
    df_mvp = df_mvp.drop_duplicates(subset=["CASEID", "LLAVE_NINIO"], keep="first")
    if antes != len(df_mvp):
        log(f"drop_duplicates: {antes} -> {len(df_mvp)} filas")

    # Tipos consistentes con el CSV final.
    df_mvp["Desnutricion_Cronica"] = df_mvp["Desnutricion_Cronica"].astype("int8")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_mvp.to_csv(OUT_PATH, index=False)
    log(f"OK. Guardado {OUT_PATH} con shape {df_mvp.shape}")
    log(f"Prevalencia Desnutricion_Cronica: "
        f"{df_mvp['Desnutricion_Cronica'].mean() * 100:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())

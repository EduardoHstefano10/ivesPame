"""Dashboard geográfico interactivo para el dataset ENDES 2024.

Muestra la cantidad de registros por departamento, provincia y distrito
usando el código UBIGEO (formato DDPPDD) y coordenadas (lat/lon).
"""
import os
import pandas as pd
import plotly.express as px
import streamlit as st


DATA_CSV = os.path.join("data", "processed", "endes_2024_unificado.csv")
DATA_GZ = os.path.join("data", "processed", "endes_2024_unificado.csv.gz")
DATA_PATH = DATA_CSV if os.path.exists(DATA_CSV) else DATA_GZ

DEPARTAMENTOS = {
    "01": "Amazonas", "02": "Áncash", "03": "Apurímac", "04": "Arequipa",
    "05": "Ayacucho", "06": "Cajamarca", "07": "Callao", "08": "Cusco",
    "09": "Huancavelica", "10": "Huánuco", "11": "Ica", "12": "Junín",
    "13": "La Libertad", "14": "Lambayeque", "15": "Lima", "16": "Loreto",
    "17": "Madre de Dios", "18": "Moquegua", "19": "Pasco", "20": "Piura",
    "21": "Puno", "22": "San Martín", "23": "Tacna", "24": "Tumbes",
    "25": "Ucayali",
}


@st.cache_data(show_spinner="Cargando datos geográficos...")
def cargar_datos_geograficos(path: str = DATA_PATH) -> pd.DataFrame:
    cols = [
        "H0_UBIGEO", "H0_LATITUDY", "H0_LONGITUDX",
        "REC91_SREGION", "zona", "Desnutricion_Cronica",
    ]
    df = pd.read_csv(path, usecols=lambda c: c in cols)
    df["UBIGEO"] = df["H0_UBIGEO"].astype(int).astype(str).str.zfill(6)
    df["dept_cod"] = df["UBIGEO"].str[:2]
    df["prov_cod"] = df["UBIGEO"].str[:4]
    df["dist_cod"] = df["UBIGEO"]
    df["departamento"] = df["dept_cod"].map(DEPARTAMENTOS).fillna("Desconocido")
    df["provincia"] = df["departamento"] + " · Prov " + df["prov_cod"].str[2:]
    df["distrito"] = df["provincia"] + " · Dist " + df["dist_cod"].str[4:]
    df = df.rename(columns={"H0_LATITUDY": "lat", "H0_LONGITUDX": "lon"})
    return df


def _kpis(df: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros", f"{len(df):,}")
    c2.metric("Departamentos", df["departamento"].nunique())
    c3.metric("Provincias", df["prov_cod"].nunique())
    c4.metric("Distritos", df["dist_cod"].nunique())


def _mapa(df: pd.DataFrame, nivel: str) -> None:
    grupo = {"Departamento": "departamento",
             "Provincia": "provincia",
             "Distrito": "distrito"}[nivel]

    agg = (
        df.groupby(grupo, as_index=False)
          .agg(registros=("UBIGEO", "size"),
               lat=("lat", "mean"),
               lon=("lon", "mean"))
          .sort_values("registros", ascending=False)
    )

    fig = px.scatter_mapbox(
        agg,
        lat="lat", lon="lon",
        size="registros", color="registros",
        color_continuous_scale="Turbo",
        size_max=40, zoom=4.2,
        hover_name=grupo,
        hover_data={"registros": True, "lat": False, "lon": False},
        mapbox_style="open-street-map",
        height=560,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, width="stretch")
    return agg


def _ranking(agg: pd.DataFrame, nivel: str, top_n: int) -> None:
    nombre_col = agg.columns[0]
    top = agg.head(top_n).iloc[::-1]
    fig = px.bar(
        top, x="registros", y=nombre_col, orientation="h",
        color="registros", color_continuous_scale="Turbo",
        text="registros", height=max(320, 22 * len(top)),
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        yaxis_title=None, xaxis_title="Registros",
        coloraxis_showscale=False,
        title=f"Top {top_n} {nivel.lower()}s por cantidad de registros",
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, width="stretch")


def _tabla(agg: pd.DataFrame, nivel: str) -> None:
    tabla = agg.drop(columns=["lat", "lon"]).reset_index(drop=True)
    tabla.index = tabla.index + 1
    tabla.index.name = "#"
    total = tabla["registros"].sum()
    tabla["%"] = (tabla["registros"] / total * 100).round(2)
    st.dataframe(tabla, width="stretch", height=420)
    st.download_button(
        "⬇️ Descargar reporte CSV",
        tabla.to_csv().encode("utf-8"),
        file_name=f"reporte_{nivel.lower()}.csv",
        mime="text/csv",
    )


def dashboard_geografico_page() -> None:
    st.title("🗺️ Dashboard Geográfico")
    st.caption(
        "Distribución de registros ENDES 2024 por departamento, provincia y distrito "
        "según UBIGEO y coordenadas geográficas."
    )

    if not os.path.exists(DATA_PATH):
        st.error(
            "No se encontró el dataset unificado. Ejecuta "
            "`python scripts/unificar_datos.py` para generarlo."
        )
        return

    df = cargar_datos_geograficos()

    with st.sidebar:
        st.markdown("### 🎚️ Filtros")
        deps = sorted(df["departamento"].unique())
        sel_dep = st.multiselect("Departamento", deps, default=deps)
        zona_opts = ["Todas"] + sorted(df["zona"].dropna().astype(str).unique().tolist()) \
            if "zona" in df.columns else ["Todas"]
        sel_zona = st.selectbox("Zona", zona_opts)
        top_n = st.slider("Top N en ranking", 5, 50, 15)

    dff = df[df["departamento"].isin(sel_dep)]
    if sel_zona != "Todas" and "zona" in dff.columns:
        dff = dff[dff["zona"].astype(str) == sel_zona]

    if dff.empty:
        st.warning("Sin registros para los filtros seleccionados.")
        return

    _kpis(dff)
    st.markdown("---")

    nivel = st.radio(
        "Nivel de agregación",
        ["Departamento", "Provincia", "Distrito"],
        horizontal=True,
    )

    agg = _mapa(dff, nivel)

    col_l, col_r = st.columns([3, 2])
    with col_l:
        _ranking(agg, nivel, top_n)
    with col_r:
        st.markdown(f"#### Reporte por {nivel.lower()}")
        _tabla(agg, nivel)

# Visor de Información Geoespacial de Precipitación y el Fenómeno ENSO
import streamlit as st
import pandas as pd
import altair as alt
import folium
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import zipfile
import tempfile
import os
import io
import numpy as np
import re
import csv

# Importaciones para Kriging
from pykrige.ok import OrdinaryKriging

try:
    from folium.plugins import ScaleControl
except ImportError:
    class ScaleControl:
        def __init__(self, *args, **kwargs): pass
        def add_to(self, m): pass

#--- Configuración de la página
st.set_page_config(layout="wide", page_title="Visor de Precipitación y ENSO")
st.markdown("""
<style>
div.block-container {padding-top: 2rem;}
.sidebar .sidebar-content { font-size: 13px; }
h1 { margin-top: 0px; padding-top: 0px; }
</style>
""", unsafe_allow_html=True)

#--- Funciones de Carga y Procesamiento ---
@st.cache_data
def load_data(file_path):
    # ... (código sin cambios)
    if file_path is None: return None
    try:
        content = file_path.getvalue()
        if not content.strip():
            st.error(f"El archivo '{file_path.name}' parece estar vacío.")
            return None
    except Exception as e:
        st.error(f"No se pudo leer el contenido del archivo '{file_path.name}': {e}")
        return None
    encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    for encoding in encodings_to_try:
        try:
            sample_str = content[:4096].decode(encoding)
            try:
                dialect = csv.Sniffer().sniff(sample_str, delimiters=';,')
                sep = dialect.delimiter
            except csv.Error:
                sep = ';' if sample_str.count(';') > sample_str.count(',') else ','
            df = pd.read_csv(io.BytesIO(content), sep=sep, encoding=encoding, low_memory=False)
            df.columns = df.columns.str.strip()
            return df
        except Exception:
            continue
    st.error(f"No se pudo decodificar el archivo '{file_path.name}'.")
    return None

# --- INICIO: FUNCIÓN load_shapefile RESTAURADA A SU VERSIÓN MÁS ROBUSTA ---
@st.cache_data
def load_shapefile(file_path):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            shp_path = [f for f in os.listdir(temp_dir) if f.endswith('.shp')][0]
            gdf = gpd.read_file(os.path.join(temp_dir, shp_path))
            gdf.columns = gdf.columns.str.strip()
            
            # Si el archivo no tiene un CRS definido, se lo asignamos explícitamente.
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:9377") # Asignación explícita y robusta
            
            return gdf.to_crs("EPSG:4326") # Transformación a WGS84
    except Exception as e:
        st.error(f"Error al procesar el shapefile: {e}")
        return None
# --- FIN: FUNCIÓN CORREGIDA ---

#--- Interfaz y Carga de Archivos ---
st.title('Visor de Precipitación y Fenómeno ENSO')
st.sidebar.header("Panel de Control")
with st.sidebar.expander("**Cargar Archivos**", expanded=True):
    st.info("Cargue los archivos de precipitación que ya contienen las columnas ENSO.")
    uploaded_file_mapa = st.file_uploader("1. Archivo de estaciones con ENSO Anual", type="csv")
    uploaded_file_precip = st.file_uploader("2. Archivo de precipitación con ENSO Mensual", type="csv")
    uploaded_file_enso = st.file_uploader("3. Archivo completo de ENSO (para análisis avanzado)", type="csv")
    uploaded_zip_shapefile = st.file_uploader("4. Shapefile de municipios (.zip)", type="zip")

if not all([uploaded_file_mapa, uploaded_file_precip, uploaded_zip_shapefile, uploaded_file_enso]):
    st.info("Por favor, suba los 4 archivos requeridos para habilitar la aplicación.")
    st.stop()

#--- Carga y Preprocesamiento de Datos ---
df_precip_anual = load_data(uploaded_file_mapa)
df_precip_mensual = load_data(uploaded_file_precip)
df_enso = load_data(uploaded_file_enso)
gdf_municipios = load_shapefile(uploaded_zip_shapefile)
if any(df is None for df in [df_precip_anual, df_precip_mensual, gdf_municipios, df_enso]):
    st.stop()
    
# Estaciones y Precipitación Anual
lon_col = next((col for col in df_precip_anual.columns if 'longitud' in col.lower() or 'lon' in col.lower()), None)
lat_col = next((col for col in df_precip_anual.columns if 'latitud' in col.lower() or 'lat' in col.lower()), None)
if not all([lon_col, lat_col]):
    st.error(f"No se encontraron las columnas de longitud y/o latitud en el archivo de estaciones.")
    st.stop()
df_precip_anual[lon_col] = pd.to_numeric(df_precip_anual[lon_col], errors='coerce')
df_precip_anual[lat_col] = pd.to_numeric(df_precip_anual[lat_col], errors='coerce')
df_precip_anual.dropna(subset=[lon_col, lat_col], inplace=True)
gdf_stations = gpd.GeoDataFrame(df_precip_anual, geometry=gpd.points_from_xy(df_precip_anual[lon_col], df_precip_anual[lat_col]), crs="EPSG:9377").to_crs("EPSG:4326")
gdf_stations['Longitud_geo'] = gdf_stations.geometry.x
gdf_stations['Latitud_geo'] = gdf_stations.geometry.y

# Precipitación Mensual (ya enriquecida)
df_precip_mensual.columns = [col.lower() for col in df_precip_mensual.columns]
year_col_precip = next((col for col in df_precip_mensual.columns if 'año' in col), None)
enso_mes_col_precip = next((col for col in df_precip_mensual.columns if 'enso_mes' in col or 'enso-mes' in col), None)
if not all([year_col_precip, enso_mes_col_precip]):
    st.error(f"No se encontraron 'año' o 'enso_mes' en el archivo de precipitación mensual.")
    st.stop()
df_precip_mensual.rename(columns={year_col_precip: 'Año', enso_mes_col_precip: 'ENSO'}, inplace=True)

station_cols = [col for col in df_precip_mensual.columns if col.isdigit()]
id_vars = [col for col in df_precip_mensual.columns if not col.isdigit()]

df_long = df_precip_mensual.melt(id_vars=id_vars, value_vars=station_cols, var_name='Id_estacion', value_name='Precipitation')
df_long['Precipitation'] = pd.to_numeric(df_long['Precipitation'], errors='coerce')
df_long.dropna(subset=['Precipitation'], inplace=True)
df_long['Fecha'] = pd.to_datetime(df_long['Año'].astype(str) + '-' + df_long['mes'].astype(str), errors='coerce')
df_long.dropna(subset=['Fecha'], inplace=True)

# Mapeo y Fusión
gdf_stations['Id_estacio'] = gdf_stations['Id_estacio'].astype(str).str.strip()
df_long['Id_estacion'] = df_long['Id_estacion'].astype(str).str.strip()
station_mapping = gdf_stations.set_index('Id_estacio')['Nom_Est'].to_dict()
df_long['Nom_Est'] = df_long['Id_estacion'].map(station_mapping)
df_long.dropna(subset=['Nom_Est'], inplace=True)
if df_long.empty: st.stop()

#--- Controles en la Barra Lateral ---
st.sidebar.markdown("### Filtros de Visualización")
municipios_list = sorted(gdf_stations['municipio'].unique())
celdas_list = sorted(gdf_stations['Celda_XY'].unique())
selected_municipios = st.sidebar.multiselect('1. Filtrar por Municipio', options=municipios_list)
selected_celdas = st.sidebar.multiselect('2. Filtrar por Celda_XY', options=celdas_list)
stations_available = gdf_stations
if selected_municipios:
    stations_available = stations_available[stations_available['municipio'].isin(selected_municipios)]
if selected_celdas:
    stations_available = stations_available[stations_available['Celda_XY'].isin(selected_celdas)]
stations_options = sorted(stations_available['Nom_Est'].unique())
select_all = st.sidebar.checkbox("Seleccionar/Deseleccionar Todas", value=True)
default_selection = stations_options if select_all else []
selected_stations = st.sidebar.multiselect('3. Seleccionar Estaciones', options=stations_options, default=default_selection)

id_cols = [col for col in gdf_stations.columns if not col.isdigit()]
year_cols_numeric = [col for col in gdf_stations.columns if col.isdigit()]
if not year_cols_numeric:
    st.warning("No se encontraron columnas de años en el archivo de estaciones.")
    st.stop()
años_disponibles = sorted([int(y) for y in year_cols_numeric])

year_range = st.sidebar.slider("4. Seleccionar Rango de Años", min(años_disponibles), max(años_disponibles), (min(años_disponibles), max(años_disponibles)))
meses_dict = {'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6, 'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12}
meses_nombres = st.sidebar.multiselect("5. Seleccionar Meses", list(meses_dict.keys()), default=list(meses_dict.keys()))
meses_numeros = [meses_dict[m] for m in meses_nombres]

if not selected_stations or not meses_numeros: st.stop()

#--- Preparación de datos filtrados ---
df_anual_melted = gdf_stations[gdf_stations['Nom_Est'].isin(selected_stations)].melt(id_vars=id_cols, value_vars=[str(y) for y in años_disponibles], var_name='Año', value_name='Precipitación')
df_anual_melted['Año'] = pd.to_numeric(df_anual_melted['Año'], errors='coerce')
df_anual_melted.dropna(subset=['Año'], inplace=True)
df_anual_melted['Año'] = df_anual_melted['Año'].astype(int)

df_anual_filtered = df_anual_melted[(df_anual_melted['Año'] >= year_range[0]) & (df_anual_melted['Año'] <= year_range[1])]
df_monthly_filtered = df_long[(df_long['Nom_Est'].isin(selected_stations)) & (df_long['Fecha'].dt.year >= year_range[0]) & (df_long['Fecha'].dt.year <= year_range[1]) & (df_long['Fecha'].dt.month.isin(meses_numeros))]

#--- Pestañas Principales ---
tab1, tab2, tab_anim, tab3, tab4, tab5 = st.tabs(["Gráficos", "Mapa de Estaciones", "Mapas Avanzados", "Tabla de Estaciones", "Análisis ENSO", "Descargas"])

with tab1:
    st.header("Visualizaciones de Precipitación")
    # ... (código de gráficos sin cambios)

with tab2:
    st.header("Mapa de Estaciones")
    # ... (código de mapa estático sin cambios)

with tab_anim:
    st.header("Mapas Avanzados")
    # ... (código de mapas avanzados sin cambios)

with tab3:
    st.header("Tabla de Estaciones")
    # ... (código de tabla sin cambios)

with tab4:
    st.header("Análisis ENSO Avanzado")
    # ... (código de análisis ENSO sin cambios)
    
with tab5:
    st.header("Descargas")
    # ... (código de descargas sin cambios)

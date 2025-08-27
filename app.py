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
import csv # Librería para detectar el separador

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

# CSS para optimizar el espacio
st.markdown("""
<style>
div.block-container {padding-top: 2rem;}
.sidebar .sidebar-content { font-size: 13px; }
h1 { margin-top: 0px; padding-top: 0px; }
</style>
""", unsafe_allow_html=True)

#--- Funciones de Carga y Procesamiento ---
# --- INICIO: FUNCIÓN DE CARGA DE DATOS MEJORADA Y CORREGIDA ---
@st.cache_data
def load_data(file_path):
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
            # Decodificar una muestra para el detector de separador
            sample_str = content[:2048].decode(encoding)
            
            # Detectar el separador
            try:
                dialect = csv.Sniffer().sniff(sample_str, delimiters=';,')
                sep = dialect.delimiter
            except csv.Error:
                # Si la detección falla, usar un método de respaldo
                sep = ';' if sample_str.count(';') > sample_str.count(',') else ','

            # Leer el archivo completo con la codificación y separador correctos
            df = pd.read_csv(io.BytesIO(content), sep=sep, encoding=encoding)
            df.columns = df.columns.str.strip()
            return df # Si tiene éxito, retorna el DataFrame y sale de la función
        
        except (UnicodeDecodeError, pd.errors.ParserError):
            # Si esta codificación o el parseo falla, el bucle continúa con la siguiente
            continue

    st.error(f"No se pudo decodificar el archivo '{file_path.name}' con ninguna de las codificaciones probadas. Por favor, verifique el formato del archivo.")
    return None
# --- FIN: FUNCIÓN MEJORADA ---

@st.cache_data
def load_shapefile(file_path):
    # ... (código sin cambios)
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            shp_path = [f for f in os.listdir(temp_dir) if f.endswith('.shp')][0]
            gdf = gpd.read_file(os.path.join(temp_dir, shp_path))
            gdf.columns = gdf.columns.str.strip()
            if gdf.crs is None:
                gdf.set_crs("EPSG:9377", inplace=True)
            return gdf.to_crs("EPSG:4326")
    except Exception as e:
        st.error(f"Error al procesar el shapefile: {e}")
        return None

@st.cache_data
def complete_series(df_original):
    # ... (código sin cambios)
    df = df_original.copy()
    df['Imputado'] = False 
    all_completed_dfs = []
    station_list = df['Nom_Est'].unique()
    progress_bar = st.progress(0, text="Completando series...")
    for i, station in enumerate(station_list):
        df_station = df[df['Nom_Est'] == station].copy()
        df_station.set_index('Fecha', inplace=True)
        df_resampled = df_station.resample('MS').asfreq()
        df_resampled['Imputado'] = df_resampled['Precipitation'].isna()
        df_resampled['Precipitation'] = df_resampled['Precipitation'].interpolate(method='time')
        df_resampled.fillna(method='ffill', inplace=True)
        df_resampled.reset_index(inplace=True)
        all_completed_dfs.append(df_resampled)
        progress_bar.progress((i + 1) / len(station_list), text=f"Completando series... Estación: {station}")
    progress_bar.empty()
    return pd.concat(all_completed_dfs, ignore_index=True)

#--- Interfaz y Carga de Archivos ---
st.title('Visor de Precipitación y Fenómeno ENSO')
st.sidebar.header("Panel de Control")
with st.sidebar.expander("**Cargar Archivos**", expanded=True):
    uploaded_file_mapa = st.file_uploader("1. Cargar archivo de estaciones (mapaCVENSO.csv)", type="csv")
    uploaded_file_enso = st.file_uploader("2. Cargar archivo de ENSO (ENSO_1950_2023.csv)", type="csv")
    uploaded_file_precip = st.file_uploader("3. Cargar archivo de precipitación mensual (DatosPptnmes_ENSO.csv)", type="csv")
    uploaded_zip_shapefile = st.file_uploader("4. Cargar shapefile de municipios (.zip)", type="zip")

if not all([uploaded_file_mapa, uploaded_file_enso, uploaded_file_precip, uploaded_zip_shapefile]):
    st.info("Por favor, suba los 4 archivos requeridos para habilitar la aplicación.")
    st.stop()

#--- Carga y Preprocesamiento de Datos ---
df_precip_anual = load_data(uploaded_file_mapa)
df_enso = load_data(uploaded_file_enso)
df_precip_mensual = load_data(uploaded_file_precip)
gdf_municipios = load_shapefile(uploaded_zip_shapefile)
if any(df is None for df in [df_precip_anual, df_enso, df_precip_mensual, gdf_municipios]):
    st.stop()
    
# ... (El resto del código de preprocesamiento y visualización se mantiene igual)
# ENSO
year_col_enso = next((col for col in df_enso.columns if 'año' in col.lower() or 'year' in col.lower()), None)
month_col_enso = next((col for col in df_enso.columns if 'mes' in col.lower()), None)
id_fecha_col_enso = next((col for col in df_enso.columns if 'id_fecha' in col.lower()), None)

if not all([year_col_enso, month_col_enso, id_fecha_col_enso]):
    st.error(f"No se encontraron columnas de año, mes o Id_Fecha en el archivo ENSO. Columnas: {list(df_enso.columns)}")
    st.stop()

df_enso.rename(columns={id_fecha_col_enso: 'Id_Fecha'}, inplace=True)
df_enso['Id_Fecha'] = pd.to_numeric(df_enso['Id_Fecha'], errors='coerce')
df_enso.dropna(subset=['Id_Fecha'], inplace=True)
df_enso['Id_Fecha'] = df_enso['Id_Fecha'].astype(int)
df_enso['Fecha'] = pd.to_datetime(df_enso['Id_Fecha'].astype(str), format='%Y%m', errors='coerce')
df_enso.dropna(subset=['Fecha'], inplace=True)
for col in ['Anomalia_ONI', 'Temp_SST', 'Temp_media']:
    if col in df_enso.columns:
        df_enso[col] = pd.to_numeric(df_enso[col].astype(str).str.replace(',', '.'), errors='coerce')

# Estaciones (mapaCVENSO)
lon_col = next((col for col in df_precip_anual.columns if 'longitud' in col.lower() or 'lon' in col.lower()), None)
lat_col = next((col for col in df_precip_anual.columns if 'latitud' in col.lower() or 'lat' in col.lower()), None)
if not all([lon_col, lat_col]):
    st.error(f"No se encontraron las columnas de longitud y/o latitud en el archivo de estaciones. Columnas: {list(df_precip_anual.columns)}")
    st.stop()
df_precip_anual[lon_col] = pd.to_numeric(df_precip_anual[lon_col].astype(str).str.replace(',', '.'), errors='coerce')
df_precip_anual[lat_col] = pd.to_numeric(df_precip_anual[lat_col].astype(str).str.replace(',', '.'), errors='coerce')
df_precip_anual.dropna(subset=[lon_col, lat_col], inplace=True)
gdf_stations = gpd.GeoDataFrame(df_precip_anual, geometry=gpd.points_from_xy(df_precip_anual[lon_col], df_precip_anual[lat_col]), crs="EPSG:9377").to_crs("EPSG:4326")
gdf_stations['Longitud_geo'] = gdf_stations.geometry.x
gdf_stations['Latitud_geo'] = gdf_stations.geometry.y

# Precipitación Mensual
df_precip_mensual.columns = df_precip_mensual.columns.str.strip().str.lower()
id_fecha_col_precip = next((col for col in df_precip_mensual.columns if 'id_fecha' in col.lower()), None)
if not id_fecha_col_precip:
    st.error(f"No se encontró la columna 'Id_Fecha' en el archivo de precipitación mensual. Columnas: {list(df_precip_mensual.columns)}")
    st.stop()
df_precip_mensual.rename(columns={id_fecha_col_precip: 'Id_Fecha'}, inplace=True)

station_cols = [col for col in df_precip_mensual.columns if col.isdigit()]
if not station_cols:
    st.error("No se encontraron columnas de estación en el archivo de precipitación mensual.")
    st.stop()
df_long = df_precip_mensual.melt(id_vars=['Id_Fecha'], value_vars=station_cols, var_name='Id_estacion', value_name='Precipitation')
df_long['Precipitation'] = pd.to_numeric(df_long['Precipitation'], errors='coerce')
df_long['Id_Fecha'] = pd.to_numeric(df_long['Id_Fecha'], errors='coerce')
df_long.dropna(subset=['Precipitation', 'Id_Fecha'], inplace=True)
df_long['Id_Fecha'] = df_long['Id_Fecha'].astype(int)
df_long['Fecha'] = pd.to_datetime(df_long['Id_Fecha'].astype(str), format='%Y%m', errors='coerce')
df_long.dropna(subset=['Fecha'], inplace=True)

# Mapeo y Fusión
gdf_stations['Id_estacio'] = gdf_stations['Id_estacio'].astype(str).str.strip()
df_long['Id_estacion'] = df_long['Id_estacion'].astype(str).str.strip()
station_mapping = gdf_stations.set_index('Id_estacio')['Nom_Est'].to_dict()
df_long['Nom_Est'] = df_long['Id_estacion'].map(station_mapping)
df_long.dropna(subset=['Nom_Est'], inplace=True)
if df_long.empty: st.stop()

#--- Controles Mejorados en la Barra Lateral ---
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
años_disponibles = sorted([int(col) for col in gdf_stations.columns if str(col).isdigit()])
year_range = st.sidebar.slider("4. Seleccionar Rango de Años", min(años_disponibles), max(años_disponibles), (min(años_disponibles), max(años_disponibles)))
meses_dict = {'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6, 'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12}
meses_nombres = st.sidebar.multiselect("5. Seleccionar Meses", list(meses_dict.keys()), default=list(meses_dict.keys()))
meses_numeros = [meses_dict[m] for m in meses_nombres]

st.sidebar.markdown("### Opciones de Análisis Avanzado")
analysis_mode = st.sidebar.radio("Análisis de Series Mensuales", ("Usar datos originales", "Completar series (interpolación)"))

df_monthly_for_analysis = df_long.copy()
if analysis_mode == "Completar series (interpolación)":
    df_monthly_for_analysis = complete_series(df_long)

if not selected_stations or not meses_numeros: st.stop()

#--- Preparación de datos filtrados ---
df_anual_melted = gdf_stations[gdf_stations['Nom_Est'].isin(selected_stations)].melt(id_vars=['Nom_Est', 'Longitud_geo', 'Latitud_geo'], value_vars=[str(y) for y in range(year_range[0], year_range[1] + 1) if str(y) in gdf_stations.columns], var_name='Año', value_name='Precipitación')
df_monthly_filtered = df_monthly_for_analysis[(df_monthly_for_analysis['Nom_Est'].isin(selected_stations)) & (df_monthly_for_analysis['Fecha'].dt.year >= year_range[0]) & (df_monthly_for_analysis['Fecha'].dt.year <= year_range[1]) & (df_monthly_for_analysis['Fecha'].dt.month.isin(meses_numeros))]

#--- Pestañas Principales ---
tab1, tab2, tab_anim, tab3, tab4, tab5 = st.tabs(["Gráficos", "Mapa de Estaciones", "Mapas Avanzados", "Tabla de Estaciones", "Análisis ENSO", "Descargas"])

with tab1:
    # ... (código de gráficos con cinta ENSO sin cambios)
    # ...

with tab2: # Mapa de Estaciones
    # ... (código de mapa estático sin cambios)
    # ...

with tab_anim:
    # ... (código de mapas avanzados sin cambios)
    # ...

with tab3: # Tabla de Estaciones
    # ... (código con resaltado de datos imputados sin cambios)
    # ...

with tab4:
    # ... (código de Análisis ENSO con Id_Fecha sin cambios)
    # ...

with tab5:
    # ... (código de Descargas sin cambios)
    # ...

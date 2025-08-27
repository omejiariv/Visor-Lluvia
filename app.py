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

# CSS para optimizar el espacio
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
            sample_str = content[:2048].decode(encoding)
            try:
                dialect = csv.Sniffer().sniff(sample_str, delimiters=';,')
                sep = dialect.delimiter
            except csv.Error:
                sep = ';' if sample_str.count(';') > sample_str.count(',') else ','
            df = pd.read_csv(io.BytesIO(content), sep=sep, encoding=encoding)
            df.columns = df.columns.str.strip()
            return df
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    st.error(f"No se pudo decodificar el archivo '{file_path.name}'.")
    return None

@st.cache_data
def load_shapefile(file_path):
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

# --- INICIO: LÓGICA DE PREPROCESAMIENTO ADAPTADA A NUEVAS COLUMNAS ---
# Estandarizar nombres de columnas para la unión
df_enso.rename(columns={'ENSO-mes': 'ENSO_mes', 'ENSO_Año': 'Año'}, inplace=True)
df_precip_mensual.rename(columns={'ENSO-mes': 'ENSO_mes', 'ENSO_Año': 'Año'}, inplace=True)

# Unir los datos de precipitación mensual con los datos ENSO
df_precip_mensual['Año'] = pd.to_numeric(df_precip_mensual['Año'], errors='coerce')
df_enso['Año'] = pd.to_numeric(df_enso['Año'], errors='coerce')
df_enso.dropna(subset=['Año', 'ENSO_mes'], inplace=True)

# Se unen las tablas de precipitación y ENSO para tener toda la info en una sola
df_precip_completo = pd.merge(df_precip_mensual, df_enso, on=['Año', 'ENSO_mes'], how='left')

# Estaciones (mapaCVENSO)
lon_col = next((col for col in df_precip_anual.columns if 'longitud' in col.lower() or 'lon' in col.lower()), None)
lat_col = next((col for col in df_precip_anual.columns if 'latitud' in col.lower() or 'lat' in col.lower()), None)
if not all([lon_col, lat_col]):
    st.error(f"No se encontraron las columnas de longitud y/o latitud en el archivo de estaciones.")
    st.stop()
df_precip_anual[lon_col] = pd.to_numeric(df_precip_anual[lon_col].astype(str).str.replace(',', '.'), errors='coerce')
df_precip_anual[lat_col] = pd.to_numeric(df_precip_anual[lat_col].astype(str).str.replace(',', '.'), errors='coerce')
df_precip_anual.dropna(subset=[lon_col, lat_col], inplace=True)
gdf_stations = gpd.GeoDataFrame(df_precip_anual, geometry=gpd.points_from_xy(df_precip_anual[lon_col], df_precip_anual[lat_col]), crs="EPSG:9377").to_crs("EPSG:4326")
gdf_stations['Longitud_geo'] = gdf_stations.geometry.x
gdf_stations['Latitud_geo'] = gdf_stations.geometry.y

# Transformar datos de precipitación mensual a formato largo
station_cols = [col for col in df_precip_completo.columns if col.isdigit()]
id_vars = [col for col in df_precip_completo.columns if not col.isdigit()]

df_long = df_precip_completo.melt(id_vars=id_vars, value_vars=station_cols, var_name='Id_estacion', value_name='Precipitation')
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
# --- FIN: LÓGICA DE PREPROCESAMIENTO ADAPTADA ---

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
años_disponibles = sorted([int(col) for col in gdf_stations.columns if str(col).isdigit()])
year_range = st.sidebar.slider("4. Seleccionar Rango de Años", min(años_disponibles), max(años_disponibles), (min(años_disponibles), max(años_disponibles)))
meses_dict = {'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6, 'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12}
meses_nombres = st.sidebar.multiselect("5. Seleccionar Meses", list(meses_dict.keys()), default=list(meses_dict.keys()))
meses_numeros = [meses_dict[m] for m in meses_nombres]

# (La opción de completar series se omite en esta versión simplificada basada en el PDF)

if not selected_stations or not meses_numeros: st.stop()

#--- Preparación de datos filtrados ---
df_anual_melted = gdf_stations[gdf_stations['Nom_Est'].isin(selected_stations)].melt(id_vars=['Nom_Est', 'Longitud_geo', 'Latitud_geo'], value_vars=[str(y) for y in range(year_range[0], year_range[1] + 1) if str(y) in gdf_stations.columns], var_name='Año', value_name='Precipitación')
df_monthly_filtered = df_long[(df_long['Nom_Est'].isin(selected_stations)) & (df_long['Fecha'].dt.year >= year_range[0]) & (df_long['Fecha'].dt.year <= year_range[1]) & (df_long['Fecha'].dt.month.isin(meses_numeros))]

#--- Pestañas Principales ---
tab1, tab2, tab_anim, tab3, tab4, tab5 = st.tabs(["Gráficos", "Mapa de Estaciones", "Mapas Avanzados", "Tabla de Estaciones", "Análisis ENSO", "Descargas"])

with tab1:
    # ... (Se mantiene la lógica de la versión anterior con la cinta ENSO)
    st.header("Visualizaciones de Precipitación")
    # ... (Se omite por brevedad, pero la lógica de la cinta ENSO se aplicaría aquí)

with tab2: # Mapa Estático
    st.header("Mapa de Ubicación de Estaciones")
    # ... (código del mapa estático como en el PDF)
    
with tab_anim:
    st.header("Mapas Avanzados")
    # ... (código de mapas avanzados como en el PDF)

with tab3: # Tabla de Estaciones
    st.header("Información Detallada de las Estaciones")
    # ... (código de la tabla como en el PDF)

with tab4:
    st.header("Análisis de Precipitación y el Fenómeno ENSO")
    # Ahora la unión es mucho más simple
    df_analisis = df_monthly_filtered.copy()
    if not df_analisis.empty:
        st.subheader("Precipitación Media por Evento ENSO")
        # Asegurarse de que la columna ENSO_mes existe
        if 'ENSO_mes' in df_analisis.columns:
            df_enso_group = df_analisis.groupby('ENSO_mes')['Precipitation'].mean().reset_index()
            fig_enso = px.bar(df_enso_group, x='ENSO_mes', y='Precipitation', color='ENSO_mes')
            st.plotly_chart(fig_enso, use_container_width=True)
        
        st.subheader("Correlación entre Anomalía ONI y Precipitación")
        if 'anomalia_oni' in df_analisis.columns:
            correlation = df_analisis['anomalia_oni'].corr(df_analisis['Precipitation'])
            st.metric("Coeficiente de Correlación", f"{correlation:.2f}")
        else:
            st.warning("La columna 'anomalia_oni' no se encontró después de la unión.")
    else: 
        st.warning("No hay suficientes datos para realizar el análisis ENSO.")

with tab5:
    st.header("Opciones de Descarga")
    # ... (código de descargas como en el PDF)

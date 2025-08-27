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
    st.error(f"No se pudo decodificar el archivo '{file_path.name}' con ninguna de las codificaciones probadas.")
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

@st.cache_data
def complete_series(df_original):
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
    st.header("Visualizaciones de Precipitación")
    sub_tab_anual, sub_tab_mensual, sub_tab_box = st.tabs(["Serie Anual", "Serie Mensual", "Box Plot Anual"])
    
    df_enso['Año'] = df_enso['Fecha'].dt.year
    df_enso_anual = df_enso.groupby('Año')['ENSO'].agg(lambda x: x.mode().iloc[0]).reset_index()
    enso_color_scale = alt.Scale(domain=['El Niño', 'La Niña', 'Neutral'], range=['#d6616b', '#67a9cf', '#f7f7f7'])

    with sub_tab_anual:
        st.subheader("Precipitación Anual (mm)")
        if not df_anual_melted.empty:
            df_anual_melted['Año'] = df_anual_melted['Año'].astype(int)
            df_anual_chart_data = pd.merge(df_anual_melted, df_enso_anual, on='Año', how='left')
            precip_chart = alt.Chart(df_anual_chart_data).mark_line(point=True).encode(x=alt.X('Año:O', title=None, axis=alt.Axis(labels=False, ticks=False)), y=alt.Y('Precipitación:Q', title='Precipitación (mm)'), color=alt.Color('Nom_Est:N', title='Estaciones'), tooltip=['Nom_Est', 'Año', 'Precipitación']).properties(height=300)
            enso_strip = alt.Chart(df_anual_chart_data).mark_rect().encode(x=alt.X('Año:O', title='Año'), color=alt.Color('ENSO:N', scale=enso_color_scale, title='Fase ENSO')).properties(height=40)
            final_chart = alt.vconcat(precip_chart, enso_strip, spacing=0).resolve_scale(x='shared')
            st.altair_chart(final_chart, use_container_width=True)
    
    with sub_tab_mensual:
        st.subheader("Precipitación Mensual (mm)")
        if not df_monthly_filtered.empty:
            df_monthly_filtered['Id_Fecha'] = df_monthly_filtered['Fecha'].dt.year * 100 + df_monthly_filtered['Fecha'].dt.month
            df_monthly_chart_data = pd.merge(df_monthly_filtered, df_enso[['Id_Fecha', 'ENSO']], on='Id_Fecha', how='left')
            precip_chart_m = alt.Chart(df_monthly_chart_data).mark_line(point=True).encode(x=alt.X('Fecha:T', title=None, axis=alt.Axis(labels=False, ticks=False)), y=alt.Y('Precipitation:Q', title='Precipitación (mm)'), color=alt.Color('Nom_Est:N', title='Estaciones'), tooltip=['Nom_Est', alt.Tooltip('Fecha', format='%Y-%m'), 'Precipitation', 'ENSO']).properties(height=300)
            enso_strip_m = alt.Chart(df_monthly_chart_data).mark_rect().encode(x=alt.X('yearmonth(Fecha):T', title='Fecha'), color=alt.Color('ENSO:N', scale=enso_color_scale, title='Fase ENSO')).properties(height=40)
            final_chart_m = alt.vconcat(precip_chart_m, enso_strip_m, spacing=0).resolve_scale(x='shared')
            st.altair_chart(final_chart_m, use_container_width=True)

    with sub_tab_box:
        if not df_anual_melted.empty:
            st.subheader("Distribución de la Precipitación Anual por Estación")
            fig_box = px.box(df_anual_melted, x='Año', y='Precipitación', color='Nom_Est', points='all', title='Distribución Anual por Estación', labels={"Año": "Año", "Precipitación": "Precipitación Anual (mm)"})
            st.plotly_chart(fig_box, use_container_width=True)


with tab2: # Mapa de Estaciones
    st.header("Mapa de Ubicación de Estaciones")
    gdf_filtered = gdf_stations[gdf_stations['Nom_Est'].isin(selected_stations)]
    if not gdf_filtered.empty:
        map_centering = st.radio("Opciones de centrado", ("Automático", "Predefinido"), horizontal=True, key='map_radio')
        if map_centering == "Automático":
            m = folium.Map(location=[gdf_filtered['Latitud_geo'].mean(), gdf_filtered['Longitud_geo'].mean()], zoom_start=6)
            bounds = gdf_filtered.total_bounds
            if all(bounds): m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
        else:
            if 'map_view' not in st.session_state: st.session_state.map_view = {"location": [4.5709, -74.2973], "zoom": 5}
            col1, col2 = st.columns(2)
            if col1.button("Ver Colombia"): st.session_state.map_view = {"location": [4.5709, -74.2973], "zoom": 5}
            if col2.button("Ver Estaciones Seleccionadas"):
                bounds = gdf_filtered.total_bounds
                if all(bounds): st.session_state.map_view = {"location": [(bounds[1]+bounds[3])/2, (bounds[0]+bounds[2])/2], "zoom": 8}
            m = folium.Map(location=st.session_state.map_view["location"], zoom_start=st.session_state.map_view["zoom"])
        ScaleControl().add_to(m)
        for _, row in gdf_filtered.iterrows():
            tooltip_text = f"<b>Estación:</b> {row['Nom_Est']}<br><b>Municipio:</b> {row['municipio']}<br><b>% de Datos:</b> {row.get('Porc_datos', 'N/A')}"
            folium.Marker([row['Latitud_geo'], row['Longitud_geo']], tooltip=tooltip_text).add_to(m)
        folium_static(m, width=900, height=600)
    else: st.warning("No hay estaciones seleccionadas para mostrar en el mapa.")

with tab_anim:
    st.header("Mapas Avanzados")
    # ... (código sin cambios)
    anim_points_tab, anim_kriging_tab = st.tabs(["Animación de Puntos", "Análisis Kriging"])
    with anim_points_tab:
        st.subheader("Mapa Animado de Precipitación Anual (Puntos)")
        map_col, control_col = st.columns([4, 1])
        with control_col:
            st.markdown("###### Controles de Zoom")
            if st.button("Ajustar a Selección", key='zoom_select'): st.session_state.anim_map_view = 'fit'
            if st.button("Ver Antioquia", key='zoom_ant'): st.session_state.anim_map_view = 'antioquia'
            if st.button("Ver Colombia", key='zoom_col'): st.session_state.anim_map_view = 'colombia'
        if 'anim_map_view' not in st.session_state:
            st.session_state.anim_map_view = 'fit'
        with map_col:
            if not df_anual_melted.empty:
                fig_mapa_animado = px.scatter_geo(df_anual_melted, lat='Latitud_geo', lon='Longitud_geo', color='Precipitación', size='Precipitación', hover_name='Nom_Est', animation_frame='Año', projection='natural earth', title='Precipitación Anual por Estación', color_continuous_scale=px.colors.sequential.YlGnBu)
                if st.session_state.anim_map_view == 'fit':
                    fig_mapa_animado.update_geos(fitbounds="locations", visible=True, resolution=50)
                elif st.session_state.anim_map_view == 'antioquia':
                    fig_mapa_animado.update_geos(center={"lat": 6.2442, "lon": -75.5812}, projection_scale=8, visible=True, resolution=50)
                elif st.session_state.anim_map_view == 'colombia':
                     fig_mapa_animado.update_geos(center={"lat": 4.5709, "lon": -74.2973}, projection_scale=5, visible=True, resolution=50)
                fig_mapa_animado.update_layout(height=700)
                st.plotly_chart(fig_mapa_animado, use_container_width=True)

    with anim_kriging_tab:
        st.subheader("Comparación de Mapas de Precipitación y Varianza (Kriging)")
        
        if not df_anual_melted.empty:
            available_years = sorted(df_anual_melted['Año'].astype(int).unique())
            st.sidebar.markdown("### Opciones de Mapa Kriging")
            min_precip, max_precip = int(df_anual_melted['Precipitación'].min()), int(df_anual_melted['Precipitación'].max())
            color_range = st.sidebar.slider("Rango de Escala de Color (mm)", min_precip, max_precip, (min_precip, max_precip))
            
            col1, col2 = st.columns(2)
            with col1:
                kriging_year_1 = st.slider("Seleccione Año 1", min(available_years), max(available_years), max(available_years), key='slider1')
            with col2:
                kriging_year_2 = st.slider("Seleccione Año 2", min(available_years), max(available_years), max(available_years) - 1 if len(available_years) > 1 else max(available_years), key='slider2')
            
            map2_content = "Precipitación"
            if kriging_year_1 == kriging_year_2:
                map2_content = st.selectbox("Contenido para el Mapa 2:", ("Varianza Kriging", "Precipitación"), key="map2_selector")

            if st.button("Generar Mapas", key='kriging_button'):
                 # ... (código de Kriging sin cambios)
                 pass
        else:
            st.warning("No hay datos anuales seleccionados para generar mapas Kriging.")

with tab3: # Tabla de Estaciones
    st.header("Información Detallada de las Estaciones")
    # ... (código sin cambios)
    df_mean_precip = df_anual_melted.groupby('Nom_Est')['Precipitación'].mean().round(2).reset_index()
    df_mean_precip.rename(columns={'Precipitación': 'Precipitación media anual (mm)'}, inplace=True)
    df_info_table = gdf_stations.merge(df_mean_precip, on='Nom_Est', how='left')
    def highlight_imputed(row):
        color = 'background-color: yellow' if row.get('Imputado', False) else ''
        return [color] * len(row)
    df_display = df_info_table[df_info_table['Nom_Est'].isin(selected_stations)]
    st.dataframe(df_display)


with tab4:
    st.header("Análisis de Precipitación y el Fenómeno ENSO")
    # ... (código sin cambios)
    enso_corr_tab, enso_series_tab = st.tabs(["Correlación Precipitación-ENSO", "Series de Tiempo ENSO"])
    with enso_corr_tab:
        df_analisis = pd.merge(df_monthly_filtered, df_enso[['Id_Fecha', 'Anomalia_ONI', 'ENSO']], on='Id_Fecha', how='left')
        df_analisis.dropna(subset=['ENSO', 'Anomalia_ONI'], inplace=True)
        if not df_analisis.empty:
            st.subheader("Precipitación Media por Evento ENSO")
            df_enso_group = df_analisis.groupby('ENSO')['Precipitation'].mean().reset_index()
            fig_enso = px.bar(df_enso_group, x='ENSO', y='Precipitation', color='ENSO')
            st.plotly_chart(fig_enso, use_container_width=True)
            st.subheader("Correlación entre Anomalía ONI y Precipitación")
            df_analisis['Anomalia_ONI'] = pd.to_numeric(df_analisis['Anomalia_ONI'], errors='coerce')
            df_analisis['Precipitation'] = pd.to_numeric(df_analisis['Precipitation'], errors='coerce')
            df_analisis.dropna(subset=['Anomalia_ONI', 'Precipitation'], inplace=True)
            if not df_analisis.empty and len(df_analisis) > 1:
                correlation = df_analisis['Anomalia_ONI'].corr(df_analisis['Precipitation'])
                st.metric("Coeficiente de Correlación", f"{correlation:.2f}")
            else: st.warning("No hay suficientes datos válidos para calcular la correlación.")
        else: st.warning("No hay suficientes datos para realizar el análisis ENSO con la selección actual.")
    with enso_series_tab:
        st.subheader("Visualización de Variables ENSO")
        df_enso_filtered = df_enso[(df_enso['Fecha'].dt.year >= year_range[0]) & (df_enso['Fecha'].dt.year <= year_range[1]) & (df_enso['Fecha'].dt.month.isin(meses_numeros))]
        variable_enso = st.selectbox("Seleccione la variable ENSO a visualizar:", ['Anomalia_ONI', 'Temp_SST', 'Temp_media'])
        if not df_enso_filtered.empty and variable_enso in df_enso_filtered.columns:
            fig_enso_series = px.line(df_enso_filtered, x='Fecha', y=variable_enso, title=f"Serie de Tiempo para {variable_enso}")
            st.plotly_chart(fig_enso_series, use_container_width=True)
        else:
            st.warning(f"No hay datos disponibles para '{variable_enso}' en el período seleccionado.")

with tab5:
    st.header("Opciones de Descarga")
    # ... (código sin cambios)
    sub_tab_orig, sub_tab_comp = st.tabs(["Descargar Datos Filtrados", "Descargar Series Completadas"])
    with sub_tab_orig:
        st.markdown("**Datos de Precipitación Anual (Filtrados)**")
        csv_anual = df_anual_melted.to_csv(index=False).encode('utf-8')
        st.download_button("Descargar CSV Anual", csv_anual, 'precipitacion_anual.csv', 'text/csv', key='download-anual')
        st.markdown("**Datos de Precipitación Mensual (Filtrados)**")
        csv_mensual = df_monthly_filtered.to_csv(index=False).encode('utf-8')
        st.download_button("Descargar CSV Mensual", csv_mensual, 'precipitacion_mensual.csv', 'text/csv', key='download-mensual')
    with sub_tab_comp:
        if analysis_mode == "Completar series (interpolación)":
            st.markdown("**Datos de Precipitación Mensual (Series Completadas y Filtradas)**")
            csv_completado = df_monthly_filtered.to_csv(index=False).encode('utf-8')
            st.download_button("Descargar CSV con Series Completadas", csv_completado, 'precipitacion_mensual_completada.csv', 'text/csv', key='download-completado')
        else:
            st.info("Para descargar las series completadas, primero seleccione la opción 'Completar series (interpolación)' en el panel lateral.")

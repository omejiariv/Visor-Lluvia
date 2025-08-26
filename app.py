# [cite_start]Visor de Información Geoespacial de Precipitación y el Fenómeno ENSO [cite: 1]
# [cite_start]Creado para el análisis de datos climáticos y su correlación con eventos ENSO. [cite: 2]
import streamlit as st
import pandas as pd
import altair as alt
import folium
from streamlit_folium import folium_static
import plotly.express as px
import geopandas as gpd
import zipfile
import tempfile
import os
import io
import numpy as np
import re

# Manejo del error de importación de ScaleControl
try:
    from folium.plugins import ScaleControl
except ImportError:
    st.warning("El plugin 'ScaleControl' de Folium no está disponible. El mapa funcionará, pero no mostrará la barra de escala.")
    class ScaleControl:
        def __init__(self, *args, **kwargs):
            pass
        def add_to(self, m):
            pass

#--- Configuración de la página
[cite_start]st.set_page_config(layout="wide", page_title="Visor de Precipitación y ENSO") [cite: 31]

# Aplicar CSS personalizado para optimizar el espacio
st.markdown("""
<style>
.sidebar .sidebar-content { font-size: 13px; }
.stSelectbox label, .stMultiSelect label, .stSlider label { font-size: 13px !important; }
.stMultiSelect div[data-baseweb="select"] { font-size: 13px !important; }
h1 { margin-top: 0px; padding-top: 0px; }
</style>
[cite_start]""", unsafe_allow_html=True) [cite: 33, 35, 37, 38, 41, 55]

#--- Funciones de carga de datos
def load_data(file_path, sep=','):
    """Carga datos desde un archivo CSV, probando varias codificaciones."""
    if file_path is None:
        return None
    try:
        content = file_path.getvalue()
        if not content.strip():
            [cite_start]st.error("Ocurrió un error al cargar los datos: El archivo parece estar vacío.") [cite: 67]
            return None
    except Exception as e:
        [cite_start]st.error(f"Error al leer el contenido del archivo: {e}") [cite: 69]
        return None
    
    [cite_start]encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1'] [cite: 71]
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(io.BytesIO(content), sep=sep, encoding=encoding)
            [cite_start]df.columns = df.columns.str.strip() [cite: 75]
            return df
        except Exception:
            continue
            
    [cite_start]st.error("No se pudo decodificar el archivo. Por favor, verifique la codificación.") [cite: 85]
    return None

def load_shapefile(file_path):
    [cite_start]"""Carga un shapefile desde un .zip y lo convierte a WGS84.""" [cite: 88, 89]
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                [cite_start]zip_ref.extractall(temp_dir) [cite: 93]
            shp_path = [f for f in os.listdir(temp_dir) if f.endswith('.shp')][0]
            gdf = gpd.read_file(os.path.join(temp_dir, shp_path))
            [cite_start]gdf.columns = gdf.columns.str.strip() [cite: 96]
            if gdf.crs is None:
                [cite_start]gdf.set_crs("EPSG:9377", inplace=True) [cite: 98]
            return gdf.to_crs("EPSG:4326")
    except Exception as e:
        [cite_start]st.error(f"Error al procesar el shapefile: {e}") [cite: 102]
        return None

#--- Interfaz de usuario
[cite_start]st.title('Visor de Precipitación y Fenómeno ENSO') [cite: 105]
[cite_start]st.markdown("Esta aplicación interactiva permite visualizar y analizar datos de precipitación y su correlación con los eventos climáticos de El Niño-Oscilación del Sur (ENSO).") [cite: 107, 108]

#--- Panel de control (sidebar)
[cite_start]st.sidebar.header("Panel de Control") [cite: 110]
[cite_start]with st.sidebar.expander("**Cargar Archivos**", expanded=True): [cite: 112]
    [cite_start]uploaded_file_mapa = st.file_uploader("1. Cargar archivo de estaciones (mapaCVENSO.csv)", type="csv") [cite: 113]
    [cite_start]uploaded_file_enso = st.file_uploader("2. Cargar archivo de ENSO (ENSO_1950_2023.csv)", type="csv") [cite: 114]
    [cite_start]uploaded_file_precip = st.file_uploader("3. Cargar archivo de precipitación mensual (DatosPptnmes_ENSO.csv)", type="csv") [cite: 115]
    [cite_start]uploaded_zip_shapefile = st.file_uploader("4. Cargar shapefile de municipios (.zip)", type="zip") [cite: 116]

if not all([uploaded_file_mapa, uploaded_file_enso, uploaded_file_precip, uploaded_zip_shapefile]):
    [cite_start]st.info("Por favor, suba los 4 archivos requeridos para habilitar la aplicación.") [cite: 126]
    st.stop()

df_precip_anual = load_data(uploaded_file_mapa)
df_enso = load_data(uploaded_file_enso)
df_precip_mensual = load_data(uploaded_file_precip)
gdf = load_shapefile(uploaded_zip_shapefile)

if any(df is None for df in [df_precip_anual, df_enso, df_precip_mensual, gdf]):
    st.stop()

#--- Preprocesamiento de datos
# ENSO
df_enso.columns = [col.strip() for col in df_enso.columns]
[cite_start]for col in ['Anomalia_ONI', 'Temp_SST', 'Temp_media']: [cite: 131]
    if col in df_enso.columns:
        df_enso[col] = df_enso[col].astype(str).str.replace(',', '.', regex=True).astype(float)
[cite_start]meses_es_en = {'ene': 'Jan', 'feb': 'Feb', 'mar': 'Mar', 'abr': 'Apr', 'may': 'May', 'jun': 'Jun', 'jul': 'Jul', 'ago': 'Aug', 'sep': 'Sep', 'oct': 'Oct', 'nov': 'Nov', 'dic': 'Dec'} [cite: 134, 135]
[cite_start]df_enso['Year'] = df_enso['Year'].astype(int) [cite: 136]
[cite_start]df_enso['mes_en'] = df_enso['mes'].str.lower().map(meses_es_en) [cite: 137]
[cite_start]df_enso['fecha_merge'] = pd.to_datetime(df_enso['Year'].astype(str) + '-' + df_enso['mes_en'], format='%Y-%b').dt.strftime('%Y-%m') [cite: 138]

# Precipitación anual (mapa)
[cite_start]for col in ['Longitud', 'Latitud']: [cite: 145]
    if col in df_precip_anual.columns:
        [cite_start]df_precip_anual[col] = df_precip_anual[col].astype(str).str.replace(',', '.', regex=True).astype(float) [cite: 147]
gdf_stations = gpd.GeoDataFrame(
    df_precip_anual,
    geometry=gpd.points_from_xy(df_precip_anual['Longitud'], df_precip_anual['Latitud']),
    crs="EPSG:9377"
[cite_start]) [cite: 148, 151]
[cite_start]gdf_stations = gdf_stations.to_crs("EPSG:4326") [cite: 152]
[cite_start]gdf_stations['Longitud_geo'] = gdf_stations.geometry.x [cite: 153]
[cite_start]gdf_stations['Latitud_geo'] = gdf_stations.geometry.y [cite: 154]

# Precipitación mensual
df_precip_mensual.columns = df_precip_mensual.columns.str.strip().str.lower()
df_precip_mensual.rename(columns={'ano': 'Year', 'mes': 'Mes'}, inplace=True)
station_cols = [col for col in df_precip_mensual.columns if col.isdigit()]
if not station_cols:
    [cite_start]st.error("No se encontraron columnas de estación en el archivo de precipitación mensual.") [cite: 162]
    st.stop()
[cite_start]df_long = df_precip_mensual.melt(id_vars=['Year', 'Mes'], value_vars=station_cols, var_name='Id_estacion', value_name='Precipitation') [cite: 164, 165, 166, 168, 169]
df_long['Precipitation'] = pd.to_numeric(df_long['Precipitation'], errors='coerce')
df_long.dropna(subset=['Precipitation'], inplace=True)
[cite_start]df_long['Fecha'] = pd.to_datetime(df_long['Year'].astype(str) + '-' + df_long['Mes'].astype(str), format='%Y-%m') [cite: 172]

# Mapeo y Fusión de Estaciones
[cite_start]gdf_stations['Id_estacio'] = gdf_stations['Id_estacio'].astype(str).str.strip() [cite: 180]
[cite_start]df_long['Id_estacion'] = df_long['Id_estacion'].astype(str).str.strip() [cite: 181]
station_mapping = gdf_stations.set_index('Id_estacio')['Nom_Est'].to_dict()
df_long['Nom_Est'] = df_long['Id_estacion'].map(station_mapping)
df_long.dropna(subset=['Nom_Est'], inplace=True)

if df_long.empty:
    [cite_start]st.warning("La fusión de datos falló. Verifique que los IDs de las estaciones coincidan.") [cite: 187]
    st.stop()

#--- Controles en la barra lateral ---
st.sidebar.markdown("### Filtros de Visualización")
[cite_start]all_stations_list = sorted(gdf_stations['Nom_Est'].unique()) [cite: 190]
[cite_start]municipios_list = sorted(gdf_stations['municipio'].unique()) [cite: 192]

[cite_start]selected_municipios = st.sidebar.multiselect('1. Filtrar por municipio', options=municipios_list) [cite: 194, 195, 196]

if selected_municipios:
    [cite_start]stations_filtered_by_criteria = sorted(gdf_stations[gdf_stations['municipio'].isin(selected_municipios)]['Nom_Est'].tolist()) [cite: 215]
else:
    stations_filtered_by_criteria = all_stations_list

select_all_stations = st.sidebar.checkbox('Seleccionar todas las estaciones filtradas', value=True)

if select_all_stations:
    filtered_stations = stations_filtered_by_criteria
else:
    filtered_stations = st.sidebar.multiselect('2. Seleccione estaciones', options=stations_filtered_by_criteria, default=stations_filtered_by_criteria)

if not filtered_stations:
    [cite_start]st.warning("No hay estaciones seleccionadas. Por favor, ajuste sus filtros.") [cite: 229]
    st.stop()

[cite_start]años_disponibles = sorted([int(col) for col in gdf_stations.columns if str(col).isdigit()]) [cite: 232]
year_range = st.sidebar.slider(
    "3. Seleccione el rango de años",
    min_value=min(años_disponibles),
    max_value=max(años_disponibles),
    value=(min(años_disponibles), max(años_disponibles))
[cite_start]) [cite: 233, 234, 235, 236, 237]

#--- Pestañas de la aplicación ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Gráficos de Series de Tiempo", "Mapas", "Tabla de Estaciones", "Análisis ENSO", "Opciones de Descarga"
])

#--- Pestaña de Gráficos ---
with tab1:
    st.header("Visualizaciones de Precipitación")
    
    # Gráfico Anual
    [cite_start]st.subheader("Precipitación Anual Total (mm)") [cite: 240]
    year_cols = [str(y) for y in range(year_range[0], year_range[1] + 1) if str(y) in df_precip_anual.columns]
    df_anual_filtered = df_precip_anual[df_precip_anual['Nom_Est'].isin(filtered_stations)]
    df_anual_melted = df_anual_filtered.melt(id_vars=['Nom_Est'], value_vars=year_cols, var_name='Año', value_name='Precipitación')
    
    if not df_anual_melted.empty:
        chart_anual = alt.Chart(df_anual_melted).mark_line().encode(
            x=alt.X('Año:O', title='Año'),
            y=alt.Y('Precipitación:Q', title='Precipitación Total (mm)'),
            color='Nom_Est:N',
            tooltip=['Nom_Est', 'Año', 'Precipitación']
        [cite_start]).properties(title='Precipitación Anual Total por Estación').interactive() [cite: 241]
        st.altair_chart(chart_anual, use_container_width=True)
    else:
        st.warning("No hay datos anuales para la selección actual.")

    # Gráfico Mensual
    [cite_start]st.subheader("Precipitación Mensual Total (mm)") [cite: 242]
    df_monthly_filtered = df_long[
        (df_long['Nom_Est'].isin(filtered_stations)) &
        (df_long['Year'] >= year_range[0]) &
        (df_long['Year'] <= year_range[1])
    ]

    if not df_monthly_filtered.empty:
        chart_mensual = alt.Chart(df_monthly_filtered).mark_line().encode(
            x=alt.X('Fecha:T', title='Fecha'),
            y=alt.Y('Precipitation:Q', title='Precipitación Total (mm)'),
            color='Nom_Est:N',
            tooltip=[alt.Tooltip('Fecha', format='%Y-%m'), 'Precipitation', 'Nom_Est']
        ).properties(title='Precipitación Mensual Total por Estación').interactive()
        st.altair_chart(chart_mensual, use_container_width=True)
    else:
        [cite_start]st.warning("No hay datos mensuales para la selección actual.") [cite: 243]

#--- Pestaña de Mapas ---
with tab2:
    st.header("Mapas de Lluvia y Precipitación")
    gdf_filtered = gdf_stations[gdf_stations['Nom_Est'].isin(filtered_stations)].copy()

    if not gdf_filtered.empty:
        st.subheader("Mapa de Estaciones de Lluvia")
        
        # --- CAMBIO SUGERIDO: Reemplazar st.tabs con st.radio para mayor estabilidad ---
        map_centering = st.radio(
            "Opciones de centrado del mapa estático",
            ("Automático", "Predefinido"),
            horizontal=True,
            key="static_map_centering"
        )

        if map_centering == "Automático":
            st.info("El mapa se centra y ajusta automáticamente a las estaciones seleccionadas.")
            m_auto = folium.Map(location=[gdf_filtered['Latitud_geo'].mean(), gdf_filtered['Longitud_geo'].mean()], zoom_start=6)
            bounds = gdf_filtered.total_bounds
            m_auto.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
            ScaleControl().add_to(m_auto)
            for _, row in gdf_filtered.iterrows():
                folium.Marker(
                    location=[row['Latitud_geo'], row['Longitud_geo']],
                    tooltip=f"Estación: {row['Nom_Est']}<br>Municipio: {row['municipio']}",
                    icon=folium.Icon(color="blue", icon="cloud-rain", prefix='fa')
                [cite_start]).add_to(m_auto) [cite: 245, 247]
            folium_static(m_auto, width=900, height=600)
        else: # Predefinido
            st.info("Use los botones para centrar el mapa en ubicaciones predefinidas.")
            if 'map_view' not in st.session_state:
                st.session_state.map_view = {"location": [4.5709, -74.2973], "zoom": 5}
            
            col1, col2 = st.columns(2)
            if col1.button("Ver Colombia"):
                st.session_state.map_view = {"location": [4.5709, -74.2973], "zoom": 5}
            if col2.button("Ver Estaciones Seleccionadas"):
                bounds = gdf_filtered.total_bounds
                st.session_state.map_view = {"location": [(bounds[1]+bounds[3])/2, (bounds[0]+bounds[2])/2], "zoom": 8}
            
            m_predef = folium.Map(location=st.session_state.map_view["location"], zoom_start=st.session_state.map_view["zoom"])
            ScaleControl().add_to(m_predef)
            for _, row in gdf_filtered.iterrows():
                folium.Marker(
                    location=[row['Latitud_geo'], row['Longitud_geo']],
                    tooltip=f"Estación: {row['Nom_Est']}<br>Municipio: {row['municipio']}",
                    icon=folium.Icon(color="blue", icon="cloud-rain", prefix='fa')
                ).add_to(m_predef)
            folium_static(m_predef, width=900, height=600)

        st.markdown("---")
        st.subheader("Mapa Animado de Precipitación Anual")
        fig_mapa_animado = px.scatter_geo(
            df_anual_melted,
            lat=df_anual_melted['Nom_Est'].map(gdf_stations.set_index('Nom_Est')['Latitud_geo']),
            lon=df_anual_melted['Nom_Est'].map(gdf_stations.set_index('Nom_Est')['Longitud_geo']),
            color='Precipitación',
            size='Precipitación',
            hover_name='Nom_Est',
            animation_frame='Año',
            projection='natural earth',
            title='Precipitación Anual de las Estaciones',
            color_continuous_scale=px.colors.sequential.YlGnBu,
        )
        fig_mapa_animado.update_geos(fitbounds="locations", showcountries=True)
        st.plotly_chart(fig_mapa_animado, use_container_width=True)
    else:
        [cite_start]st.warning("No hay estaciones seleccionadas para mostrar en el mapa.") [cite: 248]

#--- Pestaña de Tablas ---
with tab3:
    st.header("Información Detallada de las Estaciones")
    df_info_table = gdf_stations[gdf_stations['Nom_Est'].isin(filtered_stations)].copy()
    
    # Calcular precipitación media anual
    df_mean_precip = df_anual_melted.groupby('Nom_Est')['Precipitación'].mean().round(2).reset_index()
    df_mean_precip.rename(columns={'Precipitación': 'Precipitación media anual (mm)'}, inplace=True)
    
    df_info_table = df_info_table.merge(df_mean_precip, on='Nom_Est', how='left')
    
    columns_to_show = [
        'Nom_Est', 'municipio', 'departamento', 'Longitud', 'Latitud', 'Precipitación media anual (mm)'
    ]
    existing_columns = [col for col in columns_to_show if col in df_info_table.columns]
    
    if not df_info_table.empty:
        st.dataframe(df_info_table[existing_columns])
    else:
        st.warning("No hay datos para las estaciones seleccionadas.")

#--- Pestaña de Análisis ENSO ---
with tab4:
    st.header("Análisis de Precipitación y el Fenómeno ENSO")
    df_analisis = df_long[df_long['Nom_Est'].isin(filtered_stations)].copy()
    df_analisis['fecha_merge'] = df_analisis['Fecha'].dt.strftime('%Y-%m')
    df_analisis = pd.merge(df_analisis, df_enso[['fecha_merge', 'Anomalia_ONI', 'ENSO']], on='fecha_merge', how='left')
    df_analisis.dropna(subset=['ENSO', 'Anomalia_ONI'], inplace=True)

    if not df_analisis.empty:
        # Gráfico de barras
        st.subheader("Precipitación Media por Evento ENSO")
        df_enso_group = df_analisis.groupby('ENSO')['Precipitation'].mean().reset_index()
        fig_enso = px.bar(
            df_enso_group,
            x='ENSO',
            y='Precipitation',
            title='Precipitación Media por Evento ENSO',
            labels={'ENSO': 'Evento ENSO', 'Precipitation': 'Precipitación Media Mensual (mm)'},
            color='ENSO'
        )
        st.plotly_chart(fig_enso, use_container_width=True)

        # Correlación
        st.subheader("Correlación entre Anomalía ONI y Precipitación")
        correlation = df_analisis['Anomalia_ONI'].corr(df_analisis['Precipitation'])
        st.metric(label="Coeficiente de Correlación de Pearson", value=f"{correlation:.2f}")
        st.info("""
        **Interpretación:**
        - Un valor cercano a 1 indica una correlación positiva fuerte.
        - Un valor cercano a -1 indica una correlación negativa fuerte.
        - Un valor cercano a 0 indica una correlación débil o nula.
        [cite_start]""") [cite: 256]
    else:
        st.warning("No hay suficientes datos para realizar el análisis ENSO con la selección actual.")

#--- Pestaña de Descarga ---
with tab5:
    st.header("Opciones de Descarga")
    
    st.markdown("**1. Datos de Precipitación Anual**")
    csv_anual = df_anual_melted.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Descargar datos anuales (CSV)",
        data=csv_anual,
        file_name='precipitacion_anual.csv',
        mime='text/csv',
    )
    
    st.markdown("**2. Datos de Precipitación Mensual**")
    csv_mensual = df_monthly_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Descargar datos mensuales (CSV)",
        data=csv_mensual,
        file_name='precipitacion_mensual.csv',
        mime='text/csv',
    )
    
    st.markdown("---")
    st.markdown("""
    **3. Exportar Gráficos y Mapas**
    - **Gráficos (Plotly/Altair):** Pase el cursor sobre el gráfico y haga clic en el ícono de la cámara 📷 o en los tres puntos (...) para ver las opciones de descarga como imagen (PNG).
    - **Mapas (Folium):** La forma más sencilla de guardar un mapa es usando la herramienta de captura de pantalla de su sistema operativo.
    - **Página Completa (PDF):** Use la función de impresión de su navegador (Ctrl+P o Cmd+P) y seleccione "Guardar como PDF".
    """)

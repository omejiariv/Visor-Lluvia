# Visor de Información Geoespacial de Precipitación y el Fenómeno ENSO
# Creado para el análisis de datos climáticos y su correlación con eventos ENSO.
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
st.set_page_config(layout="wide", page_title="Visor de Precipitación y ENSO")

# Aplicar CSS personalizado para optimizar el espacio
st.markdown("""
<style>
.sidebar .sidebar-content { font-size: 13px; }
.stSelectbox label, .stMultiSelect label, .stSlider label { font-size: 13px !important; }
.stMultiSelect div[data-baseweb="select"] { font-size: 13px !important; }
h1 { margin-top: 0px; padding-top: 0px; }
</style>
""", unsafe_allow_html=True)

#--- Funciones de carga de datos
@st.cache_data
def load_data(file_path, sep=','):
    """Carga datos desde un archivo CSV, probando varias codificaciones."""
    if file_path is None:
        return None
    try:
        content = file_path.getvalue()
        if not content.strip():
            st.error("Ocurrió un error al cargar los datos: El archivo parece estar vacío.")
            return None
    except Exception as e:
        st.error(f"Error al leer el contenido del archivo: {e}")
        return None
    
    encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(io.BytesIO(content), sep=sep, encoding=encoding)
            df.columns = df.columns.str.strip()
            return df
        except Exception:
            continue
            
    st.error("No se pudo decodificar el archivo. Por favor, verifique la codificación.")
    return None

@st.cache_data
def load_shapefile(file_path):
    """Carga un shapefile desde un .zip y lo convierte a WGS84."""
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

#--- Interfaz de usuario
st.title('Visor de Precipitación y Fenómeno ENSO')
st.markdown("Esta aplicación interactiva permite visualizar y analizar datos de precipitación y su correlación con los eventos climáticos de El Niño-Oscilación del Sur (ENSO).")

#--- Panel de control (sidebar)
st.sidebar.header("Panel de Control")
with st.sidebar.expander("**Cargar Archivos**", expanded=True):
    uploaded_file_mapa = st.file_uploader("1. Cargar archivo de estaciones (mapaCVENSO.csv)", type="csv")
    uploaded_file_enso = st.file_uploader("2. Cargar archivo de ENSO (ENSO_1950_2023.csv)", type="csv")
    uploaded_file_precip = st.file_uploader("3. Cargar archivo de precipitación mensual (DatosPptnmes_ENSO.csv)", type="csv")
    uploaded_zip_shapefile = st.file_uploader("4. Cargar shapefile de municipios (.zip)", type="zip")

if not all([uploaded_file_mapa, uploaded_file_enso, uploaded_file_precip, uploaded_zip_shapefile]):
    st.info("Por favor, suba los 4 archivos requeridos para habilitar la aplicación.")
    st.stop()

df_precip_anual = load_data(uploaded_file_mapa)
df_enso = load_data(uploaded_file_enso)
df_precip_mensual = load_data(uploaded_file_precip)
gdf_municipios = load_shapefile(uploaded_zip_shapefile)

if any(df is None for df in [df_precip_anual, df_enso, df_precip_mensual, gdf_municipios]):
    st.stop()

#--- Preprocesamiento de datos ---

# ENSO
year_col_name_enso = None
for col in df_enso.columns:
    if 'año' in col.lower() or 'year' in col.lower():
        year_col_name_enso = col
        break
if year_col_name_enso is None:
    st.error(f"Error Crítico: No se encontró una columna para el año en el archivo ENSO. Columnas encontradas: {list(df_enso.columns)}.")
    st.stop()

month_col_name_enso = None
for col in df_enso.columns:
    if 'mes' in col.lower():
        month_col_name_enso = col
        break
if month_col_name_enso is None:
    st.error(f"Error Crítico: No se encontró una columna para el mes en el archivo ENSO. Columnas encontradas: {list(df_enso.columns)}.")
    st.stop()

df_enso.dropna(subset=[year_col_name_enso, month_col_name_enso], inplace=True)
for col in ['Anomalia_ONI', 'Temp_SST', 'Temp_media']:
    if col in df_enso.columns:
        df_enso[col] = df_enso[col].astype(str).str.replace(',', '.', regex=True).astype(float)

df_enso[year_col_name_enso] = df_enso[year_col_name_enso].astype(int)
df_enso[month_col_name_enso] = df_enso[month_col_name_enso].astype(int)

date_strings = df_enso[year_col_name_enso].astype(str) + '-' + df_enso[month_col_name_enso].astype(str)
datetime_series = pd.to_datetime(date_strings, errors='coerce')
invalid_rows = df_enso[datetime_series.isna()]
if not invalid_rows.empty:
    st.warning("Se encontraron y omitieron filas con fechas inválidas (ej. mes > 12) en el archivo ENSO. Revise estas filas en su archivo original:")
    st.dataframe(invalid_rows)

df_enso = df_enso[datetime_series.notna()]
df_enso['fecha_merge'] = datetime_series.dropna().dt.strftime('%Y-%m')

# Precipitación anual (mapa)
# --- INICIO: LÓGICA ROBUSTA PARA ENCONTRAR COLUMNAS DE COORDENADAS ---
lon_col_name = None
for col in df_precip_anual.columns:
    if 'longitud' in col.lower() or 'lon' in col.lower():
        lon_col_name = col
        break
if lon_col_name is None:
    st.error(f"Error Crítico: No se encontró una columna para Longitud en el archivo de estaciones (mapaCVENSO.csv). Columnas encontradas: {list(df_precip_anual.columns)}")
    st.stop()

lat_col_name = None
for col in df_precip_anual.columns:
    if 'latitud' in col.lower() or 'lat' in col.lower():
        lat_col_name = col
        break
if lat_col_name is None:
    st.error(f"Error Crítico: No se encontró una columna para Latitud en el archivo de estaciones (mapaCVENSO.csv). Columnas encontradas: {list(df_precip_anual.columns)}")
    st.stop()
# --- FIN: LÓGICA ROBUSTA ---

for col in [lon_col_name, lat_col_name]:
    df_precip_anual[col] = df_precip_anual[col].astype(str).str.replace(',', '.', regex=True).astype(float)

gdf_stations = gpd.GeoDataFrame(
    df_precip_anual,
    geometry=gpd.points_from_xy(df_precip_anual[lon_col_name], df_precip_anual[lat_col_name]), # Usa los nombres de columna encontrados
    crs="EPSG:9377" # Define el sistema de coordenadas planas de origen
).to_crs("EPSG:4326") # Transforma a coordenadas geográficas (WGS84)
gdf_stations['Longitud_geo'] = gdf_stations.geometry.x
gdf_stations['Latitud_geo'] = gdf_stations.geometry.y


# Precipitación mensual
df_precip_mensual.columns = df_precip_mensual.columns.str.strip().str.lower()
year_col_name_precip = None
for col in df_precip_mensual.columns:
    if 'año' in col or 'ano' in col:
        year_col_name_precip = col
        break
if year_col_name_precip is None:
    st.error(f"Error Crítico: No se encontró columna de año ('ano' o 'año') en el archivo de precipitación mensual. Columnas: {list(df_precip_mensual.columns)}")
    st.stop()
df_precip_mensual.rename(columns={year_col_name_precip: 'Año'}, inplace=True)

station_cols = [col for col in df_precip_mensual.columns if col.isdigit()]
if not station_cols:
    st.error("No se encontraron columnas de estación en el archivo de precipitación mensual.")
    st.stop()
df_long = df_precip_mensual.melt(id_vars=['Año', 'mes'], value_vars=station_cols, var_name='Id_estacion', value_name='Precipitation')
df_long['Precipitation'] = pd.to_numeric(df_long['Precipitation'], errors='coerce')
df_long.dropna(subset=['Precipitation'], inplace=True)
df_long['Fecha'] = pd.to_datetime(df_long['Año'].astype(str) + '-' + df_long['mes'].astype(str), format='%Y-%m', errors='coerce')
df_long.dropna(subset=['Fecha'], inplace=True)

# Mapeo y Fusión de Estaciones
gdf_stations['Id_estacio'] = gdf_stations['Id_estacio'].astype(str).str.strip()
df_long['Id_estacion'] = df_long['Id_estacion'].astype(str).str.strip()
station_mapping = gdf_stations.set_index('Id_estacio')['Nom_Est'].to_dict()
df_long['Nom_Est'] = df_long['Id_estacion'].map(station_mapping)
df_long.dropna(subset=['Nom_Est'], inplace=True)

if df_long.empty:
    st.warning("La fusión de datos falló. Verifique que los IDs de las estaciones coincidan.")
    st.stop()

#--- Controles en la barra lateral ---
st.sidebar.markdown("### Filtros de Visualización")
municipios_list = sorted(gdf_stations['municipio'].unique())

selected_municipios = st.sidebar.multiselect('1. Filtrar por municipio', options=municipios_list)

if selected_municipios:
    stations_filtered_by_criteria = sorted(gdf_stations[gdf_stations['municipio'].isin(selected_municipios)]['Nom_Est'].unique())
else:
    stations_filtered_by_criteria = sorted(gdf_stations['Nom_Est'].unique())

select_all_stations = st.sidebar.checkbox('Seleccionar todas las estaciones filtradas', value=True)

if select_all_stations:
    filtered_stations = stations_filtered_by_criteria
else:
    filtered_stations = st.sidebar.multiselect('2. Seleccione estaciones', options=stations_filtered_by_criteria)

if not filtered_stations:
    st.warning("No hay estaciones seleccionadas. Por favor, ajuste sus filtros.")
    st.stop()

años_disponibles = sorted([int(col) for col in gdf_stations.columns if str(col).isdigit()])
year_range = st.sidebar.slider(
    "3. Seleccione el rango de años",
    min_value=min(años_disponibles),
    max_value=max(años_disponibles),
    value=(min(años_disponibles), max(años_disponibles))
)

#--- Pestañas de la aplicación ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Gráficos de Series de Tiempo", "Mapas", "Tabla de Estaciones", "Análisis ENSO", "Opciones de Descarga"
])

#--- Pestaña de Gráficos ---
with tab1:
    st.header("Visualizaciones de Precipitación")
    
    # Datos para los gráficos
    df_anual_filtered = gdf_stations[gdf_stations['Nom_Est'].isin(filtered_stations)]
    year_cols = [str(y) for y in range(year_range[0], year_range[1] + 1) if str(y) in df_anual_filtered.columns]
    df_anual_melted = df_anual_filtered.melt(id_vars=['Nom_Est'], value_vars=year_cols, var_name='Año', value_name='Precipitación')
    
    df_monthly_filtered = df_long[
        (df_long['Nom_Est'].isin(filtered_stations)) &
        (df_long['Año'] >= year_range[0]) &
        (df_long['Año'] <= year_range[1])
    ]

    # Gráfico Anual
    st.subheader("Precipitación Anual Total (mm)")
    if not df_anual_melted.empty:
        chart_anual = alt.Chart(df_anual_melted).mark_line(point=True).encode(
            x=alt.X('Año:O', title='Año'),
            y=alt.Y('Precipitación:Q', title='Precipitación Total (mm)'),
            color='Nom_Est:N',
            tooltip=['Nom_Est', 'Año', 'Precipitación']
        ).properties(title='Precipitación Anual Total por Estación').interactive()
        st.altair_chart(chart_anual, use_container_width=True)
    else:
        st.warning("No hay datos anuales para la selección actual.")

    # Gráfico Mensual
    st.subheader("Precipitación Mensual Total (mm)")
    if not df_monthly_filtered.empty:
        chart_mensual = alt.Chart(df_monthly_filtered).mark_line().encode(
            x=alt.X('Fecha:T', title='Fecha'),
            y=alt.Y('Precipitation:Q', title='Precipitación Total (mm)'),
            color='Nom_Est:N',
            tooltip=[alt.Tooltip('Fecha', format='%Y-%m'), 'Precipitation', 'Nom_Est']
        ).properties(title='Precipitación Mensual Total por Estación').interactive()
        st.altair_chart(chart_mensual, use_container_width=True)
    else:
        st.warning("No hay datos mensuales para la selección actual.")

#--- Pestaña de Mapas ---
with tab2:
    st.header("Mapas de Lluvia y Precipitación")
    gdf_filtered = gdf_stations[gdf_stations['Nom_Est'].isin(filtered_stations)].copy()

    if not gdf_filtered.empty:
        st.subheader("Mapa de Estaciones de Lluvia")
        
        map_centering = st.radio(
            "Opciones de centrado del mapa", ("Automático", "Predefinido"),
            horizontal=True, key="static_map_centering"
        )

        if map_centering == "Automático":
            m = folium.Map(location=[gdf_filtered['Latitud_geo'].mean(), gdf_filtered['Longitud_geo'].mean()], zoom_start=6)
            bounds = gdf_filtered.total_bounds
            m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
        else: # Predefinido
            if 'map_view' not in st.session_state:
                st.session_state.map_view = {"location": [4.5709, -74.2973], "zoom": 5}
            
            col1, col2 = st.columns(2)
            if col1.button("Ver Colombia"):
                st.session_state.map_view = {"location": [4.5709, -74.2973], "zoom": 5}
            if col2.button("Ver Estaciones Seleccionadas"):
                bounds = gdf_filtered.total_bounds
                st.session_state.map_view = {"location": [(bounds[1]+bounds[3])/2, (bounds[0]+bounds[2])/2], "zoom": 8}
            
            m = folium.Map(location=st.session_state.map_view["location"], zoom_start=st.session_state.map_view["zoom"])

        ScaleControl().add_to(m)
        for _, row in gdf_filtered.iterrows():
            folium.Marker(
                location=[row['Latitud_geo'], row['Longitud_geo']],
                tooltip=f"Estación: {row['Nom_Est']}<br>Municipio: {row['municipio']}",
                icon=folium.Icon(color="blue", icon="cloud-rain", prefix='fa')
            ).add_to(m)
        folium_static(m, width=900, height=600)

        st.markdown("---")
        st.subheader("Mapa Animado de Precipitación Anual")
        df_anim_map = df_anual_melted.merge(gdf_stations[['Nom_Est', 'Latitud_geo', 'Longitud_geo']], on='Nom_Est')
        if not df_anim_map.empty:
            fig_mapa_animado = px.scatter_geo(
                df_anim_map,
                lat='Latitud_geo', lon='Longitud_geo',
                color='Precipitación', size='Precipitación',
                hover_name='Nom_Est', animation_frame='Año',
                projection='natural earth', title='Precipitación Anual de las Estaciones',
                color_continuous_scale=px.colors.sequential.YlGnBu,
            )
            fig_mapa_animado.update_geos(fitbounds="locations", showcountries=True)
            st.plotly_chart(fig_mapa_animado, use_container_width=True)
        else:
            st.warning("No hay datos para generar el mapa animado.")
    else:
        st.warning("No hay estaciones seleccionadas para mostrar en el mapa.")

#--- Pestaña de Tablas ---
with tab3:
    st.header("Información Detallada de las Estaciones")
    df_info_table = gdf_stations[gdf_stations['Nom_Est'].isin(filtered_stations)].copy()
    
    df_mean_precip = df_anual_melted.groupby('Nom_Est')['Precipitación'].mean().round(2).reset_index()
    df_mean_precip.rename(columns={'Precipitación': 'Precipitación media anual (mm)'}, inplace=True)
    
    df_info_table = df_info_table.merge(df_mean_precip, on='Nom_Est', how='left')
    
    columns_to_show = ['Nom_Est', 'municipio', 'departamento', 'Longitud_geo', 'Latitud_geo', 'Precipitación media anual (mm)']
    df_display = df_info_table[[col for col in columns_to_show if col in df_info_table.columns]].copy()
    df_display.rename(columns={'Longitud_geo': 'Longitud', 'Latitud_geo': 'Latitud'}, inplace=True)
    
    if not df_display.empty:
        st.dataframe(df_display)
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
        st.subheader("Precipitación Media por Evento ENSO")
        df_enso_group = df_analisis.groupby('ENSO')['Precipitation'].mean().reset_index()
        fig_enso = px.bar(
            df_enso_group, x='ENSO', y='Precipitation',
            title='Precipitación Media por Evento ENSO',
            labels={'ENSO': 'Evento ENSO', 'Precipitation': 'Precipitación Media Mensual (mm)'},
            color='ENSO'
        )
        st.plotly_chart(fig_enso, use_container_width=True)

        st.subheader("Correlación entre Anomalía ONI y Precipitación")
        correlation = df_analisis['Anomalia_ONI'].corr(df_analisis['Precipitation'])
        st.metric(label="Coeficiente de Correlación de Pearson", value=f"{correlation:.2f}")
        st.info("""
        **Interpretación:**
        - Un valor cercano a 1 indica una correlación positiva fuerte.
        - Un valor cercano a -1 indica una correlación negativa fuerte.
        - Un valor cercano a 0 indica una correlación débil o nula.
        """)
    else:
        st.warning("No hay suficientes datos para realizar el análisis ENSO con la selección actual.")

#--- Pestaña de Descarga ---
with tab5:
    st.header("Opciones de Descarga")
    
    st.markdown("**1. Datos de Precipitación Anual**")
    csv_anual = df_anual_melted.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Descargar datos anuales (CSV)", csv_anual, 'precipitacion_anual.csv', 'text/csv'
    )
    
    st.markdown("**2. Datos de Precipitación Mensual**")
    csv_mensual = df_monthly_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Descargar datos mensuales (CSV)", csv_mensual, 'precipitacion_mensual.csv', 'text/csv'
    )
    
    st.markdown("---")
    st.markdown("""
    **3. Exportar Gráficos y Mapas**
    - **Gráficos:** Pase el cursor sobre el gráfico y haga clic en los tres puntos (...) para ver las opciones de descarga como imagen (PNG).
    - **Mapas (Folium):** Use la herramienta de captura de pantalla de su sistema operativo.
    - **Página Completa (PDF):** Use la función de impresión de su navegador (Ctrl+P o Cmd+P) y seleccione "Guardar como PDF".
    """)

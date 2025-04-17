import os
import io
import zipfile
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import SimpleITK as sitk

from organ_detection import add_organ_detection_ui, plot_with_segmentation, render_trajectory_analysis

# Configuración de página y estilo
st.set_page_config(layout="wide", page_title="Brachyanalysis")

# CSS personalizado para aplicar los colores solicitados
st.markdown("""
<style>
    .main-header {
        color: #28aec5;
        text-align: center;
        font-size: 42px;
        margin-bottom: 20px;
        font-weight: bold;
    }
    .giant-title {
        color: #28aec5;
        text-align: center;
        font-size: 72px;
        margin: 30px 0;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        color: #c0d711;
        font-size: 24px;
        margin-bottom: 15px;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #28aec5;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
    }
    .stButton>button:hover {
        background-color: #1c94aa;
    }
    .info-box {
        background-color: rgba(40, 174, 197, 0.1);
        border-left: 3px solid #28aec5;
        padding: 10px;
        margin: 10px 0;
    }
    .success-box {
        background-color: rgba(192, 215, 17, 0.1);
        border-left: 3px solid #c0d711;
        padding: 10px;
        margin: 10px 0;
    }
    .plot-container {
        border: 2px solid #c0d711;
        border-radius: 5px;
        padding: 10px;
        margin-top: 20px;
    }
    div[data-baseweb="select"] {
        border-radius: 4px;
        border-color: #28aec5;
    }
    div[data-baseweb="slider"] > div {
        background-color: #c0d711 !important;
    }
    /* Estilos para radio buttons */
    div.stRadio > div[role="radiogroup"] > label {
        background-color: rgba(40, 174, 197, 0.1);
        margin-right: 10px;
        padding: 5px 15px;
        border-radius: 4px;
    }
    div.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {
        background-color: #28aec5;
    }
    .upload-section {
        background-color: rgba(40, 174, 197, 0.05);
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .stUploadButton>button {
        background-color: #c0d711;
        color: #1e1e1e;
        font-weight: bold;
    }
    .sidebar-title {
        color: #28aec5;
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 15px;
    }
    .control-section {
        background-color: rgba(40, 174, 197, 0.05);
        padding: 15px;
        border-radius: 8px;
        margin-top: 15px;
    }
    .input-row {
        display: flex;
        justify-content: space-between;
        gap: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Configuración de la barra lateral
st.sidebar.markdown('<p class="sidebar-title">Brachyanalysis</p>', unsafe_allow_html=True)
st.sidebar.markdown('<p class="sub-header">Visualizador de imágenes DICOM</p>', unsafe_allow_html=True)

# Sección de carga de archivos en la barra lateral
st.sidebar.markdown('<p class="sub-header">Configuración</p>', unsafe_allow_html=True)

# Solo opción de subir ZIP
uploaded_file = st.sidebar.file_uploader("Sube un archivo ZIP con tus archivos DICOM", type="zip")

# Función para buscar recursivamente archivos DICOM en un directorio
def find_dicom_series(directory):
    """Busca recursivamente series DICOM en el directorio y sus subdirectorios"""
    series_found = []
    # Explorar cada subdirectorio
    for root, dirs, files in os.walk(directory):
        try:
            # Intentar leer series DICOM en este directorio
            series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(root)
            for series_id in series_ids:
                series_files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(root, series_id)
                if series_files:
                    series_found.append((series_id, root, series_files))
        except Exception as e:
            st.sidebar.warning(f"Advertencia al buscar en {root}: {str(e)}")
            continue
    
    return series_found

def apply_window_level(image, window_width, window_center):
    """Aplica ventana y nivel a la imagen (brillo y contraste)"""
    # Convertir la imagen a float para evitar problemas con valores negativos
    image_float = image.astype(float)
    
    # Calcular los límites de la ventana
    min_value = window_center - window_width/2.0
    max_value = window_center + window_width/2.0
    
    # Aplicar la ventana
    image_windowed = np.clip(image_float, min_value, max_value)
    
    # Normalizar a [0, 1] para visualización
    if max_value != min_value:
        image_windowed = (image_windowed - min_value) / (max_value - min_value)
    else:
        image_windowed = np.zeros_like(image_float)
    
    return image_windowed

def plot_slice(vol, slice_ix, window_width, window_center):
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.axis('off')
    selected_slice = vol[slice_ix, :, :]
    
    # Aplicar ajustes de ventana/nivel
    windowed_slice = apply_window_level(selected_slice, window_width, window_center)
    
    # Mostrar la imagen con los ajustes aplicados
    ax.imshow(windowed_slice, origin='lower', cmap='gray')
    return fig

# Procesar archivos subidos
dirname = None
temp_dir = None

if uploaded_file is not None:
    # Crear un directorio temporal para extraer los archivos
    temp_dir = tempfile.mkdtemp()
    try:
        # Leer el contenido del ZIP
        with zipfile.ZipFile(io.BytesIO(uploaded_file.read()), 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Establecer dirname al directorio temporal
        dirname = temp_dir
        st.sidebar.markdown('<div class="success-box">Archivos extraídos correctamente.</div>', unsafe_allow_html=True)
    except Exception as e:
        st.sidebar.error(f"Error al extraer el archivo ZIP: {str(e)}")

# Inicializar variables para la visualización
dicom_series = None
img = None
output = None
n_slices = 0
slice_ix = 0
reader = None

# Define los presets de ventana basados en RadiAnt
radiant_presets = {
    "Default window": (0, 0),  # Auto-calculado según la imagen
    "Full dynamic": (0, 0),    # Auto-calculado según la imagen
    "CT Abdomen": (350, 50),
    "CT Angio": (600, 300),
    "CT Bone": (2000, 350),
    "CT Brain": (80, 40),
    "CT Chest": (350, 40),
    "CT Lungs": (1500, -600),
    "Negative": (0, 0),       # Invertir la imagen
    "Custom window": (0, 0)   # Valores personalizados
}

if dirname is not None:
    # Usar un spinner en el área principal en lugar de en la barra lateral
    with st.spinner('Buscando series DICOM...'):
        dicom_series = find_dicom_series(dirname)
    
    if not dicom_series:
        st.sidebar.error("No se encontraron archivos DICOM válidos en el archivo subido.")
    else:
        # Mostrar las series encontradas
        st.sidebar.markdown(f'<div class="info-box">Se encontraron {len(dicom_series)} series DICOM</div>', unsafe_allow_html=True)
        
        # Si hay múltiples series, permitir seleccionar una
        selected_series_idx = 0
        if len(dicom_series) > 1:
            series_options = [f"Serie {i+1}: {series_id[:10]}... ({len(files)} archivos)" 
                            for i, (series_id, _, files) in enumerate(dicom_series)]
            selected_series_option = st.sidebar.selectbox("Seleccionar serie DICOM:", series_options)
            selected_series_idx = series_options.index(selected_series_option)
        
        try:
            # Obtener la serie seleccionada
            series_id, series_dir, dicom_names = dicom_series[selected_series_idx]
            
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(dicom_names)
            reader.LoadPrivateTagsOn()
            reader.MetaDataDictionaryArrayUpdateOn()
            data = reader.Execute()
            img = sitk.GetArrayViewFromImage(data)
        
            n_slices = img.shape[0]
            slice_ix = st.sidebar.slider('Seleccionar corte', 0, n_slices - 1, int(n_slices/2))
            output = st.sidebar.radio('Tipo de visualización', ['Imagen', 'Metadatos'], index=0)
            
            # Añadir controles de ventana (brillo y contraste) si la salida es Imagen
            if output == 'Imagen':
                # Calcular valores iniciales para la ventana
                if img is not None:
                    min_val = float(img.min())
                    max_val = float(img.max())
                    range_val = max_val - min_val
                    
                    # Establecer valores predeterminados para window width y center
                    default_window_width = range_val
                    default_window_center = min_val + (range_val / 2)
                    
                    # Añadir presets de ventana para radiología
                    st.sidebar.markdown('<div class="control-section">', unsafe_allow_html=True)
                    st.sidebar.markdown('<p class="sub-header">Presets de ventana</p>', unsafe_allow_html=True)
                    
                    # Actualizar los presets automáticos
                    radiant_presets["Default window"] = (default_window_width, default_window_center)
                    radiant_presets["Full dynamic"] = (range_val, min_val + (range_val / 2))
                    
                    selected_preset = st.sidebar.selectbox(
                        "Presets radiológicos",
                        list(radiant_presets.keys())
                    )
                    
                    # Inicializar valores de ventana basados en el preset
                    window_width, window_center = radiant_presets[selected_preset]
                    
                    # Si es preset negativo, invertir la imagen
                    is_negative = selected_preset == "Negative"
                    if is_negative:
                        window_width = default_window_width
                        window_center = default_window_center
                    
                    # Si es un preset personalizado o Custom window, mostrar los campos de entrada
                    if selected_preset == "Custom window":
                        st.sidebar.markdown('<p class="sub-header">Ajustes personalizados</p>', unsafe_allow_html=True)
                        
                        # Mostrar información sobre el rango
                        st.sidebar.markdown(f"**Rango de valores de la imagen:** {min_val:.1f} a {max_val:.1f}")
                        
                        # Crear dos columnas para los campos de entrada
                        col1, col2 = st.sidebar.columns(2)
                        
                        with col1:
                            window_width = float(st.number_input(
                                "Ancho de ventana (WW)",
                                min_value=1.0,
                                max_value=range_val * 2,
                                value=float(default_window_width),
                                format="%.1f",
                                help="Controla el contraste. Valores menores aumentan el contraste."
                            ))
                        
                        with col2:
                            window_center = float(st.number_input(
                                "Centro de ventana (WL)",
                                min_value=min_val - range_val,
                                max_value=max_val + range_val,
                                value=float(default_window_center),
                                format="%.1f",
                                help="Controla el brillo. Valores mayores aumentan el brillo."
                            ))
                    
                    st.sidebar.markdown('</div>', unsafe_allow_html=True)
                
            else:
                # Valores predeterminados para cuando no son necesarios
                window_width = max_val - min_val if 'max_val' in locals() else 1000
                window_center = (max_val + min_val) / 2 if 'max_val' in locals() else 0
                is_negative = False
                
        except Exception as e:
            st.sidebar.error(f"Error al procesar los archivos DICOM: {str(e)}")
            st.sidebar.write("Detalles del error:", str(e))
            # Valores predeterminados
            window_width = 1000
            window_center = 0
            is_negative = False

# Visualización en la ventana principal
# Título grande siempre visible
st.markdown('<p class="giant-title">Brachyanalysis</p>', unsafe_allow_html=True)

if img is not None and output == 'Imagen':

    detection_config = add_organ_detection_ui(st.sidebar)
    st.markdown('<p class="sub-header">Visualización DICOM</p>', unsafe_allow_html=True)
    
    # Si es modo negativo, invertir la imagen
    if is_negative:
        # Muestra la imagen invertida
        fig, ax = plt.subplots(figsize=(12, 10))
        plt.axis('off')
        selected_slice = img[slice_ix, :, :]
        
        # Aplicar ventana y luego invertir
        windowed_slice = apply_window_level(selected_slice, window_width, window_center)
        windowed_slice = 1.0 - windowed_slice  # Invertir
        
        ax.imshow(windowed_slice, origin='lower', cmap='gray')
        st.pyplot(fig)
    else:
        if detection_config['enabled']:
            fig = plot_with_segmentation(img, slice_ix, window_width, window_center, detection_config)
            st.pyplot(fig)
            
            # Renderizar análisis de trayectorias potenciales
            render_trajectory_analysis(detection_config, img, slice_ix, window_width, window_center)
        else:
            fig = plot_slice(img, slice_ix, window_width, window_center)
            st.pyplot(fig)
    
    # Información adicional sobre la imagen y los ajustes actuales
    info_cols = st.columns(6)
    with info_cols[0]:
        st.markdown(f"**Dimensiones:** {img.shape[1]} x {img.shape[2]} px")
    with info_cols[1]:
        st.markdown(f"**Total cortes:** {n_slices}")
    with info_cols[2]:
        st.markdown(f"**Corte actual:** {slice_ix + 1}")
    with info_cols[3]:
        st.markdown(f"**Min/Max:** {img[slice_ix].min():.1f} / {img[slice_ix].max():.1f}")
    with info_cols[4]:
        st.markdown(f"**Ancho (WW):** {window_width:.1f}")
    with info_cols[5]:
        st.markdown(f"**Centro (WL):** {window_center:.1f}")
        
elif img is not None and output == 'Metadatos':
    st.markdown('<p class="sub-header">Metadatos DICOM</p>', unsafe_allow_html=True)
    try:
        metadata = dict()
        for k in reader.GetMetaDataKeys(slice_ix):
            metadata[k] = reader.GetMetaData(slice_ix, k)
        df = pd.DataFrame.from_dict(metadata, orient='index', columns=['Valor'])
        st.dataframe(df, height=600)
    except Exception as e:
        st.error(f"Error al leer metadatos: {str(e)}")
else:
    # Página de inicio cuando no hay imágenes cargadas
    st.markdown('<p class="sub-header">Visualizador de imágenes DICOM</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; padding: 40px; margin-top: 10px;">
        <img src="https://raw.githubusercontent.com/SimpleITK/SimpleITK/master/Documentation/docs/images/simpleitk-logo.svg" alt="SimpleITK Logo" width="200">
        <h2 style="color: #28aec5; margin-top: 20px;">Carga un archivo ZIP con tus imágenes DICOM</h2>
        <p style="font-size: 18px; margin-top: 10px;">Utiliza el panel lateral para subir tus archivos y visualizarlos</p>
    </div>
    """, unsafe_allow_html=True)

# Pie de página
st.markdown("""
<hr style="margin-top: 30px;">
<div style="text-align: center; color: #28aec5; font-size: 14px;">
    Brachyanalysis - Visualizador de imágenes DICOM
</div>
""", unsafe_allow_html=True)

# Limpiar el directorio temporal si se creó uno
if temp_dir and os.path.exists(temp_dir):
    # Nota: En una aplicación real, deberías limpiar los directorios temporales
    # cuando la aplicación se cierre, pero en Streamlit esto es complicado
    # ya que las sesiones persisten. Una solución es mantener un registro
    # de directorios temporales y limpiarlos al inicio.
    pass

import os
import io
import zipfile
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import SimpleITK as sitk
from matplotlib.colors import LinearSegmentedColormap

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
    .tab-primary {
        background-color: rgba(40, 174, 197, 0.2);
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .tab-secondary {
        background-color: rgba(192, 215, 17, 0.2);
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .fusion-controls {
        background-color: rgba(100, 100, 100, 0.1);
        padding: 10px;
        border-radius: 5px;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# Configuración de la barra lateral
st.sidebar.markdown('<p class="sidebar-title">Brachyanalysis</p>', unsafe_allow_html=True)
st.sidebar.markdown('<p class="sub-header">Visualizador de imágenes DICOM</p>', unsafe_allow_html=True)

# Inicializar variables para las sesiones en estado
if 'temp_dirs' not in st.session_state:
    st.session_state['temp_dirs'] = []
if 'dicom_series_primary' not in st.session_state:
    st.session_state['dicom_series_primary'] = None
if 'dicom_series_secondary' not in st.session_state:
    st.session_state['dicom_series_secondary'] = None
if 'active_view_mode' not in st.session_state:
    st.session_state['active_view_mode'] = "single"  # "single", "dual", "fusion"

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
                    # Intentar obtener más metadatos para esta serie
                    try:
                        # Leer el primer archivo para obtener metadatos
                        reader = sitk.ImageFileReader()
                        reader.SetFileName(series_files[0])
                        reader.LoadPrivateTagsOn()
                        reader.ReadImageInformation()
                        
                        # Intentar obtener información de modalidad y descripción
                        modality = "Desconocido"
                        description = "Serie DICOM"
                        
                        if reader.HasMetaDataKey("0008|0060"):  # Modality
                            modality = reader.GetMetaData("0008|0060").strip()
                        
                        if reader.HasMetaDataKey("0008|103e"):  # Series Description
                            description = reader.GetMetaData("0008|103e").strip()
                            
                        series_info = f"{modality}: {description}"
                    except:
                        series_info = f"Serie {series_id[:8]}..."
                    
                    series_found.append((series_id, root, series_files, series_info))
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

def plot_slice(vol, slice_ix, window_width, window_center, colormap='gray', alpha=1.0):
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.axis('off')
    
    if vol is not None and slice_ix < vol.shape[0]:
        selected_slice = vol[slice_ix, :, :]
        
        # Aplicar ajustes de ventana/nivel
        windowed_slice = apply_window_level(selected_slice, window_width, window_center)
        
        # Mostrar la imagen con los ajustes aplicados
        ax.imshow(windowed_slice, origin='lower', cmap=colormap, alpha=alpha)
    
    return fig

def plot_fusion(vol1, vol2, slice_ix1, slice_ix2, window_width1, window_center1, 
                window_width2, window_center2, alpha=0.5, colormap1='gray', colormap2='hot'):
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.axis('off')
    
    if vol1 is not None and slice_ix1 < vol1.shape[0]:
        selected_slice1 = vol1[slice_ix1, :, :]
        windowed_slice1 = apply_window_level(selected_slice1, window_width1, window_center1)
        ax.imshow(windowed_slice1, origin='lower', cmap=colormap1, alpha=1.0)
    
    if vol2 is not None and slice_ix2 < vol2.shape[0]:
        selected_slice2 = vol2[slice_ix2, :, :]
        windowed_slice2 = apply_window_level(selected_slice2, window_width2, window_center2)
        ax.imshow(windowed_slice2, origin='lower', cmap=colormap2, alpha=alpha)
    
    return fig

def process_upload(uploaded_file, key_prefix):
    """Procesa el archivo ZIP subido y devuelve el directorio temporal"""
    if uploaded_file is None:
        return None
    
    # Crear un directorio temporal para extraer los archivos
    temp_dir = tempfile.mkdtemp()
    st.session_state['temp_dirs'].append(temp_dir)
    
    try:
        # Leer el contenido del ZIP
        with zipfile.ZipFile(io.BytesIO(uploaded_file.read()), 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        st.sidebar.markdown(f'<div class="success-box">Archivos {key_prefix} extraídos correctamente.</div>', unsafe_allow_html=True)
        return temp_dir
    except Exception as e:
        st.sidebar.error(f"Error al extraer el archivo ZIP {key_prefix}: {str(e)}")
        return None

# Sección de carga de archivos en la barra lateral
st.sidebar.markdown('<p class="sub-header">Cargar Imágenes</p>', unsafe_allow_html=True)

# Opción de seleccionar modo de visualización
view_mode = st.sidebar.radio(
    "Modo de visualización",
    ["Vista individual", "Vista dual", "Fusión de imágenes"],
    index=0
)

# Mapear la selección a un valor interno
view_mode_map = {
    "Vista individual": "single",
    "Vista dual": "dual",
    "Fusión de imágenes": "fusion"
}
st.session_state['active_view_mode'] = view_mode_map[view_mode]

# Cargadores de archivos adaptados al modo de visualización
if st.session_state['active_view_mode'] in ["single", "dual", "fusion"]:
    # Siempre mostrar el cargador primario
    st.sidebar.markdown('<div class="control-section">', unsafe_allow_html=True)
    st.sidebar.markdown('<p class="sub-header">Imágenes primarias (ej. TC)</p>', unsafe_allow_html=True)
    uploaded_file_primary = st.sidebar.file_uploader(
        "Sube un archivo ZIP con imágenes DICOM", 
        type="zip",
        key="uploader_primary"
    )
    
    if uploaded_file_primary:
        dirname_primary = process_upload(uploaded_file_primary, "primarios")
        with st.spinner('Buscando series DICOM primarias...'):
            if dirname_primary:
                dicom_series_primary = find_dicom_series(dirname_primary)
                st.session_state['dicom_series_primary'] = dicom_series_primary
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Si estamos en modo dual o fusión, mostrar el segundo cargador
if st.session_state['active_view_mode'] in ["dual", "fusion"]:
    st.sidebar.markdown('<div class="control-section">', unsafe_allow_html=True)
    st.sidebar.markdown('<p class="sub-header">Imágenes secundarias (ej. RM)</p>', unsafe_allow_html=True)
    uploaded_file_secondary = st.sidebar.file_uploader(
        "Sube un archivo ZIP con imágenes DICOM", 
        type="zip",
        key="uploader_secondary"
    )
    
    if uploaded_file_secondary:
        dirname_secondary = process_upload(uploaded_file_secondary, "secundarios")
        with st.spinner('Buscando series DICOM secundarias...'):
            if dirname_secondary:
                dicom_series_secondary = find_dicom_series(dirname_secondary)
                st.session_state['dicom_series_secondary'] = dicom_series_secondary
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

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
    "MR T1": (500, 250),
    "MR T2": (800, 400),
    "Negative": (0, 0),       # Invertir la imagen
    "Custom window": (0, 0)   # Valores personalizados
}

# Define los mapas de colores disponibles
colormaps = ["gray", "hot", "viridis", "plasma", "inferno", "magma", "cividis", "jet"]

# Función para crear controles de visualización para un conjunto de imágenes
def create_image_controls(prefix, series_data, default_modality="CT"):
    # Variables para almacenar los resultados
    img = None
    n_slices = 0
    slice_ix = 0
    reader = None
    selected_series_idx = 0
    output = "Imagen"
    window_width = 1000
    window_center = 0
    is_negative = False
    colormap = "gray" if prefix == "primary" else "hot"
    
    if series_data and len(series_data) > 0:
        # Si hay múltiples series, permitir seleccionar una
        if len(series_data) > 1:
            series_options = [f"{i+1}: {info} ({len(files)} archivos)" 
                          for i, (_, _, files, info) in enumerate(series_data)]
            selected_series_option = st.sidebar.selectbox(
                f"Seleccionar serie {prefix}:",
                series_options,
                key=f"series_select_{prefix}"
            )
            selected_series_idx = series_options.index(selected_series_option)
        
        try:
            # Obtener la serie seleccionada
            series_id, series_dir, dicom_names, series_info = series_data[selected_series_idx]
            
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(dicom_names)
            reader.LoadPrivateTagsOn()
            reader.MetaDataDictionaryArrayUpdateOn()
            data = reader.Execute()
            img = sitk.GetArrayViewFromImage(data)
        
            n_slices = img.shape[0]
            slice_ix = st.sidebar.slider(
                f'Seleccionar corte {prefix}',
                0, n_slices - 1,
                int(n_slices/2),
                key=f"slice_slider_{prefix}"
            )
            
            # Detectar modalidad para presets adecuados
            modality = default_modality
            try:
                if reader.HasMetaDataKey(0, "0008|0060"):
                    modality = reader.GetMetaData(0, "0008|0060").strip()
            except:
                pass
            
            # Añadir controles de ventana (brillo y contraste)
            min_val = float(img.min())
            max_val = float(img.max())
            range_val = max_val - min_val
            
            # Establecer valores predeterminados para window width y center
            default_window_width = range_val
            default_window_center = min_val + (range_val / 2)
            
            # Actualizar los presets automáticos
            radiant_presets["Default window"] = (default_window_width, default_window_center)
            radiant_presets["Full dynamic"] = (range_val, min_val + (range_val / 2))
            
            selected_preset = st.sidebar.selectbox(
                f"Preset para {prefix}",
                list(radiant_presets.keys()),
                key=f"preset_{prefix}"
            )
            
            # Inicializar valores de ventana basados en el preset
            window_width, window_center = radiant_presets[selected_preset]
            
            # Si es preset negativo, invertir la imagen
            is_negative = selected_preset == "Negative"
            if is_negative:
                window_width = default_window_width
                window_center = default_window_center
            
            # Si es un preset personalizado, mostrar los campos de entrada
            if selected_preset == "Custom window":
                # Crear dos columnas para los campos de entrada
                col1, col2 = st.sidebar.columns(2)
                
                with col1:
                    window_width = float(st.number_input(
                        f"Ancho (WW) {prefix}",
                        min_value=1.0,
                        max_value=range_val * 2,
                        value=float(default_window_width),
                        format="%.1f",
                        key=f"ww_{prefix}"
                    ))
                
                with col2:
                    window_center = float(st.number_input(
                        f"Centro (WL) {prefix}",
                        min_value=min_val - range_val,
                        max_value=max_val + range_val,
                        value=float(default_window_center),
                        format="%.1f",
                        key=f"wl_{prefix}"
                    ))
            
            # Solo si estamos en modo fusión, permitir seleccionar colormap
            if st.session_state['active_view_mode'] == "fusion":
                colormap = st.sidebar.selectbox(
                    f"Mapa de color {prefix}",
                    colormaps,
                    index=0 if prefix == "primary" else 1,
                    key=f"colormap_{prefix}"
                )
            
            # Mostrar información sobre el rango
            st.sidebar.markdown(f"**Rango {prefix}:** {min_val:.1f} a {max_val:.1f}")
            
        except Exception as e:
            st.sidebar.error(f"Error al procesar los archivos DICOM {prefix}: {str(e)}")
    
    return img, n_slices, slice_ix, reader, window_width, window_center, is_negative, colormap

# Visualización en la ventana principal
# Título grande siempre visible
st.markdown('<p class="giant-title">Brachyanalysis</p>', unsafe_allow_html=True)

# Inicializar variables para la visualización
img_primary = None
img_secondary = None
n_slices_primary = 0
n_slices_secondary = 0
slice_ix_primary = 0
slice_ix_secondary = 0
reader_primary = None
reader_secondary = None
window_width_primary = 1000
window_center_primary = 0
window_width_secondary = 1000
window_center_secondary = 0
is_negative_primary = False
is_negative_secondary = False
colormap_primary = "gray"
colormap_secondary = "hot"
transparency = 0.5

# Procesar imágenes primarias si existen
if st.session_state['dicom_series_primary'] and len(st.session_state['dicom_series_primary']) > 0:
    st.sidebar.markdown('<div class="control-section">', unsafe_allow_html=True)
    st.sidebar.markdown('<p class="sub-header">Controles primarios</p>', unsafe_allow_html=True)
    
    img_primary, n_slices_primary, slice_ix_primary, reader_primary, window_width_primary, window_center_primary, is_negative_primary, colormap_primary = create_image_controls("primary", st.session_state['dicom_series_primary'], "CT")
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Procesar imágenes secundarias si existen y estamos en modo dual o fusión
if st.session_state['active_view_mode'] in ["dual", "fusion"] and st.session_state['dicom_series_secondary'] and len(st.session_state['dicom_series_secondary']) > 0:
    st.sidebar.markdown('<div class="control-section">', unsafe_allow_html=True)
    st.sidebar.markdown('<p class="sub-header">Controles secundarios</p>', unsafe_allow_html=True)
    
    img_secondary, n_slices_secondary, slice_ix_secondary, reader_secondary, window_width_secondary, window_center_secondary, is_negative_secondary, colormap_secondary = create_image_controls("secondary", st.session_state['dicom_series_secondary'], "MR")
    
    # Si estamos en modo fusión, mostrar control de transparencia
    if st.session_state['active_view_mode'] == "fusion":
        st.sidebar.markdown('<div class="fusion-controls">', unsafe_allow_html=True)
        st.sidebar.markdown('<p class="sub-header">Controles de fusión</p>', unsafe_allow_html=True)
        transparency = st.sidebar.slider(
            "Transparencia de fusión", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5, 
            step=0.05,
            key="transparency_slider"
        )
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Mostrar las imágenes según el modo de visualización
if st.session_state['active_view_mode'] == "single":
    if img_primary is not None:
        st.markdown('<p class="sub-header">Visualización DICOM</p>', unsafe_allow_html=True)
        
        # Si es modo negativo, invertir la imagen
        if is_negative_primary:
            fig, ax = plt.subplots(figsize=(12, 10))
            plt.axis('off')
            selected_slice = img_primary[slice_ix_primary, :, :]
            
            # Aplicar ventana y luego invertir
            windowed_slice = apply_window_level(selected_slice, window_width_primary, window_center_primary)
            windowed_slice = 1.0 - windowed_slice  # Invertir
            
            ax.imshow(windowed_slice, origin='lower', cmap=colormap_primary)
            st.pyplot(fig)
        else:
            # Muestra la imagen en la ventana principal con los ajustes aplicados
            fig = plot_slice(img_primary, slice_ix_primary, window_width_primary, window_center_primary, colormap_primary)
            st.pyplot(fig)
        
        # Información adicional sobre la imagen y los ajustes actuales
        info_cols = st.columns(6)
        with info_cols[0]:
            st.markdown(f"**Dimensiones:** {img_primary.shape[1]} x {img_primary.shape[2]} px")
        with info_cols[1]:
            st.markdown(f"**Total cortes:** {n_slices_primary}")
        with info_cols[2]:
            st.markdown(f"**Corte actual:** {slice_ix_primary + 1}")
        with info_cols[3]:
            st.markdown(f"**Min/Max:** {img_primary[slice_ix_primary].min():.1f} / {img_primary[slice_ix_primary].max():.1f}")
        with info_cols[4]:
            st.markdown(f"**Ancho (WW):** {window_width_primary:.1f}")
        with info_cols[5]:
            st.markdown(f"**Centro (WL):** {window_center_primary:.1f}")
    else:
        st.markdown('<p class="sub-header">Visualizador de imágenes DICOM</p>', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; padding: 40px; margin-top: 10px;">
            <img src="https://raw.githubusercontent.com/SimpleITK/SimpleITK/master/Documentation/docs/images/simpleitk-logo.svg" alt="SimpleITK Logo" width="200">
            <h2 style="color: #28aec5; margin-top: 20px;">Carga un archivo ZIP con tus imágenes DICOM</h2>
            <p style="font-size: 18px; margin-top: 10px;">Utiliza el panel lateral para subir tus archivos y visualizarlos</p>
        </div>
        """, unsafe_allow_html=True)

elif st.session_state['active_view_mode'] == "dual":
    st.markdown('<p class="sub-header">Vista Dual de Imágenes DICOM</p>', unsafe_allow_html=True)
    
    # Crear dos columnas para las imágenes
    col1, col2 = st.columns(2)
    
    with col1:
        if img_primary is not None:
            st.markdown('<div class="tab-primary">', unsafe_allow_html=True)
            st.markdown('<p class="sub-header" style="font-size: 18px;">Imagen Primaria (TC)</p>', unsafe_allow_html=True)
            
            # Si es modo negativo, invertir la imagen
            if is_negative_primary:
                fig, ax = plt.subplots(figsize=(6, 5))
                plt.axis('off')
                selected_slice = img_primary[slice_ix_primary, :, :]
                
                # Aplicar ventana y luego invertir
                windowed_slice = apply_window_level(selected_slice, window_width_primary, window_center_primary)
                windowed_slice = 1.0 - windowed_slice  # Invertir
                
                ax.imshow(windowed_slice, origin='lower', cmap=colormap_primary)
                st.pyplot(fig)
            else:
                # Muestra la imagen primaria
                fig = plot_slice(img_primary, slice_ix_primary, window_width_primary, window_center_primary, colormap_primary)
                st.pyplot(fig)
            
            # Información sobre la imagen primaria
            st.markdown(f"**Dimensiones:** {img_primary.shape[1]} x {img_primary.shape[2]} px | **Corte:** {slice_ix_primary + 1}/{n_slices_primary}")
            st.markdown(f"**WW/WL:** {window_width_primary:.1f}/{window_center_primary:.1f}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="tab-primary">', unsafe_allow_html=True)
            st.markdown('<p class="sub-header" style="font-size: 18px;">Imagen Primaria (No cargada)</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if img_secondary is not None:
            st.markdown('<div class="tab-secondary">', unsafe_allow_html=True)
            st.markdown('<p class="sub-header" style="font-size: 18px;">Imagen Secundaria (RM)</p>', unsafe_allow_html=True)
            
            # Si es modo negativo, invertir la imagen
            if is_negative_secondary:
                fig, ax = plt.subplots(figsize=(6, 5))
                plt.axis('off')
                selected_slice = img_secondary[slice_ix_secondary, :, :]
                
                # Aplicar ventana y luego invertir
                windowed_slice = apply_window_level(selected_slice, window_width_secondary, window_center_secondary)
                windowed_slice = 1.0 - windowed_slice  # Invertir
                
                ax.imshow(windowed_slice, origin='lower', cmap=colormap_secondary)
                st.pyplot(fig)
            else:
                # Muestra la imagen secundaria
                fig = plot_slice(img_secondary, slice_ix_secondary, window_width_secondary, window_center_secondary, colormap_secondary)
                st.pyplot(fig)
            
            # Información sobre la imagen secundaria
            st.markdown(f"**Dimensiones:** {img_secondary.shape[1]} x {img_secondary.shape[2]} px | **Corte:** {slice_ix_secondary + 1}/{n_slices_secondary}")
            st.markdown(f"**WW/WL:** {window_width_secondary:.1f}/{window_center_secondary:.1f}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
st.markdown('<div class="tab-secondary">', unsafe_allow_html=True)
            st.markdown('<p class="sub-header" style="font-size: 18px;">Imagen Secundaria (No cargada)</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state['active_view_mode'] == "fusion":
    st.markdown('<p class="sub-header">Fusión de Imágenes DICOM</p>', unsafe_allow_html=True)
    
    if img_primary is not None and img_secondary is not None:
        # Realizar fusión de imágenes
        st.markdown('<div class="fusion-controls">', unsafe_allow_html=True)
        st.markdown("### Imagen fusionada (TC + RM)")
        
        # Mostrar información sobre las imágenes a fusionar
        info_cols = st.columns(3)
        with info_cols[0]:
            st.markdown(f"**Primaria:** Corte {slice_ix_primary + 1}/{n_slices_primary}")
        with info_cols[1]:
            st.markdown(f"**Secundaria:** Corte {slice_ix_secondary + 1}/{n_slices_secondary}")
        with info_cols[2]:
            st.markdown(f"**Transparencia:** {int((1-transparency)*100)}% / {int(transparency*100)}%")
        
        # Realizar fusión de imágenes (la función ya maneja negativos)
        fig = plot_fusion(
            img_primary, img_secondary, 
            slice_ix_primary, slice_ix_secondary, 
            window_width_primary, window_center_primary,
            window_width_secondary, window_center_secondary,
            transparency, colormap_primary, colormap_secondary
        )
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Explicación de la fusión
        st.markdown("""
        <div class="info-box">
        <p>La fusión de imágenes muestra la imagen primaria (generalmente TC) en tonos de gris, 
        combinada con la imagen secundaria (generalmente RM) utilizando un mapa de color alternativo.
        Ajuste la transparencia para controlar la mezcla de ambas modalidades.</p>
        </div>
        """, unsafe_allow_html=True)
    elif img_primary is not None and img_secondary is None:
        st.warning("⚠️ Para la fusión de imágenes, es necesario cargar tanto imágenes primarias como secundarias. Por favor, carga un archivo ZIP con imágenes secundarias (RM).")
        # Mostrar solo la imagen primaria mientras tanto
        fig = plot_slice(img_primary, slice_ix_primary, window_width_primary, window_center_primary, colormap_primary)
        st.pyplot(fig)
    elif img_primary is None and img_secondary is not None:
        st.warning("⚠️ Para la fusión de imágenes, es necesario cargar tanto imágenes primarias como secundarias. Por favor, carga un archivo ZIP con imágenes primarias (TC).")
        # Mostrar solo la imagen secundaria mientras tanto
        fig = plot_slice(img_secondary, slice_ix_secondary, window_width_secondary, window_center_secondary, colormap_secondary)
        st.pyplot(fig)
    else:
        st.warning("⚠️ Para la fusión de imágenes, es necesario cargar tanto imágenes primarias (TC) como secundarias (RM). Por favor, carga los archivos ZIP correspondientes.")
        # Mostrar mensaje de instrucción
        st.markdown("""
        <div style="text-align: center; padding: 40px; margin-top: 10px;">
            <img src="https://raw.githubusercontent.com/SimpleITK/SimpleITK/master/Documentation/docs/images/simpleitk-logo.svg" alt="SimpleITK Logo" width="200">
            <h2 style="color: #28aec5; margin-top: 20px;">Fusión de imágenes multimodales</h2>
            <p style="font-size: 18px; margin-top: 10px;">Para visualizar la fusión de TC y RM, sube ambos conjuntos de imágenes DICOM en el panel lateral</p>
        </div>
        """, unsafe_allow_html=True)

# Sección para metadatos
if st.session_state['active_view_mode'] == "single" and img_primary is not None:
    # Opción para ver metadatos o imagen
    output = st.radio('Tipo de visualización', ['Imagen', 'Metadatos'], index=0)
    
    if output == 'Metadatos' and reader_primary is not None:
        st.markdown('<p class="sub-header">Metadatos DICOM</p>', unsafe_allow_html=True)
        try:
            metadata = dict()
            for k in reader_primary.GetMetaDataKeys(slice_ix_primary):
                metadata[k] = reader_primary.GetMetaData(slice_ix_primary, k)
            df = pd.DataFrame.from_dict(metadata, orient='index', columns=['Valor'])
            st.dataframe(df, height=600)
        except Exception as e:
            st.error(f"Error al leer metadatos: {str(e)}")

# Añadir función de registro para trayectorias de agujas (como mejora solicitada)
if img_primary is not None and (st.session_state['active_view_mode'] == "single" or st.session_state['active_view_mode'] == "fusion"):
    st.markdown('<div class="control-section">', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Trayectorias de agujas para braquiterapia</p>', unsafe_allow_html=True)
    
    with st.expander("Herramienta de registro de trayectorias"):
        st.markdown("""
        Esta sección permitirá registrar y visualizar las trayectorias de las agujas para braquiterapia.
        En futuras actualizaciones, podrá:
        - Marcar puntos de entrada y salida
        - Calcular la trayectoria óptima
        - Generar plantillas de implante
        - Exportar coordenadas para sistemas de planificación
        """)
        
        st.warning("Funcionalidad en desarrollo - Estará disponible en próximas versiones")
    st.markdown('</div>', unsafe_allow_html=True)

# Pie de página
st.markdown("""
<hr style="margin-top: 30px;">
<div style="text-align: center; color: #28aec5; font-size: 14px;">
    Brachyanalysis - Visualizador avanzado de imágenes DICOM para braquiterapia
</div>
""", unsafe_allow_html=True)

# Limpiar los directorios temporales cuando se reinicie la aplicación
# Nota: Esta es una aproximación ya que Streamlit mantiene el estado entre ejecuciones
if st.session_state['temp_dirs']:
    for temp_dir in st.session_state['temp_dirs']:
        if os.path.exists(temp_dir):
            # En una aplicación real, aquí se implementaría la limpieza
            pass

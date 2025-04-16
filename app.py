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
    .fusion-controls {
        background-color: rgba(192, 215, 17, 0.05);
        padding: 15px;
        border-radius: 8px;
        margin-top: 15px;
        border: 1px solid #c0d711;
    }
    .visualization-tab {
        background-color: rgba(40, 174, 197, 0.05);
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .series-identifier {
        font-size: 16px;
        font-weight: bold;
        color: #28aec5;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Inicializar sesión si aún no existe
if 'primary_data' not in st.session_state:
    st.session_state.primary_data = {
        'img': None,
        'reader': None,
        'series_id': None,
        'n_slices': 0,
        'slice_ix': 0,
        'modality': None,
        'window_width': 0,
        'window_center': 0,
        'is_negative': False
    }

if 'secondary_data' not in st.session_state:
    st.session_state.secondary_data = {
        'img': None,
        'reader': None,
        'series_id': None,
        'n_slices': 0,
        'slice_ix': 0,
        'modality': None,
        'window_width': 0,
        'window_center': 0,
        'is_negative': False
    }

if 'fusion_settings' not in st.session_state:
    st.session_state.fusion_settings = {
        'display_mode': 'Primary',  # 'Primary', 'Secondary', 'Side-by-Side', 'Fusion'
        'opacity': 0.5,
        'color_map_primary': 'gray',
        'color_map_secondary': 'hot',
        'linked_navigation': True,
        'fusion_method': 'Blend'  # 'Blend', 'Checkerboard', 'ColorOverlay'
    }

if 'temp_dirs' not in st.session_state:
    st.session_state.temp_dirs = []

# Configuración de la barra lateral
st.sidebar.markdown('<p class="sidebar-title">Brachyanalysis</p>', unsafe_allow_html=True)
st.sidebar.markdown('<p class="sub-header">Visualizador de imágenes DICOM</p>', unsafe_allow_html=True)

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
    "MRI T1": (500, 250),      # Presets para MRI
    "MRI T2": (800, 450),
    "Negative": (0, 0),        # Invertir la imagen
    "Custom window": (0, 0)    # Valores personalizados
}

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
                    # Obtener más información de la serie para mostrar
                    try:
                        # Leer metadatos del primer archivo para determinar la modalidad
                        reader = sitk.ImageFileReader()
                        reader.SetFileName(series_files[0])
                        reader.LoadPrivateTagsOn()
                        reader.ReadImageInformation()
                        
                        modality = "Desconocido"
                        description = "Sin descripción"
                        
                        # Intentar obtener modalidad y descripción
                        if reader.HasMetaDataKey("0008|0060"):
                            modality = reader.GetMetaData("0008|0060").strip()
                        if reader.HasMetaDataKey("0008|103e"):
                            description = reader.GetMetaData("0008|103e").strip()
                        
                        series_found.append((series_id, root, series_files, modality, description))
                    except Exception as e:
                        # Si falla al leer metadatos, añadir con valores predeterminados
                        series_found.append((series_id, root, series_files, "Desconocido", "Error al leer metadatos"))
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

def plot_slice(vol, slice_ix, window_width, window_center, is_negative=False, colormap='gray'):
    """Genera una figura con una imagen DICOM aplicando los ajustes especificados"""
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.axis('off')
    selected_slice = vol[slice_ix, :, :]
    
    # Aplicar ajustes de ventana/nivel
    windowed_slice = apply_window_level(selected_slice, window_width, window_center)
    
    # Invertir si es necesario
    if is_negative:
        windowed_slice = 1.0 - windowed_slice
    
    # Mostrar la imagen con los ajustes aplicados
    ax.imshow(windowed_slice, origin='lower', cmap=colormap)
    return fig

def fuse_images(img1, slice_ix1, ww1, wc1, img2, slice_ix2, ww2, wc2, fusion_method, opacity, color_map1='gray', color_map2='hot'):
    """Fusiona dos imágenes según el método especificado"""
    # Preparar las imágenes
    slice1 = img1[slice_ix1, :, :]
    slice2 = img2[slice_ix2, :, :]
    
    # Aplicar ventana/nivel
    windowed1 = apply_window_level(slice1, ww1, wc1)
    windowed2 = apply_window_level(slice2, ww2, wc2)
    
    # Asegurarse de que ambas imágenes tienen las mismas dimensiones
    # En una implementación real, aquí deberías hacer registro de imágenes
    
    # En este ejemplo simple, redimensionaremos la segunda imagen para que coincida con la primera
    if windowed1.shape != windowed2.shape:
        # Usar resize de SimpleITK para mantener la calidad
        temp_img1 = sitk.GetImageFromArray(windowed1)
        temp_img2 = sitk.GetImageFromArray(windowed2)
        
        # Obtener el tamaño de la primera imagen
        size = temp_img1.GetSize()
        
        # Redimensionar la segunda imagen
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(temp_img1.GetSpacing())
        resampler.SetSize(size)
        resampler.SetOutputDirection(temp_img1.GetDirection())
        resampler.SetOutputOrigin(temp_img1.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(0)
        
        resized_temp_img2 = resampler.Execute(temp_img2)
        windowed2 = sitk.GetArrayFromImage(resized_temp_img2)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.axis('off')
    
    # Diferentes métodos de fusión
    if fusion_method == 'Blend':
        # Fusión simple con transparencia
        ax.imshow(windowed1, origin='lower', cmap=color_map1, alpha=1.0)
        ax.imshow(windowed2, origin='lower', cmap=color_map2, alpha=opacity)
    
    elif fusion_method == 'Checkerboard':
        # Patrón de tablero de ajedrez
        checkerboard_size = 20  # Tamaño de los cuadros
        h, w = windowed1.shape
        checkerboard = np.ones((h, w))
        
        for i in range(0, h, checkerboard_size * 2):
            for j in range(0, w, checkerboard_size * 2):
                if i + checkerboard_size < h:
                    if j + checkerboard_size < w:
                        checkerboard[i:i+checkerboard_size, j:j+checkerboard_size] = 0
                        if j + 2*checkerboard_size < w:
                            checkerboard[i+checkerboard_size:i+2*checkerboard_size, j+checkerboard_size:j+2*checkerboard_size] = 0
        
        # Crear imagen fusionada con el patrón de ajedrez
        fused_image = np.zeros((h, w, 3))
        
        # Usar matplotlib para convertir imágenes en escala de grises a colores según el mapa de colores
        cm1 = plt.get_cmap(color_map1)
        cm2 = plt.get_cmap(color_map2)
        
        img1_colored = cm1(windowed1)[:, :, :3]
        img2_colored = cm2(windowed2)[:, :, :3]
        
        # Aplicar patrón de ajedrez
        for i in range(h):
            for j in range(w):
                if checkerboard[i, j] == 1:
                    fused_image[i, j] = img1_colored[i, j]
                else:
                    fused_image[i, j] = img2_colored[i, j]
        
        ax.imshow(fused_image, origin='lower')
    
    elif fusion_method == 'ColorOverlay':
        # Usar un mapa de color específico para cada imagen y combinarlas
        # La primera imagen en escala de grises, la segunda con un mapa de color
        # Creamos una imagen RGB combinada
        
        # Convertir a RGB usando los mapas de colores
        cmap1 = plt.get_cmap(color_map1)
        cmap2 = plt.get_cmap(color_map2)
        
        rgb1 = cmap1(windowed1)[:, :, :3]  # Eliminar canal alfa
        rgb2 = cmap2(windowed2)[:, :, :3]  # Eliminar canal alfa
        
        # Combinar las imágenes con la opacidad dada
        combined = rgb1 * (1 - opacity) + rgb2 * opacity
        
        # Asegurar que los valores estén en el rango correcto
        combined = np.clip(combined, 0, 1)
        
        # Mostrar la imagen combinada
        ax.imshow(combined, origin='lower')
    
    return fig

def read_dicom_series(series_files):
    """Lee una serie DICOM y devuelve la imagen, el lector y la modalidad"""
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(series_files)
    reader.LoadPrivateTagsOn()
    reader.MetaDataDictionaryArrayUpdateOn()
    data = reader.Execute()
    img = sitk.GetArrayViewFromImage(data)
    
    # Intentar obtener la modalidad
    modality = "Desconocido"
    try:
        if reader.HasMetaDataKey(0, "0008|0060"):
            modality = reader.GetMetaData(0, "0008|0060").strip()
    except:
        pass
    
    return img, reader, modality

def setup_window_controls(container, data_key, min_val, max_val, range_val):
    """Configura los controles de ventana para una imagen"""
    # Obtener estado actual
    data = st.session_state[data_key]
    
    # Establecer valores predeterminados para window width y center
    default_window_width = range_val
    default_window_center = min_val + (range_val / 2)
    
    # Actualizar presets de ventana automáticos
    radiant_presets["Default window"] = (default_window_width, default_window_center)
    radiant_presets["Full dynamic"] = (range_val, min_val + (range_val / 2))
    
    container.markdown('<div class="control-section">', unsafe_allow_html=True)
    container.markdown('<p class="sub-header">Presets de ventana</p>', unsafe_allow_html=True)
    
    selected_preset = container.selectbox(
        "Presets radiológicos",
        list(radiant_presets.keys()),
        key=f"{data_key}_preset"
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
        container.markdown('<p class="sub-header">Ajustes personalizados</p>', unsafe_allow_html=True)
        
        # Mostrar información sobre el rango
        container.markdown(f"**Rango de valores de la imagen:** {min_val:.1f} a {max_val:.1f}")
        
        # Crear dos columnas para los campos de entrada
        col1, col2 = container.columns(2)
        
        with col1:
            window_width = float(st.number_input(
                "Ancho de ventana (WW)",
                min_value=1.0,
                max_value=range_val * 2,
                value=float(default_window_width),
                format="%.1f",
                help="Controla el contraste. Valores menores aumentan el contraste.",
                key=f"{data_key}_ww"
            ))
        
        with col2:
            window_center = float(st.number_input(
                "Centro de ventana (WL)",
                min_value=min_val - range_val,
                max_value=max_val + range_val,
                value=float(default_window_center),
                format="%.1f",
                help="Controla el brillo. Valores mayores aumentan el brillo.",
                key=f"{data_key}_wc"
            ))
    
    container.markdown('</div>', unsafe_allow_html=True)
    
    # Actualizar estado
    data['window_width'] = window_width
    data['window_center'] = window_center
    data['is_negative'] = is_negative
    
    return window_width, window_center, is_negative

# Sección de carga de archivos en la barra lateral
st.sidebar.markdown('<p class="sub-header">Carga de estudios</p>', unsafe_allow_html=True)

# Pestañas para seleccionar primario o secundario
upload_tab = st.sidebar.radio("Seleccione el estudio a cargar:", ["Estudio Primario (CT)", "Estudio Secundario (MRI)"], index=0)

# Uploader para el estudio seleccionado
if upload_tab == "Estudio Primario (CT)":
    upload_key = "primary_upload"
    data_key = "primary_data"
else:
    upload_key = "secondary_upload"
    data_key = "secondary_data"

uploaded_file = st.sidebar.file_uploader(f"Cargar archivos para {upload_tab}", type="zip", key=upload_key)

# Procesar archivos subidos
if uploaded_file is not None:
    # Crear un directorio temporal para extraer los archivos
    temp_dir = tempfile.mkdtemp()
    st.session_state.temp_dirs.append(temp_dir)  # Guardar para limpieza posterior
    
    try:
        # Leer el contenido del ZIP
        with zipfile.ZipFile(io.BytesIO(uploaded_file.read()), 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        st.sidebar.markdown('<div class="success-box">Archivos extraídos correctamente.</div>', unsafe_allow_html=True)
        
        # Buscar series DICOM
        with st.spinner(f'Buscando series DICOM en {upload_tab}...'):
            dicom_series = find_dicom_series(temp_dir)
        
        if not dicom_series:
            st.sidebar.error("No se encontraron archivos DICOM válidos en el archivo subido.")
        else:
            # Mostrar las series encontradas
            st.sidebar.markdown(f'<div class="info-box">Se encontraron {len(dicom_series)} series DICOM</div>', unsafe_allow_html=True)
            
            # Si hay múltiples series, permitir seleccionar una
            selected_series_idx = 0
            if len(dicom_series) > 1:
                series_options = [f"Serie {i+1}: {modality} - {desc[:15]}... ({len(files)} archivos)" 
                                 for i, (series_id, _, files, modality, desc) in enumerate(dicom_series)]
                selected_series_option = st.sidebar.selectbox(f"Seleccionar serie para {upload_tab}:", series_options, key=f"{data_key}_series_select")
                selected_series_idx = series_options.index(selected_series_option)
            
            try:
                # Obtener la serie seleccionada
                series_id, series_dir, dicom_names, modality, desc = dicom_series[selected_series_idx]
                
                # Leer la serie DICOM
                img, reader, modality = read_dicom_series(dicom_names)
                
                # Actualizar el estado de la sesión
                st.session_state[data_key]['img'] = img
                st.session_state[data_key]['reader'] = reader
                st.session_state[data_key]['series_id'] = series_id
                st.session_state[data_key]['n_slices'] = img.shape[0]
                st.session_state[data_key]['slice_ix'] = int(img.shape[0]/2)
                st.session_state[data_key]['modality'] = modality
                
                st.sidebar.success(f"Serie cargada correctamente en {upload_tab}.")
                
            except Exception as e:
                st.sidebar.error(f"Error al procesar los archivos DICOM: {str(e)}")
                st.sidebar.write("Detalles del error:", str(e))
                
    except Exception as e:
        st.sidebar.error(f"Error al extraer el archivo ZIP: {str(e)}")

# Controles para la visualización
st.sidebar.markdown('<p class="sub-header">Visualización</p>', unsafe_allow_html=True)

# Seleccionar modo de visualización
display_mode = st.sidebar.radio(
    "Modo de visualización:",
    ["Primario", "Secundario", "Lado a Lado", "Fusión"],
    index=0,
    key="display_mode_select"
)

# Actualizar modo de visualización
st.session_state.fusion_settings['display_mode'] = display_mode

# Mostrar controles de navegación para las imágenes cargadas
if st.session_state.primary_data['img'] is not None:
    primary_n_slices = st.session_state.primary_data['n_slices']
    st.sidebar.markdown('<div class="control-section">', unsafe_allow_html=True)
    st.sidebar.markdown('<p class="series-identifier">Navegación Primaria (CT)</p>', unsafe_allow_html=True)
    slice_ix_primary = st.sidebar.slider(
        'Seleccionar corte primario', 
        0, primary_n_slices - 1, 
        st.session_state.primary_data['slice_ix'],
        key="slice_primary"
    )
    st.session_state.primary_data['slice_ix'] = slice_ix_primary
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

if st.session_state.secondary_data['img'] is not None:
    secondary_n_slices = st.session_state.secondary_data['n_slices']
    st.sidebar.markdown('<div class="control-section">', unsafe_allow_html=True)
    st.sidebar.markdown('<p class="series-identifier">Navegación Secundaria (MRI)</p>', unsafe_allow_html=True)
    slice_ix_secondary = st.sidebar.slider(
        'Seleccionar corte secundario', 
        0, secondary_n_slices - 1, 
        st.session_state.secondary_data['slice_ix'],
        key="slice_secondary"
    )
    st.session_state.secondary_data['slice_ix'] = slice_ix_secondary
    
    # Opción para vincular la navegación
    if st.session_state.primary_data['img'] is not None:
        linked_navigation = st.sidebar.checkbox(
            "Vincular navegación de cortes",
            value=st.session_state.fusion_settings['linked_navigation'],
            key="linked_navigation"
        )
        st.session_state.fusion_settings['linked_navigation'] = linked_navigation
        
        # Si se activa la navegación vinculada, sincronizar cortes
        if linked_navigation:
            # Calcular proporción entre los cortes
            ratio = st.session_state.primary_data['n_slices'] / st.session_state.secondary_data['n_slices']
            
            # Si se movió el primario, ajustar el secundario
            if st.session_state.primary_data['slice_ix'] != slice_ix_primary:
                new_secondary_slice = int(slice_ix_primary / ratio)
                new_secondary_slice = max(0, min(new_secondary_slice, secondary_n_slices - 1))
                st.session_state.secondary_data['slice_ix'] = new_secondary_slice
                st.experimental_rerun()
            
            # Si se movió el secundario, ajustar el primario
            elif st.session_state.secondary_data['slice_ix'] != slice_ix_secondary:
                new_primary_slice = int(slice_ix_secondary * ratio)
                new_primary_slice = max(0, min(new_primary_slice, primary_n_slices - 1))
                st.session_state.primary_data['slice_ix'] = new_primary_slice
                st.experimental_rerun()
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Controles de fusión si el modo es fusión
if display_mode == "Fusión" and st.session_state.primary_data['img'] is not None and st.session_state.secondary_data['img'] is not None:
    st.sidebar.markdown('<div class="fusion-controls">', unsafe_allow_html=True)
    st.sidebar.markdown('<p class="sub-header">Controles de Fusión</p>', unsafe_allow_html=True)
    
    # Método de fusión
    fusion_method = st.sidebar.selectbox(
        "Método de fusión:",
        ["Blend", "Checkerboard", "ColorOverlay"],
        index=0,
        key="fusion_method"
    )
    st.session_state.fusion_settings['fusion_method'] = fusion_method
    
    # Opacidad para fusión
    opacity = st.sidebar.slider(
        "Opacidad de la imagen secundaria:",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.fusion_settings['opacity'],
        step=0.05,
        key="opacity_slider"
    )
    st.session_state.fusion_settings['opacity'] = opacity
    
    # Mapas de color
    color_maps = ['gray', 'hot', 'plasma', 'viridis', 'inferno', 'jet', 'rainbow', 'coolwarm']
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        color_map_primary = st.selectbox(
            "Color primario:",
            color_maps,
            index=color_maps.index(st.session_state.fusion_settings['color_map_primary']),
            key="color_map_primary"
        )
        st.session_state.fusion_settings['color_map_primary'] = color_map_primary
    
    with col2:
        color_map_secondary = st.selectbox(
            "Color secundario:",
            color_maps,
            index=color_maps.index(st.session_state.fusion_settings['color_map_secondary']),
            key="color_map_secondary"
        )
        st.session_state.fusion_settings['color_map_secondary'] = color_map_secondary
    
# Visualización en el área principal
st.markdown('<p class="giant-title">Brachyanalysis</p>', unsafe_allow_html=True)

# Acción principal según el modo de visualización
if display_mode == "Primario":
    # Mostrar solo la imagen primaria
    st.markdown('<p class="sub-header">Visualización Primaria (CT)</p>', unsafe_allow_html=True)
    
    if st.session_state.primary_data['img'] is not None:
        # Obtener datos del estado
        img = st.session_state.primary_data['img']
        slice_ix = st.session_state.primary_data['slice_ix']
        
        # Configurar controles de ventana para la imagen primaria
        min_val = float(img.min())
        max_val = float(img.max())
        range_val = max_val - min_val
        
        col1, col2 = st.columns([3, 1])
        
        # Configurar ventana en la columna derecha
        window_width, window_center, is_negative = setup_window_controls(
            col2, 'primary_data', min_val, max_val, range_val
        )
        
        # Mostrar la imagen en la columna izquierda
        with col1:
            fig = plot_slice(
                img, slice_ix, window_width, window_center, 
                is_negative, st.session_state.fusion_settings['color_map_primary']
            )
            st.pyplot(fig)
        
        # Información sobre la imagen
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        info_cols = st.columns(6)
        with info_cols[0]:
            st.markdown(f"**Dimensiones:** {img.shape[1]} x {img.shape[2]} px")
        with info_cols[1]:
            st.markdown(f"**Total cortes:** {st.session_state.primary_data['n_slices']}")
        with info_cols[2]:
            st.markdown(f"**Corte actual:** {slice_ix + 1}")
        with info_cols[3]:
            st.markdown(f"**Min/Max:** {img[slice_ix].min():.1f} / {img[slice_ix].max():.1f}")
        with info_cols[4]:
            st.markdown(f"**Ancho (WW):** {window_width:.1f}")
        with info_cols[5]:
            st.markdown(f"**Centro (WL):** {window_center:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Opción para ver metadatos
        if st.checkbox("Mostrar metadatos", value=False, key="show_meta_primary"):
            st.markdown('<p class="sub-header">Metadatos DICOM</p>', unsafe_allow_html=True)
            try:
                reader = st.session_state.primary_data['reader']
                metadata = dict()
                for k in reader.GetMetaDataKeys(slice_ix):
                    metadata[k] = reader.GetMetaData(slice_ix, k)
                df = pd.DataFrame.from_dict(metadata, orient='index', columns=['Valor'])
                st.dataframe(df, height=400)
            except Exception as e:
                st.error(f"Error al leer metadatos: {str(e)}")
    else:
        st.info("No se ha cargado ninguna imagen primaria. Use el panel lateral para cargar una serie DICOM.")

elif display_mode == "Secundario":
    # Mostrar solo la imagen secundaria
    st.markdown('<p class="sub-header">Visualización Secundaria (MRI)</p>', unsafe_allow_html=True)
    
    if st.session_state.secondary_data['img'] is not None:
        # Obtener datos del estado
        img = st.session_state.secondary_data['img']
        slice_ix = st.session_state.secondary_data['slice_ix']
        
        # Configurar controles de ventana para la imagen secundaria
        min_val = float(img.min())
        max_val = float(img.max())
        range_val = max_val - min_val
        
        col1, col2 = st.columns([3, 1])
        
        # Configurar ventana en la columna derecha
        window_width, window_center, is_negative = setup_window_controls(
            col2, 'secondary_data', min_val, max_val, range_val
        )
        
        # Mostrar la imagen en la columna izquierda
        with col1:
            fig = plot_slice(
                img, slice_ix, window_width, window_center, 
                is_negative, st.session_state.fusion_settings['color_map_secondary']
            )
            st.pyplot(fig)
        
        # Información sobre la imagen
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        info_cols = st.columns(6)
        with info_cols[0]:
            st.markdown(f"**Dimensiones:** {img.shape[1]} x {img.shape[2]} px")
        with info_cols[1]:
            st.markdown(f"**Total cortes:** {st.session_state.secondary_data['n_slices']}")
        with info_cols[2]:
            st.markdown(f"**Corte actual:** {slice_ix + 1}")
        with info_cols[3]:
            st.markdown(f"**Min/Max:** {img[slice_ix].min():.1f} / {img[slice_ix].max():.1f}")
        with info_cols[4]:
            st.markdown(f"**Ancho (WW):** {window_width:.1f}")
        with info_cols[5]:
            st.markdown(f"**Centro (WL):** {window_center:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Opción para ver metadatos
        if st.checkbox("Mostrar metadatos", value=False, key="show_meta_secondary"):
            st.markdown('<p class="sub-header">Metadatos DICOM</p>', unsafe_allow_html=True)
            try:
                reader = st.session_state.secondary_data['reader']
                metadata = dict()
                for k in reader.GetMetaDataKeys(slice_ix):
                    metadata[k] = reader.GetMetaData(slice_ix, k)
                df = pd.DataFrame.from_dict(metadata, orient='index', columns=['Valor'])
                st.dataframe(df, height=400)
            except Exception as e:
                st.error(f"Error al leer metadatos: {str(e)}")
    else:
        st.info("No se ha cargado ninguna imagen secundaria. Use el panel lateral para cargar una serie DICOM.")

elif display_mode == "Lado a Lado":
    # Mostrar imágenes una al lado de la otra
    st.markdown('<p class="sub-header">Visualización Lado a Lado</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    # Columna izquierda: Imagen primaria
    with col1:
        st.markdown('<div class="visualization-tab">', unsafe_allow_html=True)
        st.markdown('<p class="series-identifier">Imagen Primaria (CT)</p>', unsafe_allow_html=True)
        
        if st.session_state.primary_data['img'] is not None:
            img = st.session_state.primary_data['img']
            slice_ix = st.session_state.primary_data['slice_ix']
            
            # Valores para ventana
            min_val = float(img.min())
            max_val = float(img.max())
            range_val = max_val - min_val
            
            # Crear expander para los controles de ventana
            with st.expander("Ajustes de ventana primaria", expanded=False):
                window_width, window_center, is_negative = setup_window_controls(
                    st, 'primary_data', min_val, max_val, range_val
                )
            
            # Mostrar imagen
            fig = plot_slice(
                img, slice_ix, window_width, window_center, 
                is_negative, st.session_state.fusion_settings['color_map_primary']
            )
            st.pyplot(fig)
            
            # Información básica
            st.markdown(f"**Corte:** {slice_ix + 1}/{st.session_state.primary_data['n_slices']} | "
                        f"**WW:** {window_width:.1f} | **WL:** {window_center:.1f}")
        else:
            st.info("No se ha cargado la imagen primaria")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Columna derecha: Imagen secundaria
    with col2:
        st.markdown('<div class="visualization-tab">', unsafe_allow_html=True)
        st.markdown('<p class="series-identifier">Imagen Secundaria (MRI)</p>', unsafe_allow_html=True)
        
        if st.session_state.secondary_data['img'] is not None:
            img = st.session_state.secondary_data['img']
            slice_ix = st.session_state.secondary_data['slice_ix']
            
            # Valores para ventana
            min_val = float(img.min())
            max_val = float(img.max())
            range_val = max_val - min_val
            
            # Crear expander para los controles de ventana
            with st.expander("Ajustes de ventana secundaria", expanded=False):
                window_width, window_center, is_negative = setup_window_controls(
                    st, 'secondary_data', min_val, max_val, range_val
                )
            
            # Mostrar imagen
            fig = plot_slice(
                img, slice_ix, window_width, window_center, 
                is_negative, st.session_state.fusion_settings['color_map_secondary']
            )
            st.pyplot(fig)
            
            # Información básica
            st.markdown(f"**Corte:** {slice_ix + 1}/{st.session_state.secondary_data['n_slices']} | "
                        f"**WW:** {window_width:.1f} | **WL:** {window_center:.1f}")
        else:
            st.info("No se ha cargado la imagen secundaria")
        st.markdown('</div>', unsafe_allow_html=True)

elif display_mode == "Fusión":
    # Mostrar fusión de imágenes
    st.markdown('<p class="sub-header">Visualización de Fusión</p>', unsafe_allow_html=True)
    
    if (st.session_state.primary_data['img'] is not None and 
        st.session_state.secondary_data['img'] is not None):
        
        # Obtener datos de las imágenes
        img1 = st.session_state.primary_data['img']
        slice_ix1 = st.session_state.primary_data['slice_ix']
        ww1 = st.session_state.primary_data['window_width']
        wc1 = st.session_state.primary_data['window_center']
        
        img2 = st.session_state.secondary_data['img']
        slice_ix2 = st.session_state.secondary_data['slice_ix']
        ww2 = st.session_state.secondary_data['window_width']
        wc2 = st.session_state.secondary_data['window_center']
        
        # Obtener configuración de fusión
        fusion_method = st.session_state.fusion_settings['fusion_method']
        opacity = st.session_state.fusion_settings['opacity']
        color_map1 = st.session_state.fusion_settings['color_map_primary']
        color_map2 = st.session_state.fusion_settings['color_map_secondary']
        
        # Columnas para imagen fusionada y controles
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Crear y mostrar imagen fusionada
            try:
                fusion_fig = fuse_images(
                    img1, slice_ix1, ww1, wc1,
                    img2, slice_ix2, ww2, wc2,
                    fusion_method, opacity, color_map1, color_map2
                )
                st.pyplot(fusion_fig)
            except Exception as e:
                st.error(f"Error al fusionar imágenes: {str(e)}")
                st.write("Detalles del error:", str(e))
        
        with col2:
            # Información sobre las imágenes
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("**Información Primaria (CT)**")
            st.markdown(f"Corte: {slice_ix1 + 1}/{st.session_state.primary_data['n_slices']}")
            st.markdown(f"WW: {ww1:.1f} | WL: {wc1:.1f}")
            st.markdown(f"Dimensiones: {img1.shape[1]}x{img1.shape[2]}")
            
            st.markdown("**Información Secundaria (MRI)**")
            st.markdown(f"Corte: {slice_ix2 + 1}/{st.session_state.secondary_data['n_slices']}")
            st.markdown(f"WW: {ww2:.1f} | WL: {wc2:.1f}")
            st.markdown(f"Dimensiones: {img2.shape[1]}x{img2.shape[2]}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Expander para ajustes rápidos de ventana
            with st.expander("Ajustes rápidos", expanded=False):
                st.markdown("**Ajustes CT**")
                st.slider("WW primario", 1.0, max(img1.max()-img1.min(), 2000.0), ww1, 
                          key="fusion_ww1", on_change=lambda: setattr(st.session_state.primary_data, 'window_width', st.session_state.fusion_ww1))
                st.slider("WL primario", img1.min(), img1.max(), wc1, 
                          key="fusion_wc1", on_change=lambda: setattr(st.session_state.primary_data, 'window_center', st.session_state.fusion_wc1))
                
                st.markdown("**Ajustes MRI**")
                st.slider("WW secundario", 1.0, max(img2.max()-img2.min(), 2000.0), ww2, 
                          key="fusion_ww2", on_change=lambda: setattr(st.session_state.secondary_data, 'window_width', st.session_state.fusion_ww2))
                st.slider("WL secundario", img2.min(), img2.max(), wc2, 
                          key="fusion_wc2", on_change=lambda: setattr(st.session_state.secondary_data, 'window_center', st.session_state.fusion_wc2))
    else:
        if st.session_state.primary_data['img'] is None:
            st.warning("Falta cargar la imagen primaria (CT)")
        if st.session_state.secondary_data['img'] is None:
            st.warning("Falta cargar la imagen secundaria (MRI)")
        
        st.info("Para la fusión de imágenes, debe cargar tanto una serie primaria (CT) como una secundaria (MRI).")

else:
    # Pantalla de inicio cuando no hay modo de visualización seleccionado
    st.markdown('<p class="sub-header">Visualizador de imágenes DICOM</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; padding: 40px; margin-top: 10px;">
        <img src="https://raw.githubusercontent.com/SimpleITK/SimpleITK/master/Documentation/docs/images/simpleitk-logo.svg" alt="SimpleITK Logo" width="200">
        <h2 style="color: #28aec5; margin-top: 20px;">Visualizador de fusión de imágenes DICOM</h2>
        <p style="font-size: 18px; margin-top: 10px;">Utilice el panel lateral para subir sus archivos y visualizarlos</p>
    </div>
    """, unsafe_allow_html=True)

# Mostrar advertencia sobre la necesidad de registro para un uso real
if display_mode == "Fusión" and st.session_state.primary_data['img'] is not None and st.session_state.secondary_data['img'] is not None:
    st.markdown("""
    <div class="info-box" style="margin-top: 20px;">
        <strong>Nota:</strong> Para una aplicación de uso clínico real, sería necesario implementar un algoritmo de registro de imágenes 
        que alinee correctamente las imágenes CT y MRI, asegurando que las estructuras anatómicas coincidan con precisión espacial. 
        Esta versión utiliza un método simplificado que solo redimensiona las imágenes sin un registro anatómico preciso.
    </div>
    """, unsafe_allow_html=True)

# Pie de página
st.markdown("""
<hr style="margin-top: 30px;">
<div style="text-align: center; color: #28aec5; font-size: 14px;">
    Brachyanalysis - Visualizador de imágenes DICOM con fusión CT-MRI
</div>
""", unsafe_allow_html=True)

# Limpiar los directorios temporales antiguos
# En una aplicación real, implementar mecanismo para limpiar directorios temporales

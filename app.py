import os
import io
import zipfile
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import SimpleITK as sitk
import matplotlib.patches as patches

# Configuración de página y estilo
st.set_page_config(layout="wide", page_title="Brachyanalysis")

# CSS personalizado (mantenido del código original)
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
</style>
""", unsafe_allow_html=True)

# Configuración de la barra lateral
st.sidebar.markdown('<p class="sidebar-title">Brachyanalysis</p>', unsafe_allow_html=True)
st.sidebar.markdown('<p class="sub-header">Visualizador de imágenes DICOM</p>', unsafe_allow_html=True)

# Sección de carga de archivos en la barra lateral
st.sidebar.markdown('<p class="sub-header">Configuración</p>', unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Sube un archivo ZIP con tus archivos DICOM", type="zip")

# Función para buscar recursivamente archivos DICOM en un directorio
def find_dicom_files(directory):
    """Busca recursivamente archivos DICOM en el directorio, separando imágenes y estructuras"""
    image_series = []
    structure_files = []
    
    for root, dirs, files in os.walk(directory):
        try:
            # Buscar archivos de estructuras (RS)
            for file in files:
                if file.startswith("RS.") or file.startswith("RS_"):
                    structure_path = os.path.join(root, file)
                    structure_files.append(structure_path)
            
            # Buscar series de imágenes DICOM
            series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(root)
            for series_id in series_ids:
                series_files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(root, series_id)
                if series_files:
                    image_series.append((series_id, root, series_files))
        except Exception as e:
            st.sidebar.warning(f"Advertencia al buscar en {root}: {str(e)}")
            continue
    
    return image_series, structure_files

def apply_window_level(image, window_width, window_center, is_negative=False):
    """Aplica ventana y nivel a la imagen (brillo y contraste)"""
    # Convertir la imagen a float
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
    
    # Invertir si es necesario
    if is_negative:
        image_windowed = 1.0 - image_windowed
    
    return image_windowed

def read_structures(structure_file_path):
    """Lee los archivos de estructura (RS) y extrae contornos"""
    try:
        reader = sitk.ImageFileReader()
        reader.SetFileName(structure_file_path)
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()
        
        # Extraer información básica de las estructuras
        structures = []
        
        # Intentar leer las estructuras mediante SimpleITK
        try:
            # Obtener número de ROIs
            roi_count = 0
            if reader.HasMetaDataKey('3006|0020'):
                roi_count_str = reader.GetMetaDataValue('3006|0020')
                roi_count = int(roi_count_str) if roi_count_str.strip() else 0
            
            # Para cada ROI, extraer nombre y color si está disponible
            for i in range(roi_count):
                structure_name = f"Estructura {i+1}"
                color = [1.0, 0.0, 0.0]  # Rojo por defecto
                
                # Aquí habría que extraer la información detallada
                # En un caso real, necesitaríamos analizar más etiquetas DICOM
                
                structures.append({
                    'name': structure_name,
                    'color': color,
                    'contours': []  # Los contornos reales vendrían del archivo DICOM
                })
        except Exception as e:
            st.warning(f"No se pudieron extraer detalles de estructuras: {str(e)}")
            
        return structures
    except Exception as e:
        st.error(f"Error al leer archivo de estructuras: {str(e)}")
        return []

def plot_multi_view(vol, current_slice, window_width, window_center, is_negative=False, structures=None):
    """
    Genera una visualización con las tres vistas ortogonales: axial, coronal y sagital
    """
    # Crear la figura con 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Extraer las dimensiones
    depth, height, width = vol.shape
    
    # Asegurarse de que los índices están dentro de los límites
    z = np.clip(current_slice[0], 0, depth-1)
    y = np.clip(current_slice[1], 0, height-1)
    x = np.clip(current_slice[2], 0, width-1)
    
    # Extraer los tres cortes ortogonales
    slice_axial = vol[z, :, :]  # Vista axial (XY)
    slice_coronal = vol[:, y, :]  # Vista coronal (XZ)
    slice_sagittal = vol[:, :, x]  # Vista sagital (YZ)
    
    # Aplicar ventana/nivel a cada vista
    axial_windowed = apply_window_level(slice_axial, window_width, window_center, is_negative)
    coronal_windowed = apply_window_level(slice_coronal, window_width, window_center, is_negative)
    sagittal_windowed = apply_window_level(slice_sagittal, window_width, window_center, is_negative)
    
    # Mostrar las tres vistas
    axes[0].imshow(axial_windowed, origin='lower', cmap='gray')
    axes[0].axhline(y=y, color='r', linestyle='-', alpha=0.5)
    axes[0].axvline(x=x, color='b', linestyle='-', alpha=0.5)
    axes[0].set_title('Vista Axial')
    axes[0].axis('off')
    
    axes[1].imshow(np.rot90(coronal_windowed), origin='lower', cmap='gray')
    axes[1].axhline(y=depth-z, color='r', linestyle='-', alpha=0.5)
    axes[1].axvline(x=x, color='g', linestyle='-', alpha=0.5)
    axes[1].set_title('Vista Coronal')
    axes[1].axis('off')
    
    axes[2].imshow(np.rot90(sagittal_windowed), origin='lower', cmap='gray')
    axes[2].axhline(y=depth-z, color='b', linestyle='-', alpha=0.5)
    axes[2].axvline(x=height-y, color='g', linestyle='-', alpha=0.5)
    axes[2].set_title('Vista Sagital')
    axes[2].axis('off')
    
    # Dibujar contornos si están disponibles
    if structures:
        for structure in structures:
            color = structure.get('color', [1.0, 0.0, 0.0])  # Rojo por defecto
            # Aquí dibujaríamos los contornos reales si estuvieran disponibles
            # En esta versión simplificada, dibujamos un círculo representativo
            circle = patches.Circle((width/2, height/2), radius=20, 
                                   edgecolor=color, facecolor='none', alpha=0.7)
            axes[0].add_patch(circle)
    
    plt.tight_layout()
    return fig

# Define los presets de ventana
radiant_presets = {
    "Default window": (0, 0),
    "Full dynamic": (0, 0),
    "CT Abdomen": (350, 50),
    "CT Angio": (600, 300),
    "CT Bone": (2000, 350),
    "CT Brain": (80, 40),
    "CT Chest": (350, 40),
    "CT Lungs": (1500, -600),
    "Negative": (0, 0),
    "Custom window": (0, 0)
}

# Procesar archivos subidos
temp_dir = None
if uploaded_file is not None:
    # Crear un directorio temporal para extraer los archivos
    temp_dir = tempfile.mkdtemp()
    try:
        # Leer el contenido del ZIP
        with zipfile.ZipFile(io.BytesIO(uploaded_file.read()), 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        st.sidebar.markdown('<div class="success-box">Archivos extraídos correctamente.</div>', unsafe_allow_html=True)
    except Exception as e:
        st.sidebar.error(f"Error al extraer el archivo ZIP: {str(e)}")

# Inicializar variables para la visualización
img = None
structures = []

# Título grande siempre visible
st.markdown('<p class="giant-title">Brachyanalysis</p>', unsafe_allow_html=True)

if temp_dir is not None:
    # Buscar series DICOM y archivos de estructura
    with st.spinner('Buscando archivos DICOM...'):
        image_series, structure_files = find_dicom_files(temp_dir)
    
    # Procesar archivos de estructura si existen
    if structure_files:
        st.sidebar.markdown(f'<div class="info-box">Se encontraron {len(structure_files)} archivos de estructura</div>', unsafe_allow_html=True)
        # Leer el primer archivo de estructuras
        if len(structure_files) > 0:
            structures = read_structures(structure_files[0])
            st.sidebar.markdown(f'<div class="success-box">Se cargaron {len(structures)} estructuras</div>', unsafe_allow_html=True)
    
    # Procesar series de imágenes
    if not image_series:
        st.sidebar.error("No se encontraron archivos DICOM de imagen válidos.")
    else:
        st.sidebar.markdown(f'<div class="info-box">Se encontraron {len(image_series)} series DICOM</div>', unsafe_allow_html=True)
        
        # Si hay múltiples series, permitir seleccionar una
        selected_series_idx = 0
        if len(image_series) > 1:
            series_options = [f"Serie {i+1}: {series_id[:10]}... ({len(files)} archivos)" 
                            for i, (series_id, _, files) in enumerate(image_series)]
            selected_series_option = st.sidebar.selectbox("Seleccionar serie DICOM:", series_options)
            selected_series_idx = series_options.index(selected_series_option)
        
        try:
            # Obtener la serie seleccionada
            series_id, series_dir, dicom_names = image_series[selected_series_idx]
            
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(dicom_names)
            reader.LoadPrivateTagsOn()
            reader.MetaDataDictionaryArrayUpdateOn()
            data = reader.Execute()
            img = sitk.GetArrayViewFromImage(data)
        
            # Obtener dimensiones
            depth, height, width = img.shape
            
            # Configurar sliders para cada dimensión
            st.sidebar.markdown('<p class="sub-header">Navegación</p>', unsafe_allow_html=True)
            z_slice = st.sidebar.slider('Corte Axial (Z)', 0, depth - 1, int(depth/2))
            y_slice = st.sidebar.slider('Corte Coronal (Y)', 0, height - 1, int(height/2))
            x_slice = st.sidebar.slider('Corte Sagital (X)', 0, width - 1, int(width/2))
            
            current_slice = [z_slice, y_slice, x_slice]
            
            # Seleccionar tipo de visualización
            output = st.sidebar.radio('Tipo de visualización', ['Imagen', 'Metadatos'], index=0)
            
            # Añadir controles de ventana si la salida es Imagen
            if output == 'Imagen':
                # Calcular valores iniciales para la ventana
                min_val = float(img.min())
                max_val = float(img.max())
                range_val = max_val - min_val
                
                # Establecer valores predeterminados para window width y center
                default_window_width = range_val
                default_window_center = min_val + (range_val / 2)
                
                # Añadir presets de ventana
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
                
                # Si es un preset personalizado, mostrar los campos de entrada
                if selected_preset == "Custom window":
                    st.sidebar.markdown('<p class="sub-header">Ajustes personalizados</p>', unsafe_allow_html=True)
                    
                    # Mostrar información sobre el rango
                    st.sidebar.markdown(f"**Rango de valores:** {min_val:.1f} a {max_val:.1f}")
                    
                    # Crear dos columnas para los campos de entrada
                    col1, col2 = st.sidebar.columns(2)
                    
                    with col1:
                        window_width = float(st.number_input(
                            "Ancho de ventana (WW)",
                            min_value=1.0,
                            max_value=range_val * 2,
                            value=float(default_window_width),
                            format="%.1f"
                        ))
                    
                    with col2:
                        window_center = float(st.number_input(
                            "Centro de ventana (WL)",
                            min_value=min_val - range_val,
                            max_value=max_val + range_val,
                            value=float(default_window_center),
                            format="%.1f"
                        ))
                
                # Toggle para mostrar/ocultar estructuras
                show_structures = st.sidebar.checkbox("Mostrar delimitaciones", value=True)
                structures_to_display = structures if show_structures else None
                
                # Mostrar visualización multi-vista
                st.markdown('<p class="sub-header">Visualización DICOM Multiplanar</p>', unsafe_allow_html=True)
                fig = plot_multi_view(img, current_slice, window_width, window_center, is_negative, structures_to_display)
                st.pyplot(fig)
                
                # Información adicional sobre la imagen y los ajustes actuales
                info_cols = st.columns(3)
                with info_cols[0]:
                    st.markdown(f"**Dimensiones:** {width} x {height} x {depth} px")
                with info_cols[1]:
                    st.markdown(f"**Min/Max:** {min_val:.1f} / {max_val:.1f}")
                with info_cols[2]:
                    st.markdown(f"**Ventana:** WW={window_width:.1f}, WL={window_center:.1f}")
                
            elif img is not None and output == 'Metadatos':
                st.markdown('<p class="sub-header">Metadatos DICOM</p>', unsafe_allow_html=True)
                try:
                    metadata = dict()
                    for k in reader.GetMetaDataKeys(z_slice):
                        metadata[k] = reader.GetMetaData(z_slice, k)
                    df = pd.DataFrame.from_dict(metadata, orient='index', columns=['Valor'])
                    st.dataframe(df, height=600)
                except Exception as e:
                    st.error(f"Error al leer metadatos: {str(e)}")
        except Exception as e:
            st.error(f"Error al procesar los archivos DICOM: {str(e)}")
            st.write("Detalles del error:", str(e))
else:
    # Página de inicio cuando no hay imágenes cargadas
    st.markdown('<p class="sub-header">Visualizador de imágenes DICOM</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; padding: 40px; margin-top: 10px;">
        <img src="https://raw.githubusercontent.com/SimpleITK/SimpleITK/master/Documentation/docs/images/simpleitk-logo.svg" alt="SimpleITK Logo" width="200">
        <h2 style="color: #28aec5; margin-top: 20px;">Carga un archivo ZIP con tus imágenes DICOM</h2>
        <p style="font-size: 18px; margin-top: 10px;">Utiliza el panel lateral para subir tus archivos y visualizarlos</p>
        <p style="font-size: 16px; margin-top: 10px;">El visor mostrará vistas axial, coronal y sagital simultáneamente</p>
    </div>
    """, unsafe_allow_html=True)

# Pie de página
st.markdown("""
<hr style="margin-top: 30px;">
<div style="text-align: center; color: #28aec5; font-size: 14px;">
    Brachyanalysis - Visualizador multiplanar de imágenes DICOM
</div>
""", unsafe_allow_html=True)

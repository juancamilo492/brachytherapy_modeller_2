import os
import io
import zipfile
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import streamlit as st
import SimpleITK as sitk
from skimage import filters, morphology, measure, segmentation
from scipy import ndimage
from sklearn.cluster import KMeans
import cv2
from scipy.spatial import distance
from mpl_toolkits.mplot3d import Axes3D

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
    .segmentation-controls {
        background-color: rgba(192, 215, 17, 0.1);
        padding: 15px;
        border-radius: 8px;
        margin-top: 15px;
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

def plot_slice(vol, slice_ix, window_width, window_center, tumor_mask=None, overlay_alpha=0.3):
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.axis('off')
    selected_slice = vol[slice_ix, :, :]
    
    # Aplicar ajustes de ventana/nivel
    windowed_slice = apply_window_level(selected_slice, window_width, window_center)
    
    # Mostrar la imagen con los ajustes aplicados
    ax.imshow(windowed_slice, origin='lower', cmap='gray')
    
    # Si hay una máscara de tumor, mostrarla como overlay
    if tumor_mask is not None and slice_ix < tumor_mask.shape[0]:
        tumor_slice = tumor_mask[slice_ix]
        if np.any(tumor_slice):  # Solo mostrar si hay algo en la máscara
            masked = np.ma.masked_where(tumor_slice == 0, tumor_slice)
            ax.imshow(masked, origin='lower', alpha=overlay_alpha, cmap='autumn')
    
    return fig

# Segmentación del tumor - Implementación de múltiples técnicas
def segment_tumor(img_array, method="threshold", params=None):
    """
    Segmentar el tumor a partir de las imágenes DICOM usando diferentes métodos
    
    Parámetros:
    - img_array: array NumPy con las imágenes DICOM (3D)
    - method: método de segmentación ("threshold", "kmeans", "region_growing", "manual")
    - params: parámetros específicos para cada método
    
    Retorna:
    - mask: máscara del tumor segmentado (mismo tamaño que img_array)
    """
    # Crear máscara del mismo tamaño que la imagen
    mask = np.zeros_like(img_array, dtype=bool)
    
    # Determinar qué método usar
    if method == "threshold":
        # Segmentación por umbral (thresholding)
        lower_threshold = params.get("lower_threshold", 100)
        upper_threshold = params.get("upper_threshold", 300)
        
        # Aplicar umbral a toda la imagen 3D
        mask = (img_array >= lower_threshold) & (img_array <= upper_threshold)
        
        # Eliminación de ruido con operaciones morfológicas
        for i in range(mask.shape[0]):
            if np.any(mask[i]):
                # Eliminar objetos pequeños
                mask[i] = morphology.remove_small_objects(mask[i], min_size=params.get("min_size", 50))
                # Cerrar huecos en los objetos
                mask[i] = morphology.closing(mask[i], morphology.disk(params.get("closing_size", 2)))
    
    elif method == "kmeans":
        # Segmentación usando K-means clustering
        n_clusters = params.get("n_clusters", 4)
        tumor_cluster = params.get("tumor_cluster", 2)  # Cluster que se considera como tumor
        selected_slice = params.get("slice", img_array.shape[0]//2)
        
        # Usar solo el slice seleccionado para el clustering
        slice_data = img_array[selected_slice].reshape(-1, 1)
        
        # Aplicar K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(slice_data)
        labels = kmeans.labels_.reshape(img_array[selected_slice].shape)
        
        # Asignar el cluster identificado como tumor a la máscara
        mask[selected_slice] = (labels == tumor_cluster)
        
        # Opcional: propagar a slices vecinos basados en similitud
        if params.get("propagate", False):
            reference_mask = mask[selected_slice]
            for i in range(selected_slice + 1, img_array.shape[0]):
                temp_data = img_array[i].reshape(-1, 1)
                temp_labels = kmeans.predict(temp_data).reshape(img_array[i].shape)
                mask[i] = (temp_labels == tumor_cluster)
                # Aplicar restricciones de continuidad espacial
                mask[i] = mask[i] & morphology.binary_dilation(reference_mask)
                reference_mask = mask[i]
                
            reference_mask = mask[selected_slice]
            for i in range(selected_slice - 1, -1, -1):
                temp_data = img_array[i].reshape(-1, 1)
                temp_labels = kmeans.predict(temp_data).reshape(img_array[i].shape)
                mask[i] = (temp_labels == tumor_cluster)
                # Aplicar restricciones de continuidad espacial
                mask[i] = mask[i] & morphology.binary_dilation(reference_mask)
                reference_mask = mask[i]
    
    elif method == "region_growing":
        # Crecimiento de región desde un punto semilla
        seed_point = params.get("seed_point", None)
        tolerance = params.get("tolerance", 10)
        
        if seed_point:
            z, y, x = seed_point
            seed_value = float(img_array[z, y, x])
            
            # Crear una máscara de región creciente para cada slice cercano al punto semilla
            for i in range(max(0, z-5), min(img_array.shape[0], z+6)):
                # Crecimiento de región en 2D para este slice
                slice_img = img_array[i].astype(float)
                segment = np.zeros_like(slice_img, dtype=bool)
                
                # Si estamos en el slice de la semilla, usar esa posición
                if i == z:
                    segment[y, x] = True
                else:
                    # Para otros slices, buscar un punto cercano con valor similar
                    if i > 0 and i < img_array.shape[0] - 1:
                        # Usar el centro de masa de la segmentación del slice anterior si existe
                        if np.any(mask[i-1 if i > z else i+1]):
                            props = measure.regionprops(mask[i-1 if i > z else i+1].astype(int))
                            if props:
                                cy, cx = props[0].centroid
                                segment[int(cy), int(cx)] = True
                
                # Solo hacer crecimiento de región si tenemos un punto de inicio
                if np.any(segment):
                    # Crecimiento de región
                    mask[i] = segmentation.flood(slice_img, tuple(np.argwhere(segment)[0]), 
                                            tolerance=tolerance)
                    
                    # Limpiar la máscara resultante
                    mask[i] = morphology.remove_small_objects(mask[i], 
                                                        min_size=params.get("min_size", 50))
                    mask[i] = morphology.closing(mask[i], 
                                            morphology.disk(params.get("closing_size", 2)))
    
    elif method == "manual":
        # Segmentación manual - dibujar el contorno en un slice específico
        roi_points = params.get("roi_points", [])
        selected_slice = params.get("slice", 0)
        
        if roi_points and len(roi_points) > 2:
            # Crear una máscara vacía para el slice seleccionado
            slice_mask = np.zeros_like(img_array[selected_slice], dtype=np.uint8)
            
            # Convertir puntos a formato numpy para OpenCV
            roi_points_np = np.array(roi_points, dtype=np.int32)
            
            # Dibujar y rellenar el polígono
            cv2.fillPoly(slice_mask, [roi_points_np], 1)
            
            # Asignar al slice correspondiente
            mask[selected_slice] = slice_mask.astype(bool)
            
            # Opcional: propagar a slices vecinos
            if params.get("propagate", False):
                propagation_range = params.get("propagation_range", 3)
                for i in range(1, propagation_range + 1):
                    if selected_slice + i < mask.shape[0]:
                        mask[selected_slice + i] = mask[selected_slice]
                    if selected_slice - i >= 0:
                        mask[selected_slice - i] = mask[selected_slice]
    
    # Opcional: post-procesamiento para cualquier método
    if params.get("post_process", True):
        # Eliminar objetos pequeños y rellenar huecos en todos los slices
        for i in range(mask.shape[0]):
            if np.any(mask[i]):
                mask[i] = morphology.remove_small_objects(mask[i], min_size=params.get("min_size", 50))
                mask[i] = morphology.remove_small_holes(mask[i], area_threshold=params.get("hole_size", 50))
    
    return mask

def project_tumor_to_sphere(tumor_mask, template_center=None, radius=None):
    """
    Proyecta el tumor segmentado a una superficie esférica
    
    Parámetros:
    - tumor_mask: máscara binaria 3D del tumor
    - template_center: centro de la esfera (x, y, z)
    - radius: radio de la esfera de proyección
    
    Retorna:
    - projected_tumor: array de la proyección del tumor sobre la esfera
    - coords: coordenadas 3D de los puntos proyectados (para visualización)
    """
    # Si no se especifica un centro, usar el centro de la imagen
    if template_center is None:
        z_dim, y_dim, x_dim = tumor_mask.shape
        template_center = (x_dim // 2, y_dim // 2, z_dim // 2)
    
    # Si no se especifica radio, estimar basado en la distancia al borde
    if radius is None:
        z_dim, y_dim, x_dim = tumor_mask.shape
        radius = min(template_center[0], template_center[1], template_center[2],
                     x_dim - template_center[0], y_dim - template_center[1], 
                     z_dim - template_center[2]) * 0.8
    
    # Obtener las coordenadas de los voxels que pertenecen al tumor
    tumor_points = np.argwhere(tumor_mask)
    
    if len(tumor_points) == 0:
        return None, None
    
    # Convertir coordenadas a formato (x, y, z) en lugar de (z, y, x)
    tumor_points_xyz = np.array([[p[2], p[1], p[0]] for p in tumor_points])
    
    # Calcular vectores desde el centro a cada punto del tumor
    vectors = tumor_points_xyz - np.array(template_center)
    
    # Calcular la longitud de cada vector
    lengths = np.sqrt(np.sum(vectors**2, axis=1))
    
    # Normalizar los vectores y escalarlos al radio deseado
    normalized_vectors = vectors / lengths[:, np.newaxis]
    projected_points = normalized_vectors * radius + np.array(template_center)
    
    # Convertir los puntos proyectados de nuevo a coordenadas (z, y, x)
    projected_points_zyx = np.array([[int(p[2]), int(p[1]), int(p[0])] for p in projected_points])
    
    # Crear una nueva máscara para los puntos proyectados
    z_dim, y_dim, x_dim = tumor_mask.shape
    projected_tumor = np.zeros((z_dim, y_dim, x_dim), dtype=bool)
    
    # Asegurarse de que los puntos estén dentro de los límites de la imagen
    valid_points = (projected_points_zyx[:, 0] >= 0) & (projected_points_zyx[:, 0] < z_dim) & \
                   (projected_points_zyx[:, 1] >= 0) & (projected_points_zyx[:, 1] < y_dim) & \
                   (projected_points_zyx[:, 2] >= 0) & (projected_points_zyx[:, 2] < x_dim)
    
    valid_projected_points = projected_points_zyx[valid_points]
    
    # Establecer los puntos proyectados como True en la nueva máscara
    for point in valid_projected_points:
        projected_tumor[point[0], point[1], point[2]] = True
    
    # También devolver las coordenadas 3D para visualización
    coords = {
        'tumor_points': tumor_points_xyz,
        'projected_points': projected_points,
        'center': template_center,
        'radius': radius
    }
    
    return projected_tumor, coords

def plot_3d_projection(coords):
    """
    Crea un gráfico 3D para visualizar la proyección del tumor a la esfera
    
    Parámetros:
    - coords: diccionario con las coordenadas del tumor y su proyección
    
    Retorna:
    - fig: figura de matplotlib con la visualización 3D
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Graficar puntos del tumor original
    ax.scatter(coords['tumor_points'][:, 0], coords['tumor_points'][:, 1], coords['tumor_points'][:, 2], 
               color='red', s=10, alpha=0.3, label='Tumor Original')
    
    # Graficar puntos proyectados
    ax.scatter(coords['projected_points'][:, 0], coords['projected_points'][:, 1], coords['projected_points'][:, 2], 
               color='blue', s=10, alpha=0.5, label='Proyección Esférica')
    
    # Graficar el centro
    ax.scatter(*coords['center'], color='green', s=100, label='Centro de Proyección')
    
    # Crear una esfera wireframe
    center = coords['center']
    radius = coords['radius']
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color='green', alpha=0.1)
    
    # Configuraciones adicionales
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Proyección del Tumor a Superficie Esférica')
    ax.legend()
    
    # Ajustar los límites para ver toda la escena
    max_range = max([
        np.max(coords['tumor_points'][:, 0]) - np.min(coords['tumor_points'][:, 0]),
        np.max(coords['tumor_points'][:, 1]) - np.min(coords['tumor_points'][:, 1]),
        np.max(coords['tumor_points'][:, 2]) - np.min(coords['tumor_points'][:, 2])
    ])
    mid_x = (np.max(coords['tumor_points'][:, 0]) + np.min(coords['tumor_points'][:, 0])) * 0.5
    mid_y = (np.max(coords['tumor_points'][:, 1]) + np.min(coords['tumor_points'][:, 1])) * 0.5
    mid_z = (np.max(coords['tumor_points'][:, 2]) + np.min(coords['tumor_points'][:, 2])) * 0.5
    
    ax.set_xlim(mid_x - max_range * 0.6, mid_x + max_range * 0.6)
    ax.set_ylim(mid_y - max_range * 0.6, mid_y + max_range * 0.6)
    ax.set_zlim(mid_z - max_range * 0.6, mid_z + max_range * 0.6)
    
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
tumor_mask = None
projected_tumor = None
projection_coords = None

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
            output = st.sidebar.radio('Tipo de visualización', ['Imagen', 'Metadatos', 'Segmentación', 'Proyección 3D'], index=0)
            
            # Calcular valores iniciales para la ventana
            if img is not None:
                min_val = float(img.min())
                max_val = float(img.max())
                range_val = max_val - min_val
                
                # Establecer valores predeterminados para window width y center
                default_window_width = range_val
                default_window_center = min_val + (range_val / 2)
                
                # Actualizar los presets automáticos
                radiant_presets["Default window"] = (default_window_width, default_window_center)
                radiant_presets["Full dynamic"] = (range_val, min_val + (range_val / 2))
            
            # Añadir controles de ventana (brillo y contraste) si la salida es Imagen
            if output == 'Imagen':
                # Añadir presets de ventana para radiología
                st.sidebar.markdown('<div class="control-section">', unsafe_allow_html=True)
                st.sidebar.markdown('<p class="sub-header">Presets de ventana</p>', unsafe_allow_html=True)
                
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
                
                # Opción para mostrar la segmentación como overlay
                st.sidebar.markdown('<div class="segmentation-controls">', unsafe_allow_html=True)
                st.sidebar.markdown('<p class="sub-header">Mostrar Segmentación</p>', unsafe_allow_html=True)
                
                show_segmentation = st.sidebar.checkbox("Mostrar tumor segmentado", value=False)
                if show_segmentation and tumor_mask is None:
                    st.sidebar.warning("Primero debes realizar la segmentación del tumor en la pestaña Segmentación")
                    
                if show_segmentation and tumor_mask is not None:
                    overlay_alpha = st.sidebar.slider("Transparencia", 0.1, 1.0, 0.3)
                else:
                    overlay_alpha = 0.3
                    
                st.sidebar.markdown('</div>', unsafe_allow_html=True)

            elif output == 'Segmentación':
                st.sidebar.markdown('<div class="segmentation-controls">', unsafe_allow_html=True)
                st.sidebar.markdown('<p class="sub-header">Parámetros de Segmentación</p>', unsafe_allow_html=True)
                
                segmentation_method = st.sidebar.selectbox(
                    "Método de segmentación",
                    ["threshold", "kmeans", "region_growing", "manual"],
                    format_func=lambda x: {
                        "threshold": "Umbralización",
                        "kmeans": "K-means clustering",
                        "region_growing": "Crecimiento de región",
                        "manual": "Segmentación manual"
                    }[x]
                )
                
                segmentation_params = {}
                
                # Parámetros específicos para cada método
                if segmentation_method == "threshold":
                    # Parámetros para umbralización
                    col1, col2 = st.sidebar.columns(2)
                    with col1:
                        lower_threshold = st.number_input(
                            "Umbral inferior",
                            min_value=float(min_val),
                            max_value=float(max_val),
                            value=float(min_val + range_val * 0.4),
                            format="%.1f"
                        )
                    with col2:
                        upper_threshold = st.number_input(
                            "Umbral superior",
                            min_value=float(min_val),
                            max_value=float(max_val),
                            value=float(min_val + range_val * 0.7),
                            format="%.1f"
                        )
                    
                    segmentation_params["lower_threshold"] = lower_threshold
                    segmentation_params["upper_threshold"] = upper_threshold
                    
                    # Parámetros avanzados
                    with st.sidebar.expander("Parámetros avanzados"):
                        min_size = st.slider("Tamaño mínimo de objeto", 10, 500, 50)
                        closing_size = st.slider("Tamaño de cierre", 1, 10, 2)
                        segmentation_params["min_size"] = min_size
                        segmentation_params["closing_size"] = closing_size
                
                elif segmentation_method == "kmeans":
                    # Parámetros para K-means
                    n_clusters = st.sidebar.slider("Número de clusters", 2, 8, 4)
                    tumor_cluster = st.sidebar.slider("Cluster del tumor (0-indexado)", 0, n_clusters-1, 2)
                    selected_kmeans_slice = st.sidebar.slider("Slice para clustering", 0, n_slices-1, slice_ix)
                    propagate = st.sidebar.checkbox("Propagar a slices vecinos", value=True)
                    
                    segmentation_params["n_clusters"] = n_clusters
                    segmentation_params["tumor_cluster"] = tumor_cluster
                    segmentation_params["slice"] = selected_kmeans_slice
                    segmentation_params["propagate"] = propagate
                
                elif segmentation_method == "region_growing":
                    # Parámetros para crecimiento de región
                    st.sidebar.markdown("**Selecciona un punto semilla en la imagen**")
                    if not hasattr(st.session_state, 'seed_point'):
                        st.session_state.seed_point = None
                    
                    # Modo de selección de punto semilla
                    st.sidebar.markdown("Navega al slice deseado y haz clic en la imagen para seleccionar un punto semilla")
                    
                    # Mostrar el punto semilla seleccionado
                    if st.session_state.seed_point:
                        z, y, x = st.session_state.seed_point
                        st.sidebar.markdown(f"Punto semilla: Slice {z}, Y={y}, X={x}")
                        seed_value = img[z, y, x] if 0 <= z < img.shape[0] and 0 <= y < img.shape[1] and 0 <= x < img.shape[2] else 0
                        st.sidebar.markdown(f"Valor en el punto: {seed_value}")
                    
                    tolerance = st.sidebar.slider("Tolerancia", 5, 100, 20)
                    
                    segmentation_params["seed_point"] = st.session_state.seed_point
                    segmentation_params["tolerance"] = tolerance
                    
                    # Parámetros avanzados
                    with st.sidebar.expander("Parámetros avanzados"):
                        min_size = st.slider("Tamaño mínimo de objeto", 10, 500, 50)
                        closing_size = st.slider("Tamaño de cierre", 1, 10, 2)
                        segmentation_params["min_size"] = min_size
                        segmentation_params["closing_size"] = closing_size
                
                elif segmentation_method == "manual":
                    # Parámetros para segmentación manual
                    st.sidebar.markdown("**Dibuja el contorno del tumor en la imagen**")
                    
                    # Inicializar puntos del ROI si no existen
                    if not hasattr(st.session_state, 'roi_points'):
                        st.session_state.roi_points = []
                    
                    # Botón para limpiar los puntos
                    if st.sidebar.button("Limpiar contorno"):
                        st.session_state.roi_points = []
                    
                    # Mostrar información sobre los puntos seleccionados
                    if st.session_state.roi_points:
                        st.sidebar.markdown(f"Puntos seleccionados: {len(st.session_state.roi_points)}")
                    
                    # Opciones de propagación
                    propagate = st.sidebar.checkbox("Propagar a slices vecinos", value=True)
                    if propagate:
                        propagation_range = st.sidebar.slider("Rango de propagación", 1, 10, 3)
                        segmentation_params["propagation_range"] = propagation_range
                    
                    segmentation_params["roi_points"] = st.session_state.roi_points
                    segmentation_params["slice"] = slice_ix
                    segmentation_params["propagate"] = propagate
                
                # Botón para ejecutar la segmentación
                if st.sidebar.button("Segmentar tumor"):
                    with st.spinner('Segmentando tumor...'):
                        tumor_mask = segment_tumor(img, method=segmentation_method, params=segmentation_params)
                        st.session_state.tumor_mask = tumor_mask
                        st.sidebar.success("Segmentación completada")
                
                # Botón para eliminar la segmentación
                if st.sidebar.button("Eliminar segmentación"):
                    tumor_mask = None
                    if hasattr(st.session_state, 'tumor_mask'):
                        del st.session_state.tumor_mask
                    st.sidebar.success("Segmentación eliminada")
                
                # Usar la segmentación guardada si existe
                if not tumor_mask and hasattr(st.session_state, 'tumor_mask'):
                    tumor_mask = st.session_state.tumor_mask
                
                st.sidebar.markdown('</div>', unsafe_allow_html=True)
                
                # Valores predeterminados para cuando se necesita ventana en la sección de segmentación
                window_width = default_window_width
                window_center = default_window_center
                is_negative = False
            
            elif output == 'Proyección 3D':
                st.sidebar.markdown('<div class="segmentation-controls">', unsafe_allow_html=True)
                st.sidebar.markdown('<p class="sub-header">Parámetros de Proyección</p>', unsafe_allow_html=True)
                
                # Verificar si existe una segmentación
                if not hasattr(st.session_state, 'tumor_mask') or st.session_state.tumor_mask is None:
                    st.sidebar.warning("Primero debes realizar la segmentación del tumor en la pestaña Segmentación")
                else:
                    # Parámetros para la proyección esférica
                    z_dim, y_dim, x_dim = img.shape
                    
                    # Opciones para el centro de la esfera
                    center_option = st.sidebar.radio(
                        "Centro de la esfera",
                        ["Centro de la imagen", "Centro del tumor", "Personalizado"]
                    )
                    
                    if center_option == "Centro de la imagen":
                        template_center = (x_dim // 2, y_dim // 2, z_dim // 2)
                    elif center_option == "Centro del tumor":
                        if hasattr(st.session_state, 'tumor_mask'):
                            # Calcular el centro del tumor
                            tumor_points = np.argwhere(st.session_state.tumor_mask)
                            if len(tumor_points) > 0:
                                # Calcular centroide (z, y, x)
                                mean_point = np.mean(tumor_points, axis=0)
                                # Convertir a (x, y, z)
                                template_center = (int(mean_point[2]), int(mean_point[1]), int(mean_point[0]))
                            else:
                                template_center = (x_dim // 2, y_dim // 2, z_dim // 2)
                                st.sidebar.warning("No se encontró tumor segmentado, usando centro de la imagen")
                        else:
                            template_center = (x_dim // 2, y_dim // 2, z_dim // 2)
                            st.sidebar.warning("No se encontró tumor segmentado, usando centro de la imagen")
                    else:  # Personalizado
                        col1, col2, col3 = st.sidebar.columns(3)
                        with col1:
                            center_x = st.number_input("Centro X", 0, x_dim-1, x_dim//2)
                        with col2:
                            center_y = st.number_input("Centro Y", 0, y_dim-1, y_dim//2)
                        with col3:
                            center_z = st.number_input("Centro Z", 0, z_dim-1, z_dim//2)
                        template_center = (center_x, center_y, center_z)
                    
                    # Opciones para el radio de la esfera
                    radius_option = st.sidebar.radio(
                        "Radio de la esfera",
                        ["Automático", "Personalizado"]
                    )
                    
                    if radius_option == "Automático":
                        radius = min(template_center[0], template_center[1], template_center[2],
                                    x_dim - template_center[0], y_dim - template_center[1], 
                                    z_dim - template_center[2]) * 0.8
                    else:  # Personalizado
                        max_possible_radius = min(template_center[0], template_center[1], template_center[2],
                                                x_dim - template_center[0], y_dim - template_center[1], 
                                                z_dim - template_center[2])
                        radius = st.sidebar.slider("Radio de la esfera", 10, int(max_possible_radius), 
                                                int(max_possible_radius * 0.8))
                    
                    # Botón para ejecutar la proyección
                    if st.sidebar.button("Proyectar tumor"):
                        with st.spinner('Proyectando tumor a esfera...'):
                            projected_tumor, projection_coords = project_tumor_to_sphere(
                                st.session_state.tumor_mask, 
                                template_center=template_center,
                                radius=radius
                            )
                            st.session_state.projected_tumor = projected_tumor
                            st.session_state.projection_coords = projection_coords
                            st.sidebar.success("Proyección completada")
                
                # Usar la proyección guardada si existe
                if not projected_tumor and hasattr(st.session_state, 'projected_tumor'):
                    projected_tumor = st.session_state.projected_tumor
                    projection_coords = st.session_state.projection_coords
                
                st.sidebar.markdown('</div>', unsafe_allow_html=True)
                
                # Valores predeterminados para cuando no son necesarios
                window_width = default_window_width
                window_center = default_window_center
                is_negative = False
                
            else:  # Para Metadatos
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
        
        # Si hay una máscara de tumor y se solicita mostrarla, hacerlo como overlay
        if tumor_mask is not None and show_segmentation:
            ax.imshow(windowed_slice, origin='lower', cmap='gray')
            tumor_slice = tumor_mask[slice_ix]
            if np.any(tumor_slice):  # Solo mostrar si hay algo en la máscara
                masked = np.ma.masked_where(tumor_slice == 0, tumor_slice)
                ax.imshow(masked, origin='lower', alpha=overlay_alpha, cmap='autumn')
        else:
            ax.imshow(windowed_slice, origin='lower', cmap='gray')
        
        st.pyplot(fig)
    else:
        # Muestra la imagen en la ventana principal con los ajustes aplicados
        # y opcionalmente con la máscara del tumor
        show_mask = tumor_mask if (show_segmentation and 'show_segmentation' in locals()) else None
        fig = plot_slice(img, slice_ix, window_width, window_center, 
                        tumor_mask=show_mask, 
                        overlay_alpha=overlay_alpha if 'overlay_alpha' in locals() else 0.3)
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

elif img is not None and output == 'Segmentación':
    st.markdown('<p class="sub-header">Segmentación del Tumor</p>', unsafe_allow_html=True)
    
    # Instrucciones diferentes según el método seleccionado
    method_instructions = {
        "threshold": "Ajusta los umbrales inferior y superior para capturar el tumor.",
        "kmeans": "Selecciona el número de clusters y cuál representa el tumor.",
        "region_growing": "Haz clic en la imagen para seleccionar un punto dentro del tumor.",
        "manual": "Haz clic en la imagen para añadir puntos y dibujar el contorno del tumor."
    }
    
    st.markdown(f"**{method_instructions.get(segmentation_method, '')}**")
    
    # Mostrar la imagen principal con opciones específicas según el método
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.axis('off')
    selected_slice = img[slice_ix, :, :]
    
    # Aplicar ventana para mejor visualización
    windowed_slice = apply_window_level(selected_slice, window_width, window_center)
    ax.imshow(windowed_slice, origin='lower', cmap='gray')
    
    # Si hay una máscara de tumor, mostrarla como overlay
    if tumor_mask is not None and slice_ix < tumor_mask.shape[0]:
        tumor_slice = tumor_mask[slice_ix]
        if np.any(tumor_slice):  # Solo mostrar si hay algo en la máscara
            masked = np.ma.masked_where(tumor_slice == 0, tumor_slice)
            ax.imshow(masked, origin='lower', alpha=0.3, cmap='autumn')
    
    # Mostrar puntos específicos según el método
    if segmentation_method == "region_growing" and hasattr(st.session_state, 'seed_point'):
        seed_z, seed_y, seed_x = st.session_state.seed_point
        if seed_z == slice_ix:  # Solo mostrar el punto si estamos en el slice correcto
            ax.plot(seed_x, seed_y, 'ro', markersize=10)
    
    elif segmentation_method == "manual" and hasattr(st.session_state, 'roi_points') and st.session_state.roi_points:
        # Dibujar los puntos y líneas del contorno
        points = np.array(st.session_state.roi_points)
        ax.plot(points[:, 0], points[:, 1], 'ro-', linewidth=2, markersize=8)
        
        # Si hay más de 2 puntos, cerrar el polígono
        if len(points) > 2:
            ax.plot([points[-1, 0], points[0, 0]], [points[-1, 1], points[0, 1]], 'ro-', linewidth=2)
    
    # Mostrar la imagen con las anotaciones
    plot_placeholder = st.empty()
    plot_placeholder.pyplot(fig)
    
    # Manejar los eventos de clic según el método
    if segmentation_method in ["region_growing", "manual"]:
        # Crear un área para capturar el clic del usuario
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            clicked = st.button("Capturar punto en la imagen" if segmentation_method == "region_growing" else "Añadir punto al contorno")
            
            if clicked:
                # En una aplicación real, se capturaría el clic del usuario
                # En Streamlit, podemos usar widgets para simular esta interacción
                coords_col1, coords_col2 = st.columns(2)
                with coords_col1:
                    x_coord = st.number_input("Coordenada X", 0, img.shape[2]-1, img.shape[2]//2)
                with coords_col2:
                    y_coord = st.number_input("Coordenada Y", 0, img.shape[1]-1, img.shape[1]//2)
                
                if st.button("Confirmar punto"):
                    if segmentation_method == "region_growing":
                        # Guardar el punto semilla (slice actual, y, x)
                        st.session_state.seed_point = (slice_ix, int(y_coord), int(x_coord))
                        st.success(f"Punto semilla establecido en: Slice {slice_ix}, Y={y_coord}, X={x_coord}")
                    else:  # manual
                        # Añadir el punto al contorno
                        if not hasattr(st.session_state, 'roi_points'):
                            st.session_state.roi_points = []
                        st.session_state.roi_points.append((int(x_coord), int(y_coord)))
                        st.success(f"Punto añadido al contorno: X={x_coord}, Y={y_coord}")
                    
                    # Recargar la página para actualizar el gráfico
                    st.experimental_rerun()
    
    # Mostrar resumen de la segmentación
    if tumor_mask is not None:
        st.markdown("### Resultados de la segmentación")
        # Calcular volumen y otras métricas
        voxel_count = np.sum(tumor_mask)
        # En un caso real, multiplicaríamos por el volumen de cada voxel
        # pero para este ejemplo, solo contamos voxels
        
        metrics_cols = st.columns(3)
        with metrics_cols[0]:
            st.metric("Voxels segmentados", f"{voxel_count}")
        with metrics_cols[1]:
            # Contar slices con tumor
            slices_with_tumor = sum(1 for i in range(tumor_mask.shape[0]) if np.any(tumor_mask[i]))
            st.metric("Slices con tumor", f"{slices_with_tumor}/{tumor_mask.shape[0]}")
        with metrics_cols[2]:
            # Volumen aproximado (suponiendo voxels de 1mm³)
            st.metric("Volumen aproximado", f"{voxel_count} mm³")

elif img is not None and output == 'Proyección 3D':
    st.markdown('<p class="sub-header">Proyección 3D del Tumor</p>', unsafe_allow_html=True)
    
    # Verificar si hay una segmentación disponible
    if not hasattr(st.session_state, 'tumor_mask') or st.session_state.tumor_mask is None:
        st.warning("No hay tumor segmentado. Por favor, realiza primero la segmentación en la pestaña correspondiente.")
    else:
        # Verificar si hay una proyección disponible
        if projection_coords is not None:
            # Visualizar la proyección 3D
            fig = plot_3d_projection(projection_coords)
            st.pyplot(fig)
            
            # Mostrar información sobre la proyección
            info_cols = st.columns(3)
            with info_cols[0]:
                st.markdown(f"**Centro de proyección:** ({projection_coords['center'][0]}, {projection_coords['center'][1]}, {projection_coords['center'][2]})")
            with info_cols[1]:
                st.markdown(f"**Radio de la esfera:** {projection_coords['radius']:.1f}")
            with info_cols[2]:
                num_points = len(projection_coords['tumor_points'])
                st.markdown(f"**Puntos proyectados:** {num_points}")
            
            # Opciones para visualizar las imágenes de la proyección
            st.markdown("### Visualización de cortes con la proyección")
            viz_option = st.radio(
                "Selecciona qué visualizar:",
                ["Tumor original", "Proyección a esfera", "Ambos superpuestos"]
            )
            
            # Mostrar visualización según la opción seleccionada
            fig2, ax2 = plt.subplots(figsize=(12, 8))
            plt.axis('off')
            selected_slice = img[slice_ix, :, :]
            windowed_slice = apply_window_level(selected_slice, window_width, window_center)
            ax2.imshow(windowed_slice, origin='lower', cmap='gray')
            
            if viz_option == "Tumor original" or viz_option == "Ambos superpuestos":
                if tumor_mask is not None and slice_ix < tumor_mask.shape[0]:
                    tumor_slice = tumor_mask[slice_ix]
                    if np.any(tumor_slice):
                        masked = np.ma.masked_where(tumor_slice == 0, tumor_slice)
                        ax2.imshow(masked, origin='lower', alpha=0.4, cmap='autumn')
            
            if viz_option == "Proyección a esfera" or viz_option == "Ambos superpuestos":
                if projected_tumor is not None and slice_ix < projected_tumor.shape[0]:
                    projected_slice = projected_tumor[slice_ix]
                    if np.any(projected_slice):
                        masked_proj = np.ma.masked_where(projected_slice == 0, projected_slice)
                        ax2.imshow(masked_proj, origin='lower', alpha=0.4, 
                                  cmap='cool' if viz_option == "Ambos superpuestos" else 'autumn')
            
            st.pyplot(fig2)
            
            # Opción para guardar resultados (en una aplicación real)
            st.markdown("### Exportar resultados")
            export_option = st.selectbox(
                "Formato de exportación:",
                ["DICOM RT Structure", "Modelo 3D (STL)", "Coordenadas (CSV)"]
            )
            
            if st.button("Exportar proyección"):
                st.info(f"La exportación en formato '{export_option}' estaría disponible en una implementación completa.")
                # Aquí se implementaría la exportación real en una aplicación completa
        else:
            st.info("Usa las opciones en el panel lateral para configurar y generar la proyección.")

else:
    # Página de inicio cuando no hay imágenes cargadas
    st.markdown('<p class="sub-header">Visualizador de imágenes DICOM</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; padding: 40px; margin-top: 10px;">
        <img src="https://raw.githubusercontent.com/SimpleITK/SimpleITK/master/Documentation/docs/images/simpleitk-logo.svg" alt="SimpleITK Logo" width="200">
        <h2 style="color: #28aec5; margin-top: 20px;">Carga un archivo ZIP con tus imágenes DICOM</h2>
        <p style="font-size: 18px; margin-top: 10px;">Utiliza el panel lateral para subir tus archivos y visualizarlos</p>
        
        <div style="margin-top: 30px; text-align: left; max-width: 800px; margin-left: auto; margin-right: auto;">
            <h3 style="color: #c0d711;">Características principales:</h3>
            <ul style="font-size: 16px; line-height: 1.6;">
                <li><strong>Visualización DICOM:</strong> Explora imágenes médicas con controles de ventana personalizables</li>
                <li><strong>Segmentación del tumor:</strong> Múltiples métodos para identificar y delimitar tumores</li>
                <li><strong>Proyección 3D:</strong> Proyecta el tumor a una superficie esférica para planificación de braquiterapia</li>
                <li><strong>Exportación:</strong> Guarda los resultados en formatos estándar médicos</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Pie de página
st.markdown("""
<hr style="margin-top: 30px;">
<div style="text-align: center; color: #28aec5; font-size: 14px;">
    Brachyanalysis - Visualizador de imágenes DICOM con segmentación y proyección para braquiterapia
</div>
""", unsafe_allow_html=True)

# Capturar eventos de clic en la imagen cuando sea necesario
if img is not None and output == 'Segmentación' and segmentation_method in ["region_growing", "manual"]:
    # En un entorno web real, esta funcionalidad requeriría JavaScript para capturar clics
    # En Streamlit, esto es una limitación, por lo que usamos los campos de entrada como alternativa
    pass

# Limpiar el directorio temporal si se creó uno
if temp_dir and os.path.exists(temp_dir):
    # Nota: En una aplicación real, deberías limpiar los directorios temporales
    # cuando la aplicación se cierre, pero en Streamlit esto es complicado
    # ya que las sesiones persisten. Una solución es mantener un registro
    # de directorios temporales y limpiarlos al inicio.
    pass

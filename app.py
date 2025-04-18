import os
import io
import zipfile
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import SimpleITK as sitk
from sklearn.cluster import KMeans
from scipy.optimize import minimize
from scipy import ndimage

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

# Funciones para segmentación de tumores
def threshold_based_segmentation(img_3d, slice_ix, min_threshold=None, max_threshold=None):
    """
    Segmenta el tumor basado en umbrales de intensidad
    
    Parameters:
    -----------
    img_3d : numpy.ndarray
        Imagen 3D (serie DICOM completa)
    slice_ix : int
        Índice del corte actual
    min_threshold : float
        Umbral mínimo de intensidad (si es None, se calcula automáticamente)
    max_threshold : float
        Umbral máximo de intensidad (si es None, se calcula automáticamente)
    
    Returns:
    --------
    mask : numpy.ndarray
        Máscara binaria del tumor segmentado
    """
    # Seleccionar el corte actual
    img_slice = img_3d[slice_ix].copy()
    
    # Calcular umbrales automáticamente si no son proporcionados
    if min_threshold is None or max_threshold is None:
        # Calcular histograma
        hist, bin_edges = np.histogram(img_slice.flatten(), bins=256)
        
        # Suponer que el tumor tiene un rango de intensidad particular
        # Por defecto, tomamos valores en el rango superior del histograma
        if min_threshold is None:
            # Usar el percentil 80 como umbral mínimo por defecto
            min_threshold = np.percentile(img_slice, 80)
        
        if max_threshold is None:
            # Usar el valor máximo como umbral máximo
            max_threshold = np.max(img_slice)
    
    # Crear máscara binaria basada en umbrales
    mask = (img_slice >= min_threshold) & (img_slice <= max_threshold)
    
    # Procesamiento morfológico para eliminar pequeños objetos y rellenar huecos
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3)))
    
    # Eliminar componentes pequeños (mantener solo los más grandes)
    labeled_mask, num_features = ndimage.label(mask)
    if num_features > 0:
        sizes = ndimage.sum(mask, labeled_mask, range(1, num_features + 1))
        if len(sizes) > 0:
            # Mantener solo el componente más grande (asumiendo que es el tumor)
            largest_component = np.argmax(sizes) + 1
            mask = labeled_mask == largest_component
    
    return mask

def region_growing_segmentation(img_3d, slice_ix, seed_point=None, threshold_factor=0.3):
    """
    Segmenta el tumor usando el algoritmo de crecimiento de regiones
    
    Parameters:
    -----------
    img_3d : numpy.ndarray
        Imagen 3D (serie DICOM completa)
    slice_ix : int
        Índice del corte actual
    seed_point : tuple
        Punto de inicio (y, x) para el crecimiento. Si es None, se usa el centro de la imagen
    threshold_factor : float
        Factor que determina el rango de intensidades aceptables
    
    Returns:
    --------
    mask : numpy.ndarray
        Máscara binaria del tumor segmentado
    """
    # Seleccionar el corte actual
    img_slice = img_3d[slice_ix].copy()
    
    # Si no se proporciona un punto semilla, usar el centro de la imagen
    if seed_point is None:
        seed_point = (img_slice.shape[0] // 2, img_slice.shape[1] // 2)
    
    # Obtener el valor de intensidad en el punto semilla
    seed_intensity = img_slice[seed_point]
    
    # Definir umbrales para el crecimiento de regiones
    lower_threshold = seed_intensity - threshold_factor * seed_intensity
    upper_threshold = seed_intensity + threshold_factor * seed_intensity
    
    # Inicializar la máscara y la cola de píxeles a procesar
    mask = np.zeros_like(img_slice, dtype=bool)
    processed = np.zeros_like(img_slice, dtype=bool)
    
    # Añadir el punto semilla a la cola
    points_to_process = [seed_point]
    mask[seed_point] = True
    processed[seed_point] = True
    
    # Definir movimientos en 4 direcciones (arriba, abajo, izquierda, derecha)
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Algoritmo de crecimiento de regiones
    while points_to_process:
        y, x = points_to_process.pop(0)
        
        for dy, dx in neighbors:
            new_y, new_x = y + dy, x + dx
            
            # Verificar si está dentro de los límites
            if (0 <= new_y < img_slice.shape[0] and 0 <= new_x < img_slice.shape[1] and
                not processed[new_y, new_x]):
                
                processed[new_y, new_x] = True
                # Verificar si cumple con los umbrales
                if lower_threshold <= img_slice[new_y, new_x] <= upper_threshold:
                    mask[new_y, new_x] = True
                    points_to_process.append((new_y, new_x))
    
    # Procesamiento morfológico para mejorar la segmentación
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3)))
    
    return mask

def project_tumor_to_sphere(tumor_mask, img_3d, slice_ix, template_center=None, radius=None):
    """
    Proyecta la máscara del tumor a una superficie esférica
    
    Parameters:
    -----------
    tumor_mask : numpy.ndarray
        Máscara binaria del tumor segmentado
    img_3d : numpy.ndarray
        Imagen 3D completa
    slice_ix : int
        Índice del corte actual
    template_center : tuple
        Centro de la plantilla (y, x). Si es None, se usa el centro de la imagen
    radius : float
        Radio de la esfera. Si es None, se calcula automáticamente
    
    Returns:
    --------
    projected_points : list
        Lista de puntos proyectados (theta, phi) donde el tumor está presente
    """
    if template_center is None:
        # Usar el centro de la imagen como centro de la plantilla
        template_center = (tumor_mask.shape[0] // 2, tumor_mask.shape[1] // 2)
    
    # Encontrar los puntos del tumor (donde la máscara es True)
    tumor_points = np.where(tumor_mask)
    y_points, x_points = tumor_points
    
    if len(y_points) == 0:
        return []  # No hay puntos del tumor para proyectar
    
    # Calcular coordenadas relativas al centro de la plantilla
    y_rel = y_points - template_center[0]
    x_rel = x_points - template_center[1]
    
    # Si el radio no se proporciona, calcularlo basado en la distancia máxima
    if radius is None:
        distances = np.sqrt(y_rel**2 + x_rel**2)
        radius = np.max(distances) * 1.1  # Añadir un 10% extra
    
    # Crear una matriz 3D para representar los puntos en el espacio
    # La coordenada z es la profundidad en la serie de imágenes
    z_rel = np.full_like(y_rel, slice_ix - img_3d.shape[0] // 2)
    
    # Calcular coordenadas esféricas (r, theta, phi)
    # r es la distancia desde el origen
    r = np.sqrt(x_rel**2 + y_rel**2 + z_rel**2)
    
    # theta es el ángulo en el plano xy (azimut)
    theta = np.arctan2(y_rel, x_rel)
    
    # phi es el ángulo desde el eje z (elevación)
    phi = np.arccos(z_rel / np.maximum(r, 1e-10))
    
    # Proyectar a la superficie de la esfera (r = radius)
    projected_points = [(t, p) for t, p in zip(theta, phi)]
    
    return projected_points

def calculate_terminal_points(projected_points, num_needles=9):
    """
    Calcula los puntos terminales óptimos usando k-means clustering
    
    Parameters:
    -----------
    projected_points : list
        Lista de puntos proyectados (theta, phi)
    num_needles : int
        Número de agujas (clusters) a calcular
    
    Returns:
    --------
    terminal_points : numpy.ndarray
        Array de puntos terminales (theta, phi)
    """
    if not projected_points:
        return np.array([])
    
    # Convertir a array numpy
    points_array = np.array(projected_points)
    
    # Ajustar el número de clusters si hay pocos puntos
    actual_num_needles = min(num_needles, len(projected_points))
    
    if actual_num_needles == 0:
        return np.array([])
    
    # Aplicar k-means clustering
    kmeans = KMeans(n_clusters=actual_num_needles, random_state=42)
    kmeans.fit(points_array)
    
    # Obtener los centros de los clusters como puntos terminales
    terminal_points = kmeans.cluster_centers_
    
    return terminal_points

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

def plot_slice_with_segmentation(vol, slice_ix, window_width, window_center, segmentation_mask=None, projected_points=None):
    """
    Visualiza un corte con la segmentación superpuesta
    
    Parameters:
    -----------
    vol : numpy.ndarray
        Volumen 3D de la imagen
    slice_ix : int
        Índice del corte a visualizar
    window_width : float
        Ancho de ventana para ajustes de visualización
    window_center : float
        Centro de ventana para ajustes de visualización
    segmentation_mask : numpy.ndarray
        Máscara de segmentación (opcional)
    projected_points : list
        Lista de puntos proyectados (opcional)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figura con la visualización
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.axis('off')
    selected_slice = vol[slice_ix, :, :]
    
    # Aplicar ajustes de ventana/nivel
    windowed_slice = apply_window_level(selected_slice, window_width, window_center)
    
    # Mostrar la imagen con los ajustes aplicados
    ax.imshow(windowed_slice, origin='lower', cmap='gray')
    
    # Si hay una máscara de segmentación, superponerla
    if segmentation_mask is not None:
        # Crear una versión RGB de la imagen
        segmentation_overlay = np.zeros((*windowed_slice.shape, 4))
        segmentation_overlay[..., 0] = 1.0  # R (rojo)
        segmentation_overlay[..., 1] = 0.0  # G (verde)
        segmentation_overlay[..., 2] = 0.0  # B (azul)
        segmentation_overlay[..., 3] = 0.3 * segmentation_mask  # Transparencia (alpha)
        
        # Superponer la máscara
        ax.imshow(segmentation_overlay, origin='lower')
    
    # Si hay puntos proyectados, mostrarlos
    if projected_points and len(projected_points) > 0:
        # Convertir coordenadas esféricas a cartesianas
        center = (selected_slice.shape[1] // 2, selected_slice.shape[0] // 2)
        for theta, phi in projected_points:
            # Cálculo simplificado para mostrar en 2D
            x = center[0] + 50 * np.cos(theta)
            y = center[1] + 50 * np.sin(theta)
            ax.scatter(x, y, c='g', s=20)
    
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
projected_points = None
terminal_points = None

# Añadir configuración para la segmentación
segmentation_method = st.sidebar.radio(
    "Método de segmentación",
    ["Ninguno", "Umbral", "Crecimiento de regiones"],
    index=0
)

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
            output = st.sidebar.radio('Tipo de visualización', ['Imagen', 'Metadatos', 'Segmentación'], index=0)
            
            # Calcular valores iniciales para la ventana
            if img is not None:
                min_val = float(img.min())
                max_val = float(img.max())
                range_val = max_val - min_val
                
                # Establecer valores predeterminados para window width y center
                default_window_width = range_val
                default_window_center = min_val + (range_val / 2)
                
                # Ajustar los presets automáticos
                radiant_presets["Default window"] = (default_window_width, default_window_center)
                radiant_presets["Full dynamic"] = (range_val, min_val + (range_val / 2))
            
            # Añadir controles de ventana (brillo y contraste) si la salida es Imagen o Segmentación
            if output in ['Imagen', 'Segmentación']:
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
                
                # Si la salida es Segmentación, añadir controles específicos de segmentación
                if output == 'Segmentación':
                    st.sidebar.markdown('<div class="control-section">', unsafe_allow_html=True)
                    st.sidebar.markdown('<p class="sub-header">Parámetros de segmentación</p>', unsafe_allow_html=True)
                    
                    if segmentation_method == "Umbral":
                        # Parámetros para segmentación por umbral
                        min_threshold = st.sidebar.slider(
                            "Umbral mínimo", 
                            min_value=float(min_val),
                            max_value=float(max_val),
                            value=float(min_val + 0.8 * range_val),  # Por defecto, 80% del rango
                            format="%.1f"
                        )
                        
                        max_threshold = st.sidebar.slider(
                            "Umbral máximo",
                            min_value=float(min_val),
                            max_value=float(max_val),
                            value=float(max_val),
                            format="%.1f"
                        )
                        
                        # Segmentar el tumor usando umbrales
                        tumor_mask = threshold_based_segmentation(
                            img, slice_ix, min_threshold, max_threshold
                        )
                        
                    elif segmentation_method == "Crecimiento de regiones":
                        # Parámetros para segmentación por crecimiento de regiones
                        threshold_factor = st.sidebar.slider(
                            "Factor de umbral",
                            min_value=0.05,
                            max_value=0.5,
                            value=0.3,
                            step=0.05
                        )
                        
                        # Obtener la posición del punto semilla mediante un clic en la imagen
                        st.sidebar.markdown("**Punto semilla:**")
                        st.sidebar.markdown("Para seleccionar, haga clic en la imagen")
                        
                        seed_point = None
                        # Por defecto, usar el centro de la imagen
                        y_center, x_center = img[slice_ix].shape[0] // 2, img[slice_ix].shape[1] // 2
                        
                        seed_y = st.sidebar.slider(
                            "Y", 
                            min_value=0,
                            max_value=img[slice_ix].shape[0] - 1,
                            value=y_center
                        )
                        
                        seed_x = st.sidebar.slider(
                            "X", 
                            min_value=0,
                            max_value=img[slice_ix].shape[1] - 1,
                            value=x_center
                        )
                        
                        seed_point = (seed_y, seed_x)
                        
                        # Segmentar el tumor usando crecimiento de regiones
                        tumor_mask = region_growing_segmentation(
                            img, slice_ix, seed_point, threshold_factor
                        )
                    
                    # Proyectar el tumor a una superficie esférica si hay una segmentación
                    if tumor_mask is not None and np.any(tumor_mask):
                        st.sidebar.markdown('<p class="sub-header">Proyección del tumor</p>', unsafe_allow_html=True)
                        
                        # Botón para realizar la proyección
                        if st.sidebar.button("Proyectar tumor"):
                            # Calcular centro de la plantilla y radio
                            template_center = (tumor_mask.shape[0] // 2, tumor_mask.shape[1] // 2)
                            
                            # Proyectar el tumor
                            projected_points = project_tumor_to_sphere(
                                tumor_mask, img, slice_ix, template_center
                            )
                            
                            # Calcular puntos terminales
                            num_needles = st.sidebar.slider(
                                "Número de agujas",
                                min_value=1,
                                max_value=15,
                                value=9
                            )
                            
                            terminal_points = calculate_terminal_points(
                                projected_points, num_needles
                            )
                            
                            st.sidebar.success(f"Se encontraron {len(projected_points)} puntos proyectados")
                            st.sidebar.success(f"Se calcularon {len(terminal_points)} puntos terminales")
                    
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

# Mostrar la visualización principal
if img is not None:
    if output == 'Imagen':
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
            # Muestra la imagen en la ventana principal con los ajustes aplicados
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
            
    elif output == 'Metadatos':
        st.markdown('<p class="sub-header">Metadatos DICOM</p>', unsafe_allow_html=True)
        try:
            metadata = dict()
            for k in reader.GetMetaDataKeys(slice_ix):
                metadata[k] = reader.GetMetaData(slice_ix, k)
            df = pd.DataFrame.from_dict(metadata, orient='index', columns=['Valor'])
            st.dataframe(df, height=600)
        except Exception as e:
            st.error(f"Error al leer metadatos: {str(e)}")
            
    elif output == 'Segmentación':
        st.markdown('<p class="sub-header">Segmentación del Tumor</p>', unsafe_allow_html=True)
        
        # Visualizar la imagen con la segmentación sobrepuesta
        if tumor_mask is not None:
            fig = plot_slice_with_segmentation(
                img, slice_ix, window_width, window_center, 
                segmentation_mask=tumor_mask, 
                projected_points=terminal_points if terminal_points is not None else None
            )
            st.pyplot(fig)
            
            # Mostrar estadísticas sobre la segmentación
            if np.any(tumor_mask):
                tumor_pixels = np.sum(tumor_mask)
                tumor_percentage = (tumor_pixels / tumor_mask.size) * 100
                
                stats_cols = st.columns(3)
                with stats_cols[0]:
                    st.markdown(f"**Píxeles del tumor:** {tumor_pixels}")
                with stats_cols[1]:
                    st.markdown(f"**Porcentaje del área:** {tumor_percentage:.2f}%")
                with stats_cols[2]:
                    st.markdown(f"**Método:** {segmentation_method}")
                
                # Si hay puntos proyectados, mostrar información sobre ellos
                if projected_points and len(projected_points) > 0:
                    st.markdown('<p class="sub-header">Proyección del Tumor</p>', unsafe_allow_html=True)
                    st.markdown(f"**Puntos proyectados:** {len(projected_points)}")
                    
                    if terminal_points is not None and len(terminal_points) > 0:
                        # Convertir los puntos terminales a posiciones en la imagen
                        st.markdown(f"**Puntos terminales calculados:** {len(terminal_points)}")
                        
                        # Mostrar coordenadas de los puntos terminales
                        terminal_df = pd.DataFrame({
                            'Punto': [f'Punto {i+1}' for i in range(len(terminal_points))],
                            'Theta': [f"{point[0]:.4f}" for point in terminal_points],
                            'Phi': [f"{point[1]:.4f}" for point in terminal_points]
                        })
                        st.dataframe(terminal_df)
                        
                        # Opción para exportar los resultados
                        if st.button("Exportar resultados de la segmentación"):
                            # Crear un CSV con los resultados
                            csv = terminal_df.to_csv(index=False)
                            st.download_button(
                                label="Descargar CSV de puntos",
                                data=csv,
                                file_name="puntos_terminales.csv",
                                mime="text/csv",
                            )
            else:
                st.warning("No se detectaron píxeles del tumor con los parámetros actuales. Ajusta los parámetros e intenta de nuevo.")
        else:
            st.info("Selecciona un método de segmentación y ajusta los parámetros en el panel lateral.")
            
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

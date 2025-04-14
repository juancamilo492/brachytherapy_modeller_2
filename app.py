import os
import io
import zipfile
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import SimpleITK as sitk
from scipy.ndimage import label, binary_dilation
import cv2
from sklearn.cluster import KMeans

# Configuración de página y estilo
st.set_page_config(layout="wide", page_title="Brachyanalysis - Sistema Experto")

# CSS personalizado
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
    .warning-box {
        background-color: rgba(255, 193, 7, 0.1);
        border-left: 3px solid #ffc107;
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
    .analysis-result {
        background-color: rgba(192, 215, 17, 0.1);
        padding: 15px;
        border-radius: 8px;
        margin-top: 15px;
    }
    .needle-path {
        stroke: #ff5733;
        stroke-width: 2px;
    }
    .tumor-boundary {
        stroke: #c0d711;
        stroke-width: 2px;
        fill: rgba(192, 215, 17, 0.2);
    }
    .risk-organ {
        stroke: #ff9800;
        stroke-width: 1px;
        fill: rgba(255, 152, 0, 0.2);
    }
    .target-volume {
        stroke: #28aec5;
        stroke-width: 1px;
        fill: rgba(40, 174, 197, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Configuración de la barra lateral
st.sidebar.markdown('<p class="sidebar-title">Brachyanalysis</p>', unsafe_allow_html=True)
st.sidebar.markdown('<p class="sub-header">Sistema Experto para Braquiterapia</p>', unsafe_allow_html=True)

# Sección de carga de archivos en la barra lateral
st.sidebar.markdown('<p class="sub-header">Configuración</p>', unsafe_allow_html=True)

# Solo opción de subir ZIP
uploaded_file = st.sidebar.file_uploader("Sube un archivo ZIP con tus archivos DICOM", type="zip")

# Constantes y umbrales
HU_TUMOR_MIN = 60  # Valor HU mínimo aproximado para tejido tumoral
HU_TUMOR_MAX = 120  # Valor HU máximo aproximado para tejido tumoral

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

def plot_slice(vol, slice_ix, overlay=None, needle_paths=None):
    """
    Visualiza un corte DICOM con superposiciones opcionales
    
    Args:
        vol: Volumen 3D de la imagen
        slice_ix: Índice del corte a visualizar
        overlay: Matriz de segmentación opcional (mismo tamaño que el corte)
        needle_paths: Lista de trayectorias de agujas a mostrar
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    selected_slice = vol[slice_ix, :, :]
    
    # Mostrar la imagen base en escala de grises
    ax.imshow(selected_slice, origin='lower', cmap='gray')
    
    # Si hay una superposición de segmentación, mostrarla con transparencia
    if overlay is not None:
        # Crear una máscara RGB para la superposición
        mask_rgb = np.zeros((*overlay.shape, 4))
        
        # Tumor en verde con transparencia
        tumor_mask = overlay == 1
        mask_rgb[tumor_mask, 0] = 0.2  # R
        mask_rgb[tumor_mask, 1] = 0.8  # G
        mask_rgb[tumor_mask, 2] = 0.2  # B
        mask_rgb[tumor_mask, 3] = 0.5  # Alpha
        
        # Órganos de riesgo en rojo con transparencia
        risk_mask = overlay == 2
        mask_rgb[risk_mask, 0] = 0.8  # R
        mask_rgb[risk_mask, 1] = 0.2  # G
        mask_rgb[risk_mask, 2] = 0.2  # B
        mask_rgb[risk_mask, 3] = 0.5  # Alpha
        
        ax.imshow(mask_rgb, origin='lower')
    
    # Si hay trayectorias de agujas, mostrarlas como líneas
    if needle_paths and slice_ix in needle_paths:
        for path in needle_paths[slice_ix]:
            ax.plot([path['start'][1], path['end'][1]], 
                    [path['start'][0], path['end'][0]], 
                    'r-', linewidth=2)
            # Punto de entrada marcado con un círculo
            ax.plot(path['start'][1], path['start'][0], 'ro', markersize=5)
            # Punto objetivo marcado con una X
            ax.plot(path['end'][1], path['end'][0], 'rx', markersize=5)
    
    plt.axis('off')
    return fig

def segment_tumor(img_slice, threshold_min=HU_TUMOR_MIN, threshold_max=HU_TUMOR_MAX):
    """
    Segmenta el tumor en un corte DICOM basado en umbrales de intensidad y ajuste por clustering
    
    Args:
        img_slice: Corte 2D de la imagen
        threshold_min: Valor mínimo HU para tejido tumoral
        threshold_max: Valor máximo HU para tejido tumoral
    
    Returns:
        Máscara binaria donde el tumor es 1, el resto 0
    """
    # Aplicar umbral inicial basado en valores HU
    mask = ((img_slice >= threshold_min) & (img_slice <= threshold_max)).astype(np.uint8)
    
    # Aplicar operaciones morfológicas para limpiar la máscara
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Etiquetar componentes conectados
    labeled_array, num_features = label(mask)
    
    # Si hay múltiples regiones, quedarse con las más grandes (probablemente el tumor)
    if num_features > 0:
        # Contar pixeles en cada región
        sizes = np.bincount(labeled_array.ravel())
        
        # Ignorar el fondo (etiqueta 0)
        if len(sizes) > 1:
            sizes[0] = 0
            max_label = np.argmax(sizes)
            
            # Crear máscara solo con la región más grande
            mask = (labeled_array == max_label).astype(np.uint8)
    
    return mask

def detect_organs_at_risk(img_slice):
    """
    Detecta órganos de riesgo como vejiga y recto en un corte DICOM
    
    Args:
        img_slice: Corte 2D de la imagen
    
    Returns:
        Máscara donde los órganos de riesgo son 1, el resto 0
    """
    # Normalizar la imagen para aplicar K-means
    normalized = cv2.normalize(img_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Aplicar filtro medio para suavizar la imagen
    smoothed = cv2.medianBlur(normalized, 5)
    
    # Redimensionar para acelerar el proceso
    small = cv2.resize(smoothed, (smoothed.shape[1]//2, smoothed.shape[0]//2))
    
    # Preparar datos para K-means
    pixels = small.reshape(-1, 1).astype(np.float32)
    
    # Realizar K-means con 3 clústeres (fondo, tejido normal, órganos de riesgo)
    k = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Identificar el clúster más oscuro (probablemente contiene órganos de riesgo)
    darkest_cluster = np.argmin(centers)
    
    # Crear máscara para el clúster identificado
    mask = (labels.reshape(small.shape) == darkest_cluster).astype(np.uint8)
    
    # Redimensionar de vuelta al tamaño original
    mask = cv2.resize(mask, (img_slice.shape[1], img_slice.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Operaciones morfológicas para mejorar la máscara
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def plan_needle_trajectories(tumor_mask, risk_mask, slice_shape):
    """
    Planifica trayectorias de agujas para braquiterapia evitando órganos de riesgo
    
    Args:
        tumor_mask: Máscara binaria del tumor
        risk_mask: Máscara binaria de órganos de riesgo
        slice_shape: Dimensiones del corte (altura, anchura)
    
    Returns:
        Lista de trayectorias de agujas (punto inicial y final)
    """
    # Si no hay tumor detectado, retornar lista vacía
    if not np.any(tumor_mask):
        return []
    
    # Encontrar el centro del tumor
    tumor_indices = np.where(tumor_mask > 0)
    if len(tumor_indices[0]) == 0:
        return []
    
    tumor_center_y = int(np.mean(tumor_indices[0]))
    tumor_center_x = int(np.mean(tumor_indices[1]))
    
    # Definir posibles puntos de entrada (parte inferior de la imagen)
    entry_y = slice_shape[0] - 20  # 20 píxeles desde el borde inferior
    potential_entries = []
    
    # Crear posibles puntos de entrada espaciados
    for x in range(20, slice_shape[1]-20, 40):
        potential_entries.append((entry_y, x))
    
    # Planificar trayectorias de aguja
    trajectories = []
    
    for entry_point in potential_entries:
        # Vector de dirección hacia el centro del tumor
        dir_y = tumor_center_y - entry_point[0]
        dir_x = tumor_center_x - entry_point[1]
        
        # Normalizar el vector de dirección
        magnitude = np.sqrt(dir_y**2 + dir_x**2)
        if magnitude == 0:
            continue
        
        dir_y /= magnitude
        dir_x /= magnitude
        
        # Comprobar si la trayectoria intersecta con órganos de riesgo
        safe_path = True
        
        # Simulamos la trayectoria punto por punto
        test_y, test_x = entry_point
        while 0 <= int(test_y) < slice_shape[0] and 0 <= int(test_x) < slice_shape[1]:
            if risk_mask[int(test_y), int(test_x)] > 0:
                safe_path = False
                break
            
            # Si alcanzamos el tumor, hemos encontrado un objetivo
            if tumor_mask[int(test_y), int(test_x)] > 0:
                break
            
            # Avanzar a lo largo de la trayectoria
            test_y += dir_y * 5  # Avanzamos de 5 en 5 píxeles para eficiencia
            test_x += dir_x * 5
        
        # Si la trayectoria es segura y termina en el tumor, la agregamos
        if safe_path and tumor_mask[int(test_y), int(test_x)] > 0:
            trajectories.append({
                'start': (entry_point[0], entry_point[1]),
                'end': (int(test_y), int(test_x))
            })
    
    # Limitar a un máximo de 5 agujas
    if len(trajectories) > 5:
        trajectories = trajectories[:5]
    
    return trajectories

def calculate_dose_coverage(tumor_mask, needle_paths, slice_shape):
    """
    Calcula una estimación de la cobertura de dosis para el tumor
    
    Args:
        tumor_mask: Máscara binaria del tumor
        needle_paths: Lista de trayectorias de agujas
        slice_shape: Dimensiones del corte
    
    Returns:
        Porcentaje de cobertura estimado del tumor
    """
    # Si no hay agujas o tumor, la cobertura es 0%
    if not needle_paths or not np.any(tumor_mask):
        return 0.0
    
    # Crear una matriz de dosis inicializada en ceros
    dose_map = np.zeros(slice_shape, dtype=np.float32)
    
    # Simulación simple de distribución de dosis por aguja
    # Para cada aguja, creamos un gradiente de dosis que disminuye con la distancia
    for path in needle_paths:
        target_y, target_x = path['end']
        
        # Para cada píxel en la imagen
        y_indices, x_indices = np.ogrid[:slice_shape[0], :slice_shape[1]]
        
        # Calcular distancia cuadrática (más eficiente que raíz cuadrada)
        distance_squared = (y_indices - target_y)**2 + (x_indices - target_x)**2
        
        # La dosis disminuye con el cuadrado de la distancia (ley inversa del cuadrado)
        # Radio efectivo de 30 píxeles para la dosis
        effective_radius = 30
        dose_contribution = np.maximum(0, 1 - distance_squared / (effective_radius**2))
        
        # Acumular dosis
        dose_map += dose_contribution
    
    # Normalizar el mapa de dosis a un máximo de 1.0
    if np.max(dose_map) > 0:
        dose_map /= np.max(dose_map)
    
    # Calcular cobertura del tumor (qué porcentaje recibe al menos el 90% de la dosis prescrita)
    tumor_pixels = np.sum(tumor_mask)
    if tumor_pixels > 0:
        covered_pixels = np.sum((tumor_mask > 0) & (dose_map >= 0.9))
        coverage_percent = (covered_pixels / tumor_pixels) * 100
    else:
        coverage_percent = 0.0
    
    return coverage_percent

def analyze_clinical_significance(coverage, num_needles):
    """
    Analiza la significancia clínica del plan de tratamiento
    
    Args:
        coverage: Porcentaje de cobertura del tumor
        num_needles: Número de agujas en el plan
    
    Returns:
        Diccionario con evaluación clínica
    """
    result = {
        "calidad_cobertura": "",
        "complejidad_plan": "",
        "recomendacion": ""
    }
    
    # Evaluar calidad de cobertura
    if coverage >= 95:
        result["calidad_cobertura"] = "Excelente"
    elif coverage >= 90:
        result["calidad_cobertura"] = "Buena"
    elif coverage >= 80:
        result["calidad_cobertura"] = "Aceptable"
    else:
        result["calidad_cobertura"] = "Insuficiente"
    
    # Evaluar complejidad del plan
    if num_needles <= 2:
        result["complejidad_plan"] = "Baja"
    elif num_needles <= 4:
        result["complejidad_plan"] = "Media"
    else:
        result["complejidad_plan"] = "Alta"
    
    # Generar recomendación
    if coverage < 80:
        result["recomendacion"] = "Se recomienda revisar el plan para mejorar la cobertura del tumor."
    elif coverage < 90 and num_needles < 4:
        result["recomendacion"] = "Considere agregar agujas adicionales para mejorar la cobertura."
    elif coverage >= 95 and num_needles > 4:
        result["recomendacion"] = "Posible simplificación del plan reduciendo el número de agujas."
    else:
        result["recomendacion"] = "Plan balanceado en términos de cobertura y complejidad."
    
    return result

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
segmentation_data = {}
needle_trajectories = {}

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
            
            # Opciones de visualización y análisis
            output = st.sidebar.radio('Modo de visualización', 
                                  ['Imagen básica', 'Segmentación y análisis', 'Planificación de agujas', 'Metadatos'], 
                                  index=0)
            
            # Si se selecciona segmentación o planificación, realizar análisis del corte actual
            if output in ['Segmentación y análisis', 'Planificación de agujas'] and slice_ix not in segmentation_data:
                with st.spinner('Analizando imágenes...'):
                    # Segmentar tumor
                    tumor_mask = segment_tumor(img[slice_ix])
                    
                    # Detectar órganos de riesgo
                    risk_mask = detect_organs_at_risk(img[slice_ix])
                    
                    # Crear máscara combinada para visualización
                    combined_mask = np.zeros_like(tumor_mask)
                    combined_mask[tumor_mask > 0] = 1  # Tumor
                    combined_mask[risk_mask > 0] = 2   # Órganos de riesgo
                    
                    # Almacenar segmentación
                    segmentation_data[slice_ix] = {
                        'tumor_mask': tumor_mask,
                        'risk_mask': risk_mask,
                        'combined_mask': combined_mask
                    }
            
            # Si se selecciona planificación de agujas, planificar trayectorias
            if output == 'Planificación de agujas' and slice_ix not in needle_trajectories:
                with st.spinner('Planificando trayectorias de agujas...'):
                    if slice_ix in segmentation_data:
                        tumor_mask = segmentation_data[slice_ix]['tumor_mask']
                        risk_mask = segmentation_data[slice_ix]['risk_mask']
                        
                        # Planificar trayectorias de agujas
                        paths = plan_needle_trajectories(tumor_mask, risk_mask, img[slice_ix].shape)
                        needle_trajectories[slice_ix] = paths
                    else:
                        needle_trajectories[slice_ix] = []
                        
        except Exception as e:
            st.sidebar.error(f"Error al procesar los archivos DICOM: {str(e)}")
            st.sidebar.write("Detalles del error:", str(e))

# Visualización en la ventana principal
# Título grande siempre visible
st.markdown('<p class="giant-title">Brachyanalysis</p>', unsafe_allow_html=True)

if img is not None:
    if output == 'Imagen básica':
        st.markdown('<p class="sub-header">Visualización DICOM</p>', unsafe_allow_html=True)
        
        # Muestra la imagen en la ventana principal
        fig = plot_slice(img, slice_ix)
        st.pyplot(fig)
        
        # Información adicional sobre la imagen
        info_cols = st.columns(4)
        with info_cols[0]:
            st.markdown(f"**Dimensiones:** {img.shape[1]} x {img.shape[2]} px")
        with info_cols[1]:
            st.markdown(f"**Total cortes:** {n_slices}")
        with info_cols[2]:
            st.markdown(f"**Corte actual:** {slice_ix + 1}")
        with info_cols[3]:
            st.markdown(f"**Min/Max:** {img[slice_ix].min():.1f} / {img[slice_ix].max():.1f}")
    
    elif output == 'Segmentación y análisis':
        st.markdown('<p class="sub-header">Segmentación y análisis del tumor</p>', unsafe_allow_html=True)
        
        if slice_ix in segmentation_data:
            combined_mask = segmentation_data[slice_ix]['combined_mask']
            
            # Mostrar imagen con segmentación superpuesta
            fig = plot_slice(img, slice_ix, combined_mask)
            st.pyplot(fig)
            
            # Información de la segmentación
            info_cols = st.columns(3)
            
            tumor_pixels = np.sum(segmentation_data[slice_ix]['tumor_mask'])
            risk_pixels = np.sum(segmentation_data[slice_ix]['risk_mask'])
            
            # Estimación del volumen basada en el área (simplificación)
            pixel_area_mm2 = 1.0  # Esto debería obtenerse de los metadatos DICOM
            tumor_area_mm2 = tumor_pixels * pixel_area_mm2
            
            with info_cols[0]:
                st.markdown(f"**Área del tumor:** {tumor_area_mm2:.2f} mm²")
            with info_cols[1]:
                st.markdown(f"**Píxeles del tumor:** {tumor_pixels}")
            with info_cols[2]:
                st.markdown(f"**Órganos de riesgo detectados:** {'Sí' if risk_pixels > 0 else 'No'}")
            
            # Análisis de la forma del tumor
            if tumor_pixels > 0:
                st.markdown("### Análisis morfológico del tumor")
                
                # Cálculo de características básicas
                tumor_indices = np.where(segmentation_data[slice_ix]['tumor_mask'] > 0)
                tumor_center_y = int(np.mean(tumor_indices[0]))
                tumor_center_x = int(np.mean(tumor_indices[1]))
                
                # Distancia máxima desde el centro (estimación de radio)
                max_dist = 0
                for y, x in zip(tumor_indices[0], tumor_indices[1]):
                    dist = np.sqrt((y - tumor_center_y)**2 + (x - tumor_center_x)**2)
                    max_dist = max(max_dist, dist)
                
# Relación área/perímetro (forma)
                tumor_mask = segmentation_data[slice_ix]['tumor_mask']
                contours, _ = cv2.findContours(tumor_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    perimeter = cv2.arcLength(contours[0], True)
                    area = cv2.contourArea(contours[0])
                    circularity = 0
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    shape_cols = st.columns(4)
                    with shape_cols[0]:
                        st.markdown(f"**Centro del tumor:** ({tumor_center_x}, {tumor_center_y})")
                    with shape_cols[1]:
                        st.markdown(f"**Radio aproximado:** {max_dist:.2f} px")
                    with shape_cols[2]:
                        st.markdown(f"**Circularidad:** {circularity:.2f}")
                    with shape_cols[3]:
                        if circularity > 0.8:
                            shape_type = "Redondeado"
                        elif circularity > 0.6:
                            shape_type = "Ovalado"
                        else:
                            shape_type = "Irregular"
                        st.markdown(f"**Morfología:** {shape_type}")
                
                # Identificación de posibles áreas de infiltración
                st.markdown("### Análisis de homogeneidad e infiltración")
                
                # Muestrear valores de intensidad dentro del tumor
                tumor_values = img[slice_ix][tumor_mask > 0]
                if len(tumor_values) > 0:
                    tumor_mean = np.mean(tumor_values)
                    tumor_std = np.std(tumor_values)
                    
                    # Grado de homogeneidad (CV = desviación estándar / media)
                    homogeneity = tumor_std / tumor_mean if tumor_mean > 0 else 0
                    
                    homog_cols = st.columns(3)
                    with homog_cols[0]:
                        st.markdown(f"**Intensidad media:** {tumor_mean:.2f} HU")
                    with homog_cols[1]:
                        st.markdown(f"**Desviación estándar:** {tumor_std:.2f} HU")
                    with homog_cols[2]:
                        if homogeneity < 0.1:
                            homog_type = "Alta homogeneidad"
                        elif homogeneity < 0.2:
                            homog_type = "Homogeneidad media"
                        else:
                            homog_type = "Baja homogeneidad (posible infiltración)"
                        st.markdown(f"**Heterogeneidad:** {homog_type}")
        else:
            st.warning("No se ha realizado segmentación para este corte.")
    
    elif output == 'Planificación de agujas':
        st.markdown('<p class="sub-header">Planificación de agujas para braquiterapia</p>', unsafe_allow_html=True)
        
        if slice_ix in segmentation_data and slice_ix in needle_trajectories:
            combined_mask = segmentation_data[slice_ix]['combined_mask']
            paths = needle_trajectories[slice_ix]
            
            # Crear diccionario para visualizar las trayectorias
            paths_dict = {slice_ix: paths} if paths else {}
            
            # Mostrar imagen con segmentación y trayectorias
            fig = plot_slice(img, slice_ix, combined_mask, paths_dict)
            st.pyplot(fig)
            
            # Información sobre las trayectorias planificadas
            st.markdown(f"### Plan de agujas para el corte {slice_ix + 1}")
            st.markdown(f"**Número de agujas planificadas:** {len(paths)}")
            
            if paths:
                # Calcular la cobertura estimada de dosis
                tumor_mask = segmentation_data[slice_ix]['tumor_mask']
                coverage = calculate_dose_coverage(tumor_mask, paths, img[slice_ix].shape)
                
                # Análisis de significancia clínica
                clinical_analysis = analyze_clinical_significance(coverage, len(paths))
                
                st.markdown(f"**Cobertura estimada del tumor:** {coverage:.1f}%")
                
                # Mostrar tabla con coordenadas de las agujas
                st.markdown("#### Coordenadas de las agujas planificadas")
                
                needle_data = []
                for i, path in enumerate(paths):
                    needle_data.append({
                        "Aguja": i+1,
                        "Punto de entrada (x, y)": f"({path['start'][1]}, {path['start'][0]})",
                        "Punto objetivo (x, y)": f"({path['end'][1]}, {path['end'][0]})"
                    })
                
                st.table(pd.DataFrame(needle_data))
                
                # Mostrar análisis clínico
                st.markdown('<div class="analysis-result">', unsafe_allow_html=True)
                st.markdown("### Análisis clínico del plan")
                st.markdown(f"**Calidad de cobertura:** {clinical_analysis['calidad_cobertura']}")
                st.markdown(f"**Complejidad del plan:** {clinical_analysis['complejidad_plan']}")
                st.markdown(f"**Recomendación:** {clinical_analysis['recomendacion']}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Botón para exportar el plan
                if st.button("Exportar plan de agujas (CSV)"):
                    # Crear DataFrame con las coordenadas
                    export_data = pd.DataFrame({
                        "Aguja": [i+1 for i in range(len(paths))],
                        "Entrada_X": [path['start'][1] for path in paths],
                        "Entrada_Y": [path['start'][0] for path in paths],
                        "Objetivo_X": [path['end'][1] for path in paths],
                        "Objetivo_Y": [path['end'][0] for path in paths]
                    })
                    
                    # Convertir a CSV
                    csv = export_data.to_csv(index=False)
                    
                    # Crear botón de descarga
                    st.download_button(
                        label="Descargar CSV",
                        data=csv,
                        file_name=f"plan_agujas_corte_{slice_ix+1}.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("No se han podido calcular trayectorias seguras para este corte. Posibles causas:")
                st.markdown("""
                - El tumor no es visible en este corte
                - Los órganos de riesgo bloquean todas las posibles trayectorias
                - El tumor está fuera del alcance de las agujas del template
                """)
                
                st.markdown('<div class="warning-box">Se recomienda seleccionar otro corte para la planificación.</div>', unsafe_allow_html=True)
        else:
            st.warning("No se ha realizado segmentación o planificación para este corte.")
    
    elif output == 'Metadatos':
        st.markdown('<p class="sub-header">Metadatos DICOM</p>', unsafe_allow_html=True)
        try:
            metadata = dict()
            for k in reader.GetMetaDataKeys(slice_ix):
                metadata[k] = reader.GetMetaData(slice_ix, k)
            
            # Filtrar y organizar los metadatos más relevantes
            important_tags = [
                "0008|0060", # Modalidad
                "0018|0050", # Espesor de corte
                "0018|0088", # Espaciado entre cortes
                "0018|1100", # Tiempo de reconstrucción
                "0020|0032", # Posición del paciente
                "0020|0037", # Orientación del paciente
                "0028|0002", # Muestras por pixel
                "0028|0010", # Filas
                "0028|0011", # Columnas
                "0028|0030", # Espaciado de pixel
                "0028|0100", # Bits asignados
                "0028|0101", # Bits almacenados
                "0028|0102", # Bit más significativo
                "0028|0103", # Representación de pixel
                "0028|1050", # Ventana centro
                "0028|1051", # Ventana ancho
                "0028|1052", # Rescale interceptación
                "0028|1053"  # Rescale pendiente
            ]
            
            # Separar metadatos importantes
            important_metadata = {k: metadata[k] for k in important_tags if k in metadata}
            other_metadata = {k: v for k, v in metadata.items() if k not in important_tags}
            
            # Mostrar metadatos importantes en formato más accesible
            st.markdown("### Metadatos relevantes para braquiterapia")
            
            meta_cols = st.columns(2)
            with meta_cols[0]:
                for tag, value in list(important_metadata.items())[:len(important_metadata)//2]:
                    tag_name = tag.replace("|", ":")
                    st.markdown(f"**{tag_name}:** {value}")
            
            with meta_cols[1]:
                for tag, value in list(important_metadata.items())[len(important_metadata)//2:]:
                    tag_name = tag.replace("|", ":")
                    st.markdown(f"**{tag_name}:** {value}")
            
            # Mostrar todos los metadatos en un DataFrame
            st.markdown("### Todos los metadatos DICOM")
            df = pd.DataFrame.from_dict(metadata, orient='index', columns=['Valor'])
            st.dataframe(df, height=400)
            
            # Añadir botón para descargar los metadatos
            if st.button("Exportar metadatos (CSV)"):
                csv = df.to_csv()
                st.download_button(
                    label="Descargar CSV",
                    data=csv,
                    file_name=f"metadatos_dicom_corte_{slice_ix+1}.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error al leer metadatos: {str(e)}")
else:
    # Página de inicio cuando no hay imágenes cargadas
    st.markdown('<p class="sub-header">Sistema Experto para Braquiterapia</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; padding: 40px; margin-top: 10px;">
        <img src="https://raw.githubusercontent.com/SimpleITK/SimpleITK/master/Documentation/docs/images/simpleitk-logo.svg" alt="SimpleITK Logo" width="200">
        <h2 style="color: #28aec5; margin-top: 20px;">Carga un archivo ZIP con tus imágenes DICOM</h2>
        <p style="font-size: 18px; margin-top: 10px;">Utiliza el panel lateral para subir tus archivos y visualizarlos</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Información sobre el sistema
    st.markdown('<p class="sub-header">Acerca de este sistema</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <p><strong>Brachyanalysis</strong> es un sistema experto para análisis de imágenes DICOM en braquiterapia de cáncer de cérvix que permite:</p>
        <ul>
            <li>Visualizar imágenes médicas en formato DICOM</li>
            <li>Segmentar automáticamente el tumor y órganos de riesgo</li>
            <li>Planificar trayectorias de agujas para braquiterapia</li>
            <li>Estimar la cobertura de dosis en el tumor</li>
            <li>Analizar la significancia clínica del plan de tratamiento</li>
        </ul>
        <p>Para comenzar, sube un archivo ZIP con imágenes DICOM de un estudio de braquiterapia.</p>
    </div>
    """, unsafe_allow_html=True)

# Agregar sección para análisis 3D avanzado (disponible en todos los modos)
if img is not None and n_slices > 1:
    st.markdown('<p class="sub-header">Análisis 3D</p>', unsafe_allow_html=True)
    
    # Opción de activar/desactivar el análisis 3D
    if st.button("Realizar análisis 3D del volumen"):
        with st.spinner("Procesando volumen 3D..."):
            # Seleccionar un rango apropiado de cortes para analizar
            start_slice = max(0, slice_ix - 5)
            end_slice = min(n_slices, slice_ix + 5)
            
            # Variables para almacenar resultados
            tumor_volumes = []
            tumor_areas = []
            tumor_centers = []
            
            # Procesar cada corte
            for s in range(start_slice, end_slice):
                # Segmentar el tumor en este corte
                if s not in segmentation_data:
                    tumor_mask = segment_tumor(img[s])
                    segmentation_data[s] = {
                        'tumor_mask': tumor_mask,
                        'risk_mask': detect_organs_at_risk(img[s]),
                        'combined_mask': None
                    }
                else:
                    tumor_mask = segmentation_data[s]['tumor_mask']
                
                # Calcular área y volumen estimado
                area_pixels = np.sum(tumor_mask)
                if area_pixels > 0:
                    # Estimar área en mm² (usando un factor de escala ficticio para este ejemplo)
                    # En un sistema real, esto se obtendría de los metadatos DICOM
                    pixel_area_mm2 = 1.0  # mm² por pixel
                    slice_thickness_mm = 3.0  # mm por corte
                    
                    area_mm2 = area_pixels * pixel_area_mm2
                    volume_mm3 = area_mm2 * slice_thickness_mm
                    
                    # Calcular centro del tumor
                    tumor_indices = np.where(tumor_mask > 0)
                    if len(tumor_indices[0]) > 0:
                        center_y = int(np.mean(tumor_indices[0]))
                        center_x = int(np.mean(tumor_indices[1]))
                        
                        tumor_areas.append(area_mm2)
                        tumor_volumes.append(volume_mm3)
                        tumor_centers.append((s, center_y, center_x))
            
            # Visualizar resultados
            if tumor_volumes:
                # Volumen total estimado
                total_volume = sum(tumor_volumes)
                
                # Mostrar resultados
                st.markdown(f"### Resultados del análisis 3D")
                st.markdown(f"**Volumen estimado del tumor:** {total_volume:.2f} mm³")
                st.markdown(f"**Cortes con tumor detectado:** {len(tumor_volumes)} de {end_slice - start_slice}")
                
                # Graficar el volumen del tumor por corte
                st.markdown("#### Distribución del volumen del tumor por corte")
                
                # Preparar datos para el gráfico
                volume_df = pd.DataFrame({
                    'Corte': list(range(start_slice, start_slice + len(tumor_volumes))),
                    'Volumen (mm³)': tumor_volumes
                })
                
                # Crear gráfico
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(volume_df['Corte'], volume_df['Volumen (mm³)'], color='#28aec5')
                ax.set_xlabel('Número de corte')
                ax.set_ylabel('Volumen del tumor (mm³)')
                ax.set_title('Distribución del volumen del tumor por corte')
                ax.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig)
                
                # Recomendar cortes para planificación
                best_slices = volume_df.sort_values('Volumen (mm³)', ascending=False).head(3)['Corte'].tolist()
                
                st.markdown("#### Recomendación para planificación")
                st.markdown(f"""
                Basado en la distribución del volumen del tumor, se recomienda realizar la planificación de agujas
                en los siguientes cortes: **{', '.join([str(s) for s in best_slices])}**
                """)
                
                # Añadir un botón para generar informe completo
                if st.button("Generar informe completo"):
                    # En un sistema real, esto generaría un informe PDF
                    st.markdown('<div class="success-box">Informe generado con éxito</div>', unsafe_allow_html=True)
                    
                    # Preparar datos para el informe
                    informe_data = {
                        "ID Paciente": "ANON12345",  # En un sistema real, esto se obtendría de los metadatos DICOM
                        "Fecha estudio": "13/04/2025",  # Se obtendría de los metadatos
                        "Volumen tumor": f"{total_volume:.2f} mm³",
                        "Cortes recomendados": ", ".join([str(s) for s in best_slices]),
                        "Número de agujas": sum([len(needle_trajectories.get(s, [])) for s in best_slices]),
                        "Cobertura estimada": f"{calculate_dose_coverage(segmentation_data.get(best_slices[0], {}).get('tumor_mask', np.array([])), needle_trajectories.get(best_slices[0], []), img[best_slices[0] if best_slices else 0].shape):.1f}%"
                    }
                    
                    # Mostrar datos del informe
                    st.table(pd.DataFrame([informe_data]))
            else:
                st.warning("No se detectó tejido tumoral en los cortes analizados.")

# Pie de página
st.markdown("""
<hr style="margin-top: 30px;">
<div style="text-align: center; color: #28aec5; font-size: 14px;">
    Brachyanalysis - Sistema Experto para Braquiterapia de Cáncer de Cérvix
</div>
""", unsafe_allow_html=True)

# Limpiar el directorio temporal si se creó uno
if temp_dir and os.path.exists(temp_dir):
    # Nota: En una aplicación real, deberías limpiar los directorios temporales
    # cuando la aplicación se cierre, pero en Streamlit esto es complicado
    # ya que las sesiones persisten.
    pass

# Funciones adicionales para un sistema más completo

def register_template_coordinates(img_slice, template_pos):
    """
    Registra las coordenadas del template en relación con la imagen
    
    Args:
        img_slice: Corte 2D de la imagen
        template_pos: Posición del template (x, y)
    
    Returns:
        Matriz de transformación para convertir coordenadas de imagen a template
    """
    # Implementación básica de registro de coordenadas
    # En un sistema real, esto sería más complejo y preciso
    
    # Crear matriz de transformación simple (solo traslación)
    transform = np.array([
        [1, 0, -template_pos[0]],
        [0, 1, -template_pos[1]],
        [0, 0, 1]
    ])
    
    return transform

def convert_to_template_coordinates(img_coords, transform):
    """
    Convierte coordenadas de la imagen a coordenadas del template
    
    Args:
        img_coords: Coordenadas en la imagen (y, x)
        transform: Matriz de transformación
    
    Returns:
        Coordenadas en el sistema del template
    """
    # Convertir coordenadas de imagen a coordenadas homogéneas
    homog_coords = np.array([img_coords[1], img_coords[0], 1])
    
    # Aplicar transformación
    template_coords = np.dot(transform, homog_coords)
    
    # Convertir de nuevo a coordenadas cartesianas
    return (template_coords[1], template_coords[0])

def optimize_needle_distribution(tumor_mask, risk_mask, num_needles=5):
    """
    Optimiza la distribución de agujas para maximizar la cobertura del tumor
    
    Args:
        tumor_mask: Máscara binaria del tumor
        risk_mask: Máscara binaria de órganos de riesgo
        num_needles: Número máximo de agujas
    
    Returns:
        Lista de coordenadas de agujas optimizadas
    """
    # En un sistema real, esto utilizaría algoritmos de optimización avanzados
    # como algoritmos genéticos o optimización por enjambre de partículas
    
    # Implementación básica: dividir el tumor en regiones y colocar una aguja en cada región
    
    # Etiquetar componentes conectados del tumor
    tumor_labels, num_regions = label(tumor_mask)
    
    # Si hay más regiones que agujas, usar solo las regiones más grandes
    if num_regions > num_needles:
        # Contar píxeles por región
        region_sizes = np.bincount(tumor_labels.ravel())[1:]  # Ignorar el fondo (0)
        
        # Obtener las N regiones más grandes
        largest_regions = np.argsort(region_sizes)[-num_needles:] + 1  # +1 porque las etiquetas empiezan en 1
    else:
        largest_regions = range(1, num_regions + 1)
    
    # Colocar una aguja en el centro de cada región
    needle_coords = []
    for region_id in largest_regions:
        region_mask = (tumor_labels == region_id)
        
        # Encontrar el centro de la región
        region_indices = np.where(region_mask)
        if len(region_indices[0]) > 0:
            center_y = int(np.mean(region_indices[0]))
            center_x = int(np.mean(region_indices[1]))
            
            needle_coords.append((center_y, center_x))
    
    return needle_coords

def simulate_dose_distribution(img_shape, needle_coords, dose_prescription):
    """
    Simula la distribución de dosis para un conjunto de agujas
    
    Args:
        img_shape: Dimensiones de la imagen
        needle_coords: Lista de coordenadas de agujas
        dose_prescription: Dosis prescrita en Gy
    
    Returns:
        Matriz de dosis simulada
    """
    # Crear matriz de dosis vacía
    dose_matrix = np.zeros(img_shape, dtype=np.float32)
    
    # Para cada aguja, simular la distribución de dosis
    for y, x in needle_coords:
        # Crear una matriz de distancia desde esta aguja
        y_indices, x_indices = np.ogrid[:img_shape[0], :img_shape[1]]
        distances = np.sqrt((y_indices - y)**2 + (x_indices - x)**2)
        
        # Modelo simplificado de dosis: inversamente proporcional al cuadrado de la distancia
        # D(r) = D0 * (r0/r)^2 * e^(-μ*r)
        r0 = 10.0  # Distancia de referencia en píxeles
        mu = 0.1   # Coeficiente de atenuación
        
        # Calcular contribución de dosis de esta aguja
        dose_contribution = dose_prescription * (r0 / np.maximum(distances, r0))**2 * np.exp(-mu * distances)
        
        # Añadir a la matriz de dosis total
        dose_matrix += dose_contribution
    
    return dose_matrix

def evaluate_plan_quality(dose_matrix, tumor_mask, risk_mask, prescribed_dose):
    """
    Evalúa la calidad del plan de braquiterapia
    
    Args:
        dose_matrix: Matriz de dosis simulada
        tumor_mask: Máscara binaria del tumor
        risk_mask: Máscara binaria de órganos de riesgo
        prescribed_dose: Dosis prescrita en Gy
    
    Returns:
        Diccionario con métricas de calidad del plan
    """
    # Inicializar resultados
    results = {}
    
    # Cobertura del tumor (V100)
    tumor_pixels = np.sum(tumor_mask)
    if tumor_pixels > 0:
        # Calcular qué porcentaje del tumor recibe al menos la dosis prescrita
        v100 = np.sum((tumor_mask > 0) & (dose_matrix >= prescribed_dose)) / tumor_pixels * 100
        results["V100"] = v100
        
        # Calcular qué porcentaje del tumor recibe al menos el 90% de la dosis prescrita
        v90 = np.sum((tumor_mask > 0) & (dose_matrix >= 0.9 * prescribed_dose)) / tumor_pixels * 100
        results["V90"] = v90
    else:
        results["V100"] = 0
        results["V90"] = 0
    
    # Dosis a órganos de riesgo
    risk_pixels = np.sum(risk_mask)
    if risk_pixels > 0:
        # Dosis máxima a órganos de riesgo
        max_oar_dose = np.max(dose_matrix[risk_mask > 0]) if np.any(risk_mask > 0) else 0
        results["Max_OAR_Dose"] = max_oar_dose
        
        # Porcentaje de órganos de riesgo que reciben más del 70% de la dosis prescrita
        oar_v70 = np.sum((risk_mask > 0) & (dose_matrix >= 0.7 * prescribed_dose)) / risk_pixels * 100
        results["OAR_V70"] = oar_v70
    else:
        results["Max_OAR_Dose"] = 0
        results["OAR_V70"] = 0
    
    # Índice de conformidad (CI)
    # CI = (Volumen que recibe dosis prescrita) / (Volumen del tumor)
    volume_prescribed = np.sum(dose_matrix >= prescribed_dose)
    if tumor_pixels > 0 and volume_prescribed > 0:
        ci = (np.sum((tumor_mask > 0) & (dose_matrix >= prescribed_dose)) / tumor_pixels) / (volume_prescribed / np.sum(tumor_mask > 0))
        results["CI"] = ci
    else:
        results["CI"] = 0
    
    # Índice de homogeneidad (HI)
    # HI = (D2% - D98%) / D50%
    if tumor_pixels > 0:
        dose_in_tumor = dose_matrix[tumor_mask > 0]
        if len(dose_in_tumor) > 0:
            d2 = np.percentile(dose_in_tumor, 98)
            d98 = np.percentile(dose_in_tumor, 2)
            d50 = np.percentile(dose_in_tumor, 50)
            
            if d50 > 0:
                hi = (d2 - d98) / d50
                results["HI"] = hi
            else:
                results["HI"] = 0
        else:
            results["HI"] = 0
    else:
        results["HI"] = 0
    
    return results



import numpy as np
import cv2
import matplotlib.pyplot as plt
import SimpleITK as sitk
from skimage import filters, measure, segmentation, morphology
import streamlit as st

def add_organ_detection_ui(sidebar):
    """Añade controles de UI para detección de órganos en la barra lateral"""
    sidebar.markdown('<div class="control-section">', unsafe_allow_html=True)
    sidebar.markdown('<p class="sub-header">Detección de órganos</p>', unsafe_allow_html=True)
    
    detection_enabled = sidebar.checkbox("Activar detección de órganos", value=False)
    
    detection_options = sidebar.expander("Opciones de detección", expanded=False)
    with detection_options:
        tumor_threshold = st.slider(
            "Umbral para tumores", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.65, 
            step=0.05,
            help="Valores más altos hacen la detección más restrictiva"
        )
        
        organ_sensitivity = st.slider(
            "Sensibilidad para órganos", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.5, 
            step=0.1,
            help="Sensibilidad del algoritmo para detectar órganos"
        )
        
        min_area = st.slider(
            "Área mínima", 
            min_value=100, 
            max_value=5000, 
            value=500, 
            step=100,
            help="Tamaño mínimo de estructuras a detectar"
        )
        
        show_labels = st.checkbox("Mostrar etiquetas", value=True)
        
    sidebar.markdown('</div>', unsafe_allow_html=True)
    
    return {
        'enabled': detection_enabled,
        'tumor_threshold': tumor_threshold,
        'organ_sensitivity': organ_sensitivity,
        'min_area': min_area,
        'show_labels': show_labels
    }

def simple_organ_segmentation(img_slice, window_width, window_center, config):
    """
    Segmentación simple para detectar potenciales órganos y tumores
    
    Args:
        img_slice: Slice de la imagen DICOM
        window_width, window_center: Parámetros de visualización
        config: Configuración de detección
    
    Returns:
        Imagen original con superposición de segmentaciones
    """
    # Normalizar la imagen usando los parámetros de ventana
    normalized_img = apply_window_level(img_slice, window_width, window_center)
    
    # Convertir a uint8 para procesamiento de imágenes
    img_uint8 = (normalized_img * 255).astype(np.uint8)
    
    # 1. Detección simplificada de tumores (estructuras hiperdensas)
    # Aplicar umbral para segmentar posibles tumores (valores altos en CT)
    tumor_threshold = config['tumor_threshold']
    tumor_mask = (normalized_img > tumor_threshold).astype(np.uint8)
    
    # Eliminar pequeños objetos
    tumor_mask = morphology.remove_small_objects(
        tumor_mask.astype(bool), 
        min_size=config['min_area']
    ).astype(np.uint8)
    
    # 2. Detección de otros órganos
    # Usar Otsu para segmentación general
    thresh_otsu = filters.threshold_otsu(normalized_img)
    organ_mask = (normalized_img > (thresh_otsu * config['organ_sensitivity'])).astype(np.uint8)
    
    # Eliminar pequeños objetos
    organ_mask = morphology.remove_small_objects(
        organ_mask.astype(bool), 
        min_size=config['min_area']
    ).astype(np.uint8)
    
    # Excluir tumores de los órganos
    organ_mask[tumor_mask == 1] = 0
    
    # 3. Etiquetar los órganos
    labeled_organs, num_organs = measure.label(organ_mask, background=0, return_num=True)
    labeled_tumors, num_tumors = measure.label(tumor_mask, background=0, return_num=True)
    
    # Preparar imagen a color para visualización
    img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
    
    # Superponer máscaras con transparencia
    # Órganos en verde semi-transparente
    overlay = img_rgb.copy()
    
    # Colorear órganos detectados
    organ_props = measure.regionprops(labeled_organs)
    for i, prop in enumerate(organ_props):
        # Asignar colores diferentes a diferentes órganos
        color = [(0, 255, 0), (0, 200, 200), (200, 200, 0)][i % 3]  # Verde, cian, amarillo
        overlay[labeled_organs == prop.label] = color
    
    # Tumores en rojo
    tumor_props = measure.regionprops(labeled_tumors)
    for prop in tumor_props:
        overlay[labeled_tumors == prop.label] = (255, 0, 0)  # Rojo
    
    # Fusionar con imagen original con transparencia
    alpha = 0.3
    cv2.addWeighted(overlay, alpha, img_rgb, 1 - alpha, 0, img_rgb)
    
    # Añadir etiquetas si está habilitado
    if config['show_labels']:
        # Etiquetar órganos
        for i, prop in enumerate(organ_props):
            y, x = prop.centroid
            cv2.putText(
                img_rgb, 
                f"Órgano {i+1}", 
                (int(x), int(y)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                1
            )
        
        # Etiquetar tumores
        for i, prop in enumerate(tumor_props):
            y, x = prop.centroid
            cv2.putText(
                img_rgb, 
                f"Tumor {i+1}", 
                (int(x), int(y)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                1
            )
    
    return img_rgb, num_organs, num_tumors

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

def plot_with_segmentation(vol, slice_ix, window_width, window_center, detection_config):
    """Genera gráfico con segmentación de órganos"""
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.axis('off')
    
    selected_slice = vol[slice_ix, :, :]
    
    if detection_config['enabled']:
        # Procesar imagen con detección de órganos
        img_with_detection, num_organs, num_tumors = simple_organ_segmentation(
            selected_slice, 
            window_width, 
            window_center,
            detection_config
        )
        
        # Mostrar imagen procesada
        ax.imshow(img_with_detection, origin='lower')
        
        # Añadir texto de resumen en la esquina
        summary_text = f"Detectados: {num_organs} órganos, {num_tumors} tumores"
        ax.text(
            10, 30, summary_text, 
            color='white', fontsize=12, 
            bbox=dict(facecolor='#28aec5', alpha=0.5)
        )
    else:
        # Mostrar imagen normal con ajustes de ventana/nivel
        windowed_slice = apply_window_level(selected_slice, window_width, window_center)
        ax.imshow(windowed_slice, origin='lower', cmap='gray')
    
    return fig

def render_trajectory_analysis(detection_config, vol, slice_ix, window_width, window_center):
    """Renderiza análisis de trayectorias potenciales si la detección está activada"""
    if not detection_config['enabled']:
        return
    
    st.markdown('<p class="sub-header">Análisis de trayectorias potenciales</p>', unsafe_allow_html=True)
    
    # Simulación simple de análisis de trayectorias
    selected_slice = vol[slice_ix, :, :]
    
    # Crear un análisis muy básico de posibles trayectorias
    # (esto es solo ilustrativo, un algoritmo real sería más complejo)
    
    # 1. Procesar imagen para segmentación
    _, num_organs, num_tumors = simple_organ_segmentation(
        selected_slice, window_width, window_center, detection_config
    )
    
    if num_tumors == 0:
        st.warning("No se detectaron tumores en este corte. Ajuste los parámetros de detección o seleccione otro corte.")
        return
    
    # 2. Simulación de planificación de trayectorias
    st.markdown(
        f"""
        <div class="info-box">
            <p><strong>Análisis preliminar:</strong></p>
            <ul>
                <li>Tumores detectados: {num_tumors}</li>
                <li>Órganos cercanos: {num_organs}</li>
                <li>Corte actual: {slice_ix + 1}</li>
            </ul>
            <p>Se recomienda la planificación de {min(num_tumors + 2, 8)} agujas para cubrir adecuadamente la región tumoral.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # 3. Mostrar botón para generar template (simulado)
    if st.button("Generar template preliminar"):
        st.success("Template preliminar generado basado en el análisis actual.")
        st.info("En un sistema completo, esto generaría un modelo 3D basado en los tumores detectados, similar al descrito en el artículo, utilizando algoritmos k-means y el método del gradiente conjugado.")

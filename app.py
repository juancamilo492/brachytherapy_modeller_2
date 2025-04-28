import os
import io
import zipfile
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import SimpleITK as sitk
import pydicom
import matplotlib.patches as patches
import time
from functools import lru_cache

# Configuración de Streamlit
st.set_page_config(page_title="Brachyanalysis", layout="wide")

# --- CSS personalizado ---
st.markdown("""
<style>
    .giant-title { color: #28aec5; font-size: 64px; text-align: center; margin-bottom: 30px; }
    .sidebar-title { color: #28aec5; font-size: 28px; font-weight: bold; margin-bottom: 15px; }
    .info-box { background-color: #eef9fb; border-left: 5px solid #28aec5; padding: 10px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<p class="giant-title">Brachyanalysis</p>', unsafe_allow_html=True)

# --- Inicialización de estado de sesión ---
if 'volume' not in st.session_state:
    st.session_state.volume = None
if 'volume_info' not in st.session_state:
    st.session_state.volume_info = None
if 'structures' not in st.session_state:
    st.session_state.structures = None
if 'processed_contours' not in st.session_state:
    st.session_state.processed_contours = {}
if 'figures' not in st.session_state:
    st.session_state.figures = {'axial': None, 'coronal': None, 'sagittal': None}
if 'slice_info' not in st.session_state:
    st.session_state.slice_info = {'axial': 0, 'coronal': 0, 'sagittal': 0}

# --- Sidebar ---
st.sidebar.markdown('<p class="sidebar-title">Configuración</p>', unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Sube un archivo ZIP con tus imágenes DICOM", type="zip")

# --- Funciones auxiliares para cargar archivos DICOM y estructuras ---

def extract_zip(uploaded_zip):
    """Extrae archivos de un ZIP subido"""
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(io.BytesIO(uploaded_zip.read()), 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    return temp_dir

def find_dicom_series(directory):
    """Busca archivos DICOM y los agrupa por SeriesInstanceUID"""
    series = {}
    structures = []

    for root, _, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)
            try:
                dcm = pydicom.dcmread(path, force=True, stop_before_pixels=True)
                modality = getattr(dcm, 'Modality', '')

                if modality == 'RTSTRUCT' or file.startswith('RS'):
                    structures.append(path)
                elif modality in ['CT', 'MR', 'PT', 'US']:
                    uid = getattr(dcm, 'SeriesInstanceUID', 'unknown')
                    if uid not in series:
                        series[uid] = []
                    series[uid].append(path)
            except Exception:
                pass  # Ignorar archivos no DICOM

    return series, structures

# --- Parte 2: Carga de imágenes y estructuras ---

def load_dicom_series(file_list):
    """Carga imágenes DICOM como volumen 3D con manejo mejorado de errores"""
    dicom_files = []
    for file_path in file_list:
        try:
            dcm = pydicom.dcmread(file_path, force=True)
            if hasattr(dcm, 'pixel_array'):
                dicom_files.append((file_path, dcm))
        except Exception:
            continue

    if not dicom_files:
        return None, None

    # Ordenar por InstanceNumber
    dicom_files.sort(key=lambda x: getattr(x[1], 'InstanceNumber', 0))
    
    # Encontrar la forma más común
    shape_counts = {}
    for _, dcm in dicom_files:
        shape = dcm.pixel_array.shape
        shape_counts[shape] = shape_counts.get(shape, 0) + 1
    
    best_shape = max(shape_counts, key=shape_counts.get)
    slices = [d[1].pixel_array for d in dicom_files if d[1].pixel_array.shape == best_shape]

    # Crear volumen 3D
    volume = np.stack(slices)

    # Extraer información de spacings
    sample = dicom_files[0][1]
    pixel_spacing = getattr(sample, 'PixelSpacing', [1,1])
    # Asegurarse de que pixel_spacing sea una lista regular de Python
    pixel_spacing = list(map(float, pixel_spacing))
    slice_thickness = float(getattr(sample, 'SliceThickness', 1))
    
    # Corregido: pixel_spacing ya está convertido a lista regular
    spacing = pixel_spacing + [slice_thickness]
    
    origin = getattr(sample, 'ImagePositionPatient', [0,0,0])
    direction = getattr(sample, 'ImageOrientationPatient', [1,0,0,0,1,0])

    direction_matrix = np.array([
        [direction[0], direction[3], 0],
        [direction[1], direction[4], 0],
        [direction[2], direction[5], 1]
    ])

    # Añadir posiciones Z de cada corte para uso posterior
    slice_positions = []
    for _, dcm in dicom_files:
        if hasattr(dcm, 'ImagePositionPatient'):
            slice_positions.append(float(dcm.ImagePositionPatient[2]))
        else:
            slice_positions.append(0.0)

    volume_info = {
        'spacing': spacing,
        'origin': origin,
        'direction': direction_matrix,
        'size': volume.shape,
        'slice_positions': slice_positions
    }
    
    # Debug info
    st.success(f"Volumen cargado: {volume.shape}, spacing: {spacing}")
    
    return volume, volume_info

def load_rtstruct(file_path):
    """Carga contornos RTSTRUCT con mejor manejo de errores y debug"""
    try:
        struct = pydicom.dcmread(file_path)
        structures = {}
        
        if not hasattr(struct, 'ROIContourSequence'):
            st.warning("El archivo RTSTRUCT no contiene secuencia ROIContour")
            return structures
        
        # Mapeo de ROI Number a ROI Name
        roi_names = {roi.ROINumber: roi.ROIName for roi in struct.StructureSetROISequence}
        
        # Debug info
        st.info(f"Estructuras encontradas: {', '.join(roi_names.values())}")
        
        for roi in struct.ROIContourSequence:
            color = np.array(roi.ROIDisplayColor) / 255.0 if hasattr(roi, 'ROIDisplayColor') else np.random.rand(3)
            contours = []
            
            if hasattr(roi, 'ContourSequence'):
                contour_count = 0
                for contour in roi.ContourSequence:
                    pts = np.array(contour.ContourData).reshape(-1, 3)
                    contours.append({'points': pts, 'z': np.mean(pts[:,2])})
                    contour_count += 1
                
                roi_name = roi_names.get(roi.ReferencedROINumber, f"ROI-{roi.ReferencedROINumber}")
                structures[roi_name] = {'color': color, 'contours': contours}
                st.info(f"Estructura {roi_name}: {contour_count} contornos cargados")
            
        return structures
    except Exception as e:
        st.error(f"Error leyendo estructura: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

# --- Parte 3: Funciones de visualización OPTIMIZADAS ---

def apply_window(img, window_center, window_width):
    """Aplica ventana de visualización a la imagen"""
    img = img.astype(np.float32)

    min_value = window_center - window_width / 2
    max_value = window_center + window_width / 2

    img = np.clip(img, min_value, max_value)  # Recortar intensidades
    img = (img - min_value) / (max_value - min_value)  # Normalizar 0-1
    img = np.clip(img, 0, 1)  # Garantizar dentro [0,1]

    return img

# Función para procesar contornos una vez y cachearlos
@lru_cache(maxsize=64)
def process_contour_slice(plane, slice_pos, structure_key, tolerance=2.0):
    """
    Procesa y cachea los contornos para un plano y posición específicos.
    Devuelve los puntos en coordenadas de píxel listos para dibujar.
    """
    if not st.session_state.structures or structure_key not in st.session_state.structures:
        return []
    
    struct = st.session_state.structures[structure_key]
    volume_info = st.session_state.volume_info
    origin = np.array(volume_info['origin'])
    spacing = np.array(volume_info['spacing'])
    
    result_contours = []
    
    for contour in struct['contours']:
        raw_points = contour['points']
        
        if plane == 'axial':
            # Verificar cercanía en Z
            contour_z_values = raw_points[:, 2]
            min_z = np.min(contour_z_values)
            max_z = np.max(contour_z_values)
            
            if (min_z - tolerance <= slice_pos <= max_z + tolerance or
                abs(contour['z'] - slice_pos) <= tolerance):
                
                pixel_points = np.zeros((raw_points.shape[0], 2))
                pixel_points[:, 0] = (raw_points[:, 0] - origin[0]) / spacing[0]
                pixel_points[:, 1] = (raw_points[:, 1] - origin[1]) / spacing[1]
                
                if len(pixel_points) >= 3:
                    result_contours.append(pixel_points)
                    
        elif plane == 'coronal':
            # Verificar cercanía en Y
            mask = np.abs(raw_points[:, 1] - slice_pos) < spacing[1] * 0.5
            if np.sum(mask) >= 3:
                selected_points = raw_points[mask]
                pixel_points = np.zeros((selected_points.shape[0], 2))
                pixel_points[:, 0] = (selected_points[:, 0] - origin[0]) / spacing[0]  # X
                pixel_points[:, 1] = (selected_points[:, 2] - origin[2]) / spacing[2]  # Z
                
                result_contours.append(pixel_points)
                
        elif plane == 'sagittal':
            # Verificar cercanía en X
            mask = np.abs(raw_points[:, 0] - slice_pos) < spacing[0] * 0.5
            if np.sum(mask) >= 3:
                selected_points = raw_points[mask]
                pixel_points = np.zeros((selected_points.shape[0], 2))
                pixel_points[:, 0] = (selected_points[:, 1] - origin[1]) / spacing[1]  # Y
                pixel_points[:, 1] = (selected_points[:, 2] - origin[2]) / spacing[2]  # Z
                
                result_contours.append(pixel_points)
    
    return result_contours

# Función para obtener una imagen de corte específica con ventana aplicada
def get_slice_image(volume, slice_idx, plane, window, invert_colors=False):
    """Obtener una imagen de corte con la ventana y color aplicados"""
    # Obtener la imagen del corte
    if plane == 'axial':
        img = volume[slice_idx, :, :]
    elif plane == 'coronal':
        img = volume[:, slice_idx, :]
    elif plane == 'sagittal':
        img = volume[:, :, slice_idx]
    else:
        raise ValueError("Plano inválido")

    # Aplicar ventana
    img = apply_window(img, window[1], window[0])
    if invert_colors:
        img = 1.0 - img
    
    return img

# Versión optimizada de draw_slice que reutiliza figura existente
def update_slice_view(plane, slice_idx, window, structures_visible=True, linewidth=2, invert_colors=False):
    """
    Actualiza un corte existente en lugar de crear uno nuevo cada vez.
    Mucho más eficiente para cambios de corte.
    """
    volume = st.session_state.volume
    volume_info = st.session_state.volume_info
    structures = st.session_state.structures
    
    # Obtener o inicializar figura si es necesario
    if st.session_state.figures[plane] is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.axis('off')
        st.session_state.figures[plane] = {'fig': fig, 'ax': ax, 'image': None, 'contours': [], 'text': []}
    else:
        fig = st.session_state.figures[plane]['fig']
        ax = st.session_state.figures[plane]['ax']
        
        # Limpiar contenido anterior
        if st.session_state.figures[plane]['image'] is not None:
            st.session_state.figures[plane]['image'].remove()
        for patch in st.session_state.figures[plane]['contours']:
            patch.remove()
        for text in st.session_state.figures[plane]['text']:
            text.remove()
        
        st.session_state.figures[plane]['contours'] = []
        st.session_state.figures[plane]['text'] = []
    
    # Calcular posición física actual
    origin = np.array(volume_info['origin'])
    spacing = np.array(volume_info['spacing'])
    
    if plane == 'axial':
        if 'slice_positions' in volume_info and len(volume_info['slice_positions']) > slice_idx:
            current_slice_pos = volume_info['slice_positions'][slice_idx]
        else:
            current_slice_pos = origin[2] + slice_idx * spacing[2]
        coord_label = f"Z: {current_slice_pos:.2f} mm"
    elif plane == 'coronal':
        current_slice_pos = origin[1] + slice_idx * spacing[1]
        coord_label = f"Y: {current_slice_pos:.2f} mm"
    elif plane == 'sagittal':
        current_slice_pos = origin[0] + slice_idx * spacing[0]
        coord_label = f"X: {current_slice_pos:.2f} mm"
    
    # Obtener y mostrar imagen
    img = get_slice_image(volume, slice_idx, plane, window, invert_colors)
    st.session_state.figures[plane]['image'] = ax.imshow(img, cmap='gray')
    
    # Añadir texto informativo
    text1 = ax.text(5, 15, f"{plane} - slice {slice_idx}", color='white',
            bbox=dict(facecolor='black', alpha=0.5))
    text2 = ax.text(5, 30, coord_label, color='yellow',
            bbox=dict(facecolor='black', alpha=0.5))
    st.session_state.figures[plane]['text'] = [text1, text2]
    
    # Mostrar contornos optimizados si están activados
    if structures_visible and structures:
        for name, struct in structures.items():
            # Usar la función cacheada para procesar contornos
            contours = process_contour_slice(plane, current_slice_pos, name, spacing[2] * 2.0)
            color = struct['color']
            
            if contours:
                for contour_points in contours:
                    if len(contour_points) >= 3:
                        polygon = patches.Polygon(contour_points, closed=True,
                                               fill=False, edgecolor=color,
                                               linewidth=linewidth)
                        ax.add_patch(polygon)
                        st.session_state.figures[plane]['contours'].append(polygon)
                
                # Añadir texto del nombre si hay contornos
                text = ax.text(img.shape[1]/2, img.shape[0]/2, name,
                        color=color, fontsize=8, ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.7))
                st.session_state.figures[plane]['text'].append(text)
    
    return fig

# --- Parte 4: Interfaz principal OPTIMIZADA ---

if uploaded_file:
    # Si hay un nuevo archivo, recargar todo
    if 'last_file' not in st.session_state or st.session_state.last_file != uploaded_file.name:
        st.session_state.last_file = uploaded_file.name
        
        # Limpiar figuras existentes
        st.session_state.figures = {'axial': None, 'coronal': None, 'sagittal': None}
        st.session_state.processed_contours = {}
        
        # Extraer y procesar nuevos datos
        temp_dir = extract_zip(uploaded_file)
        series_dict, structure_files = find_dicom_series(temp_dir)
        
        if series_dict:
            series_options = list(series_dict.keys())
            selected_series = st.sidebar.selectbox("Selecciona la serie", series_options)
            dicom_files = series_dict[selected_series]
            
            # Cargar volumen e información
            st.session_state.volume, st.session_state.volume_info = load_dicom_series(dicom_files)
            
            # Cargar estructuras si están disponibles
            st.session_state.structures = None
            if structure_files:
                st.session_state.structures = load_rtstruct(structure_files[0])
                if st.session_state.structures:
                    st.success(f"✅ Se cargaron {len(st.session_state.structures)} estructuras.")
                else:
                    st.warning("⚠️ No se encontraron estructuras RTSTRUCT.")
        else:
            st.warning("No se encontraron imágenes DICOM en el ZIP.")
    
    # Si tenemos volumen cargado, mostrar controles de visualización
    if st.session_state.volume is not None:
        volume = st.session_state.volume
        volume_info = st.session_state.volume_info
        
        st.sidebar.markdown('<p class="sidebar-title">Visualización</p>', unsafe_allow_html=True)
        
        # Definir límites de los sliders
        max_axial = volume.shape[0] - 1
        max_coronal = volume.shape[1] - 1
        max_sagittal = volume.shape[2] - 1
        
        st.sidebar.markdown("#### Selección de cortes")
        st.sidebar.markdown("#### Opciones avanzadas")
        sync_slices = st.sidebar.checkbox("Sincronizar cortes", value=True)
        invert_colors = st.sidebar.checkbox("Invertir colores (Negativo)", value=False)
        
        # Control optimizado de cortes - un solo widget para cada dirección
        if sync_slices:
            unified_idx = st.sidebar.slider(
                "Corte (sincronizado)",
                min_value=0,
                max_value=max(max_axial, max_coronal, max_sagittal),
                value=max_axial // 2,
                step=1,
                key="unified_slice_slider"
            )
            axial_idx = min(unified_idx, max_axial)
            coronal_idx = min(unified_idx, max_coronal)
            sagittal_idx = min(unified_idx, max_sagittal)
        else:
            axial_idx = st.sidebar.slider(
                "Corte axial (Z)",
                min_value=0,
                max_value=max_axial,
                value=st.session_state.slice_info['axial'],
                step=1,
                key="axial_slider"
            )
            coronal_idx = st.sidebar.slider(
                "Corte coronal (Y)",
                min_value=0,
                max_value=max_coronal,
                value=st.session_state.slice_info['coronal'],
                step=1,
                key="coronal_slider"
            )
            sagittal_idx = st.sidebar.slider(
                "Corte sagital (X)",
                min_value=0,
                max_value=max_sagittal,
                value=st.session_state.slice_info['sagittal'],
                step=1,
                key="sagittal_slider"
            )
        
        # Actualizar el estado de los cortes
        st.session_state.slice_info = {
            'axial': axial_idx,
            'coronal': coronal_idx,
            'sagittal': sagittal_idx
        }
        
        window_option = st.sidebar.selectbox(
            "Tipo de ventana",
            ["Default", "Cerebro (Brain)", "Pulmón (Lung)", "Hueso (Bone)", "Abdomen", "Mediastino (Mediastinum)",
             "Hígado (Liver)", "Tejido blando (Soft Tissue)", "Columna blanda (Spine Soft)",
             "Columna ósea (Spine Bone)", "Aire (Air)", "Grasa (Fat)", "Metal", "Personalizado"]
        )
        
        if window_option == "Default":
            # Obtener valores de ventana de la imagen DICOM
            try:
                sample = dicom_files[0]
                dcm = pydicom.dcmread(sample, force=True)
                window_width = getattr(dcm, 'WindowWidth', [400])[0] if hasattr(dcm, 'WindowWidth') else 400
                window_center = getattr(dcm, 'WindowCenter', [40])[0] if hasattr(dcm, 'WindowCenter') else 40
                # Asegurar que los valores son números
                if isinstance(window_width, (list, tuple)):
                    window_width = window_width[0]
                if isinstance(window_center, (list, tuple)):
                    window_center = window_center[0]
            except Exception:
                window_width, window_center = 400, 40  # Valores por defecto si hay algún error
        # Configuración de ventana
        elif window_option == "Cerebro (Brain)":
            window_width, window_center = 80, 40
        elif window_option == "Pulmón (Lung)":
            window_width, window_center = 1500, -600
        elif window_option == "Hueso (Bone)":
            window_width, window_center = 1500, 300
        elif window_option == "Abdomen":
            window_width, window_center = 400, 60
        elif window_option == "Mediastino (Mediastinum)":
            window_width, window_center = 400, 40
        elif window_option == "Hígado (Liver)":
            window_width, window_center = 150, 70
        elif window_option == "Tejido blando (Soft Tissue)":
            window_width, window_center = 350, 50
        elif window_option == "Columna blanda (Spine Soft)":
            window_width, window_center = 350, 50
        elif window_option == "Columna ósea (Spine Bone)":
            window_width, window_center = 1500, 300
        elif window_option == "Aire (Air)":
            window_width, window_center = 2000, -1000
        elif window_option == "Grasa (Fat)":
            window_width, window_center = 200, -100
        elif window_option == "Metal":
            window_width, window_center = 4000, 1000
        elif window_option == "Personalizado":
            window_center = st.sidebar.number_input("Window Center (WL)", value=40)
            window_width = st.sidebar.number_input("Window Width (WW)", value=400)
        
        show_structures = st.sidebar.checkbox("Mostrar estructuras", value=True)
        linewidth = st.sidebar.slider("Grosor líneas", 1, 8, 2)
        
        # Mostrar las imágenes en tres columnas
        col1, col2, col3 = st.columns(3)
        
        # Cachear los parámetros de ventana para evitar recálculos
        window_params = (window_width, window_center)
        
        with col1:
            st.markdown("### Axial")
            fig_axial = update_slice_view(
                'axial', axial_idx, window_params, 
                structures_visible=show_structures,
                linewidth=linewidth,
                invert_colors=invert_colors
            )
            st.pyplot(fig_axial)
        
        with col2:
            st.markdown("### Coronal")
            fig_coronal = update_slice_view(
                'coronal', coronal_idx, window_params,
                structures_visible=show_structures,
                linewidth=linewidth,
                invert_colors=invert_colors
            )
            st.pyplot(fig_coronal)
        
        with col3:
            st.markdown("### Sagital")
            fig_sagittal = update_slice_view(
                'sagittal', sagittal_idx, window_params,
                structures_visible=show_structures,
                linewidth=linewidth,
                invert_colors=invert_colors
            )
            st.pyplot(fig_sagittal)
            
# Botón para cargar datos de prueba (para demostración)
if not uploaded_file:
    st.markdown("""
    <div class="info-box">
        <h3>Cómo usar Brachyanalysis</h3>
        <p>1. Sube un archivo ZIP que contenga tus imágenes DICOM y (opcionalmente) estructuras.</p>
        <p>2. Selecciona la serie de imágenes para visualizar.</p>
        <p>3. Utiliza los controles de la barra lateral para ajustar la visualización.</p>
    </div>
    """, unsafe_allow_html=True)

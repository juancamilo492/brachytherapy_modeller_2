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
    """Improved function to load contours from RTSTRUCT file"""
    try:
        struct = pydicom.dcmread(file_path)
        structures = {}
        
        if not hasattr(struct, 'ROIContourSequence'):
            st.warning("The RTSTRUCT file doesn't contain a ROIContour sequence")
            return structures
        
        # Map ROI Number to ROI Name
        roi_names = {}
        if hasattr(struct, 'StructureSetROISequence'):
            for roi in struct.StructureSetROISequence:
                if hasattr(roi, 'ROINumber') and hasattr(roi, 'ROIName'):
                    roi_names[roi.ROINumber] = roi.ROIName
        
        # Debug info
        roi_list = list(roi_names.values())
        if roi_list:
            st.info(f"Found structures: {', '.join(roi_list)}")
        
        # Process each ROI in the contour sequence
        for roi in struct.ROIContourSequence:
            # Default color if not specified
            if hasattr(roi, 'ROIDisplayColor'):
                color = np.array(roi.ROIDisplayColor) / 255.0
            else:
                color = np.random.rand(3)  # Random color
                
            contours = []
            
            if hasattr(roi, 'ContourSequence'):
                contour_count = 0
                
                for contour in roi.ContourSequence:
                    # Only process contours that have contour data
                    if hasattr(contour, 'ContourData') and contour.ContourData:
                        # Make sure we have the right number of points
                        num_points = len(contour.ContourData) // 3
                        if num_points * 3 == len(contour.ContourData):
                            pts = np.array(contour.ContourData).reshape(num_points, 3)
                            contours.append({
                                'points': pts,
                                'z': np.mean(pts[:, 2])  # Average z position
                            })
                            contour_count += 1
                
                # Use ROI name if available, otherwise use ROI number
                if hasattr(roi, 'ReferencedROINumber'):
                    roi_num = roi.ReferencedROINumber
                    roi_name = roi_names.get(roi_num, f"ROI-{roi_num}")
                    
                    structures[roi_name] = {
                        'color': color,
                        'contours': contours
                    }
                    
                    # Provide info about loaded contours
                    if contour_count > 0:
                        st.info(f"Loaded {contour_count} contours for structure '{roi_name}'")
        
        return structures
    
    except Exception as e:
        st.error(f"Error reading RTSTRUCT: {e}")
        import traceback
        st.error(traceback.format_exc())
        return {}

# --- Parte 3: Funciones de visualización ---

def patient_to_voxel(point, volume_info):
    """
    Converts a single point from patient coordinate space to voxel indices.
    Returns the coordinates in order expected by the current view (x,y).
    """
    # Extract necessary info
    origin = np.array(volume_info['origin'])
    spacing = np.array(volume_info['spacing'])
    
    # Basic transformation (physical space to index space)
    voxel_point = (point - origin) / spacing
    
    # Round to nearest integer for index
    voxel_point = np.round(voxel_point).astype(int)
    
    return voxel_point


def apply_window(img, window_center, window_width):
    """Aplica ventana de visualización a la imagen"""
    img = img.astype(np.float32)

    min_value = window_center - window_width / 2
    max_value = window_center + window_width / 2

    img = np.clip(img, min_value, max_value)  # Recortar intensidades
    img = (img - min_value) / (max_value - min_value)  # Normalizar 0-1
    img = np.clip(img, 0, 1)  # Garantizar dentro [0,1]

    return img


def draw_slice(volume, slice_idx, plane, structures, volume_info, window, linewidth=2, show_names=True, invert_colors=False):
    """
    Improved version of draw_slice that correctly handles contours on different planes
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.axis('off')

    # Get slice image
    if plane == 'axial':
        img = volume[slice_idx, :, :]
        slice_pos = slice_idx * volume_info['spacing'][2] + volume_info['origin'][2]
        view_axes = [0, 1]  # X,Y axes for this view
        depth_axis = 2  # Z is depth axis
    elif plane == 'coronal':
        img = volume[:, slice_idx, :]
        slice_pos = slice_idx * volume_info['spacing'][1] + volume_info['origin'][1]
        view_axes = [0, 2]  # X,Z axes 
        depth_axis = 1  # Y is depth axis
    elif plane == 'sagittal':
        img = volume[:, :, slice_idx]
        slice_pos = slice_idx * volume_info['spacing'][0] + volume_info['origin'][0]
        view_axes = [1, 2]  # Y,Z axes
        depth_axis = 0  # X is depth axis
    else:
        raise ValueError("Invalid plane")

    # Apply window
    img_windowed = apply_window(img, window[1], window[0])
    if invert_colors:
        img_windowed = 1.0 - img_windowed

    # Show base image
    ax.imshow(img_windowed, cmap='gray')

    # Draw contours
    if structures:
        for name, struct in structures.items():
            drawn_contours = 0
            all_x_points = []
            all_y_points = []
            
            for contour in struct['contours']:
                points = contour['points']
                
                # Check if this contour should be shown in this slice
                # Use the contour's position along the depth axis
                mean_depth_pos = np.mean(points[:, depth_axis])
                slice_thickness = volume_info['spacing'][depth_axis]
                
                # Only show contours that are close to this slice
                if abs(mean_depth_pos - slice_pos) <= slice_thickness:
                    # Project points to 2D for this view
                    points_2d = []
                    
                    for point in points:
                        # Convert to voxel space
                        voxel_point = patient_to_voxel(point, volume_info)
                        # Extract the 2 relevant axes for this view
                        point_2d = [voxel_point[view_axes[0]], voxel_point[view_axes[1]]]
                        points_2d.append(point_2d)
                    
                    # Only draw if we have enough points
                    if len(points_2d) >= 3:
                        points_2d = np.array(points_2d)
                        # Flip y-coordinates for proper display (matplotlib coordinate system)
                        if plane == 'axial':
                            # For axial view, flip y to match image display
                            points_2d[:, 1] = img.shape[0] - points_2d[:, 1]
                        
                        # Create and add polygon
                        polygon = patches.Polygon(points_2d, closed=True, 
                                                 fill=False, edgecolor=struct['color'], 
                                                 linewidth=linewidth)
                        ax.add_patch(polygon)
                        drawn_contours += 1
                        
                        # Collect points for calculating center
                        all_x_points.extend(points_2d[:, 0])
                        all_y_points.extend(points_2d[:, 1])
            
            # Show structure name at centroid of all points
            if drawn_contours > 0 and show_names and all_x_points and all_y_points:
                x_center = np.mean(all_x_points)
                y_center = np.mean(all_y_points)
                ax.text(x_center, y_center, name, 
                       color=struct['color'], fontsize=9, weight='bold',
                       ha='center', va='center', bbox=dict(facecolor='black', alpha=0.2))

    # Add slice information
    plt.text(5, 15, f"{plane} - slice {slice_idx}", color='white', 
             bbox=dict(facecolor='black', alpha=0.5))

    plt.tight_layout()
    return fig


# --- Parte 4: Interfaz principal ---

if uploaded_file:
    temp_dir = extract_zip(uploaded_file)
    series_dict, structure_files = find_dicom_series(temp_dir)

    if series_dict:
        series_options = list(series_dict.keys())
        selected_series = st.sidebar.selectbox("Selecciona la serie", series_options)
        dicom_files = series_dict[selected_series]

        # Cargar el volumen e información
        volume, volume_info = load_dicom_series(dicom_files)

        # Cargar estructuras si están disponibles
        structures = None
        if structure_files:
            structures = load_rtstruct(structure_files[0])
            if structures:
                st.success(f"✅ Se cargaron {len(structures)} estructuras.")
            else:
                st.warning("⚠️ No se encontraron estructuras RTSTRUCT.")

        if volume is not None:
            st.sidebar.markdown('<p class="sidebar-title">Visualización</p>', unsafe_allow_html=True)

            # Definir límites de los sliders
            max_axial = volume.shape[0] - 1
            max_coronal = volume.shape[1] - 1
            max_sagittal = volume.shape[2] - 1

            st.sidebar.markdown("#### Selección de cortes")
            st.sidebar.markdown("#### Opciones avanzadas")
            sync_slices = st.sidebar.checkbox("Sincronizar cortes", value=True)
            invert_colors = st.sidebar.checkbox("Invertir colores (Negativo)", value=False)

            if sync_slices:
                unified_idx = st.sidebar.slider(
                    "Corte (sincronizado)",
                    min_value=0,
                    max_value=max(max_axial, max_coronal, max_sagittal),
                    value=max_axial // 2,
                    step=1
                )
                unified_idx = st.sidebar.number_input(
                    "Corte (sincronizado)",
                    min_value=0,
                    max_value=max(max_axial, max_coronal, max_sagittal),
                    value=unified_idx,
                    step=1
                )
                axial_idx = min(unified_idx, max_axial)
                coronal_idx = min(unified_idx, max_coronal)
                sagittal_idx = min(unified_idx, max_sagittal)
            else:
                axial_idx = st.sidebar.slider(
                    "Corte axial (Z)",
                    min_value=0,
                    max_value=max_axial,
                    value=max_axial // 2,
                    step=1
                )
                axial_idx = st.sidebar.number_input(
                    "Corte axial (Z)",
                    min_value=0,
                    max_value=max_axial,
                    value=axial_idx,
                    step=1
                )
                coronal_idx = st.sidebar.slider(
                    "Corte coronal (Y)",
                    min_value=0,
                    max_value=max_coronal,
                    value=max_coronal // 2,
                    step=1
                )
                coronal_idx = st.sidebar.number_input(
                    "Corte coronal (Y)",
                    min_value=0,
                    max_value=max_coronal,
                    value=coronal_idx,
                    step=1
                )
                sagittal_idx = st.sidebar.slider(
                    "Corte sagital (X)",
                    min_value=0,
                    max_value=max_sagittal,
                    value=max_sagittal // 2,
                    step=1
                )
                sagittal_idx = st.sidebar.number_input(
                    "Corte sagital (X)",
                    min_value=0,
                    max_value=max_sagittal,
                    value=sagittal_idx,
                    step=1
                )

            window_option = st.sidebar.selectbox(
                "Tipo de ventana",
                ["Default", "Cerebro (Brain)", "Pulmón (Lung)", "Hueso (Bone)", "Abdomen", "Mediastino (Mediastinum)",
                 "Hígado (Liver)", "Tejido blando (Soft Tissue)", "Columna blanda (Spine Soft)",
                 "Columna ósea (Spine Bone)", "Aire (Air)", "Grasa (Fat)", "Metal", "Personalizado"]
            )


            if window_option == "Default":
                # Obtener valores de ventana de la imagen DICOM
                sample = dicom_files[0]
                try:
                    dcm = pydicom.dcmread(sample, force=True)
                    window_width = getattr(dcm, 'WindowWidth', [400])[0] if hasattr(dcm, 'WindowWidth') else 400
                    window_center = getattr(dcm, 'WindowCenter', [40])[0] if hasattr(dcm, 'WindowCenter') else 40
                    # Asegurar que los valores son números, ya que pueden ser múltiples o estar en formato especial
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

            with col1:
                st.markdown("### Axial")
                fig_axial = draw_slice(
                    volume, axial_idx, 'axial', 
                    structures if show_structures else None, 
                    volume_info, 
                    (window_width, window_center),
                    linewidth=linewidth,
                    invert_colors=invert_colors
                )
                st.pyplot(fig_axial)

            with col2:
                st.markdown("### Coronal")
                fig_coronal = draw_slice(
                    volume, coronal_idx, 'coronal',
                    structures if show_structures else None,
                    volume_info,
                    (window_width, window_center),
                    linewidth=linewidth,
                    invert_colors=invert_colors
                )
                st.pyplot(fig_coronal)

            with col3:
                st.markdown("### Sagital")
                fig_sagittal = draw_slice(
                    volume, sagittal_idx, 'sagittal',
                    structures if show_structures else None,
                    volume_info,
                    (window_width, window_center),
                    linewidth=linewidth,
                    invert_colors=invert_colors
                )
                st.pyplot(fig_sagittal)
    else:
        st.warning("No se encontraron imágenes DICOM en el ZIP.")

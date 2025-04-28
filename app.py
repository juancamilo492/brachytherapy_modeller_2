import os
import io
import zipfile
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pydicom
import matplotlib.patches as patches
from pydicom.valuerep import DSfloat

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
    pixel_spacing = list(map(float, pixel_spacing))
    slice_thickness = float(getattr(sample, 'SliceThickness', 1))
    spacing = pixel_spacing + [slice_thickness]
    origin = getattr(sample, 'ImagePositionPatient', [0,0,0])
    direction = getattr(sample, 'ImageOrientationPatient', [1,0,0,0,1,0])

    direction_matrix = np.array([
        [direction[0], direction[3], 0],
        [direction[1], direction[4], 0],
        [direction[2], direction[5], 1]
    ])

    # Añadir posiciones Z de cada corte
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
        
        roi_names = {roi.ROINumber: roi.ROIName for roi in struct.StructureSetROISequence}
        st.info(f"Estructuras encontradas: {', '.join(roi_names.values())}")
        
        for roi in struct.ROIContourSequence:
            color = np.array(roi.ROIDisplayColor) / 255.0 if hasattr(roi, 'ROIDisplayColor') else np.random.rand(3)
            contours = []
            
            if hasattr(roi, 'ContourSequence'):
                for contour in roi.ContourSequence:
                    pts = np.array(contour.ContourData).reshape(-1, 3)
                    contours.append({'points': pts, 'z': np.mean(pts[:,2])})
                
                roi_name = roi_names.get(roi.ReferencedROINumber, f"ROI-{roi.ReferencedROINumber}")
                structures[roi_name] = {'color': color, 'contours': contours}
            
        return structures
    except Exception as e:
        st.error(f"Error leyendo estructura: {e}")
        return {}

# --- Parte 3: Funciones de visualización ---

def patient_to_voxel(points, volume_info):
    """Convierte puntos del espacio del paciente al espacio de vóxeles"""
    origin = np.asarray(volume_info['origin'], dtype=np.float32)
    spacing = np.asarray(volume_info['spacing'], dtype=np.float32)
    direction = volume_info['direction']
    
    inv_direction = np.linalg.inv(direction)
    adjusted_points = np.zeros_like(points, dtype=np.float32)
    
    for i in range(len(points)):
        vec = np.dot(inv_direction, points[i] - origin)
        adjusted_points[i] = vec / spacing
        
    return adjusted_points

def apply_window(img, window_center, window_width):
    """Aplica ventana de visualización a la imagen"""
    img = img.astype(np.float32)
    min_value = window_center - window_width / 2
    max_value = window_center + window_width / 2
    img = np.clip(img, min_value, max_value)
    img = (img - min_value) / (max_value - min_value)
    img = np.clip(img, 0, 1)
    return img

def draw_slice(volume, slice_idx, plane, structures, volume_info, window, linewidth=2, show_names=True, invert_colors=False):
    """Dibuja un corte con contornos asegurando que todos se muestren"""
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.axis('off')

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

    # Mostrar imagen base
    ax.imshow(img, cmap='gray')

    # Dibujar contornos
    if structures:
        origin = np.array(volume_info['origin'])
        spacing = np.array(volume_info['spacing'])
        
        for name, struct in structures.items():
            contours_drawn = 0
            for contour in struct['contours']:
                raw_points = contour['points']
                points = (raw_points - origin) / spacing

                # Determinar si el contorno intersecta con el slice actual
                if plane == 'axial':
                    slice_pos = slice_idx * spacing[2] + origin[2]
                    mask = np.abs(raw_points[:, 2] - slice_pos) < spacing[2] / 2
                    if np.any(mask):
                        pts = points[mask][:, [0, 1]]
                        if len(pts) >= 3:
                            polygon = patches.Polygon(pts, closed=True, fill=False, 
                                                    edgecolor=struct['color'], linewidth=linewidth)
                            ax.add_patch(polygon)
                            contours_drawn += 1

                elif plane == 'coronal':
                    slice_pos = slice_idx * spacing[1] + origin[1]
                    mask = np.abs(raw_points[:, 1] - slice_pos) < spacing[1] / 2
                    if np.any(mask):
                        pts = points[mask][:, [0, 2]]
                        if len(pts) >= 3:
                            polygon = patches.Polygon(pts, closed=True, fill=False, 
                                                    edgecolor=struct['color'], linewidth=linewidth)
                            ax.add_patch(polygon)
                            contours_drawn += 1

                elif plane == 'sagittal':
                    slice_pos = slice_idx * spacing[0] + origin[0]
                    mask = np.abs(raw_points[:, 0] - slice_pos) < spacing[0] / 2
                    if np.any(mask):
                        pts = points[mask][:, [1, 2]]
                        if len(pts) >= 3:
                            polygon = patches.Polygon(pts, closed=True, fill=False, 
                                                    edgecolor=struct['color'], linewidth=linewidth)
                            ax.add_patch(polygon)
                            contours_drawn += 1

            # Mostrar nombre si se dibujaron contornos
            if contours_drawn > 0 and show_names:
                ax.text(img.shape[1]/2, img.shape[0]/2, f"{name} ({contours_drawn})", 
                       color=struct['color'], fontsize=8,
                       ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))

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
                ["Default", "Cerebro (Brain)", "Pulmón (Lung)", "Hueso (Bone)", "Abdomen"]
            )

            if window_option == "Default":
                sample = dicom_files[0]
                dcm = pydicom.dcmread(sample, force=True)
                # Manejar WindowWidth y WindowCenter correctamente
                window_width = getattr(dcm, 'WindowWidth', 400)
                window_center = getattr(dcm, 'WindowCenter', 40)
                # Convertir DSfloat o listas a valores numéricos
                if isinstance(window_width, (list, tuple)):
                    window_width = float(window_width[0])
                elif isinstance(window_width, DSfloat):
                    window_width = float(window_width)
                else:
                    window_width = float(window_width)
                if isinstance(window_center, (list, tuple)):
                    window_center = float(window_center[0])
                elif isinstance(window_center, DSfloat):
                    window_center = float(window_center)
                else:
                    window_center = float(window_center)
            elif window_option == "Cerebro (Brain)":
                window_width, window_center = 80, 40
            elif window_option == "Pulmón (Lung)":
                window_width, window_center = 1500, -600
            elif window_option == "Hueso (Bone)":
                window_width, window_center = 1500, 300
            elif window_option == "Abdomen":
                window_width, window_center = 400, 60

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

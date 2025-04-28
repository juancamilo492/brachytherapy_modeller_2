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
    """Carga imágenes DICOM como volumen 3D usando la lógica del código original"""
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

    # Ordenar por InstanceNumber o, si no está disponible, por posición Z
    try:
        dicom_files.sort(key=lambda x: getattr(x[1], 'InstanceNumber', 0))
    except:
        # Backup: ordenar por la posición Z si falló la ordenación por InstanceNumber
        dicom_files.sort(key=lambda x: float(getattr(x[1], 'ImagePositionPatient', [0, 0, 0])[2]))
    
    # Encontrar la forma más común
    shape_counts = {}
    for _, dcm in dicom_files:
        shape = dcm.pixel_array.shape
        shape_counts[shape] = shape_counts.get(shape, 0) + 1
    
    best_shape = max(shape_counts, key=shape_counts.get)
    
    # Filtrar a solo aquellas imágenes con la forma más común
    filtered_files = [(path, dcm) for path, dcm in dicom_files if dcm.pixel_array.shape == best_shape]
    slices = [d[1].pixel_array for d in filtered_files]
    
    # Crear volumen 3D
    volume = np.stack(slices)
    
    # Normalizar el rango de valores para visualización
    if volume.max() > 0:
        volume = volume.astype(np.float32)

    # Extraer información importante de referencia del DICOM
    sample = filtered_files[0][1]
    pixel_spacing = getattr(sample, 'PixelSpacing', [1, 1])
    slice_thickness = getattr(sample, 'SliceThickness', 1)
    
    # Calcular el espaciado real entre rebanadas si está disponible
    if len(filtered_files) > 1:
        pos_first = getattr(filtered_files[0][1], 'ImagePositionPatient', [0, 0, 0])
        pos_last = getattr(filtered_files[-1][1], 'ImagePositionPatient', [0, 0, 0])
        if pos_first and pos_last:
            slice_spacing = abs(pos_last[2] - pos_first[2]) / (len(filtered_files) - 1)
            if slice_spacing > 0:
                slice_thickness = slice_spacing

    spacing = pixel_spacing + [slice_thickness]
    origin = getattr(sample, 'ImagePositionPatient', [0, 0, 0])
    direction = getattr(sample, 'ImageOrientationPatient', [1, 0, 0, 0, 1, 0])

    # Crear matriz de dirección completa
    direction_matrix = np.zeros((3, 3))
    direction_matrix[0, 0] = direction[0]
    direction_matrix[0, 1] = direction[1]
    direction_matrix[0, 2] = direction[2]
    direction_matrix[1, 0] = direction[3]
    direction_matrix[1, 1] = direction[4]
    direction_matrix[1, 2] = direction[5]
    # La tercera fila es el producto vectorial para asegurar un sistema de coordenadas derecho
    direction_matrix[2, :] = np.cross(direction_matrix[0, :], direction_matrix[1, :])

    # Crear mapa de posiciones de cada corte para uso en la conversión de coordenadas
    slice_positions = []
    for _, dcm in filtered_files:
        pos = getattr(dcm, 'ImagePositionPatient', origin)
        slice_positions.append(pos[2])  # Guardar posición Z

    volume_info = {
        'spacing': spacing,
        'origin': origin,
        'direction': direction_matrix,
        'size': volume.shape,
        'slice_positions': slice_positions
    }
    
    return volume, volume_info

def load_rtstruct(file_path):
    """Carga contornos RTSTRUCT usando la lógica del código original"""
    try:
        struct = pydicom.dcmread(file_path)
        structures = {}
        
        if not hasattr(struct, 'ROIContourSequence'):
            return structures
        
        # Mapeo de ROI Number a ROI Name
        roi_names = {roi.ROINumber: roi.ROIName for roi in struct.StructureSetROISequence}
        
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
        st.warning(f"Error leyendo estructura: {e}")
        return None

# --- Parte 3: Funciones de visualización ---

def patient_to_voxel(points, volume_info):
    """
    Convierte puntos del espacio del paciente (físico) al espacio de vóxeles (índices de la matriz).
    Esta función mejorada tiene en cuenta la orientación de la imagen.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Se esperaba (N, 3) puntos, recibido {points.shape}")

    origin = np.asarray(volume_info['origin'], dtype=np.float32)
    spacing = np.asarray(volume_info['spacing'], dtype=np.float32)
    direction = volume_info['direction']
    
    # Invertir la matriz de dirección para transformar del espacio físico al espacio del voxel
    inv_direction = np.linalg.inv(direction)
    
    # Transponer para procesamiento vectorial
    adjusted_points = np.zeros_like(points)
    
    for i in range(len(points)):
        # Primero aplicar la matriz de dirección inversa
        vec = np.dot(inv_direction, points[i] - origin)
        # Luego aplicar el espaciado
        adjusted_points[i] = vec / spacing
        
    return adjusted_points


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
    Dibuja un corte específico del volumen con los contornos de las estructuras.
    Versión mejorada para transformar correctamente las coordenadas.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.axis('off')

    # Obtener la imagen del corte y ajustar según el plano
    if plane == 'axial':
        img = volume[slice_idx, :, :].copy()
        # Transformar a la orientación correcta de visualización
        img = np.rot90(img, k=1)  # Puede requerir ajuste según orientación
    elif plane == 'coronal':
        img = volume[:, slice_idx, :].copy()
        img = np.rot90(img, k=1)  # Rotar para visualización correcta
    elif plane == 'sagittal':
        img = volume[:, :, slice_idx].copy()
        img = np.rot90(img)
    else:
        raise ValueError("Plano inválido: debe ser 'axial', 'coronal' o 'sagittal'")

    # Aplicar ventana de visualización
    img = apply_window(img, window[1], window[0])
    if invert_colors:
        img = 1.0 - img

    # Mostrar imagen base
    ax.imshow(img, cmap='gray', origin='lower')

    # Dibujar contornos si hay estructuras disponibles
    if structures:
        slice_z_position = None
        if plane == 'axial' and 'slice_positions' in volume_info:
            # Obtener la posición Z física para este slice si está disponible
            slice_z_position = volume_info['slice_positions'][slice_idx] if slice_idx < len(volume_info['slice_positions']) else None
        
        for name, struct in structures.items():
            all_pts = []  # Para calcular centro de la estructura

            for contour in struct['contours']:
                # Transformar los puntos a coordenadas de vóxel
                voxels = patient_to_voxel(contour['points'], volume_info)
                
                # Para cada plano, adaptar cómo seleccionamos y mostramos los puntos
                if plane == 'axial':
                    # Filtrar los contornos por su posición Z física 
                    # o usar índice del corte si no hay posiciones guardadas
                    if slice_z_position is not None:
                        # Usar tolerancia para contornos que están cerca del plano
                        mask = np.abs(contour['points'][:, 2] - slice_z_position) < 1.0
                    else:
                        # Método alternativo basado en índices
                        mask = np.abs(voxels[:, 0] - slice_idx) < 0.5
                        
                    # Si hay puntos, mostrarlos transformando las coordenadas
                    if np.any(mask):
                        # Para vista axial, usamos Y, X (después de transformar)
                        pts = voxels[mask][:, [1, 2]]
                        pts = np.fliplr(pts)  # Ajustar orientación si es necesario
                        
                        if len(pts) >= 3:  # Necesitamos al menos 3 puntos para un polígono
                            polygon = patches.Polygon(pts, closed=True, fill=False, 
                                                     edgecolor=struct['color'], linewidth=linewidth)
                            ax.add_patch(polygon)
                            all_pts.append(pts)
                
                elif plane == 'coronal':
                    # Para vista coronal, filtramos por coordenada Y
                    mask = np.abs(voxels[:, 1] - slice_idx) < 0.5
                    if np.any(mask):
                        # Usar X y Z (transformado)
                        pts = voxels[mask][:, [0, 2]]
                        pts = np.fliplr(pts)  # Ajustar según sea necesario
                        
                        if len(pts) >= 3:
                            polygon = patches.Polygon(pts, closed=True, fill=False, 
                                                     edgecolor=struct['color'], linewidth=linewidth)
                            ax.add_patch(polygon)
                            all_pts.append(pts)
                
                elif plane == 'sagittal':
                    # Para vista sagital, filtramos por coordenada X
                    mask = np.abs(voxels[:, 2] - slice_idx) < 0.5
                    if np.any(mask):
                        # Usar Y y Z (transformado)
                        pts = voxels[mask][:, [0, 1]]
                        pts = np.fliplr(pts)  # Ajustar según sea necesario
                        
                        if len(pts) >= 3:
                            polygon = patches.Polygon(pts, closed=True, fill=False, 
                                                     edgecolor=struct['color'], linewidth=linewidth)
                            ax.add_patch(polygon)
                            all_pts.append(pts)

            # Dibujar nombre de la estructura una vez, en el centro del conjunto de puntos
            if show_names and all_pts and len(all_pts) > 0:
                try:
                    all_pts_concat = np.vstack(all_pts)
                    center = np.mean(all_pts_concat, axis=0)
                    ax.text(center[0], center[1], name, color=struct['color'], fontsize=8,
                           ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
                except:
                    pass  # Evitar errores si no podemos calcular el centro

    # Ajustar límites para coincidir bien
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(0, img.shape[0])

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

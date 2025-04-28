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
    spacing = list(getattr(sample, 'PixelSpacing', [1,1])) + [getattr(sample, 'SliceThickness', 1)]
    origin = getattr(sample, 'ImagePositionPatient', [0,0,0])
    direction = getattr(sample, 'ImageOrientationPatient', [1,0,0,0,1,0])

    direction_matrix = np.array([
        [direction[0], direction[3], 0],
        [direction[1], direction[4], 0],
        [direction[2], direction[5], 1]
    ])

    volume_info = {
        'spacing': spacing,
        'origin': origin,
        'direction': direction_matrix,
        'size': volume.shape
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
            
            structures[roi_names[roi.ReferencedROINumber]] = {'color': color, 'contours': contours}
        
        return structures
    except Exception as e:
        st.warning(f"Error leyendo estructura: {e}")
        return None

# --- Parte 3: Funciones de visualización ---

def patient_to_voxel(points, volume_info):
    """Convierte puntos de coordenadas paciente a coordenadas de voxel"""
    spacing = np.array(volume_info['spacing'])
    origin = np.array(volume_info['origin'])
    coords = (points - origin) / spacing
    return coords

def apply_window(img, window_center, window_width):
    """Aplica ventana de visualización (WW/WL)"""
    ww, wc = window_width, window_center
    img = np.clip(img, wc - ww/2, wc + ww/2)
    img = (img - img.min()) / (img.max() - img.min())
    return img

def draw_slice(volume, slice_idx, plane, structures, volume_info, window, linewidth=2, show_names=True, invert_colors=False):
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

    # Mostrar imagen
    ax.imshow(img, cmap='gray', origin='lower')

    # Dibujar contornos
    if structures:
        for name, struct in structures.items():
            all_pts = []  # para calcular centro global después
            for contour in struct['contours']:
                voxels = patient_to_voxel(contour['points'], volume_info)

                if plane == 'axial':
                    mask = np.isclose(voxels[:, 2], slice_idx, atol=0.5)
                    pts = voxels[mask][:, [0,1]]
                elif plane == 'coronal':
                    mask = np.isclose(voxels[:, 1], slice_idx, atol=0.5)
                    pts = voxels[mask][:, [0,2]]
                elif plane == 'sagittal':
                    mask = np.isclose(voxels[:, 0], slice_idx, atol=0.5)
                    pts = voxels[mask][:, [1,2]]

                if len(pts) >= 3:
                    polygon = patches.Polygon(pts, closed=True, fill=False, edgecolor=struct['color'], linewidth=linewidth)
                    ax.add_patch(polygon)
                    all_pts.append(pts)

            # Dibujar nombre una vez por estructura
            if show_names and all_pts:
                all_pts_concat = np.vstack(all_pts)
                center = np.mean(all_pts_concat, axis=0)
                ax.text(center[0], center[1], name, color=struct['color'], fontsize=8,
                        ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    # Ajustar límites para coincidir
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(0, img.shape[0])

    return fig



# --- Parte 4: Interfaz principal ---

if uploaded_file:
    temp_dir = extract_zip(uploaded_file)
    series_dict, structure_files = find_dicom_series(temp_dir)

    if series_dict:
        series_options = list(series_dict.keys())
        selected_series = st.sidebar.selectbox("Selecciona la serie", series_options)
        dicom_files = series_dict[selected_series]

        # Usar la función de carga del código original
        volume, volume_info = load_dicom_series(dicom_files)

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

            show_structures = st.sidebar.checkbox("Mostrar estructuras", value=False)
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

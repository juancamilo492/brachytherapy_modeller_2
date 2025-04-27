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

def load_image_series(file_list):
    """Carga una serie de imágenes DICOM como volumen 3D"""
    try:
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(file_list)
        image = reader.Execute()

        array = sitk.GetArrayFromImage(image)  # (slices, rows, cols)
        spacing = image.GetSpacing()           # (x, y, z)
        origin = image.GetOrigin()
        direction = image.GetDirection()

        volume_info = {
            'spacing': spacing,
            'origin': origin,
            'direction': direction,
            'size': array.shape
        }
        return array, volume_info
    except Exception as e:
        st.error(f"Error cargando imágenes DICOM: {e}")
        return None, None

def load_rtstruct(file_path):
    """Carga un archivo RTSTRUCT y extrae contornos"""
    try:
        struct = pydicom.dcmread(file_path, force=True)
        structures = {}

        # Mapeo de ROI Number a ROI Name
        roi_names = {}
        if hasattr(struct, 'StructureSetROISequence'):
            for roi in struct.StructureSetROISequence:
                roi_names[roi.ROINumber] = roi.ROIName

        # Extraer contornos
        if hasattr(struct, 'ROIContourSequence'):
            for roi in struct.ROIContourSequence:
                roi_number = roi.ReferencedROINumber
                name = roi_names.get(roi_number, f'ROI-{roi_number}')
                
                # Extraemos el color si está disponible
                color = np.array(roi.ROIDisplayColor) / 255.0 if hasattr(roi, 'ROIDisplayColor') else np.random.rand(3)
                
                contours = []
                if hasattr(roi, 'ContourSequence'):
                    for contour in roi.ContourSequence:
                        coords = np.array(contour.ContourData).reshape(-1, 3)  # (N, 3)
                        contours.append({'points': coords, 'z': np.mean(coords[:,2])})

                structures[name] = {'color': color, 'contours': contours}

        return structures
    except Exception as e:
        st.warning(f"Error leyendo estructura: {e}")
        return None

# --- Parte 3: Funciones de visualización ---

def apply_window(image, window_center, window_width):
    """Aplica ventana de visualización (WW/WL)"""
    img = image.astype(np.float32)
    min_val = window_center - window_width / 2
    max_val = window_center + window_width / 2
    img = np.clip((img - min_val) / (max_val - min_val), 0, 1)
    return img

def patient_to_voxel(points, volume_info):
    """Convierte puntos de coordenadas paciente a coordenadas de voxel"""
    spacing = np.array(volume_info['spacing'])
    origin = np.array(volume_info['origin'])
    coords = (points - origin) / spacing
    return coords

def draw_slice(volume, slice_idx, plane, structures, volume_info, window, linewidth=2, show_names=True, invert_colors=False):
    """Dibuja un corte 2D y superpone contornos"""
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.axis('off')

    if plane == 'axial':
        img = volume[slice_idx,:,:]
    elif plane == 'coronal':
        img = volume[:,slice_idx,:]
    elif plane == 'sagittal':
        img = volume[:,:,slice_idx]
    else:
        raise ValueError("Plano inválido")

    # Aplicar ventana
    ww, wc = window
    img = np.clip(img, wc - ww/2, wc + ww/2)
    img = (img - img.min()) / (img.max() - img.min())
    
    # Invertir colores si está activado
    if invert_colors:
        img = 1.0 - img

    ax.imshow(img, cmap='gray', origin='lower')

    if structures and show_names:
        for name, struct in structures.items():
            for contour in struct['contours']:
                voxels = patient_to_voxel(contour['points'], volume_info)
                if plane == 'axial':
                    mask = np.isclose(voxels[:,2], slice_idx, atol=1)
                    pts = voxels[mask][:, [1,0]]
                elif plane == 'coronal':
                    mask = np.isclose(voxels[:,1], slice_idx, atol=1)
                    pts = voxels[mask][:, [2,0]]
                elif plane == 'sagittal':
                    mask = np.isclose(voxels[:,0], slice_idx, atol=1)
                    pts = voxels[mask][:, [2,1]]

                if len(pts) >= 3:
                    polygon = patches.Polygon(pts, closed=True, fill=False, edgecolor=struct['color'], linewidth=linewidth)
                    ax.add_patch(polygon)
                    if show_names:
                        center = np.mean(pts, axis=0)
                        ax.text(center[0], center[1], name, color=struct['color'], fontsize=8, ha='center', va='center',
                                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    return fig

# --- Parte 4: Interfaz principal ---

if uploaded_file:
    temp_dir = extract_zip(uploaded_file)
    series_dict, structure_files = find_dicom_series(temp_dir)

    if series_dict:
        series_options = list(series_dict.keys())
        selected_series = st.sidebar.selectbox("Selecciona la serie", series_options)
        dicom_files = series_dict[selected_series]

        image_3d, volume_info = load_image_series(dicom_files)

        structures = None
        if structure_files:
            structures = load_rtstruct(structure_files[0])
            if structures:
                st.success(f"✅ Se cargaron {len(structures)} estructuras.")
            else:
                st.warning("⚠️ No se encontraron estructuras RTSTRUCT.")

        if image_3d is not None:
            st.sidebar.markdown('<p class="sidebar-title">Visualización</p>', unsafe_allow_html=True)

            max_axial = image_3d.shape[0] - 1
            max_coronal = image_3d.shape[1] - 1
            max_sagittal = image_3d.shape[2] - 1

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
                ["Cerebro (Brain)", "Pulmón (Lung)", "Hueso (Bone)", "Abdomen", "Mediastino (Mediastinum)",
                 "Hígado (Liver)", "Tejido blando (Soft Tissue)", "Columna blanda (Spine Soft)",
                 "Columna ósea (Spine Bone)", "Aire (Air)", "Grasa (Fat)", "Metal", "Personalizado"]
            )

            if window_option == "Cerebro (Brain)":
                window_center, window_width = 40, 80
            elif window_option == "Pulmón (Lung)":
                window_center, window_width = -600, 1500
            elif window_option == "Hueso (Bone)":
                window_center, window_width = 300, 1500
            elif window_option == "Abdomen":
                window_center, window_width = 60, 400
            elif window_option == "Mediastino (Mediastinum)":
                window_center, window_width = 40, 400
            elif window_option == "Hígado (Liver)":
                window_center, window_width = 70, 150
            elif window_option == "Tejido blando (Soft Tissue)":
                window_center, window_width = 50, 350
            elif window_option == "Columna blanda (Spine Soft)":
                window_center, window_width = 50, 350
            elif window_option == "Columna ósea (Spine Bone)":
                window_center, window_width = 300, 1500
            elif window_option == "Aire (Air)":
                window_center, window_width = -1000, 2000
            elif window_option == "Grasa (Fat)":
                window_center, window_width = -100, 200
            elif window_option == "Metal":
                window_center, window_width = 1000, 4000
            elif window_option == "Personalizado":
                window_center = st.sidebar.number_input("Window Center (WL)", value=40)
                window_width = st.sidebar.number_input("Window Width (WW)", value=400)

            show_structures = st.sidebar.checkbox("Mostrar estructuras", value=False)
            linewidth = st.sidebar.slider("Grosor líneas", 1, 8, 2)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("### Axial")
                fig_axial = draw_slice(
                    image_3d, axial_idx, 'axial', 
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
                    image_3d, coronal_idx, 'coronal',
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
                    image_3d, sagittal_idx, 'sagittal',
                    structures if show_structures else None,
                    volume_info,
                    (window_width, window_center),
                    linewidth=linewidth,
                    invert_colors=invert_colors
                )
                st.pyplot(fig_sagittal)
    else:
        st.warning("No se encontraron imágenes DICOM en el ZIP.")

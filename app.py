# --- Parte 1: Configuración inicial + carga de archivos ---

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
                contours = []

                if hasattr(roi, 'ContourSequence'):
                    for contour in roi.ContourSequence:
                        coords = np.array(contour.ContourData).reshape(-1, 3)  # (N, 3)
                        contours.append(coords)

                structures[name] = contours

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

def plot_slice(image_3d, volume_info, index, plane='axial', structures=None, window_center=40, window_width=400, show_structures=False):
    """Dibuja un slice específico de la imagen (axial, sagital o coronal)"""

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.axis('off')

    # Seleccionar el corte según la vista
    if plane == 'axial':
        img = image_3d[index, :, :]
    elif plane == 'sagittal':
        img = image_3d[:, :, index]
    elif plane == 'coronal':
        img = image_3d[:, index, :]
    else:
        raise ValueError(f"Plano no reconocido: {plane}")

    # Aplicar ventana
    img = apply_window(img, window_center, window_width)

    # Mostrar imagen base
    ax.imshow(img, cmap='gray', origin='lower')

    # Si se deben mostrar estructuras
    if show_structures and structures:
        plot_contours(ax, structures, index, volume_info, plane)

    return fig

def plot_contours(ax, structures, index, volume_info, plane):
    """Dibuja los contornos de las estructuras sobre el ax"""

    # Espaciado entre píxeles
    spacing_x, spacing_y, spacing_z = volume_info['spacing']
    size_slices, size_rows, size_cols = volume_info['size']

    for name, contours in structures.items():
        for contour in contours:
            if contour.shape[1] != 3:
                continue  # saltar contornos inválidos

            # Separar coordenadas
            xs, ys, zs = contour[:, 0], contour[:, 1], contour[:, 2]

            # Según el plano, seleccionar los puntos que caen en el corte actual
            if plane == 'axial':
                # Z es el índice del slice
                positions = zs / spacing_z
                if np.any(np.isclose(positions, index, atol=1)):
                    x = xs / spacing_x
                    y = ys / spacing_y
                    ax.plot(x, y, linewidth=1.5)
            elif plane == 'sagittal':
                positions = xs / spacing_x
                if np.any(np.isclose(positions, index, atol=1)):
                    y = ys / spacing_y
                    z = zs / spacing_z
                    ax.plot(y, z, linewidth=1.5)
            elif plane == 'coronal':
                positions = ys / spacing_y
                if np.any(np.isclose(positions, index, atol=1)):
                    x = xs / spacing_x
                    z = zs / spacing_z
                    ax.plot(x, z, linewidth=1.5)



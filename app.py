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


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
    """Carga una serie de imágenes DICOM como volumen 3D usando pydicom."""
    try:
        dicom_files = []
        for path in file_list:
            dcm = pydicom.dcmread(path, force=True)
            if hasattr(dcm, 'pixel_array'):
                dicom_files.append((path, dcm))

        if not dicom_files:
            st.error("No se encontraron imágenes DICOM válidas.")
            return None, None

        # Sort by InstanceNumber
        dicom_files.sort(key=lambda x: getattr(x[1], 'InstanceNumber', 0))

        # Validate shapes
        shape_counts = {}
        for _, dcm in dicom_files:
            shape = dcm.pixel_array.shape
            shape_counts[shape] = shape_counts.get(shape, 0) + 1
        best_shape = max(shape_counts, key=shape_counts.get)
        slices = [d[1].pixel_array for d in dicom_files if d[1].pixel_array.shape == best_shape]

        # Stack slices into a 3D array
        array = np.stack(slices)  # Shape: (Z, Y, X)

        # Extract metadata
        sample = dicom_files[0][1]
        spacing = list(getattr(sample, 'PixelSpacing', [1.0, 1.0])) + [getattr(sample, 'SliceThickness', 1.0)]
        origin = getattr(sample, 'ImagePositionPatient', [0, 0, 0])
        direction = getattr(sample, 'ImageOrientationPatient', [1, 0, 0, 0, 1, 0])

        # Convert direction to a list if it's a MultiValue object
        if isinstance(direction, pydicom.multival.MultiValue):
            direction = list(direction)

        # Ensure spacing values are valid
        spacing = [max(s, 0.1) for s in spacing]  # Avoid zero or negative spacing

        volume_info = {
            'spacing': spacing,
            'origin': origin,
            'direction': direction,
            'size': array.shape
        }

        # Log diagnostic information
        st.write(f"Loaded image series (pydicom):")
        st.write(f"- Array shape: {array.shape} (Z, Y, X)")
        st.write(f"- Spacing: {spacing} (X, Y, Z) mm")
        st.write(f"- Origin: {origin}")
        st.write(f"- Direction: {direction}")
        st.write(f"- Min/Max pixel values: {array.min()}, {array.max()}")

        # Warn about high anisotropy
        if spacing[2] / min(spacing[0], spacing[1]) > 5:
            st.warning(f"High anisotropy detected: Z-spacing ({spacing[2]:.2f} mm) is much larger than X/Y ({spacing[0]:.2f}, {spacing[1]:.2f} mm). Resampling recommended.")

        # Optional: Resample to isotropic voxels
        if spacing[0] != spacing[1] or spacing[1] != spacing[2]:
            # Convert array back to SimpleITK image for resampling
            image = sitk.GetImageFromArray(array)
            image.SetSpacing(spacing)
            image.SetOrigin(origin)
            # Extend direction to 3x3 matrix
            direction_3x3 = direction + [0, 0, 1]  # Now safe to concatenate
            image.SetDirection(direction_3x3)

            # Resample to isotropic spacing (use the smallest spacing)
            target_spacing = min(spacing)
            size = [int(s * sp / target_spacing) for s, sp in zip(array.shape[::-1], spacing)]  # (X, Y, Z)
            size = size[::-1]  # (Z, Y, X)
            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing([target_spacing] * 3)
            resampler.SetSize(size)
            resampler.SetOutputOrigin(image.GetOrigin())
            resampler.SetOutputDirection(image.GetDirection())
            resampler.SetInterpolator(sitk.sitkLinear)
            image = resampler.Execute(image)

            array = sitk.GetArrayFromImage(image)
            spacing = [target_spacing] * 3
            volume_info['spacing'] = spacing
            volume_info['size'] = array.shape
            st.info(f"Resampled to isotropic spacing: {target_spacing} mm, new shape: {array.shape}")

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
    """Aplica ventana de visualización (WW/WL) con manejo de valores extremos."""
    img = image.astype(np.float32)
    # Clip to a reasonable range for CT (-1000 to 3000 HU)
    img = np.clip(img, -1000, 3000)
    min_val = window_center - window_width / 2
    max_val = window_center + window_width / 2
    img = np.clip((img - min_val) / (max_val - min_val), 0, 1)
    return img

def plot_slice(image_3d, volume_info, index, plane='axial', structures=None, window_center=40, window_width=400, show_structures=False, invert_colors=False, debug_raw=False):
    """Dibuja un corte específico en el plano correcto."""
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.axis('off')

    # Get voxel spacing and size
    spacing_x, spacing_y, spacing_z = volume_info['spacing']
    size_slices, size_rows, size_cols = volume_info['size']

    # Validate index
    max_idx = {'axial': size_slices, 'coronal': size_rows, 'sagittal': size_cols}
    if index >= max_idx[plane]:
        st.error(f"Index {index} out of bounds for {plane} plane (max: {max_idx[plane] - 1})")
        return fig

    # Extract slice
    if plane == 'axial':
        slice_img = image_3d[index, :, :]  # Shape: (Y, X)
        aspect = spacing_x / spacing_y
    elif plane == 'coronal':
        slice_img = image_3d[:, index, :]  # Shape: (Z, X)
        slice_img = slice_img.T  # Transpose to (X, Z)
        aspect = spacing_z / spacing_x
    elif plane == 'sagittal':
        slice_img = image_3d[:, :, index]  # Shape: (Z, Y)
        slice_img = slice_img.T  # Transpose to (Y, Z)
        aspect = spacing_z / spacing_y
    else:
        raise ValueError(f"Plano no reconocido: {plane}")

    # Log slice information
    st.write(f"{plane} slice at index {index}:")
    st.write(f"- Shape: {slice_img.shape}")
    st.write(f"- Aspect ratio: {aspect:.2f}")
    st.write(f"- Min/Max values: {slice_img.min()}, {slice_img.max()}")

    # Check for valid slice data
    if slice_img.size == 0 or np.all(slice_img == 0):
        st.error(f"Invalid slice data for {plane} plane at index {index}")
        return fig

    if debug_raw:
        ax.imshow(slice_img, cmap='gray', origin='lower')
        st.write(f"Debug: Raw {plane} slice at index {index}, shape: {slice_img.shape}")
        return fig

    # Apply windowing
    img = apply_window(slice_img, window_center, window_width)

    # Invert colors if enabled
    if invert_colors:
        img = 1.0 - img

    # Display with aspect ratio
    ax.imshow(img, cmap='gray', origin='lower', aspect=aspect)

    # Draw contours if enabled
    if show_structures and structures:
        plot_contours(ax, structures, index, volume_info, plane)

    return fig

    # Log slice information
    st.write(f"{plane} slice at index {index}:")
    st.write(f"- Shape: {slice_img.shape}")
    st.write(f"- Aspect ratio: {aspect:.2f}")
    st.write(f"- Min/Max values: {slice_img.min()}, {slice_img.max()}")

    # Check for valid slice data
    if slice_img.size == 0 or np.all(slice_img == 0):
        st.error(f"Invalid slice data for {plane} plane at index {index}")
        return fig

    if debug_raw:
        ax.imshow(slice_img, cmap='gray', origin='lower')
        st.write(f"Debug: Raw {plane} slice at index {index}, shape: {slice_img.shape}")
        return fig

    # Apply windowing
    img = apply_window(slice_img, window_center, window_width)

    # Invert colors if enabled
    if invert_colors:
        img = 1.0 - img

    # Display with aspect ratio
    ax.imshow(img, cmap='gray', origin='lower', aspect=aspect)

    # Draw contours if enabled
    if show_structures and structures:
        plot_contours(ax, structures, index, volume_info, plane)

    return fig

def patient_to_voxel(points, volume_info):
    """Convierte puntos de coordenadas paciente a coordenadas de voxel."""
    spacing = np.array(volume_info['spacing'])
    origin = np.array(volume_info['origin'])
    coords = (points - origin) / spacing
    return coords

def plot_contours(ax, structures, index, volume_info, plane):
    """Dibuja los contornos de las estructuras sobre el ax."""
    spacing_x, spacing_y, spacing_z = volume_info['spacing']
    size_slices, size_rows, size_cols = volume_info['size']

    for name, contours in structures.items():
        for contour in contours:
            if contour.shape[1] != 3:
                continue  # Skip invalid contours

            # Convert physical coordinates to voxel indices
            voxels = patient_to_voxel(contour, volume_info)
            x_pixels, y_pixels, z_pixels = voxels[:, 0], voxels[:, 1], voxels[:, 2]

            # Select points based on the plane and slice index
            if plane == 'axial':
                mask = np.isclose(z_pixels, index, atol=1)
                pts = voxels[mask][:, [0, 1]]  # (X, Y)
            elif plane == 'coronal':
                mask = np.isclose(y_pixels, index, atol=1)
                pts = voxels[mask][:, [0, 2]]  # (X, Z)
            elif plane == 'sagittal':
                mask = np.isclose(x_pixels, index, atol=1)
                pts = voxels[mask][:, [1, 2]]  # (Y, Z)
            else:
                continue

            # Draw contour as a polygon if enough points
            if len(pts) >= 3:
                polygon = patches.Polygon(pts, closed=True, fill=False, edgecolor='red', linewidth=1.5)
                ax.add_patch(polygon)


# --- Parte 4: Interfaz principal ---

if uploaded_file:
    # Extraer y cargar los datos
    temp_dir = extract_zip(uploaded_file)
    series_dict, structure_files = find_dicom_series(temp_dir)

    if series_dict:
        # Selección de serie si hay varias
        series_options = list(series_dict.keys())
        selected_series = st.sidebar.selectbox("Selecciona la serie", series_options)
        dicom_files = series_dict[selected_series]

        # Cargar imágenes
        image_3d, volume_info = load_image_series(dicom_files)

        # Cargar estructuras si existen
        structures = None
        if structure_files:
            structures = load_rtstruct(structure_files[0])

        if image_3d is not None:
            # Sidebar: Configuración
            st.sidebar.markdown('<p class="sidebar-title">Visualización</p>', unsafe_allow_html=True)

            max_axial = image_3d.shape[0] - 1
            max_coronal = image_3d.shape[1] - 1
            max_sagittal = image_3d.shape[2] - 1


            st.sidebar.markdown("#### Selección de cortes")
            st.sidebar.markdown("#### Opciones avanzadas")
            sync_slices = st.sidebar.checkbox("Sincronizar cortes", value=True)
            invert_colors = st.sidebar.checkbox("Invertir colores (Negativo)", value=False)
            
            if sync_slices:
                unified_idx = st.sidebar.number_input(
                    "Corte (sincronizado)", min_value=0, max_value=max(max_axial, max_coronal, max_sagittal),
                    value=max_axial // 2, step=1
                )
                axial_idx = unified_idx
                coronal_idx = unified_idx
                sagittal_idx = unified_idx
            else:
                axial_idx = st.sidebar.number_input(
                    "Corte axial (Z)", min_value=0, max_value=max_axial, value=max_axial // 2, step=1
                )
                coronal_idx = st.sidebar.number_input(
                    "Corte coronal (Y)", min_value=0, max_value=max_coronal, value=max_coronal // 2, step=1
                )
                sagittal_idx = st.sidebar.number_input(
                    "Corte sagital (X)", min_value=0, max_value=max_sagittal, value=max_sagittal // 2, step=1
                )

            
            # Opciones de ventana predeterminadas
            window_option = st.sidebar.selectbox(
                "Tipo de ventana", 
                ["Cerebro (Brain)", "Pulmón (Lung)", "Hueso (Bone)", "Abdomen", "Mediastino (Mediastinum)", 
                 "Hígado (Liver)", "Tejido blando (Soft Tissue)", "Columna blanda (Spine Soft)", 
                 "Columna ósea (Spine Bone)", "Aire (Air)", "Grasa (Fat)", "Metal", "Personalizado"]
            )

            # Asignar valores predeterminados
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

            # Mostrar vistas
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("### Axial")
                fig_axial = plot_slice(
                    image_3d, volume_info, axial_idx,
                    plane='axial', structures=structures,
                    window_center=window_center, window_width=window_width,
                    show_structures=show_structures
                )
                st.pyplot(fig_axial)

            with col2:
                st.markdown("### Coronal")
                fig_coronal = plot_slice(
                    image_3d, volume_info, coronal_idx,
                    plane='coronal', structures=structures,
                    window_center=window_center, window_width=window_width,
                    show_structures=show_structures
                )
                st.pyplot(fig_coronal)

            with col3:
                st.markdown("### Sagital")
                fig_sagittal = plot_slice(
                    image_3d, volume_info, sagittal_idx,
                    plane='sagittal', structures=structures,
                    window_center=window_center, window_width=window_width,
                    show_structures=show_structures
                )
                st.pyplot(fig_sagittal)

    else:
        st.warning("No se encontraron imágenes DICOM en el ZIP.")



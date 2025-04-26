import os
import io
import zipfile
import tempfile
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import SimpleITK as sitk
import pydicom
import matplotlib.patches as patches

# ----------------------------
# Funciones principales
# ----------------------------

def load_dicom_series(directory):
    """Carga manualmente imágenes DICOM desde un directorio, tolerante a tamaños distintos"""
    dicom_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)
            try:
                dcm = pydicom.dcmread(path, force=True)
                if hasattr(dcm, 'pixel_array'):
                    dicom_files.append((path, dcm))
            except Exception:
                continue
    
    if not dicom_files:
        return None, None

    # Ordenar por InstanceNumber si existe
    dicom_files.sort(key=lambda x: getattr(x[1], 'InstanceNumber', 0))
    
    # Agrupar por tamaño
    shape_counts = {}
    for _, dcm in dicom_files:
        shape = dcm.pixel_array.shape
        shape_counts[shape] = shape_counts.get(shape, 0) + 1
    
    # Buscar el tamaño más común
    best_shape = max(shape_counts, key=shape_counts.get)

    # Solo usar imágenes del tamaño correcto
    slices = [d[1].pixel_array for d in dicom_files if d[1].pixel_array.shape == best_shape]

    # Crear volumen
    volume = np.stack(slices)

    # Obtener info de volumen
    sample = dicom_files[0][1]
    spacing = getattr(sample, 'PixelSpacing', [1,1]) + [getattr(sample, 'SliceThickness', 1)]
    origin = getattr(sample, 'ImagePositionPatient', [0,0,0])
    direction = getattr(sample, 'ImageOrientationPatient', [1,0,0,0,1,0])
    direction = np.array([
        direction[0], direction[3], 0,
        direction[1], direction[4], 0,
        direction[2], direction[5], 1
    ])
    
    volume_info = {
        'spacing': spacing,
        'origin': origin,
        'direction': direction,
        'sitk_image': None
    }
    return volume, volume_info



def load_rtstruct(path, volume_info):
    """Carga contornos RTSTRUCT"""
    struct = pydicom.dcmread(path)
    structures = {}
    if not hasattr(struct, 'ROIContourSequence'):
        return structures
    
    roi_names = {roi.ROINumber: roi.ROIName for roi in struct.StructureSetROISequence}
    for roi in struct.ROIContourSequence:
        color = np.array(roi.ROIDisplayColor) / 255.0 if hasattr(roi, 'ROIDisplayColor') else np.random.rand(3)
        contours = []
        for contour in roi.ContourSequence:
            pts = np.array(contour.ContourData).reshape(-1, 3)  # (N, 3)
            contours.append({'points': pts, 'z': np.mean(pts[:, 2])})
        structures[roi_names[roi.ReferencedROINumber]] = {'color': color, 'contours': contours}
    return structures

def patient_to_voxel_coords(points, volume_info):
    """Transforma coordenadas de paciente a índices de voxel"""
    image = volume_info['sitk_image']
    return [image.TransformPhysicalPointToIndex(tuple(pt)) for pt in points]

def draw_slice_with_structures(volume, slice_idx, plane, structures, volume_info, window, show_names=True):
    """Dibuja un corte específico y superpone estructuras"""
    fig, ax = plt.subplots(figsize=(8,8))
    plt.axis('off')
    
    if plane == 'Axial':
        img = volume[slice_idx,:,:]
    elif plane == 'Coronal':
        img = volume[:,slice_idx,:]
    elif plane == 'Sagital':
        img = volume[:,:,slice_idx]
    else:
        raise ValueError("Plano inválido")
    
    # Aplicar ventana
    ww, wc = window
    img = np.clip(img, wc - ww/2, wc + ww/2)
    img = (img - img.min()) / (img.max() - img.min())
    ax.imshow(img, cmap='gray', origin='lower')
    
    # Dibujar contornos
    if structures:
        for name, struct in structures.items():
            for contour in struct['contours']:
                voxels = np.array(patient_to_voxel_coords(contour['points'], volume_info))
                if plane == 'Axial':
                    mask = np.isclose(voxels[:,2], slice_idx, atol=1)
                    pts = voxels[mask][:, [1,0]]  # row, col
                elif plane == 'Coronal':
                    mask = np.isclose(voxels[:,1], slice_idx, atol=1)
                    pts = voxels[mask][:, [2,0]]  # slice, col
                elif plane == 'Sagital':
                    mask = np.isclose(voxels[:,0], slice_idx, atol=1)
                    pts = voxels[mask][:, [2,1]]  # slice, row
                
                if len(pts) >= 3:
                    polygon = patches.Polygon(pts, closed=True, fill=False, edgecolor=struct['color'], linewidth=2)
                    ax.add_patch(polygon)
                    if show_names:
                        center = np.mean(pts, axis=0)
                        ax.text(center[0], center[1], name, color=struct['color'], fontsize=8, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    return fig

# ----------------------------
# Streamlit App
# ----------------------------

st.set_page_config(layout="wide", page_title="Brachyanalysis V2")

st.title("🔷 Brachyanalysis - Visualizador DICOM Mejorado")

uploaded_zip = st.file_uploader("Sube tu archivo ZIP de DICOM + RTSTRUCT", type="zip")

if uploaded_zip:
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    volume, volume_info = load_dicom_series(temp_dir)
    
    struct_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(temp_dir) for f in filenames if f.endswith('.dcm')]
    structures = None
    for s in struct_files:
        try:
            dcm = pydicom.dcmread(s, stop_before_pixels=True)
            if getattr(dcm, 'Modality', '') == 'RTSTRUCT':
                structures = load_rtstruct(s, volume_info)
                break
        except:
            continue
    
    if volume is not None:
        mode = st.sidebar.selectbox("Vista:", ["Axial", "Coronal", "Sagital", "3 vistas"])
        ww = st.sidebar.slider("Ancho de ventana (WW)", 1, 4000, 350)
        wc = st.sidebar.slider("Centro de ventana (WC)", -1000, 1000, 50)
        show_structs = st.sidebar.checkbox("Mostrar estructuras", value=True)
        show_names = st.sidebar.checkbox("Mostrar nombres de estructuras", value=True)
        
        if mode != "3 vistas":
            if mode == "Axial":
                n = volume.shape[0]
            elif mode == "Coronal":
                n = volume.shape[1]
            elif mode == "Sagital":
                n = volume.shape[2]
            slice_idx = st.slider(f"Slice {mode}", 0, n-1, n//2)
            fig = draw_slice_with_structures(volume, slice_idx, mode, structures if show_structs else None, volume_info, (ww, wc), show_names)
            st.pyplot(fig)
        else:
            slices = [volume.shape[0]//2, volume.shape[1]//2, volume.shape[2]//2]
            planes = ["Axial", "Coronal", "Sagital"]
            cols = st.columns(3)
            for i, plane in enumerate(planes):
                with cols[i]:
                    st.write(f"**{plane}**")
                    fig = draw_slice_with_structures(volume, slices[i], plane, structures if show_structs else None, volume_info, (ww, wc), show_names)
                    st.pyplot(fig)
    else:
        st.error("No se encontró una serie DICOM válida en el ZIP.")

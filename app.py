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
from pydicom.dataset import Dataset
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.colors as mcolors

# Configuración de página
st.set_page_config(layout="wide", page_title="Brachyanalysis")

# CSS personalizado
st.markdown("""
<style>
    .main-header {color: #28aec5; text-align: center; font-size: 42px; margin-bottom: 20px; font-weight: bold;}
    .giant-title {color: #28aec5; text-align: center; font-size: 72px; margin: 30px 0; font-weight: bold; text-transform: uppercase; letter-spacing: 2px; text-shadow: 2px 2px 4px rgba(0,0,0,0.1);}
    .sub-header {color: #c0d711; font-size: 24px; margin-bottom: 15px; font-weight: bold;}
    .stButton>button {background-color: #28aec5; color: white; border: none; border-radius: 4px; padding: 8px 16px;}
    .stButton>button:hover {background-color: #1c94aa;}
    .info-box {background-color: rgba(40, 174, 197, 0.1); border-left: 3px solid #28aec5; padding: 10px; margin: 10px 0;}
    .success-box {background-color: rgba(192, 215, 17, 0.1); border-left: 3px solid #c0d711; padding: 10px; margin: 10px 0;}
    .sidebar-title {color: #28aec5; font-size: 28px; font-weight: bold; margin-bottom: 15px;}
</style>
""", unsafe_allow_html=True)

# Configuración de la barra lateral
st.sidebar.markdown('<p class="sidebar-title">Brachyanalysis</p>', unsafe_allow_html=True)
st.sidebar.markdown('<p class="sub-header">Visualizador DICOM</p>', unsafe_allow_html=True)

# Carga de archivos
uploaded_file = st.sidebar.file_uploader("Sube un archivo ZIP con tus archivos DICOM", type="zip")

# Funciones de procesamiento DICOM
def find_dicom_files(directory):
    """Busca archivos DICOM en el directorio y clasifica por tipo"""
    img_series = []
    struct_files = []
    dose_files = []
    plan_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # Intenta leer el header para identificar el tipo de archivo
                dcm = pydicom.dcmread(file_path, force=True, stop_before_pixels=True)
                
                # Clasificar por modalidad
                if hasattr(dcm, 'Modality'):
                    if dcm.Modality == 'RTSTRUCT':
                        struct_files.append(file_path)
                    elif dcm.Modality == 'RTDOSE':
                        dose_files.append(file_path)
                    elif dcm.Modality == 'RTPLAN':
                        plan_files.append(file_path)
                    elif dcm.Modality in ['CT', 'MR', 'PT', 'US']:
                        series_uid = dcm.SeriesInstanceUID if hasattr(dcm, 'SeriesInstanceUID') else 'unknown'
                        img_series.append((file_path, series_uid, dcm.Modality))
            except Exception:
                # Si no se puede leer como DICOM, ignorar
                continue
    
    # Agrupar por SeriesInstanceUID
    series_dict = {}
    for file_path, series_uid, modality in img_series:
        if series_uid not in series_dict:
            series_dict[series_uid] = {'files': [], 'modality': modality}
        series_dict[series_uid]['files'].append(file_path)
    
    # Ordenar archivos de cada serie
    for series in series_dict.values():
        try:
            # Intentar ordenar por número de instancia
            series['files'] = sorted(series['files'], 
                                    key=lambda x: pydicom.dcmread(x, force=True, stop_before_pixels=True).InstanceNumber 
                                    if hasattr(pydicom.dcmread(x, force=True, stop_before_pixels=True), 'InstanceNumber') 
                                    else 0)
        except:
            # Si falla, mantener el orden original
            pass
    
    return series_dict, struct_files, dose_files, plan_files

def get_image_orientation(dcm):
    """Extrae la orientación de la imagen"""
    if hasattr(dcm, 'ImageOrientationPatient'):
        return np.array(dcm.ImageOrientationPatient)
    else:
        return np.array([1, 0, 0, 0, 1, 0])  # Default orientation

def get_image_position(dcm):
    """Extrae la posición de la imagen"""
    if hasattr(dcm, 'ImagePositionPatient'):
        return np.array(dcm.ImagePositionPatient)
    else:
        return np.array([0, 0, 0])  # Default position

def get_pixel_spacing(dcm):
    """Extrae el espaciado de píxeles"""
    if hasattr(dcm, 'PixelSpacing'):
        return np.array(dcm.PixelSpacing)
    else:
        return np.array([1, 1])  # Default spacing

def load_image_series(files):
    """Carga una serie de archivos DICOM como una imagen 3D"""
    try:
        # Primero intentamos con SimpleITK
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(files)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        image = reader.Execute()
        
        # Extraer metadatos importantes para transformaciones
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        direction = image.GetDirection()
        
        # Crear matriz de transformación
        transform_matrix = np.array([
            [direction[0]*spacing[0], direction[3]*spacing[1], direction[6]*spacing[2], origin[0]],
            [direction[1]*spacing[0], direction[4]*spacing[1], direction[7]*spacing[2], origin[1]],
            [direction[2]*spacing[0], direction[5]*spacing[1], direction[8]*spacing[2], origin[2]],
            [0, 0, 0, 1]
        ])
        
        # Obtener array de la imagen
        img_array = sitk.GetArrayFromImage(image)
        
        # Crear información del volumen para mapeo
        volume_info = {
            'spacing': spacing,
            'origin': origin,
            'direction': direction,
            'transform': transform_matrix,
            'inverse_transform': np.linalg.inv(transform_matrix),
            'dimensions': img_array.shape
        }
        
        return reader, img_array, volume_info
    
    except Exception as e:
        st.sidebar.error(f"Error al cargar imagen con SimpleITK: {str(e)}")
        try:
            # Fallback: utilizar pydicom directamente
            first_dcm = pydicom.dcmread(files[0])
            
            # Extraer información de posición y orientación
            orientation = get_image_orientation(first_dcm)
            position = get_image_position(first_dcm)
            spacing_xy = get_pixel_spacing(first_dcm)
            
            # Calcular spacing Z basado en archivos consecutivos
            if len(files) > 1:
                second_dcm = pydicom.dcmread(files[1])
                second_position = get_image_position(second_dcm)
                spacing_z = np.linalg.norm(position - second_position)
            else:
                spacing_z = 1.0  # Default si solo hay un archivo
            
            # Cargar todos los slices
            slices = []
            for file in files:
                dcm = pydicom.dcmread(file)
                if hasattr(dcm, 'pixel_array'):
                    slices.append(dcm.pixel_array)
            
            # Convertir a array 3D
            if slices:
                img_array = np.stack(slices)
                
                # Calcular transformación
                row_vec = orientation[:3]
                col_vec = orientation[3:]
                normal_vec = np.cross(row_vec, col_vec)
                
                # Construir matriz de transformación
                transform_matrix = np.zeros((4, 4))
                transform_matrix[:3, 0] = row_vec * spacing_xy[0]
                transform_matrix[:3, 1] = col_vec * spacing_xy[1]
                transform_matrix[:3, 2] = normal_vec * spacing_z
                transform_matrix[:3, 3] = position
                transform_matrix[3, 3] = 1.0
                
                # Información del volumen
                volume_info = {
                    'spacing': (spacing_xy[0], spacing_xy[1], spacing_z),
                    'origin': position,
                    'direction': np.concatenate([row_vec, col_vec, normal_vec]),
                    'transform': transform_matrix,
                    'inverse_transform': np.linalg.inv(transform_matrix),
                    'dimensions': img_array.shape
                }
                
                return None, img_array, volume_info
            else:
                return None, None, None
        except Exception as e:
            st.sidebar.error(f"Error al cargar imagen con pydicom: {str(e)}")
            return None, None, None

def load_structure_file(file_path, volume_info=None):
    """Carga un archivo RTStruct y extrae las estructuras"""
    try:
        struct = pydicom.dcmread(file_path, force=True)
        
        if not hasattr(struct, 'ROIContourSequence'):
            st.sidebar.warning("Archivo RTStruct no contiene contornos")
            return None
        
        structures = {}
        roi_names = {}
        
        # Mapear ROI números a nombres
        if hasattr(struct, 'StructureSetROISequence'):
            for roi in struct.StructureSetROISequence:
                if hasattr(roi, 'ROINumber') and hasattr(roi, 'ROIName'):
                    roi_names[roi.ROINumber] = roi.ROIName
        
        # Verificar si hay referencia a serie de imagen
        referenced_frame_uid = None
        if hasattr(struct, 'ReferencedFrameOfReferenceSequence'):
            for ref_frame in struct.ReferencedFrameOfReferenceSequence:
                if hasattr(ref_frame, 'RTReferencedStudySequence'):
                    for ref_study in ref_frame.RTReferencedStudySequence:
                        if hasattr(ref_study, 'RTReferencedSeriesSequence'):
                            for ref_series in ref_study.RTReferencedSeriesSequence:
                                if hasattr(ref_series, 'SeriesInstanceUID'):
                                    referenced_frame_uid = ref_series.SeriesInstanceUID
        
        # Extraer contornos
        for roi in struct.ROIContourSequence:
            if not hasattr(roi, 'ContourSequence'):
                continue
                
            roi_number = roi.ReferencedROINumber if hasattr(roi, 'ReferencedROINumber') else 0
            name = roi_names.get(roi_number, f"ROI-{roi_number}")
            
            # Obtener color si está disponible
            if hasattr(roi, 'ROIDisplayColor'):
                color = [float(c)/255 for c in roi.ROIDisplayColor]
            else:
                # Generar un color aleatorio pero reproducible
                hash_val = hash(name) % 1000
                r = (hash_val % 255) / 255.0
                g = ((hash_val * 3) % 255) / 255.0
                b = ((hash_val * 7) % 255) / 255.0
                color = [r, g, b]
            
            contours = []
            for contour in roi.ContourSequence:
                if not hasattr(contour, 'ContourData') or not hasattr(contour, 'ContourGeometricType'):
                    continue
                    
                points = contour.ContourData
                if len(points) >= 6:  # Asegurar que hay al menos 2 puntos (x,y,z)
                    # Agrupar puntos en coordenadas (x,y,z)
                    coords = np.array([(points[i], points[i+1], points[i+2]) 
                                     for i in range(0, len(points), 3)])
                    
                    # Identificar el plano Z de este contorno
                    z_values = coords[:, 2]
                    z_plane = np.mean(z_values)
                    
                    contours.append({
                        'coords': coords,
                        'z': z_plane
                    })
            
            structures[name] = {
                'contours': contours,
                'color': color,
                'referenced_uid': referenced_frame_uid
            }
        
        return structures
        
    except Exception as e:
        st.sidebar.error(f"Error al cargar estructuras: {str(e)}")
        st.sidebar.write(f"Detalles: {e}")
        return None

def apply_window_level(image, window_width, window_center):
    """Aplica ventana y nivel a la imagen"""
    image_float = image.astype(float)
    min_value = window_center - window_width/2.0
    max_value = window_center + window_width/2.0
    image_windowed = np.clip(image_float, min_value, max_value)
    if max_value != min_value:
        image_windowed = (image_windowed - min_value) / (max_value - min_value)
    else:
        image_windowed = np.zeros_like(image_float)
    return image_windowed

def patient_to_image_coords(patient_points, volume_info):
    """Convierte coordenadas de paciente a coordenadas de imagen"""
    # Asegurar que la entrada es un array numpy
    pts = np.array(patient_points)
    
    # Añadir columna de 1s para transformación homogénea
    if pts.shape[1] == 3:
        pts_homog = np.column_stack([pts, np.ones(pts.shape[0])])
    else:
        pts_homog = pts
    
    # Aplicar transformación inversa
    img_points_homog = np.dot(volume_info['inverse_transform'], pts_homog.T).T
    
    # Obtener coordenadas de imagen (sin la última columna)
    img_points = img_points_homog[:, :3]
    
    # Redondear a índices enteros
    img_indices = np.round(img_points).astype(int)
    
    # Convertir a formato (slice, row, col) desde (x, y, z)
    # En la mayoría de los casos necesitamos cambiar el orden de los ejes
    dims = volume_info['dimensions']
    # Esto depende de la orientación específica, ajustar según sea necesario
    indices_swapped = np.zeros_like(img_indices)
    
    # Por defecto: x->col, y->row, z->slice
    indices_swapped[:, 0] = dims[1] - img_indices[:, 1]  # row (invertido para visualización correcta)
    indices_swapped[:, 1] = img_indices[:, 0]           # col
    indices_swapped[:, 2] = dims[0] - img_indices[:, 2]  # slice (posiblemente invertido)
    
    return indices_swapped

def get_slice_z_position(slice_idx, volume_info):
    """Obtiene la posición Z del slice en el espacio del paciente"""
    # Crear punto en coordenadas de imagen en el centro del slice
    dims = volume_info['dimensions']
    image_point = np.array([dims[2]//2, dims[1]//2, slice_idx, 1])
    
    # Transformar a coordenadas del paciente
    patient_point = np.dot(volume_info['transform'], image_point)
    
    return patient_point[2]  # Valor Z

def plot_with_structures(vol, slice_ix, window_width, window_center, 
                         structures=None, volume_info=None, z_tolerance=3.0,
                         show_struct_names=True):
    """Dibuja un slice con estructuras sobrepuestas"""
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.axis('off')
    
    # Mostrar imagen base
    selected_slice = vol[slice_ix, :, :]
    windowed_slice = apply_window_level(selected_slice, window_width, window_center)
    ax.imshow(windowed_slice, origin='lower', cmap='gray')
    
    # Si no hay estructuras o info del volumen, salir
    if not structures or not volume_info:
        return fig
    
    # Obtener posición Z del slice actual en coords del paciente
    z_pos = get_slice_z_position(slice_ix, volume_info)
    
    # Para mostrar nombres de estructuras
    struct_names_drawn = set()
    
    # Dibujar estructuras
    for name, struct in structures.items():
        color = struct['color']
        
        for contour in struct['contours']:
            # Verificar si el contorno está cerca del slice actual
            if abs(contour['z'] - z_pos) <= z_tolerance:
                # Convertir contorno a coordenadas de imagen
                pts = contour['coords']
                img_points = patient_to_image_coords(pts, volume_info)
                
                # Extraer solo las filas y columnas (ignorando Z para dibujo 2D)
                points_2d = img_points[:, :2]
                
                # Crear y dibujar polígono si hay suficientes puntos
                if len(points_2d) > 2:
                    polygon = patches.Polygon(points_2d, 
                                             closed=True, 
                                             fill=False, 
                                             edgecolor=color, 
                                             linewidth=2)
                    ax.add_patch(polygon)
                    
                    # Mostrar nombre si aún no se ha mostrado para esta estructura
                    if show_struct_names and name not in struct_names_drawn:
                        # Usar el centro del contorno para posicionar el texto
                        center = np.mean(points_2d, axis=0)
                        ax.text(center[0], center[1], name, color=color, 
                                fontsize=9, ha='center', va='center',
                                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                        struct_names_drawn.add(name)
    
    return fig

# Presets de ventana
radiant_presets = {
    "Default window": (0, 0),
    "CT Abdomen": (350, 50),
    "CT Angio": (600, 300),
    "CT Bone": (2000, 350),
    "CT Brain": (80, 40),
    "CT Chest": (350, 40),
    "CT Lungs": (1500, -600),
    "Negative": (0, 0),
    "Custom window": (0, 0)
}

# Procesar archivos subidos
if uploaded_file is not None:
    # Crear directorio temporal
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Extraer ZIP
        with zipfile.ZipFile(io.BytesIO(uploaded_file.read()), 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        st.sidebar.markdown('<div class="success-box">Archivos extraídos correctamente.</div>', unsafe_allow_html=True)
        
        # Buscar archivos DICOM e identificar tipos
        with st.spinner('Buscando archivos DICOM...'):
            series_dict, struct_files, dose_files, plan_files = find_dicom_files(temp_dir)
        
        if not series_dict:
            st.sidebar.error("No se encontraron imágenes DICOM válidas.")
        else:
            # Mostrar series encontradas
            st.sidebar.markdown(f'<div class="info-box">Se encontraron {len(series_dict)} series de imágenes</div>', unsafe_allow_html=True)
            
            # Seleccionar serie
            series_options = [f"{files['modality']} Serie {i+1}: {uid[:8]}... ({len(files['files'])} imágenes)" 
                            for i, (uid, files) in enumerate(series_dict.items())]
            
            selected_series_option = st.sidebar.selectbox("Seleccionar serie:", series_options)
            selected_idx = series_options.index(selected_series_option)
            selected_series_uid = list(series_dict.keys())[selected_idx]
            
            # Cargar imágenes
            reader, img, volume_info = load_image_series(series_dict[selected_series_uid]['files'])
            
            if img is not None:
                n_slices = img.shape[0]
                slice_ix = st.sidebar.slider('Seleccionar corte', 0, n_slices - 1, int(n_slices/2))
                output = st.sidebar.radio('Visualización', ['Imagen', 'Metadatos'], index=0)
                
                # Cargar estructuras si existen
                structures = None
                if struct_files:
                    st.sidebar.markdown(f'<div class="info-box">Archivos de estructuras: {len(struct_files)}</div>', unsafe_allow_html=True)
                    show_structures = st.sidebar.checkbox("Mostrar estructuras", value=True)
                    show_names = st.sidebar.checkbox("Mostrar nombres", value=True)
                    z_tolerance = st.sidebar.slider("Tolerancia de plano (mm)", 0.5, 10.0, 3.0)
                    
                    if show_structures:
                        for struct_file in struct_files:
                            try:
                                structures = load_structure_file(struct_file, volume_info)
                                if structures:
                                    st.sidebar.success(f"Estructuras cargadas: {len(structures)} ROIs")
                                    # Mostrar lista de estructuras encontradas
                                    struct_names = list(structures.keys())
                                    if len(struct_names) > 0:
                                        st.sidebar.info(f"Nombres: {', '.join(struct_names[:5])}" + 
                                                      ("..." if len(struct_names) > 5 else ""))
                                    break  # Usar el primer archivo de estructuras válido
                            except Exception as e:
                                st.sidebar.warning(f"No se pudo leer estructura: {e}")
                
                # Ajustes de ventana para visualización
                if output == 'Imagen':
                    min_val = float(img.min())
                    max_val = float(img.max())
                    range_val = max_val - min_val
                    
                    default_window_width = range_val
                    default_window_center = min_val + (range_val / 2)
                    
                    # Actualizar presets automáticos
                    radiant_presets["Default window"] = (default_window_width, default_window_center)
                    
                    selected_preset = st.sidebar.selectbox(
                        "Presets radiológicos",
                        list(radiant_presets.keys())
                    )
                    
                    # Inicializar valores de ventana
                    window_width, window_center = radiant_presets[selected_preset]
                    is_negative = selected_preset == "Negative"
                    
                    if is_negative:
                        window_width = default_window_width
                        window_center = default_window_center
                    
                    # Ajustes personalizados
                    if selected_preset == "Custom window":
                        st.sidebar.markdown('<p class="sub-header">Ajustes personalizados</p>', unsafe_allow_html=True)
                        st.sidebar.markdown(f"**Rango de valores:** {min_val:.1f} a {max_val:.1f}")
                        
                        col1, col2 = st.sidebar.columns(2)
                        with col1:
                            window_width = float(st.number_input("Ancho (WW)", min_value=1.0, max_value=range_val * 2, value=float(default_window_width), format="%.1f"))
                        with col2:
                            window_center = float(st.number_input("Centro (WL)", min_value=min_val - range_val, max_value=max_val + range_val, value=float(default_window_center), format="%.1f"))
    
    except Exception as e:
        st.sidebar.error(f"Error al procesar archivos: {str(e)}")
        img = None

# Visualización principal
st.markdown('<p class="giant-title">Brachyanalysis</p>', unsafe_allow_html=True)

if 'img' in locals() and img is not None:
    if output == 'Imagen':
        st.markdown('<p class="sub-header">Visualización DICOM</p>', unsafe_allow_html=True)
        
        # Dibujar imagen con/sin estructuras
        if 'is_negative' in locals() and is_negative:
            # Invertir la imagen
            fig, ax = plt.subplots(figsize=(12, 10))
            plt.axis('off')
            selected_slice = img[slice_ix, :, :]
            windowed_slice = apply_window_level(selected_slice, window_width, window_center)
            windowed_slice = 1.0 - windowed_slice  # Invertir
            ax.imshow(windowed_slice, origin='lower', cmap='gray')
            st.pyplot(fig)
        else:
            # Imagen normal con estructuras si hay
            if ('structures' in locals() and structures and 'volume_info' in locals() and 
                'show_structures' in locals() and show_structures):
                fig = plot_with_structures(img, slice_ix, window_width, window_center, 
                                          structures, volume_info, 
                                          z_tolerance=z_tolerance if 'z_tolerance' in locals() else 3.0,
                                          show_struct_names='show_names' in locals() and show_names)
            else:
                fig, ax = plt.subplots(figsize=(12, 10))
                plt.axis('off')
                selected_slice = img[slice_ix, :, :]
                windowed_slice = apply_window_level(selected_slice, window_width, window_center)
                ax.imshow(windowed_slice, origin='lower', cmap='gray')
            st.pyplot(fig)
        
        # Información adicional
        info_cols = st.columns(6)
        with info_cols[0]:
            st.markdown(f"**Dimensiones:** {img.shape[1]} x {img.shape[2]} px")
        with info_cols[1]:
            st.markdown(f"**Total cortes:** {n_slices}")
        with info_cols[2]:
            st.markdown(f"**Corte actual:** {slice_ix + 1}")
        with info_cols[3]:
            st.markdown(f"**Min/Max:** {img[slice_ix].min():.1f} / {img[slice_ix].max():.1f}")
        with info_cols[4]:
            st.markdown(f"**Ancho (WW):** {window_width:.1f}")
        with info_cols[5]:
            st.markdown(f"**Centro (WL):** {window_center:.1f}")
            
    elif output == 'Metadatos' and 'reader' in locals() and reader is not None:
        st.markdown('<p class="sub-header">Metadatos DICOM</p>', unsafe_allow_html=True)
        try:
            metadata = dict()
            for k in reader.GetMetaDataKeys(slice_ix):
                metadata[k] = reader.GetMetaData(slice_ix, k)
            df = pd.DataFrame.from_dict(metadata, orient='index', columns=['Valor'])
            st.dataframe(df, height=600)
        except Exception as e:
            # Fallback a PyDICOM para metadatos
            try:
                dcm_file = series_dict[selected_series_uid]['files'][slice_ix]
                dcm = pydicom.dcmread(dcm_file)
                metadata = {f"{tag.name} ({tag.tag})": str(tag.value) for tag in dcm}
                df = pd.DataFrame.from_dict(metadata, orient='index', columns=['Valor'])
                st.dataframe(df, height=600)
            except Exception as e2:
                st.error(f"Error al leer metadatos: {str(e2)}")
else:
    # Página de inicio
    st.markdown('<p class="sub-header">Visualizador de imágenes DICOM</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; padding: 40px; margin-top: 10px;">
        <h2 style="color: #28aec5; margin-top: 20px;">Carga un archivo ZIP con tus imágenes DICOM</h2>
        <p style="font-size: 18px; margin-top: 10px;">Utiliza el panel lateral para subir tus archivos y visualizarlos</p>
    </div>
    """, unsafe_allow_html=True)

# Pie de página
st.markdown("""
<hr style="margin-top: 30px;">
<div style="text-align: center; color: #28aec5; font-size: 14px;">
    Brachyanalysis - Visualizador de imágenes DICOM
</div>
""", unsafe_allow_html=True)

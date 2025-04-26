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
    
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # Intenta leer el header para identificar el tipo de archivo
                dcm = pydicom.dcmread(file_path, force=True, stop_before_pixels=True)
                
                # Identificar archivos RTStruct
                if hasattr(dcm, 'Modality') and dcm.Modality == 'RTSTRUCT':
                    struct_files.append(file_path)
                # Para otros tipos de DICOM (imágenes)
                elif hasattr(dcm, 'SOPClassUID'):
                    # Solo agregar si parece ser una imagen
                    if (hasattr(dcm, 'Modality') and 
                        dcm.Modality in ['CT', 'MR', 'PT', 'US']):
                        series_uid = dcm.SeriesInstanceUID if hasattr(dcm, 'SeriesInstanceUID') else 'unknown'
                        img_series.append((file_path, series_uid, dcm.Modality if hasattr(dcm, 'Modality') else 'unknown'))
            except Exception:
                # Si no se puede leer como DICOM, ignorar
                continue
    
    # Agrupar por SeriesInstanceUID
    series_dict = {}
    for file_path, series_uid, modality in img_series:
        if series_uid not in series_dict:
            series_dict[series_uid] = {'files': [], 'modality': modality}
        series_dict[series_uid]['files'].append(file_path)
    
    return series_dict, struct_files

def load_image_series(files):
    """Carga una serie de archivos DICOM como una imagen 3D"""
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    try:
        image = reader.Execute()
        return reader, sitk.GetArrayFromImage(image)
    except Exception as e:
        st.sidebar.error(f"Error al cargar imagen: {str(e)}")
        return None, None

def load_structure_file(file_path, img_data=None):
    """Carga un archivo RTStruct y extrae las estructuras"""
    try:
        struct = pydicom.dcmread(file_path, force=True)
        
        if not hasattr(struct, 'ROIContourSequence'):
            return None
        
        structures = {}
        roi_names = {}
        
        # Mapear ROI números a nombres
        if hasattr(struct, 'StructureSetROISequence'):
            for roi in struct.StructureSetROISequence:
                if hasattr(roi, 'ROINumber') and hasattr(roi, 'ROIName'):
                    roi_names[roi.ROINumber] = roi.ROIName
        
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
                color = [1.0, 0.0, 0.0]  # Rojo por defecto
            
            contours = []
            for contour in roi.ContourSequence:
                if not hasattr(contour, 'ContourData') or not hasattr(contour, 'ContourGeometricType'):
                    continue
                    
                points = contour.ContourData
                if contour.ContourGeometricType == 'CLOSED_PLANAR':
                    # Agrupar puntos en coordenadas (x,y,z)
                    coords = [(points[i], points[i+1], points[i+2]) 
                              for i in range(0, len(points), 3)]
                    
                    contours.append({
                        'coords': np.array(coords),
                        'z': coords[0][2] if coords else 0  # Coord Z del primer punto
                    })
            
            structures[name] = {
                'contours': contours,
                'color': color
            }
        
        return structures
        
    except Exception as e:
        st.sidebar.error(f"Error al cargar estructuras: {str(e)}")
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

def plot_with_structures(vol, slice_ix, window_width, window_center, structures=None, image_to_patient=None):
    """Dibuja un slice con estructuras sobrepuestas"""
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.axis('off')
    
    # Mostrar imagen base
    selected_slice = vol[slice_ix, :, :]
    windowed_slice = apply_window_level(selected_slice, window_width, window_center)
    ax.imshow(windowed_slice, origin='lower', cmap='gray')
    
    # Dibujar estructuras si hay
    if structures and image_to_patient:
        # Obtener posición Z del slice actual en coords del paciente
        z_pos = image_to_patient(0, 0, slice_ix)[2]
        z_tolerance = 2.0  # Tolerancia en mm
        
        for name, struct in structures.items():
            color = struct['color']
            
            for contour in struct['contours']:
                # Solo dibujar si está cerca del slice actual
                if abs(contour['z'] - z_pos) <= z_tolerance:
                    points = contour['coords']
                    
                    # Convertir puntos a coordenadas de imagen
                    image_points = []
                    for p in points:
                        i, j, _ = patient_to_image(p, image_to_patient)
                        image_points.append((j, i))  # Note: j,i order for display
                    
                    # Crear y dibujar polígono
                    if len(image_points) > 2:
                        polygon = patches.Polygon(image_points, 
                                                 closed=True, 
                                                 fill=False, 
                                                 edgecolor=color, 
                                                 linewidth=2)
                        ax.add_patch(polygon)
    
    return fig

def patient_to_image(patient_point, image_to_patient):
    """Convierte coordenadas de paciente a índices de imagen"""
    # Esta es una aproximación simple - para una implementación completa
    # necesitarías usar la matriz de transformación inversa
    px, py, pz = patient_point
    
    # Para simplificar, asumimos que las coordenadas están ya alineadas
    # En un caso real, necesitarías usar la matriz de transformación inversa
    origin = image_to_patient(0, 0, 0)
    spacing = [1.0, 1.0, 1.0]  # Esto debe venir de los metadatos
    
    # Calcular índices
    i = int(round((px - origin[0]) / spacing[0]))
    j = int(round((py - origin[1]) / spacing[1]))
    k = int(round((pz - origin[2]) / spacing[2]))
    
    return i, j, k

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
            series_dict, struct_files = find_dicom_files(temp_dir)
        
        if not series_dict:
            st.sidebar.error("No se encontraron imágenes DICOM válidas.")
        else:
            # Mostrar series encontradas
            st.sidebar.markdown(f'<div class="info-box">Se encontraron {len(series_dict)} series de imágenes</div>', unsafe_allow_html=True)
            
            # Seleccionar serie
            series_options = [f"{modality} Serie {i+1}: {uid[:8]}... ({len(files['files'])} imágenes)" 
                            for i, (uid, files) in enumerate(series_dict.items()) 
                            for modality in [files['modality']]]
            
            selected_series_option = st.sidebar.selectbox("Seleccionar serie:", series_options)
            selected_idx = series_options.index(selected_series_option)
            selected_series_uid = list(series_dict.keys())[selected_idx]
            
            # Cargar imágenes
            reader, img = load_image_series(series_dict[selected_series_uid]['files'])
            
            if img is not None:
                n_slices = img.shape[0]
                slice_ix = st.sidebar.slider('Seleccionar corte', 0, n_slices - 1, int(n_slices/2))
                output = st.sidebar.radio('Visualización', ['Imagen', 'Metadatos'], index=0)
                
                # Cargar estructuras si existen
                structures = None
                if struct_files:
                    st.sidebar.markdown(f'<div class="info-box">Archivos de estructuras: {len(struct_files)}</div>', unsafe_allow_html=True)
                    show_structures = st.sidebar.checkbox("Mostrar estructuras", value=True)
                    
                    if show_structures:
                        for struct_file in struct_files:
                            try:
                                structures = load_structure_file(struct_file)
                                if structures:
                                    break  # Usar el primer archivo de estructuras válido
                            except Exception as e:
                                st.sidebar.warning(f"No se pudo leer estructura: {e}")
                
                # Definir función de transformación simple para mapeo
                def image_to_patient(i, j, k):
                    # Leer origen y espaciado desde los metadatos si es posible
                    # Esta es una aproximación simplificada
                    try:
                        origin = [0, 0, 0]
                        spacing = [1, 1, 1]
                        x = origin[0] + i * spacing[0]
                        y = origin[1] + j * spacing[1]
                        z = origin[2] + k * spacing[2]
                        return (x, y, z)
                    except:
                        return (i, j, k)  # Fallback
                
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
            if 'structures' in locals() and structures and 'image_to_patient' in locals():
                fig = plot_with_structures(img, slice_ix, window_width, window_center, structures, image_to_patient)
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
            st.error(f"Error al leer metadatos: {str(e)}")
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

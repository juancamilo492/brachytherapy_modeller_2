import os
import io
import zipfile
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import monai
from monai.networks.nets import UNet
from monai.transforms import (
    AddChannel, ScaleIntensity, ToTensor, Compose, LoadImage,
    Orientation, Spacing, Resize, AsDiscrete
)
from monai.transforms.compose import MapTransform
from monai.data import Dataset, DataLoader
from monai.inferers import sliding_window_inference
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import binary_erosion, binary_dilation

class AddChanneld(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = AddChannel()(d[key])
        return d

class ScaleIntensityd(MapTransform):
    def __init__(self, keys, minv=0.0, maxv=1.0):
        super().__init__(keys)
        self.minv = minv
        self.maxv = maxv
    
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = ScaleIntensity(minv=self.minv, maxv=self.maxv)(d[key])
        return d

class ToTensord(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = ToTensor()(d[key])
        return d


# Configuración de página y estilo
st.set_page_config(layout="wide", page_title="Brachyanalysis")

# CSS personalizado (se mantiene el mismo)
st.markdown("""
<style>
    .main-header {
        color: #28aec5;
        text-align: center;
        font-size: 42px;
        margin-bottom: 20px;
        font-weight: bold;
    }
    .giant-title {
        color: #28aec5;
        text-align: center;
        font-size: 72px;
        margin: 30px 0;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        color: #c0d711;
        font-size: 24px;
        margin-bottom: 15px;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #28aec5;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
    }
    .stButton>button:hover {
        background-color: #1c94aa;
    }
    .info-box {
        background-color: rgba(40, 174, 197, 0.1);
        border-left: 3px solid #28aec5;
        padding: 10px;
        margin: 10px 0;
    }
    .success-box {
        background-color: rgba(192, 215, 17, 0.1);
        border-left: 3px solid #c0d711;
        padding: 10px;
        margin: 10px 0;
    }
    .plot-container {
        border: 2px solid #c0d711;
        border-radius: 5px;
        padding: 10px;
        margin-top: 20px;
    }
    div[data-baseweb="select"] {
        border-radius: 4px;
        border-color: #28aec5;
    }
    div[data-baseweb="slider"] > div {
        background-color: #c0d711 !important;
    }
    /* Estilos para radio buttons */
    div.stRadio > div[role="radiogroup"] > label {
        background-color: rgba(40, 174, 197, 0.1);
        margin-right: 10px;
        padding: 5px 15px;
        border-radius: 4px;
    }
    div.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {
        background-color: #28aec5;
    }
    .upload-section {
        background-color: rgba(40, 174, 197, 0.05);
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .stUploadButton>button {
        background-color: #c0d711;
        color: #1e1e1e;
        font-weight: bold;
    }
    .sidebar-title {
        color: #28aec5;
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 15px;
    }
    .control-section {
        background-color: rgba(40, 174, 197, 0.05);
        padding: 15px;
        border-radius: 8px;
        margin-top: 15px;
    }
    .input-row {
        display: flex;
        justify-content: space-between;
        gap: 10px;
        margin-bottom: 10px;
    }
    .segmentation-control {
        background-color: rgba(192, 215, 17, 0.1);
        padding: 15px;
        border-radius: 8px;
        margin-top: 15px;
    }
    .tabs-custom {
        margin-bottom: 20px;
    }
    .tabs-custom button {
        background-color: #f0f0f0;
        border-radius: 5px 5px 0 0;
    }
    .tabs-custom button[aria-selected="true"] {
        background-color: #28aec5;
        color: white;
    }
    .needle-planning {
        background-color: rgba(40, 174, 197, 0.05);
        padding: 20px;
        border-radius: 8px;
        margin: 20px 0;
    }
    .template-preview {
        border: 2px solid #c0d711;
        border-radius: 8px;
        padding: 15px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Definición del modelo U-Net para segmentación de tumores cervicales
class CervixTumorUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=2):
        super(CervixTumorUNet, self).__init__()
        self.model = UNet(
            spatial_dims=3,
            in_channels=n_channels,
            out_channels=n_classes,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2
        )
    
    def forward(self, x):
        return self.model(x)

# Definición del modelo para la segmentación multi-órganos (OARs)
class OARsSegmentationModel(nn.Module):
    def __init__(self, n_channels=1, n_classes=5):  # 5 clases: fondo, vejiga, recto, intestino delgado, colon
        super(OARsSegmentationModel, self).__init__()
        self.model = UNet(
            spatial_dims=3,
            in_channels=n_channels,
            out_channels=n_classes,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2
        )
    
    def forward(self, x):
        return self.model(x)

# Función para cargar los modelos pre-entrenados
@st.cache_resource
def load_models():
    # Modelo para tumores cervicales
    tumor_model = CervixTumorUNet()
    try:
        # En una implementación real, cargarías los pesos pre-entrenados
        # tumor_model.load_state_dict(torch.load("tumor_model_weights.pth"))
        pass
    except:
        st.warning("Modelo de segmentación de tumores simulado (no se cargaron pesos reales)")

    # Modelo para órganos de riesgo
    oars_model = OARsSegmentationModel()
    try:
        # En una implementación real, cargarías los pesos pre-entrenados
        # oars_model.load_state_dict(torch.load("oars_model_weights.pth"))
        pass
    except:
        st.warning("Modelo de segmentación de OARs simulado (no se cargaron pesos reales)")
    
    return tumor_model, oars_model

# Pipeline de transformaciones para preprocesar la imagen para segmentación
def get_transform_pipeline():
    return Compose([
        AddChanneld(keys=["image"]),
        ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        Resized(keys=["image"], spatial_size=(128, 128, 128)),
        ToTensord(keys=["image"])
    ])

# Función para buscar recursivamente archivos DICOM en un directorio
def find_dicom_series(directory):
    """Busca recursivamente series DICOM en el directorio y sus subdirectorios"""
    series_found = []
    # Explorar cada subdirectorio
    for root, dirs, files in os.walk(directory):
        try:
            # Intentar leer series DICOM en este directorio
            series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(root)
            for series_id in series_ids:
                series_files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(root, series_id)
                if series_files:
                    series_found.append((series_id, root, series_files))
        except Exception as e:
            st.sidebar.warning(f"Advertencia al buscar en {root}: {str(e)}")
            continue
    
    return series_found

def apply_window_level(image, window_width, window_center):
    """Aplica ventana y nivel a la imagen (brillo y contraste)"""
    # Convertir la imagen a float para evitar problemas con valores negativos
    image_float = image.astype(float)
    
    # Calcular los límites de la ventana
    min_value = window_center - window_width/2.0
    max_value = window_center + window_width/2.0
    
    # Aplicar la ventana
    image_windowed = np.clip(image_float, min_value, max_value)
    
    # Normalizar a [0, 1] para visualización
    if max_value != min_value:
        image_windowed = (image_windowed - min_value) / (max_value - min_value)
    else:
        image_windowed = np.zeros_like(image_float)
    
    return image_windowed

def plot_slice(vol, slice_ix, window_width, window_center, segmentation=None, oars=None):
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.axis('off')
    selected_slice = vol[slice_ix, :, :]
    
    # Aplicar ajustes de ventana/nivel
    windowed_slice = apply_window_level(selected_slice, window_width, window_center)
    
    # Mostrar la imagen con los ajustes aplicados
    ax.imshow(windowed_slice, origin='lower', cmap='gray')
    
    # Superponer segmentación de tumor si existe
    if segmentation is not None:
        tumor_mask = segmentation[slice_ix, :, :].astype(float)
        # Crear máscara de contorno para mejor visualización
        if np.any(tumor_mask):
            contour = binary_dilation(tumor_mask) & ~binary_erosion(tumor_mask)
            ax.imshow(np.ma.masked_where(~contour, contour), origin='lower', cmap='autumn', alpha=0.7)
    
    # Superponer segmentaciones de órganos de riesgo si existen
    if oars is not None:
        oar_colors = ['cyan', 'magenta', 'yellow', 'green']
        for i, color in enumerate(oar_colors):
            if i < oars.shape[0]:  # Asegurarse de que hay suficientes canales
                oar_mask = oars[i, slice_ix, :, :]
                if np.any(oar_mask):
                    contour = binary_dilation(oar_mask) & ~binary_erosion(oar_mask)
                    ax.imshow(np.ma.masked_where(~contour, contour), origin='lower', cmap=color, alpha=0.5)
    
    return fig

# Función para realizar segmentación de tumor
def segment_tumor(image_data, model):
    # Aquí iría el código real para aplicar el modelo a los datos
    # En esta versión de simulación, creamos una segmentación sintética
    
    # En un caso real:
    """
    model.eval()
    with torch.no_grad():
        data = {"image": image_data}
        transforms = get_transform_pipeline()
        data = transforms(data)
        
        pred = sliding_window_inference(
            data["image"].unsqueeze(0),
            roi_size=(96, 96, 96),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5
        )
        
        # Convertir a numpy para visualización
        pred_np = AsDiscrete(threshold=0.5)(pred[0, 1]).cpu().numpy()
        # Redimensionar a tamaño original
        pred_np = resize_segmentation(pred_np, image_data.shape)
        return pred_np
    """
    
    # Simular un tumor en el centro
    segmentation = np.zeros_like(image_data)
    center_z, center_y, center_x = np.array(image_data.shape) // 2
    radius = min(image_data.shape) // 10
    
    z, y, x = np.ogrid[:image_data.shape[0], :image_data.shape[1], :image_data.shape[2]]
    tumor_mask = (z - center_z)**2 + (y - center_y)**2 + (x - center_x)**2 <= radius**2
    segmentation[tumor_mask] = 1
    
    return segmentation

# Función para segmentar órganos de riesgo
def segment_oars(image_data, model):
    # Simulación de segmentación de órganos de riesgo
    oars_segmentation = np.zeros((4, *image_data.shape), dtype=np.uint8)
    
    # Simular vejiga
    center_z, center_y, center_x = np.array(image_data.shape) // 2
    center_y -= image_data.shape[1] // 6  # Desplazar hacia arriba
    
    z, y, x = np.ogrid[:image_data.shape[0], :image_data.shape[1], :image_data.shape[2]]
    bladder_mask = ((z - center_z)**2)/9 + ((y - center_y)**2)/4 + ((x - center_x)**2)/4 <= (image_data.shape[1] // 8)**2
    oars_segmentation[0, bladder_mask] = 1
    
    # Simular recto
    center_y += image_data.shape[1] // 3  # Desplazar abajo
    rectum_mask = ((z - center_z)**2)/9 + ((y - center_y)**2)/4 + ((x - center_x)**2)/4 <= (image_data.shape[1] // 10)**2
    oars_segmentation[1, rectum_mask] = 1
    
    # Simular intestino
    center_x -= image_data.shape[2] // 4  # Desplazar a la izquierda
    intestine_mask = ((z - center_z)**2)/16 + ((y - (center_y - 20))**2)/9 + ((x - center_x)**2)/9 <= (image_data.shape[1] // 12)**2
    oars_segmentation[2, intestine_mask] = 1
    
    # Simular colon
    center_x += image_data.shape[2] // 2  # Desplazar a la derecha
    colon_mask = ((z - center_z)**2)/16 + ((y - (center_y - 20))**2)/9 + ((x - center_x)**2)/9 <= (image_data.shape[1] // 12)**2
    oars_segmentation[3, colon_mask] = 1
    
    return oars_segmentation

# Función para generar trayectoria de agujas
def generate_needle_paths(tumor_segmentation, oars_segmentation=None, n_needles=5):
    # En un caso real, se utilizaría un algoritmo de planificación para optimizar
    # las trayectorias basado en la segmentación del tumor y órganos de riesgo
    
    # Encontrar el centro del tumor
    z_indices, y_indices, x_indices = np.where(tumor_segmentation > 0)
    if len(z_indices) == 0:
        return []
    
    tumor_center = np.array([
        np.mean(z_indices),
        np.mean(y_indices),
        np.mean(x_indices)
    ])
    
    # Generar puntos equidistantes alrededor del centro del tumor
    needle_paths = []
    radius = min(len(z_indices), len(y_indices), len(x_indices)) // 3
    
    for i in range(n_needles):
        angle = 2 * np.pi * i / n_needles
        
        # Punto dentro del tumor
        target_point = tumor_center + np.array([0, radius * np.cos(angle), radius * np.sin(angle)])
        
        # Punto de entrada (desde abajo)
        entry_z = tumor_center[0] + tumor_segmentation.shape[0] // 4
        entry_point = np.array([entry_z, target_point[1], target_point[2]])
        
        needle_paths.append({
            'entry': entry_point.astype(int),
            'target': target_point.astype(int)
        })
    
    return needle_paths

# Función para generar una vista previa del template 3D
def generate_template_preview(needle_paths, image_shape):
    # Convertir las coordenadas de aguja a un formato adecuado para FreeCAD
    template_data = {
        'holes': []
    }
    
    for path in needle_paths:
        # Normalizar coordenadas a dimensiones reales (en mm)
        entry = path['entry']
        normalized_coords = [
            float(entry[2]) / image_shape[2] * 50 - 25,  # X: -25 a 25 mm
            float(entry[1]) / image_shape[1] * 50 - 25   # Y: -25 a 25 mm
        ]
        
        template_data['holes'].append({
            'x': normalized_coords[0],
            'y': normalized_coords[1],
            'diameter': 1.5  # Diámetro del orificio en mm
        })
    
    return template_data

# Función para visualizar el template en 3D
def visualize_template(template_data):
    # Crear una figura 3D con Plotly
    fig = go.Figure()
    
    # Añadir la base del template como una caja
    fig.add_trace(go.Mesh3d(
        x=[25, 25, -25, -25, 25, 25, -25, -25],
        y=[25, -25, -25, 25, 25, -25, -25, 25],
        z=[0, 0, 0, 0, 5, 5, 5, 5],
        i=[0, 0, 0, 1, 4, 4, 4, 5],
        j=[1, 2, 4, 5, 5, 6, 7, 6],
        k=[2, 3, 7, 6, 6, 7, 3, 2],
        opacity=0.4,
        color='#28aec5'
    ))
    
    # Añadir los orificios como cilindros
    for hole in template_data['holes']:
        x, y = hole['x'], hole['y']
        r = hole['diameter'] / 2
        
        # Crear una malla cilíndrica
        theta = np.linspace(0, 2*np.pi, 20)
        z = np.linspace(0, 5, 2)
        theta_grid, z_grid = np.meshgrid(theta, z)
        
        x_cylinder = x + r * np.cos(theta_grid)
        y_cylinder = y + r * np.sin(theta_grid)
        z_cylinder = z_grid
        
        fig.add_trace(go.Surface(
            x=x_cylinder,
            y=y_cylinder,
            z=z_cylinder,
            colorscale=[[0, '#c0d711'], [1, '#c0d711']],
            showscale=False
        ))
    
    # Configurar el layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X (mm)", range=[-30, 30]),
            yaxis=dict(title="Y (mm)", range=[-30, 30]),
            zaxis=dict(title="Z (mm)", range=[0, 10]),
            aspectmode="data"
        ),
        scene_camera=dict(
            eye=dict(x=1.5, y=-1.5, z=1.2)
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=600
    )
    
    return fig

# Función para exportar los datos a formato FreeCAD
def export_to_freecad(template_data):
    # Generar un script de Python para FreeCAD
    freecad_script = f"""# Script de FreeCAD para crear un template de braquiterapia
import FreeCAD as App
import Part
import Draft

# Crear documento
doc = App.newDocument("BrachytherapyTemplate")

# Crear la base del template
base = Part.makeBox(50, 50, 5)
template = doc.addObject("Part::Feature", "Template")
template.Shape = base

# Crear orificios para las agujas
holes = []
"""

    # Añadir código para cada orificio
    for i, hole in enumerate(template_data['holes']):
        freecad_script += f"""
# Orificio {i+1}
hole{i} = Part.makeCylinder({hole['diameter']/2}, 5, App.Vector({hole['x']+25}, {hole['y']+25}, 0), App.Vector(0, 0, 1))
holes.append(hole{i})
"""

    # Finalizar el script para realizar la operación booleana
    freecad_script += """
# Realizar la operación booleana para sustraer los orificios
combined_holes = holes[0]
for hole in holes[1:]:
    combined_holes = combined_holes.fuse(hole)

result = template.Shape.cut(combined_holes)
final_template = doc.addObject("Part::Feature", "FinalTemplate")
final_template.Shape = result

# Ocultar objetos intermedios
template.ViewObject.Visibility = False

# Guardar el archivo
doc.recompute()
doc.saveAs("BrachytherapyTemplate.FCStd")
"""
    
    return freecad_script

# Configuración de la barra lateral
st.sidebar.markdown('<p class="sidebar-title">Brachyanalysis</p>', unsafe_allow_html=True)
st.sidebar.markdown('<p class="sub-header">Visualizador y planificador para braquiterapia</p>', unsafe_allow_html=True)

# Sección de carga de archivos en la barra lateral
st.sidebar.markdown('<p class="sub-header">Configuración</p>', unsafe_allow_html=True)

# Solo opción de subir ZIP
uploaded_file = st.sidebar.file_uploader("Sube un archivo ZIP con tus archivos DICOM", type="zip")

# Procesar archivos subidos
dirname = None
temp_dir = None

if uploaded_file is not None:
    # Crear un directorio temporal para extraer los archivos
    temp_dir = tempfile.mkdtemp()
    try:
        # Leer el contenido del ZIP
        with zipfile.ZipFile(io.BytesIO(uploaded_file.read()), 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Establecer dirname al directorio temporal
        dirname = temp_dir
        st.sidebar.markdown('<div class="success-box">Archivos extraídos correctamente.</div>', unsafe_allow_html=True)
    except Exception as e:
        st.sidebar.error(f"Error al extraer el archivo ZIP: {str(e)}")

# Inicializar variables para la visualización
dicom_series = None
img = None
output = None
n_slices = 0
slice_ix = 0
reader = None
tumor_segmentation = None
oars_segmentation = None
needle_paths = []

# Cargar modelos de segmentación
tumor_model, oars_model = load_models()

# Define los presets de ventana basados en RadiAnt
radiant_presets = {
    "Default window": (0, 0),  # Auto-calculado según la imagen
    "Full dynamic": (0, 0),    # Auto-calculado según la imagen
    "CT Abdomen": (350, 50),
    "CT Angio": (600, 300),
    "CT Bone": (2000, 350),
    "CT Brain": (80, 40),
    "CT Chest": (350, 40),
    "CT Lungs": (1500, -600),
    "Negative": (0, 0),       # Invertir la imagen
    "Custom window": (0, 0)   # Valores personalizados
}

if dirname is not None:
    # Usar un spinner en el área principal en lugar de en la barra lateral
    with st.spinner('Buscando series DICOM...'):
        dicom_series = find_dicom_series(dirname)
    
    if not dicom_series:
        st.sidebar.error("No se encontraron archivos DICOM válidos en el archivo subido.")
    else:
        # Mostrar las series encontradas
        st.sidebar.markdown(f'<div class="info-box">Se encontraron {len(dicom_series)} series DICOM</div>', unsafe_allow_html=True)
        
        # Si hay múltiples series, permitir seleccionar una
        selected_series_idx = 0
        if len(dicom_series) > 1:
            series_options = [f"Serie {i+1}: {series_id[:10]}... ({len(files)} archivos)" 
                            for i, (series_id, _, files) in enumerate(dicom_series)]
            selected_series_option = st.sidebar.selectbox("Seleccionar serie DICOM:", series_options)
            selected_series_idx = series_options.index(selected_series_option)
        
        try:
            # Obtener la serie seleccionada
            series_id, series_dir, dicom_names = dicom_series[selected_series_idx]
            
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(dicom_names)
            reader.LoadPrivateTagsOn()
            reader.MetaDataDictionaryArrayUpdateOn()
            data = reader.Execute()
            img = sitk.GetArrayViewFromImage(data)
        
            n_slices = img.shape[0]
            slice_ix = st.sidebar.slider('Seleccionar corte', 0, n_slices - 1, int(n_slices/2))
            
            # Modificar opciones de visualización para incluir segmentación y planificación
            output = st.sidebar.radio('Modo de visualización', 
                                     ['Imagen', 'Segmentación', 'Planificación', 'Metadatos'], 
                                     index=0)
            
            # Añadir controles de ventana (brillo y contraste)
            # Calcular valores iniciales para la ventana
            if img is not None:
                min_val = float(img.min())
                max_val = float(img.max())
                range_val = max_val - min_val
                
                # Establecer valores predeterminados para window width y center
                default_window_width = range_val
                default_window_center = min_val + (range_val / 2)
                
                # Añadir presets de ventana para radiología
                st.sidebar.markdown('<div class="control-section">', unsafe_allow_html=True)
                st.sidebar.markdown('<p class="sub-header">Presets de ventana</p>', unsafe_allow_html=True)
                
                # Actualizar los presets automáticos
                radiant_presets["Default window"] = (default_window_width, default_window_center)
                radiant_presets["Full dynamic"] = (range_val, min_val + (range_val / 2))
                
                selected_preset = st.sidebar.selectbox(
                    "Presets radiológicos",
                    list(radiant_presets.keys())
                )
                
                # Inicializar valores de ventana basados en el preset
                window_width, window_center = radiant_presets[selected_preset]
                
                # Si es preset negativo, invertir la imagen
                is_negative = selected_preset == "Negative"
                if is_negative:
                    window_width = default_window_width
                    window_center = default_window_center
                
                # Si es un preset personalizado o Custom window, mostrar los campos de entrada
                if selected_preset == "Custom window":
                    st.sidebar.markdown('<p class="sub-header">Ajustes personalizados</p>', unsafe_allow_html=True)
                    
                    # Mostrar información sobre el rango
                    st.sidebar.markdown(f"**Rango de valores de la imagen:** {min_val:.1f} a {max_val:.1f}")
                    
                    # Crear dos columnas para los campos de entrada
                    col1, col2 = st.sidebar.columns(2)
                    
                    with col1:
                        window_width = float(st.number_input(
                            "Ancho de ventana (WW)",
                            min_value=1.0,
                            max_value=range_val * 2,
                            value=float(default_window_width),
                            format="%.1f",
                            help="Controla el contraste. Valores menores aumentan el contraste."
                        ))
                    
                    with col2:
                        window_center = float(st.number_input(
                            "Centro de ventana (WL)",
                            min_value=min_val - range_val,
                            max_value=max_val + range_val,
                            value=float(default_window_center),
                            format="%.1f",
                            help="Controla el brillo. Valores mayores aumentan el brillo."
                        ))
                
                st.sidebar.markdown('</div>', unsafe_allow_html=True)
                
                # Añadir controles para la segmentación si estamos en ese modo
                if output == 'Segmentación':
                    st.sidebar.markdown('<div class="segmentation-control">', unsafe_allow_html=True)
                    st.sidebar.markdown('<p class="sub-header">Opciones de segmentación</p>', unsafe_allow_html=True)
                    
                    # Botones para ejecutar la segmentación
                    run_tumor_segmentation = st.sidebar.button("Segmentar tumor cervical")
                    run_oars_segmentation = st.sidebar.button("Segmentar órganos de riesgo")
                    
                    if run_tumor_segmentation:
                        with st.spinner('Segmentando tumor...'):
                            tumor_segmentation = segment_tumor(img, tumor_model)
                            st.sidebar.success("Segmentación de tumor completada")
                    
                    if run_oars_segmentation:
                        with st.spinner('Segmentando órganos de riesgo...'):
                            oars_segmentation = segment_oars(img, oars_model)
                            st.sidebar.success("Segmentación de OARs completada")
                    
                    # Opciones de visualización
                    show_tumor = st.sidebar.checkbox("Mostrar tumor", value=True)
                    show_oars = st.sidebar.checkbox("Mostrar OARs", value=True)
                    
                    # Selector de órganos específicos si se han segmentado
                    if oars_segmentation is not None and show_oars:
                        oar_names = ["Vejiga", "Recto", "Intestino delgado", "Colon"]
                        selected_oars = st.sidebar.multiselect(
                            "Órganos de riesgo a mostrar",
                            oar_names,
                            default=oar_names
                        )
                        
                        # Crear máscara según selección
                        oars_mask = np.zeros_like(oars_segmentation)
                        for i, name in enumerate(oar_names):
                            if name in selected_oars:
                                oars_mask[i] = oars_segmentation[i]
                        
                        # Reemplazar segmentación con la máscara filtrada
                        oars_segmentation = oars_mask if any(selected_oars) else None
                    
                    st.sidebar.markdown('</div>', unsafe_allow_html=True)
                
                # Añadir controles para la planificación si estamos en ese modo
                if output == 'Planificación':
                    st.sidebar.markdown('<div class="segmentation-control">', unsafe_allow_html=True)
                    st.sidebar.markdown('<p class="sub-header">Planificación de agujas</p>', unsafe_allow_html=True)
                    
                    # Asegurarse de que tengamos segmentación de tumor
                    if tumor_segmentation is None:
                        with st.spinner('Segmentando tumor...'):
                            tumor_segmentation = segment_tumor(img, tumor_model)
                    
                    # Asegurarse de que tengamos segmentación de OARs
                    if oars_segmentation is None:
                        with st.spinner('Segmentando órganos de riesgo...'):
                            oars_segmentation = segment_oars(img, oars_model)
                    
                    # Número de agujas
                    n_needles = st.sidebar.slider('Número de agujas', 3, 15, 9)
                    
                    # Botón para generar trayectorias
                    if st.sidebar.button("Generar trayectorias de aguja"):
                        with st.spinner('Calculando trayectorias óptimas...'):
                            needle_paths = generate_needle_paths(tumor_segmentation, oars_segmentation, n_needles)
                            st.sidebar.success(f"Se generaron {len(needle_paths)} trayectorias")
                    
                    st.sidebar.markdown('</div>', unsafe_allow_html=True)
            
            else:
                # Valores predeterminados para cuando no son necesarios
                window_width = 1000
                window_center = 0
                is_negative = False
                
        except Exception as e:
            st.sidebar.error(f"Error al procesar los archivos DICOM: {str(e)}")
            st.sidebar.write("Detalles del error:", str(e))
            # Valores predeterminados
            window_width = 1000
            window_center = 0
            is_negative = False

# Visualización en la ventana principal
# Título grande siempre visible
st.markdown('<p class="giant-title">Brachyanalysis</p>', unsafe_allow_html=True)

if img is not None:
    if output == 'Imagen':
        st.markdown('<p class="sub-header">Visualización DICOM</p>', unsafe_allow_html=True)
        
        # Si es modo negativo, invertir la imagen
        if is_negative:
            # Muestra la imagen invertida
            fig, ax = plt.subplots(figsize=(12, 10))
            plt.axis('off')
            selected_slice = img[slice_ix, :, :]
            
            # Aplicar ventana y luego invertir
            windowed_slice = apply_window_level(selected_slice, window_width, window_center)
            windowed_slice = 1.0 - windowed_slice  # Invertir
            
            ax.imshow(windowed_slice, origin='lower', cmap='gray')
            st.pyplot(fig)
        else:
            # Muestra la imagen en la ventana principal con los ajustes aplicados
            fig = plot_slice(img, slice_ix, window_width, window_center)
            st.pyplot(fig)
        
    elif output == 'Segmentación':
        st.markdown('<p class="sub-header">Segmentación de tumores y órganos de riesgo</p>', unsafe_allow_html=True)
        
        # Mostrar visualización con segmentaciones superpuestas
        seg_to_display = tumor_segmentation if show_tumor else None
        oars_to_display = oars_segmentation if show_oars else None
        
        # Mostrar la imagen con las segmentaciones
        fig = plot_slice(img, slice_ix, window_width, window_center, seg_to_display, oars_to_display)
        st.pyplot(fig)
        
        # Mostrar información adicional sobre la segmentación
        col1, col2 = st.columns(2)
        with col1:
            if tumor_segmentation is not None:
                tumor_volume = np.sum(tumor_segmentation) * 0.001  # Convertir a ml (asumiendo voxeles de 1mm³)
                st.markdown(f"**Volumen del tumor:** {tumor_volume:.2f} ml")
                st.markdown(f"**Ubicación central:** Z={int(np.mean(np.where(tumor_segmentation)[0]))}, "
                          f"Y={int(np.mean(np.where(tumor_segmentation)[1]))}, "
                          f"X={int(np.mean(np.where(tumor_segmentation)[2]))}")
        
        with col2:
            if oars_segmentation is not None:
                oar_names = ["Vejiga", "Recto", "Intestino delgado", "Colon"]
                for i, name in enumerate(oar_names):
                    if i < oars_segmentation.shape[0]:
                        oar_volume = np.sum(oars_segmentation[i]) * 0.001  # Convertir a ml
                        st.markdown(f"**Volumen {name}:** {oar_volume:.2f} ml")
        
    elif output == 'Planificación':
        st.markdown('<p class="sub-header">Planificación de trayectorias de agujas</p>', unsafe_allow_html=True)
        
        # Crear pestañas para diferentes vistas
        tabs = st.tabs(["Trayectorias 2D", "Visualización 3D", "Template 3D", "Exportar"])
        
        with tabs[0]:
            # Mostrar trayectorias de agujas en 2D
            fig, ax = plt.subplots(figsize=(12, 10))
            plt.axis('off')
            
            # Mostrar imagen base
            selected_slice = img[slice_ix, :, :]
            windowed_slice = apply_window_level(selected_slice, window_width, window_center)
            ax.imshow(windowed_slice, origin='lower', cmap='gray')
            
            # Mostrar tumor
            if tumor_segmentation is not None:
                tumor_slice = tumor_segmentation[slice_ix, :, :]
                if np.any(tumor_slice):
                    contour = binary_dilation(tumor_slice) & ~binary_erosion(tumor_slice)
                    ax.imshow(np.ma.masked_where(~contour, contour), origin='lower', cmap='autumn', alpha=0.7)
            
            # Mostrar órganos de riesgo
            if oars_segmentation is not None:
                oar_colors = ['cyan', 'magenta', 'yellow', 'green']
                for i, color in enumerate(oar_colors):
                    if i < oars_segmentation.shape[0]:
                        oar_slice = oars_segmentation[i, slice_ix, :, :]
                        if np.any(oar_slice):
                            contour = binary_dilation(oar_slice) & ~binary_erosion(oar_slice)
                            ax.imshow(np.ma.masked_where(~contour, contour), origin='lower', cmap=color, alpha=0.5)
            
            # Dibujar trayectorias de agujas
            if needle_paths:
                for i, path in enumerate(needle_paths):
                    # Verificar si la trayectoria pasa por este corte
                    entry = path['entry']
                    target = path['target']
                    
                    # Dibujar solo si la trayectoria pasa cerca de este corte
                    z_min = min(entry[0], target[0])
                    z_max = max(entry[0], target[0])
                    if z_min <= slice_ix <= z_max or abs(slice_ix - z_min) <= 3 or abs(slice_ix - z_max) <= 3:
                        # Calcular punto de intersección con este corte
                        if entry[0] != target[0]:  # Evitar división por cero
                            t = (slice_ix - entry[0]) / (target[0] - entry[0])
                            intersection_y = entry[1] + t * (target[1] - entry[1])
                            intersection_x = entry[2] + t * (target[2] - entry[2])
                            
                            # Dibujar punto de intersección de la aguja
                            ax.scatter(intersection_x, intersection_y, color='#c0d711', s=50, marker='o', edgecolors='black', label=f'Aguja {i+1}')
                            ax.text(intersection_x + 10, intersection_y + 10, f'{i+1}', color='white', fontsize=12, bbox=dict(facecolor='#28aec5', alpha=0.7))
            
            st.pyplot(fig)
            
            # Mostrar información sobre las trayectorias
            if needle_paths:
                st.markdown('<div class="needle-planning">', unsafe_allow_html=True)
                st.markdown('<p class="sub-header">Información de trayectorias</p>', unsafe_allow_html=True)
                
                needle_data = []
                for i, path in enumerate(needle_paths):
                    entry = path['entry']
                    target = path['target']
                    length = np.sqrt(np.sum((entry - target)**2)) * 0.1  # Convertir a cm
                    needle_data.append({
                        'Número': i+1,
                        'Entrada (Z,Y,X)': f"{entry[0]}, {entry[1]}, {entry[2]}",
                        'Objetivo (Z,Y,X)': f"{target[0]}, {target[1]}, {target[2]}",
                        'Longitud (cm)': f"{length:.1f}"
                    })
                
                df = pd.DataFrame(needle_data)
                st.dataframe(df)
                st.markdown('</div>', unsafe_allow_html=True)
        
        with tabs[1]:
            # Visualización 3D de trayectorias (usando Plotly)
            if needle_paths:
                # Crear figura 3D
                fig = go.Figure()
                
                # Añadir tumor como superficie isométrica
                if tumor_segmentation is not None:
                    z_coords, y_coords, x_coords = np.where(tumor_segmentation > 0)
                    fig.add_trace(go.Scatter3d(
                        x=x_coords, y=y_coords, z=z_coords,
                        mode='markers',
                        marker=dict(
                            size=3,
                            color='red',
                            opacity=0.3
                        ),
                        name='Tumor'
                    ))
                
                # Añadir órganos de riesgo como nubes de puntos
                if oars_segmentation is not None:
                    oar_colors = ['cyan', 'magenta', 'yellow', 'green']
                    oar_names = ["Vejiga", "Recto", "Intestino delgado", "Colon"]
                    
                    for i, (color, name) in enumerate(zip(oar_colors, oar_names)):
                        if i < oars_segmentation.shape[0]:
                            # Submuestrear para mejorar rendimiento
                            z_coords, y_coords, x_coords = np.where(oars_segmentation[i] > 0)
                            sample_rate = max(1, len(z_coords) // 1000)  # Limitar a ~1000 puntos
                            
                            fig.add_trace(go.Scatter3d(
                                x=x_coords[::sample_rate], 
                                y=y_coords[::sample_rate], 
                                z=z_coords[::sample_rate],
                                mode='markers',
                                marker=dict(
                                    size=2,
                                    color=color,
                                    opacity=0.2
                                ),
                                name=name
                            ))
                
                # Añadir trayectorias de agujas
                for i, path in enumerate(needle_paths):
                    entry = path['entry']
                    target = path['target']
                    
                    fig.add_trace(go.Scatter3d(
                        x=[entry[2], target[2]],
                        y=[entry[1], target[1]],
                        z=[entry[0], target[0]],
                        mode='lines+markers',
                        line=dict(color='#c0d711', width=5),
                        marker=dict(size=6, color=['#28aec5', '#c0d711']),
                        name=f'Aguja {i+1}'
                    ))
                
                # Configurar layout
                fig.update_layout(
                    scene=dict(
                        xaxis_title='X',
                        yaxis_title='Y',
                        zaxis_title='Z',
                        aspectmode='data'
                    ),
                    margin=dict(l=0, r=0, b=0, t=0),
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    ),
                    height=700
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Genera trayectorias de agujas primero para visualizar en 3D")
        
        with tabs[2]:
            # Visualización del template 3D
            if needle_paths:
                # Generar datos para el template
                template_data = generate_template_preview(needle_paths, img.shape)
                
                # Visualizar el template en 3D
                fig = visualize_template(template_data)
                st.plotly_chart(fig, use_container_width=True)
                
                # Mostrar información del template
                st.markdown('<div class="template-preview">', unsafe_allow_html=True)
                st.markdown(f"**Template con {len(template_data['holes'])} orificios**")
                st.markdown("Dimensiones: 50mm x 50mm x 5mm")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("Genera trayectorias de agujas primero para crear el template")
        
        with tabs[3]:
            # Opciones de exportación
            if needle_paths:
                st.markdown('<p class="sub-header">Exportar datos para fabricación</p>', unsafe_allow_html=True)
                
                # Generar datos para el template
                template_data = generate_template_preview(needle_paths, img.shape)
                
                # Generar script para FreeCAD
                freecad_script = export_to_freecad(template_data)
                
                # Opciones de exportación
                export_format = st.radio("Formato de exportación", ["Script FreeCAD", "Coordenadas CSV"])
                
                if export_format == "Script FreeCAD":
                    st.code(freecad_script, language="python")
                    
                    # Crear botón de descarga para el script
                    freecad_bytes = freecad_script.encode()
                    st.download_button(
                        label="Descargar script FreeCAD",
                        data=freecad_bytes,
                        file_name="template_freecad.py",
                        mime="text/plain"
                    )
                else:
                    # Crear CSV con datos de coordenadas
                    csv_data = "hole_id,x_mm,y_mm,diameter_mm\n"
                    for i, hole in enumerate(template_data['holes']):
                        csv_data += f"{i+1},{hole['x']},{hole['y']},{hole['diameter']}\n"
                    
                    st.text_area("Datos CSV", csv_data, height=200)
                    
                    # Crear botón de descarga para CSV
                    csv_bytes = csv_data.encode()
                    st.download_button(
                        label="Descargar coordenadas CSV",
                        data=csv_bytes,
                        file_name="template_coords.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("Genera trayectorias de agujas primero para exportar datos")
    
    elif output == 'Metadatos':
        st.markdown('<p class="sub-header">Metadatos DICOM</p>', unsafe_allow_html=True)
        try:
            metadata = dict()
            for k in reader.GetMetaDataKeys(slice_ix):
                metadata[k] = reader.GetMetaData(slice_ix, k)
            df = pd.DataFrame.from_dict(metadata, orient='index', columns=['Valor'])
            st.dataframe(df, height=600)
        except Exception as e:
            st.error(f"Error al leer metadatos: {str(e)}")
    
    # Información adicional sobre la imagen y los ajustes actuales
    if output in ['Imagen', 'Segmentación']:
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
else:
    # Página de inicio cuando no hay imágenes cargadas
    st.markdown('<p class="sub-header">Visualizador de imágenes DICOM</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; padding: 40px; margin-top: 10px;">
        <img src="https://raw.githubusercontent.com/SimpleITK/SimpleITK/master/Documentation/docs/images/simpleitk-logo.svg" alt="SimpleITK Logo" width="200">
        <h2 style="color: #28aec5; margin-top: 20px;">Carga un archivo ZIP con tus imágenes DICOM</h2>
        <p style="font-size: 18px; margin-top: 10px;">Utiliza el panel lateral para subir tus archivos y visualizarlos</p>
        <p style="font-size: 16px; margin-top: 20px;">La aplicación incluye:</p>
        <ul style="list-style-type: none; display: inline-block; text-align: left;">
            <li>✓ Segmentación automática de tumores cervicales</li>
            <li>✓ Identificación de órganos de riesgo</li>
            <li>✓ Planificación de trayectorias de agujas</li>
            <li>✓ Generación de templates 3D</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Pie de página
st.markdown("""
<hr style="margin-top: 30px;">
<div style="text-align: center; color: #28aec5; font-size: 14px;">
    Brachyanalysis - Visualizador y planificador para braquiterapia
</div>
""", unsafe_allow_html=True)

# Limpiar el directorio temporal si se creó uno
if temp_dir and os.path.exists(temp_dir):
    # Nota: En una aplicación real, deberías limpiar los directorios temporales
    # cuando la aplicación se cierre, pero en Streamlit esto es complicado
    # ya que las sesiones persisten. Una solución es mantener un registro
    # de directorios temporales y limpiarlos al inicio.
    pass

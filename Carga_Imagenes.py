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
    """Carga imágenes DICOM como volumen 3D con manejo mejorado de errores"""
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
    pixel_spacing = getattr(sample, 'PixelSpacing', [1,1])
    # Asegurarse de que pixel_spacing sea una lista regular de Python
    pixel_spacing = list(map(float, pixel_spacing))
    slice_thickness = float(getattr(sample, 'SliceThickness', 1))
    
    # Corregido: pixel_spacing ya está convertido a lista regular
    spacing = pixel_spacing + [slice_thickness]
    
    origin = getattr(sample, 'ImagePositionPatient', [0,0,0])
    direction = getattr(sample, 'ImageOrientationPatient', [1,0,0,0,1,0])

    direction_matrix = np.array([
        [direction[0], direction[3], 0],
        [direction[1], direction[4], 0],
        [direction[2], direction[5], 1]
    ])

    # Añadir posiciones Z de cada corte para uso posterior
    slice_positions = []
    for _, dcm in dicom_files:
        if hasattr(dcm, 'ImagePositionPatient'):
            slice_positions.append(float(dcm.ImagePositionPatient[2]))
        else:
            slice_positions.append(0.0)

    volume_info = {
        'spacing': spacing,
        'origin': origin,
        'direction': direction_matrix,
        'size': volume.shape,
        'slice_positions': slice_positions
    }
        
    return volume, volume_info

def load_rtstruct(file_path):
    """Carga contornos RTSTRUCT con mejor manejo de errores y debug"""
    try:
        struct = pydicom.dcmread(file_path)
        structures = {}
        
        if not hasattr(struct, 'ROIContourSequence'):
            st.warning("El archivo RTSTRUCT no contiene secuencia ROIContour")
            return structures
        
        # Mapeo de ROI Number a ROI Name
        roi_names = {roi.ROINumber: roi.ROIName for roi in struct.StructureSetROISequence}
        
        for roi in struct.ROIContourSequence:
            color = np.array(roi.ROIDisplayColor) / 255.0 if hasattr(roi, 'ROIDisplayColor') else np.random.rand(3)
            contours = []
            
            if hasattr(roi, 'ContourSequence'):
                contour_count = 0
                for contour in roi.ContourSequence:
                    pts = np.array(contour.ContourData).reshape(-1, 3)
                    contours.append({'points': pts, 'z': np.mean(pts[:,2])})
                    contour_count += 1
                
                roi_name = roi_names.get(roi.ReferencedROINumber, f"ROI-{roi.ReferencedROINumber}")
                structures[roi_name] = {'color': color, 'contours': contours}
            
        return structures
    except Exception as e:
        st.error(f"Error leyendo estructura: {e}")
        import traceback
        st.code(traceback.format_exc())
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
    Dibuja un corte de un volumen con contornos superpuestos, funcionando para planos axial, coronal y sagital.
    """
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

    # Mostrar imagen base
    ax.imshow(img, cmap='gray')

    # Mostrar texto de slice y posición física
    origin = np.array(volume_info['origin'])
    spacing = np.array(volume_info['spacing'])
    
    if plane == 'axial':
        if 'slice_positions' in volume_info and len(volume_info['slice_positions']) > slice_idx:
            current_slice_pos = volume_info['slice_positions'][slice_idx]
        else:
            current_slice_pos = origin[2] + slice_idx * spacing[2]
        coord_label = f"Z: {current_slice_pos:.2f} mm"
    elif plane == 'coronal':
        current_slice_pos = origin[1] + slice_idx * spacing[1]
        coord_label = f"Y: {current_slice_pos:.2f} mm"
    elif plane == 'sagittal':
        current_slice_pos = origin[0] + slice_idx * spacing[0]
        coord_label = f"X: {current_slice_pos:.2f} mm"

    ax.text(5, 15, f"{plane} - slice {slice_idx}", color='white',
            bbox=dict(facecolor='black', alpha=0.5))
    ax.text(5, 30, coord_label, color='yellow',
            bbox=dict(facecolor='black', alpha=0.5))

    # Dibujar contornos si existen estructuras
    if structures:
        for name, struct in structures.items():
            contour_drawn = 0
            color = struct['color']

            for contour in struct['contours']:
                raw_points = contour['points']

                if plane == 'axial':
                    # Verificar cercanía en Z
                    contour_z_values = raw_points[:, 2]
                    min_z = np.min(contour_z_values)
                    max_z = np.max(contour_z_values)
                    tolerance = spacing[2] * 2.0

                    if (min_z - tolerance <= current_slice_pos <= max_z + tolerance or
                        abs(contour['z'] - current_slice_pos) <= tolerance):

                        pixel_points = np.zeros((raw_points.shape[0], 2))
                        pixel_points[:, 0] = (raw_points[:, 0] - origin[0]) / spacing[0]
                        pixel_points[:, 1] = (raw_points[:, 1] - origin[1]) / spacing[1]

                        if len(pixel_points) >= 3:
                            polygon = patches.Polygon(pixel_points, closed=True,
                                                       fill=False, edgecolor=color,
                                                       linewidth=linewidth)
                            ax.add_patch(polygon)
                            contour_drawn += 1

                elif plane == 'coronal':
                    # Verificar cercanía en Y
                    mask = np.abs(raw_points[:, 1] - current_slice_pos) < spacing[1]
                    if np.sum(mask) >= 3:
                        selected_points = raw_points[mask]
                        pixel_points = np.zeros((selected_points.shape[0], 2))
                        pixel_points[:, 0] = (selected_points[:, 0] - origin[0]) / spacing[0]  # X
                        pixel_points[:, 1] = (selected_points[:, 2] - origin[2]) / spacing[2]  # Z

                        polygon = patches.Polygon(pixel_points, closed=True,
                                                   fill=False, edgecolor=color,
                                                   linewidth=linewidth)
                        ax.add_patch(polygon)
                        contour_drawn += 1

                elif plane == 'sagittal':
                    # Verificar cercanía en X
                    mask = np.abs(raw_points[:, 0] - current_slice_pos) < spacing[0]
                    if np.sum(mask) >= 3:
                        selected_points = raw_points[mask]
                        pixel_points = np.zeros((selected_points.shape[0], 2))
                        pixel_points[:, 0] = (selected_points[:, 1] - origin[1]) / spacing[1]  # Y
                        pixel_points[:, 1] = (selected_points[:, 2] - origin[2]) / spacing[2]  # Z

                        polygon = patches.Polygon(pixel_points, closed=True,
                                                   fill=False, edgecolor=color,
                                                   linewidth=linewidth)
                        ax.add_patch(polygon)
                        contour_drawn += 1

            if contour_drawn > 0 and show_names:
                ax.text(img.shape[1]/2, img.shape[0]/2, f"{name} ({contour_drawn})",
                        color=color, fontsize=8, ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.7))

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

            st.title("Generador de cilindro con punta tipo tampón (FreeCAD)")
            st.write("Esta aplicación genera código para crear un cilindro con una punta redondeada en FreeCAD.")
            
            # Parámetros en centímetros
            col1, col2 = st.columns(2)
            with col1:
                diametro_cm = st.slider("Diámetro (cm)", min_value=1.0, max_value=12.0, value=3.0, step=0.1)
                st.write(f"Diámetro seleccionado: {diametro_cm} cm ({diametro_cm*10} mm)")
            
            with col2:
                longitud_cm = st.slider("Longitud total (cm)", min_value=2.0, max_value=20.0, value=5.0, step=0.1)
                st.write(f"Longitud seleccionada: {longitud_cm} cm ({longitud_cm*10} mm)")
            
            # Opción para personalizar la proporción de la punta
            with st.expander("Opciones avanzadas"):
                prop_punta = st.slider("Proporción de la punta (%)", min_value=10, max_value=50, value=20, step=5)
                st.write(f"La punta ocupará el {prop_punta}% de la longitud total")
            
            # Convertir a milímetros
            diametro_mm = round(diametro_cm * 10, 2)
            longitud_mm = round(longitud_cm * 10, 2)
            
            # Definir la longitud del cuerpo y la altura de la punta redondeada
            altura_punta = round(longitud_mm * prop_punta/100, 2)
            altura_cuerpo = round(longitud_mm - altura_punta, 2)
            
            codigo = f"""import FreeCAD as App
            import Part
            
            # Crear un nuevo documento
            doc = App.newDocument()
            
            # Parámetros
            diametro = {diametro_mm}
            radio = diametro / 2
            altura_total = {longitud_mm}
            altura_cuerpo = {altura_cuerpo}
            altura_punta = {altura_punta}
            
            # Crear cuerpo cilíndrico
            cuerpo = Part.makeCylinder(radio, altura_cuerpo)
            
            # Crear punta redondeada (semiesfera)
            centro_semiesfera = App.Vector(0, 0, altura_cuerpo)
            punta = Part.makeSphere(radio, centro_semiesfera)
            
            # Cortar la mitad inferior de la esfera
            box = Part.makeBox(diametro*2, diametro*2, altura_cuerpo)
            box.translate(App.Vector(-diametro, -diametro, -altura_cuerpo))
            punta = punta.cut(box)
            
            # Unir cilindro y punta
            objeto_final = cuerpo.fuse(punta)
            
            # Crear un objeto en el documento de FreeCAD
            objeto = doc.addObject("Part::Feature", "CilindroConPunta")
            objeto.Shape = objeto_final
            
            # Actualizar el documento
            doc.recompute()
            
            # Vista - Solo si estamos en la interfaz gráfica
            if App.GuiUp:
                import FreeCADGui as Gui
                App.activeDocument().recompute()
                Gui.activeDocument().activeView().viewAxonometric()
                Gui.SendMsgToActiveView("ViewFit")
            
            print("Objeto creado con éxito con las siguientes dimensiones:")
            print(f"- Diámetro: {{diametro}} mm")
            print(f"- Altura total: {{altura_total}} mm")
            print(f"- Altura del cuerpo: {{altura_cuerpo}} mm")
            print(f"- Altura de la punta: {{altura_punta}} mm")
            """
            
            # Mostrar el código
            st.subheader("Código FreeCAD generado")
            st.code(codigo, language="python")
                    
            # Botón de descarga
            st.download_button(
                label="Descargar código FreeCAD (.py)",
                data=codigo,
                file_name="cilindro_punta_redondeada.py",
                mime="text/x-python"
            )
            
    else:
        st.warning("No se encontraron imágenes DICOM en el ZIP.")

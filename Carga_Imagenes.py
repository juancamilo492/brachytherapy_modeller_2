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
import plotly.graph_objects as go

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
    pixel_spacing = list(map(float, pixel_spacing))
    slice_thickness = float(getattr(sample, 'SliceThickness', 1))
    
    spacing = pixel_spacing + [slice_thickness]
    
    origin = getattr(sample, 'ImagePositionPatient', [0,0,0])
    direction = getattr(sample, 'ImageOrientationPatient', [1,0,0,0,1,0])

    direction_matrix = np.array([
        [direction[0], direction[3], 0],
        [direction[1], direction[4], 0],
        [direction[2], direction[5], 1]
    ])

    # Añadir posiciones Z de cada corte
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
        
        roi_names = {roi.ROINumber: roi.ROIName for roi in struct.StructureSetROISequence}
        
        for roi in struct.ROIContourSequence:
            color = np.array(roi.ROIDisplayColor) / 255.0 if hasattr(roi, 'ROIDisplayColor') else np.random.rand(3)
            contours = []
            
            if hasattr(roi, 'ContourSequence'):
                for contour in roi.ContourSequence:
                    pts = np.array(contour.ContourData).reshape(-1, 3)
                    contours.append({'points': pts, 'z': np.mean(pts[:,2])})
                
                roi_name = roi_names.get(roi.ReferencedROINumber, f"ROI-{roi.ReferencedROINumber}")
                structures[roi_name] = {'color': color, 'contours': contours}
            
        return structures
    except Exception as e:
        st.error(f"Error leyendo estructura: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

# --- Parte 3: Funciones de visualización y análisis ---

def patient_to_voxel(points, volume_info):
    """Convierte puntos del espacio del paciente al espacio de vóxeles"""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Se esperaba (N, 3) puntos, recibido {points.shape}")

    origin = np.asarray(volume_info['origin'], dtype=np.float32)
    spacing = np.asarray(volume_info['spacing'], dtype=np.float32)
    direction = volume_info['direction']
    
    inv_direction = np.linalg.inv(direction)
    
    adjusted_points = np.zeros_like(points)
    
    for i in range(len(points)):
        vec = np.dot(inv_direction, points[i] - origin)
        adjusted_points[i] = vec / spacing
        
    return adjusted_points

def apply_window(img, window_center, window_width):
    """Aplica ventana de visualización a la imagen"""
    img = img.astype(np.float32)
    min_value = window_center - window_width / 2
    max_value = window_center + window_width / 2
    img = np.clip(img, min_value, max_value)
    img = (img - min_value) / (max_value - min_value)
    img = np.clip(img, 0, 1)
    return img

def point_in_polygon(point, polygon):
    """Determina si un punto 2D está dentro de un polígono 2D"""
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        if ((polygon[i][1] > y) != (polygon[j][1] > y)) and \
           (x < (polygon[j][0] - polygon[i][0]) * (y - polygon[i][1]) / 
           (polygon[j][1] - polygon[i][1] + 1e-10) + polygon[i][0]):
            inside = not inside
        j = i
    return inside

def line_intersects_contour(line_start, line_end, contour_points, margin=2.0):
    """Verifica si una línea 3D intersecta un contorno con margen"""
    from scipy.spatial import distance
    line_vec = line_end - line_start
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return False
    line_dir = line_vec / line_len
    for point in contour_points:
        t = np.dot(point - line_start, line_dir)
        if 0 <= t <= line_len:
            proj_point = line_start + t * line_dir
            dist = distance.euclidean(point, proj_point)
            if dist < margin:
                return True
    return False

def compute_needle_trajectories(num_needles, cylinder_diameter, cylinder_length, structures, volume_info, ctv_centroid_offset=[0,0,0]):
    """Calcula trayectorias de agujas para alcanzar CTV evitando órganos sanos"""
    needle_diameter = 3.0  # mm
    needle_radius = needle_diameter / 2
    cylinder_radius = cylinder_diameter / 2
    needle_ring_radius = cylinder_radius * 0.7  # 70% del radio del cilindro

    # Encontrar CTV y calcular su centroide
    ctv_structure = None
    for name, struct in structures.items():
        if name.startswith("CTV_"):
            ctv_structure = struct
            break
    
    if not ctv_structure:
        st.error("No se encontró estructura CTV")
        return [], [], None

    # Calcular centroide de CTV
    all_points = np.concatenate([c['points'] for c in ctv_structure['contours']])
    ctv_centroid = np.mean(all_points, axis=0) + np.array(ctv_centroid_offset)

    # Generar posiciones de entrada de agujas en la base del cilindro, alineadas con CTV
    needle_entries = []
    needle_trajectories = []
    angles = np.linspace(0, 2 * np.pi, num_needles, endpoint=False)
    for angle in angles:
        x = needle_ring_radius * np.cos(angle) + ctv_centroid[0]
        y = needle_ring_radius * np.sin(angle) + ctv_centroid[1]
        z = ctv_centroid[2] - cylinder_length  # Base del cilindro por debajo del CTV
        needle_entries.append([x, y, z])

    # Calcular trayectorias
    healthy_structures = {name: s for name, s in structures.items() if not name.startswith("CTV_")}
    spacing = volume_info['spacing']
    margin = 2.0  # Margen de 2 mm

    for entry in needle_entries:
        entry = np.array(entry)
        target = ctv_centroid
        feasible = True
        best_trajectory = None
        best_angle = 0

        # Probar trayectoria recta y ajustes angulares
        for angle_adj in [0, 5, -5, 10, -10, 15, -15]:
            # Rotar el vector de dirección
            dir_vec = target - entry
            if angle_adj != 0:
                theta = np.deg2rad(angle_adj)
                rot_matrix = np.array([
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]
                ])
                dir_vec = rot_matrix @ dir_vec
            dir_vec = dir_vec / np.linalg.norm(dir_vec)
            end_point = entry + dir_vec * cylinder_length * 2  # Extender más allá del cilindro

            # Verificar colisiones con órganos sanos
            intersects = False
            for name, struct in healthy_structures.items():
                for contour in struct['contours']:
                    if line_intersects_contour(entry, end_point, contour['points'], margin):
                        intersects = True
                        break
                if intersects:
                    break
            
            if not intersects:
                best_trajectory = [entry, end_point]
                best_angle = angle_adj
                break

        if best_trajectory:
            needle_trajectories.append({
                'entry': best_trajectory[0],
                'end': best_trajectory[1],
                'feasible': True,
                'angle_adjustment': best_angle
            })
        else:
            needle_trajectories.append({
                'entry': entry,
                'end': end_point,
                'feasible': False,
                'angle_adjustment': 0
            })

    return needle_entries, needle_trajectories, ctv_centroid

def draw_slice(volume, slice_idx, plane, structures, volume_info, window, needle_trajectories=None, ctv_centroid=None, cylinder_diameter=None, cylinder_length=None, show_cylinder_2d=False, linewidth=2, show_names=True, invert_colors=False):
    """Dibuja un corte con contornos, trayectorias de agujas y cilindro"""
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.axis('off')

    if plane == 'axial':
        img = volume[slice_idx, :, :]
    elif plane == 'coronal':
        img = volume[:, slice_idx, :]
    elif plane == 'sagittal':
        img = volume[:, :, slice_idx]
    else:
        raise ValueError("Plano inválido")

    img = apply_window(img, window[1], window[0])
    if invert_colors:
        img = 1.0 - img

    ax.imshow(img, cmap='gray')

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

    ax.text(5, 15, f"{plane} - slice {slice_idx}", color='white', bbox=dict(facecolor='black', alpha=0.5))
    ax.text(5, 30, coord_label, color='yellow', bbox=dict(facecolor='black', alpha=0.5))

    if structures:
        for name, struct in structures.items():
            contour_drawn = 0
            color = struct['color']

            for contour in struct['contours']:
                raw_points = contour['points']

                if plane == 'axial':
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
                            polygon = patches.Polygon(pixel_points, closed=True, fill=False, edgecolor=color, linewidth=linewidth)
                            ax.add_patch(polygon)
                            contour_drawn += 1

                elif plane == 'coronal':
                    mask = np.abs(raw_points[:, 1] - current_slice_pos) < spacing[1]
                    if np.sum(mask) >= 3:
                        selected_points = raw_points[mask]
                        pixel_points = np.zeros((selected_points.shape[0], 2))
                        pixel_points[:, 0] = (selected_points[:, 0] - origin[0]) / spacing[0]
                        pixel_points[:, 1] = (selected_points[:, 2] - origin[2]) / spacing[2]
                        polygon = patches.Polygon(pixel_points, closed=True, fill=False, edgecolor=color, linewidth=linewidth)
                        ax.add_patch(polygon)
                        contour_drawn += 1

                elif plane == 'sagittal':
                    mask = np.abs(raw_points[:, 0] - current_slice_pos) < spacing[0]
                    if np.sum(mask) >= 3:
                        selected_points = raw_points[mask]
                        pixel_points = np.zeros((selected_points.shape[0], 2))
                        pixel_points[:, 0] = (selected_points[:, 1] - origin[1]) / spacing[1]
                        pixel_points[:, 1] = (selected_points[:, 2] - origin[2]) / spacing[2]
                        polygon = patches.Polygon(pixel_points, closed=True, fill=False, edgecolor=color, linewidth=linewidth)
                        ax.add_patch(polygon)
                        contour_drawn += 1

            if contour_drawn > 0 and show_names:
                ax.text(img.shape[1]/2, img.shape[0]/2, f"{name} ({contour_drawn})", color=color, fontsize=8, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))

    if needle_trajectories:
        for traj in needle_trajectories:
            start = traj['entry']
            end = traj['end']
            color = 'green' if traj['feasible'] else 'red'

            # Convertir a coordenadas de píxeles
            start_voxel = patient_to_voxel(np.array([start]), volume_info)[0]
            end_voxel = patient_to_voxel(np.array([end]), volume_info)[0]

            if plane == 'axial':
                z_voxel = slice_idx
                if (start_voxel[2] <= z_voxel <= end_voxel[2] or end_voxel[2] <= z_voxel <= start_voxel[2]):
                    t = (z_voxel - start_voxel[2]) / (end_voxel[2] - start_voxel[2] + 1e-10)
                    x = start_voxel[0] + t * (end_voxel[0] - start_voxel[0])
                    y = start_voxel[1] + t * (end_voxel[1] - start_voxel[1])
                    ax.plot([x], [y], marker='o', color=color, markersize=4)
            elif plane == 'coronal':
                y_voxel = slice_idx
                if (start_voxel[1] <= y_voxel <= end_voxel[1] or end_voxel[1] <= y_voxel <= start_voxel[1]):
                    t = (y_voxel - start_voxel[1]) / (end_voxel[1] - start_voxel[1] + 1e-10)
                    x = start_voxel[0] + t * (end_voxel[0] - start_voxel[0])
                    z = start_voxel[2] + t * (end_voxel[2] - start_voxel[2])
                    ax.plot([x], [z], marker='o', color=color, markersize=4)
            elif plane == 'sagittal':
                x_voxel = slice_idx
                if (start_voxel[0] <= x_voxel <= end_voxel[0] or end_voxel[0] <= x_voxel <= start_voxel[0]):
                    t = (x_voxel - start_voxel[0]) / (end_voxel[0] - start_voxel[0] + 1e-10)
                    y = start_voxel[1] + t * (end_voxel[1] - start_voxel[1])
                    z = start_voxel[2] + t * (end_voxel[2] - start_voxel[2])
                    ax.plot([y], [z], marker='o', color=color, markersize=4)

    if show_cylinder_2d and ctv_centroid is not None and cylinder_diameter is not None and cylinder_length is not None:
        cylinder_radius = cylinder_diameter / 2
        z_base = ctv_centroid[2] - cylinder_length
        z_top = ctv_centroid[2]

        if plane == 'axial':
            if z_base <= current_slice_pos <= z_top:
                center_x = (ctv_centroid[0] - origin[0]) / spacing[0]
                center_y = (ctv_centroid[1] - origin[1]) / spacing[1]
                radius_pixel = cylinder_radius / spacing[0]  # Asumiendo spacing[0] == spacing[1]
                circle = patches.Circle((center_x, center_y), radius_pixel, fill=False, edgecolor='blue', linewidth=linewidth, linestyle='--')
                ax.add_patch(circle)
        elif plane == 'coronal':
            if abs(ctv_centroid[1] - current_slice_pos) <= cylinder_radius:
                x_min = (ctv_centroid[0] - cylinder_radius - origin[0]) / spacing[0]
                x_max = (ctv_centroid[0] + cylinder_radius - origin[0]) / spacing[0]
                z_min = (z_base - origin[2]) / spacing[2]
                z_max = (z_top - origin[2]) / spacing[2]
                width = x_max - x_min
                height = z_max - z_min
                rect = patches.Rectangle((x_min, z_min), width, height, fill=False, edgecolor='blue', linewidth=linewidth, linestyle='--')
                ax.add_patch(rect)
        elif plane == 'sagittal':
            if abs(ctv_centroid[0] - current_slice_pos) <= cylinder_radius:
                y_min = (ctv_centroid[1] - cylinder_radius - origin[1]) / spacing[1]
                y_max = (ctv_centroid[1] + cylinder_radius - origin[1]) / spacing[1]
                z_min = (z_base - origin[2]) / spacing[2]
                z_max = (z_top - origin[2]) / spacing[2]
                width = y_max - y_min
                height = z_max - z_min
                rect = patches.Rectangle((y_min, z_min), width, height, fill=False, edgecolor='blue', linewidth=linewidth, linestyle='--')
                ax.add_patch(rect)

    plt.tight_layout()
    return fig

def draw_3d_visualization(structures, needle_trajectories, volume_info, cylinder_diameter, cylinder_length, ctv_centroid):
    """Crea una visualización 3D de estructuras, trayectorias de agujas y cilindro usando Plotly"""
    fig = go.Figure()

    # Visualizar estructuras (contornos)
    if structures:
        for name, struct in structures.items():
            color = struct['color']
            rgb_color = f"rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})"
            opacity = 0.3 if name == "BODY" else 0.6
            for contour in struct['contours']:
                pts = contour['points']
                # Submuestrear para rendimiento
                if len(pts) > 1000:
                    indices = np.random.choice(len(pts), 1000, replace=False)
                    pts = pts[indices]
                fig.add_trace(go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    mode='lines',
                    line=dict(color=rgb_color, width=2),
                    name=name,
                    opacity=opacity
                ))

    # Visualizar trayectorias de agujas
    if needle_trajectories:
        for i, traj in enumerate(needle_trajectories):
            start = traj['entry']
            end = traj['end']
            color = 'green' if traj['feasible'] else 'red'
            fig.add_trace(go.Scatter3d(
                x=[start[0], end[0]],
                y=[start[1], end[1]],
                z=[start[2], end[2]],
                mode='lines',
                line=dict(color=color, width=8),
                name=f"Needle {i+1} ({'Feasible' if traj['feasible'] else 'Infeasible'})"
            ))

    # Visualizar el cilindro como una malla semi-transparente
    cylinder_radius = cylinder_diameter / 2
    z_base = ctv_centroid[2] - cylinder_length
    z_top = ctv_centroid[2]

    # Crear puntos para el cilindro
    theta = np.linspace(0, 2*np.pi, 20)
    z = np.linspace(z_base, z_top, 10)
    theta, z = np.meshgrid(theta, z)
    x = cylinder_radius * np.cos(theta) + ctv_centroid[0]
    y = cylinder_radius * np.sin(theta) + ctv_centroid[1]
    fig.add_trace(go.Mesh3d(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        opacity=0.4,
        color='#4682B4',  # SteelBlue (grisáceo-azul)
        name='Cylinder',
        showlegend=True
    ))

    # Configurar el diseño
    fig.update_layout(
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data'
        ),
        showlegend=True,
        height=600
    )

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
            st.sidebar.markdown("**Visualización 3D (cilindro y trayectorias)**")
            show_3d_visualization = st.sidebar.checkbox("Mostrar visualización 3D", value=False)
            sync_slices = st.sidebar.checkbox("Sincronizar cortes", value=True)
            invert_colors = st.sidebar.checkbox("Invertir colores (Negativo)", value=False)
            show_structures = st.sidebar.checkbox("Mostrar estructuras", value=True)
            show_needle_trajectories = st.sidebar.checkbox("Mostrar trayectorias de agujas", value=True)
            show_cylinder_2d = st.sidebar.checkbox("Mostrar cilindro en 2D", value=True)

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
                    if isinstance(window_width, (list, tuple)):
                        window_width = window_width[0]
                    if isinstance(window_center, (list, tuple)):
                        window_center = window_center[0]
                except Exception:
                    window_width, window_center = 400, 40
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

            linewidth = st.sidebar.slider("Grosor líneas", 1, 8, 2)

            # Parámetros de agujas y tándem
            st.sidebar.markdown('<p class="sidebar-title">Configuración de Agujas</p>', unsafe_allow_html=True)
            num_needles = st.sidebar.slider("Número de agujas", min_value=1, max_value=12, value=6, step=1)
            tandem_diameter = st.sidebar.number_input("Diámetro del tándem (mm)", min_value=1.0, max_value=10.0, value=5.0, step=0.1)
            ctv_offset_x = st.sidebar.number_input("Offset CTV X (mm)", min_value=-50.0, max_value=50.0, value=0.0, step=0.1)
            ctv_offset_y = st.sidebar.number_input("Offset CTV Y (mm)", min_value=-50.0, max_value=50.0, value=0.0, step=0.1)
            ctv_offset_z = st.sidebar.number_input("Offset CTV Z (mm)", min_value=-50.0, max_value=50.0, value=0.0, step=0.1)

            # Generador de cilindro con agujas
            st.title("Generador de cilindro con punta tipo tampón (FreeCAD)")
            st.write("Esta aplicación genera código para crear un cilindro con una punta redondeada y orificios para agujas y tándem en FreeCAD.")

            # Parámetros en centímetros
            col1, col2 = st.columns(2)
            with col1:
                diametro_cm = st.slider("Diámetro (cm)", min_value=1.0, max_value=12.0, value=3.0, step=0.1)
                st.write(f"Diámetro seleccionado: {diametro_cm} cm ({diametro_cm*10} mm)")
            
            with col2:
                longitud_cm = st.slider("Longitud total (cm)", min_value=2.0, max_value=20.0, value=5.0, step=0.1)
                st.write(f"Longitud seleccionada: {longitud_cm} cm ({longitud_cm*10} mm)")
            
            with st.expander("Opciones avanzadas"):
                prop_punta = st.slider("Proporción de la punta (%)", min_value=10, max_value=50, value=20, step=5)
                st.write(f"La punta ocupará el {prop_punta}% de la longitud total")

            # Convertir a milímetros
            diametro_mm = round(diametro_cm * 10, 2)
            longitud_mm = round(longitud_cm * 10, 2)
            altura_punta = round(longitud_mm * prop_punta/100, 2)
            altura_cuerpo = round(longitud_mm - altura_punta, 2)
            needle_diameter = 3.0

            # Calcular trayectorias de agujas
            needle_entries, needle_trajectories, ctv_centroid = [], [], None
            if structures:
                needle_entries, needle_trajectories, ctv_centroid = compute_needle_trajectories(
                    num_needles, diametro_mm, longitud_mm, structures, volume_info,
                    ctv_centroid_offset=[ctv_offset_x, ctv_offset_y, ctv_offset_z]
                )
                if needle_trajectories:
                    infeasible = [t for t in needle_trajectories if not t['feasible']]
                    if infeasible:
                        st.warning(f"⚠️ {len(infeasible)} trayectorias de agujas intersectan órganos sanos.")

            # Mostrar las imágenes en tres columnas
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("### Axial")
                fig_axial = draw_slice(
                    volume, axial_idx, 'axial',
                    structures if show_structures else None,
                    volume_info,
                    (window_width, window_center),
                    needle_trajectories if show_needle_trajectories else None,
                    ctv_centroid,
                    diametro_mm,
                    longitud_mm,
                    show_cylinder_2d,
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
                    needle_trajectories if show_needle_trajectories else None,
                    ctv_centroid,
                    diametro_mm,
                    longitud_mm,
                    show_cylinder_2d,
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
                    needle_trajectories if show_needle_trajectories else None,
                    ctv_centroid,
                    diametro_mm,
                    longitud_mm,
                    show_cylinder_2d,
                    linewidth=linewidth,
                    invert_colors=invert_colors
                )
                st.pyplot(fig_sagittal)

            # Visualización 3D en la interfaz principal
            if show_3d_visualization and structures:
                st.markdown("### Visualización 3D")
                fig_3d = draw_3d_visualization(
                    structures,
                    needle_trajectories,
                    volume_info,
                    diametro_mm,
                    longitud_mm,
                    ctv_centroid
                )
                st.plotly_chart(fig_3d, use_container_width=True)

            # Generar código FreeCAD
            needle_code = ""
            needle_positions_str = ""
            for i, (entry, traj) in enumerate(zip(needle_entries, needle_trajectories)):
                feasible = traj['feasible']
                x, y = entry[0], entry[1]
                angle_adj = traj['angle_adjustment']
                needle_positions_str += f"  - Aguja {i+1}: (x={x:.2f}, y={y:.2f}, z=0), Ángulo ajuste: {angle_adj}°, {'Válida' if feasible else 'Inválida'}\n"
                if feasible:
                    # Crear cilindro para la aguja
                    needle_code += f"""
# Aguja {i+1}
needle_{i+1} = Part.makeCylinder({needle_diameter/2}, {longitud_mm*1.2}, App.Vector({x}, {y}, -{longitud_mm*0.1}), App.Vector(0, 0, 1))
cylinder = cylinder.cut(needle_{i+1})
"""
            
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
tandem_diameter = {tandem_diameter}
needle_diameter = {needle_diameter}
num_needles = {num_needles}

# Crear cuerpo cilíndrico
cylinder = Part.makeCylinder(radio, altura_cuerpo)

# Crear punta redondeada (semiesfera)
centro_semiesfera = App.Vector(0, 0, altura_cuerpo)
punta = Part.makeSphere(radio, centro_semiesfera)

# Cortar la mitad inferior de la esfera
box = Part.makeBox(diametro*2, diametro*2, altura_cuerpo)
box.translate(App.Vector(-diametro, -diametro, -altura_cuerpo))
punta = punta.cut(box)

# Unir cilindro y punta
cylinder = cylinder.fuse(punta)

# Crear orificio para el tándem
tandem = Part.makeCylinder(tandem_diameter/2, altura_total*1.2, App.Vector(0, 0, -altura_total*0.1), App.Vector(0, 0, 1))
cylinder = cylinder.cut(tandem)

# Crear orificios para las agujas
{needle_code}

# Crear un objeto en el documento de FreeCAD
objeto = doc.addObject("Part::Feature", "CilindroConPunta")
objeto.Shape = cylinder

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
print(f"- Diámetro del tándem: {{tandem_diameter}} mm")
print(f"- Número de agujas: {{num_needles}}")
print("Posiciones de agujas:")
{needle_positions_str}
"""
            # Mostrar el código
            st.subheader("Código FreeCAD generado")
            st.code(codigo, language="python")
                    
            # Botón de descarga para el código FreeCAD
            st.download_button(
                label="Descargar código FreeCAD (.py)",
                data=codigo,
                file_name="cilindro_punta_redondeada.py",
                mime="text/x-python"
            )

            # Generar y descargar reporte de agujas
            report = f"""Reporte de Agujas
================
Diámetro del cilindro: {diametro_mm} mm
Longitud total: {longitud_mm} mm
Diámetro del tándem: {tandem_diameter} mm
Número de agujas: {num_needles}
Diámetro de agujas: {needle_diameter} mm

Posiciones y estado de las agujas:
{needle_positions_str}
"""
            st.download_button(
                label="Descargar reporte de agujas (.txt)",
                data=report,
                file_name="reporte_agujas.txt",
                mime="text/plain"
            )

    else:
        st.warning("No se encontraron imágenes DICOM en el ZIP.")

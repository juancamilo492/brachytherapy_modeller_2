import streamlit as st
import base64
import numpy as np
import math

def generate_download_link(code):
    """Genera un link para descargar el código como archivo .py"""
    b64 = base64.b64encode(code.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="template_braquiterapia.py">Descargar código</a>'
    return href

st.image("Banner.png")

st.title("Generador de Guías de Braquiterapia")
st.subheader("Personalización de parámetros para impresión 3D")

# Parámetros generales - eliminando los parámetros de curvatura
st.header("Parámetros Generales")
col1, col2 = st.columns(2)
with col1:
    diametro_base = st.number_input("Diámetro de la base (mm)", value=50.0, step=0.5)
    altura_base = st.number_input("Altura de la base (mm)", value=10.0, step=0.5)
    
# Valores predeterminados para la curvatura (no visibles para el usuario)
radio_curvatura = 150.0
angulo_curvatura = 30.0

# Parámetros del cilindro central
st.header("Cilindro Central")
col1, col2 = st.columns(2)
with col1:
    diametro_cilindro = st.number_input("Diámetro del cilindro (mm)", value=20.0, step=0.5)
    altura_cilindro = st.number_input("Altura del cilindro (mm)", value=80.0, step=0.5)
    
with col2:
    diametro_orificio_central = st.number_input("Diámetro del orificio central (mm)", value=5.0, step=0.1)
    forma_punta = st.selectbox("Forma de la punta del cilindro", ["Redondeada", "Cónica", "Plana"])

# Parámetros de los orificios para agujas
st.header("Orificios para Agujas")
col1, col2 = st.columns(2)
with col1:
    num_orificios_base = st.slider("Número de orificios en la base", 5, 30, 15)
    diametro_orificios = st.number_input("Diámetro de los orificios (mm)", value=1.5, step=0.1)

with col2:
    incluir_agujas_cilindro = st.checkbox("Incluir agujas que atraviesen el cilindro", value=True)
    num_orificios_cilindro = st.slider("Número de orificios en el cilindro", 0, 12, 6, disabled=not incluir_agujas_cilindro)

# Patrones de distribución
patron_distribucion = st.selectbox("Patrón de distribución de orificios", 
                                  ["Equidistante", "Rectangular", "Circular", "Hexagonal"])

distribucion_equidistante = st.slider("Distancia entre orificios (mm)", 3.0, 10.0, 5.0, 0.5, 
                                     disabled=patron_distribucion != "Equidistante")

# Opciones avanzadas
with st.expander("Opciones avanzadas"):
    tolerancia = st.number_input("Tolerancia para impresión 3D (mm)", value=0.1, step=0.01)
    material = st.selectbox("Material recomendado", ["PLA", "ABS", "PETG", "Resina médica"])
    infill = st.slider("Porcentaje de relleno recomendado", 50, 100, 80)
    
    altura_max_agujas = st.slider("Altura máxima de las agujas (mm)", 
                                value=int(altura_cilindro * 0.9), 
                                min_value=int(altura_base), 
                                max_value=int(altura_base + altura_cilindro))

# Botón para generar código
if st.button("Generar Código FreeCAD"):
    # Generar el código Python para FreeCAD
    code = f"""# Código generado automáticamente para FreeCAD
# Template de Braquiterapia para tratamiento de cáncer de cérvix
# Parámetros personalizados:
# - Diámetro base: {diametro_base} mm
# - Altura base: {altura_base} mm
# - Diámetro cilindro: {diametro_cilindro} mm
# - Altura cilindro: {altura_cilindro} mm
# - Patrón distribución: {patron_distribucion}
# - Incluir agujas en cilindro: {"Sí" if incluir_agujas_cilindro else "No"}

import FreeCAD as App
import Part
import Draft
import math
from FreeCAD import Base
import random

# Crear un nuevo documento
doc = App.newDocument("TemplateBraquiterapia")

# Calcular altura total según la forma de la punta
altura_total = {altura_base + altura_cilindro}
if "{forma_punta}" == "Redondeada" or "{forma_punta}" == "Cónica":
    altura_punta = {diametro_cilindro/2}
    altura_total += altura_punta
else:
    altura_punta = 0

# Crear la base con curvatura
def crear_base_curva():
    # Crear un cilindro base
    cilindro = Part.makeCylinder({diametro_base/2}, {altura_base})
    
    # Parámetros de curvatura predefinidos
    radio_curvatura = 150.0  # mm
    angulo_curvatura = 30.0  # grados
    
    # Crear un plano de corte para la curvatura
    plano = Part.makeBox({diametro_base*2}, {diametro_base*2}, {diametro_base*2})
    plano.translate(Base.Vector(-{diametro_base}, -{diametro_base}, -radio_curvatura))
    plano.rotate(Base.Vector(0,0,0), Base.Vector(1,0,0), angulo_curvatura)
    
    # Cortar la base con el plano
    base_curva = cilindro.cut(plano)
    return base_curva

# Crear el cilindro central
def crear_cilindro_central():
    cilindro = Part.makeCylinder({diametro_cilindro/2}, {altura_cilindro})
    cilindro.translate(Base.Vector(0, 0, {altura_base}))
    
    # Crear la punta según la forma seleccionada
    if "{forma_punta}" == "Redondeada":
        punta = Part.makeSphere({diametro_cilindro/2})
        punta.translate(Base.Vector(0, 0, {altura_base + altura_cilindro}))
    elif "{forma_punta}" == "Cónica":
        punta = Part.makeCone({diametro_cilindro/2}, 0, {diametro_cilindro/2})
        punta.translate(Base.Vector(0, 0, {altura_base + altura_cilindro}))
    
    # Combinar cilindro y punta si no es plana
    if "{forma_punta}" != "Plana":
        cilindro = cilindro.fuse(punta)
    
    return cilindro

# Crear el orificio central
def crear_orificio_central():
    # Crear un orificio que sea significativamente más largo que la altura total
    # para asegurar que atraviese completamente todas las partes
    margen_seguridad = 20  # mm adicionales en cada extremo
    longitud_orificio = altura_total + 2 * margen_seguridad
    
    # Posicionar el orificio para que sobresalga por ambos extremos
    orificio = Part.makeCylinder({diametro_orificio_central/2}, longitud_orificio)
    orificio.translate(Base.Vector(0, 0, -margen_seguridad))
    
    return orificio

# Crear los orificios para las agujas
def crear_orificios_agujas_base():
    orificios = []
    
    # Margen de seguridad para asegurar que los orificios atraviesen completamente
    margen_seguridad = 20  # mm adicionales en cada extremo
    longitud_orificio = altura_total + 2 * margen_seguridad
    
    if "{patron_distribucion}" == "Equidistante":
        # Anillos concéntricos con distancia definida entre orificios
        distancia_entre_orificios = {distribucion_equidistante}
        radio_inicial = {diametro_cilindro/2} + distancia_entre_orificios
        radio_final = {diametro_base/2} - distancia_entre_orificios
        
        # Calcular número de anillos basado en la distancia entre orificios
        num_anillos = max(1, int((radio_final - radio_inicial) / distancia_entre_orificios))
        
        count = 0
        for i in range(num_anillos):
            radio_anillo = radio_inicial + i * (radio_final - radio_inicial) / max(1, num_anillos - 1)
            
            # Calcular número de orificios en este anillo
            circunferencia = 2 * math.pi * radio_anillo
            orificios_en_anillo = max(1, int(circunferencia / distancia_entre_orificios))
            
            if count + orificios_en_anillo > {num_orificios_base}:
                orificios_en_anillo = {num_orificios_base} - count
            
            for j in range(orificios_en_anillo):
                if count < {num_orificios_base}:
                    angulo = j * 360 / orificios_en_anillo
                    x = radio_anillo * math.cos(math.radians(angulo))
                    y = radio_anillo * math.sin(math.radians(angulo))
                    
                    # Crear orificio que sobresalga por ambos extremos
                    orificio = Part.makeCylinder({diametro_orificios/2}, longitud_orificio)
                    orificio.translate(Base.Vector(x, y, -margen_seguridad))
                    orificios.append(orificio)
                    count += 1
    
    elif "{patron_distribucion}" == "Rectangular":
        # Calcular cuántas filas y columnas necesitamos
        filas = int(math.sqrt({num_orificios_base}))
        cols = int({num_orificios_base} / filas) + 1
        
        # Espacio entre orificios
        espacio_x = {diametro_base - diametro_cilindro} / 2 / (cols + 1)
        espacio_y = {diametro_base - diametro_cilindro} / 2 / (filas + 1)
        
        # Posición inicial
        x_start = -{diametro_base/2} + {diametro_cilindro/2} + espacio_x
        y_start = -{diametro_base/2} + {diametro_cilindro/2} + espacio_y
        
        count = 0
        for i in range(filas):
            for j in range(cols):
                if count < {num_orificios_base}:
                    x = x_start + j * espacio_x * 2
                    y = y_start + i * espacio_y * 2
                    
                    # Verificar si está dentro del área útil (fuera del cilindro central)
                    if (x**2 + y**2)**0.5 > {diametro_cilindro/2 + diametro_orificios}:
                        # Crear orificio que sobresalga por ambos extremos
                        orificio = Part.makeCylinder({diametro_orificios/2}, longitud_orificio)
                        orificio.translate(Base.Vector(x, y, -margen_seguridad))
                        orificios.append(orificio)
                        count += 1
    
    elif "{patron_distribucion}" == "Circular":
        # Distribuir en anillos concéntricos
        radio_primer_anillo = {diametro_cilindro/2} + {diametro_orificios} * 2
        radio_ultimo_anillo = {diametro_base/2} - {diametro_orificios} * 2
        
        num_anillos = 3
        orificios_por_anillo = {num_orificios_base} // num_anillos
        
        for anillo in range(num_anillos):
            radio = radio_primer_anillo + anillo * (radio_ultimo_anillo - radio_primer_anillo)/(num_anillos-1)
            for i in range(orificios_por_anillo):
                angulo = i * 360 / orificios_por_anillo
                x = radio * math.cos(math.radians(angulo))
                y = radio * math.sin(math.radians(angulo))
                
                # Crear orificio que sobresalga por ambos extremos
                orificio = Part.makeCylinder({diametro_orificios/2}, longitud_orificio)
                orificio.translate(Base.Vector(x, y, -margen_seguridad))
                orificios.append(orificio)
    
    elif "{patron_distribucion}" == "Hexagonal":
        # Patrón hexagonal con distancia entre orificios
        radio_inicial = {diametro_cilindro/2} + {diametro_orificios} * 2
        distancia = {distribucion_equidistante}
        radio_util = {diametro_base/2} - {diametro_orificios} * 2
        
        # Usar coordenadas hexagonales
        orificios_creados = 0
        for q in range(-10, 11):
            for r in range(-10, 11):
                # Calcular coordenadas cartesianas desde coordenadas hexagonales
                x = distancia * (3/2 * q)
                y = distancia * (math.sqrt(3)/2 * q + math.sqrt(3) * r)
                
                # Verificar si está dentro del área útil
                distancia_al_centro = math.sqrt(x**2 + y**2)
                if (distancia_al_centro <= radio_util and 
                    distancia_al_centro >= radio_inicial and 
                    orificios_creados < {num_orificios_base}):
                    # Crear orificio que sobresalga por ambos extremos
                    orificio = Part.makeCylinder({diametro_orificios/2}, longitud_orificio)
                    orificio.translate(Base.Vector(x, y, -margen_seguridad))
                    orificios.append(orificio)
                    orificios_creados += 1
                    
    return orificios

# Crear orificios que atraviesan el cilindro
def crear_orificios_cilindro():
    orificios = []
    
    # Margen de seguridad para asegurar que los orificios atraviesen completamente
    margen_seguridad = 20  # mm adicionales en cada extremo
    longitud_orificio = altura_total + 2 * margen_seguridad
    
    if {incluir_agujas_cilindro}:
        # Distribuir orificios equidistantes alrededor del cilindro
        for i in range({num_orificios_cilindro}):
            # Calcular posición angular
            angulo = i * 360 / {num_orificios_cilindro}
            
            # Calcular posición en el perímetro del cilindro
            radio = {diametro_cilindro/2} * 0.7  # Usar 70% del radio para mantener distancia del borde
            x = radio * math.cos(math.radians(angulo))
            y = radio * math.sin(math.radians(angulo))
            
            # Crear orificio que sobresalga por ambos extremos
            orificio = Part.makeCylinder({diametro_orificios/2}, longitud_orificio)
            orificio.translate(Base.Vector(x, y, -margen_seguridad))
            orificios.append(orificio)
    
    return orificios

# Ejecutar el modelado
base = crear_base_curva()
base_obj = doc.addObject("Part::Feature", "Base")
base_obj.Shape = base

cilindro = crear_cilindro_central()
cilindro_obj = doc.addObject("Part::Feature", "Cilindro")
cilindro_obj.Shape = cilindro

# Fusionar base y cilindro
template = base.fuse(cilindro)

# Crear orificios
orificios = []

# Crear el orificio central
orificio_central = crear_orificio_central()
orificios.append(orificio_central)

# Crear orificios para agujas en la base
orificios_base = crear_orificios_agujas_base()
orificios.extend(orificios_base)

# Crear orificios que atraviesan el cilindro
orificios_cilindro = crear_orificios_cilindro()
orificios.extend(orificios_cilindro)

# Cortar todos los orificios de una sola vez
for orificio in orificios:
    template = template.cut(orificio)

# Crear el objeto final
template_obj = doc.addObject("Part::Feature", "TemplateBraquiterapia")
template_obj.Shape = template

# Verificar que los orificios están correctamente hechos
# Añadir un paso de refinamiento de malla para mejorar la representación de los orificios
try:
    import MeshPart
    mesh = doc.addObject("Mesh::Feature", "Mesh")
    mesh.Mesh = MeshPart.meshFromShape(Shape=template, LinearDeflection=0.1, AngularDeflection=0.1)
    # Opcional: Refinar la malla cerca de los orificios
    mesh.Mesh.refine()
except:
    print("No se pudo refinar la malla. Continúa con el modelo normal.")

# Actualizar la vista
doc.recompute()
try:
    import FreeCADGui
    FreeCADGui.SendMsgToActiveView("ViewFit")
    # Mostrar bordes para mejor visualización de los orificios
    FreeCADGui.ActiveDocument.ActiveView.setEdgesDisplayMode("Flat Lines")
except:
    pass

print("Template de braquiterapia generado con éxito")
"""
    
    st.code(code, language="python")
    
    # Generar enlace de descarga
    st.markdown(generate_download_link(code), unsafe_allow_html=True)
    
    # Instrucciones para el usuario
    st.success("Código generado con éxito. Copia este código o descárgalo y pégalo en la consola Python de FreeCAD.")
    
    with st.expander("Instrucciones de uso"):
        st.markdown("""
        ### Pasos para usar este código en FreeCAD:
        
        1. Abre FreeCAD
        2. Ve a Ver > Paneles > Consola Python
        3. Copia el código generado
        4. Pégalo en la consola Python y presiona Enter
        5. El modelo se generará automáticamente
        6. Puedes exportarlo como archivo STL para impresión 3D
        
        ### Recomendaciones:
        - Utiliza FreeCAD versión 0.19 o superior
        - Verifica las dimensiones antes de imprimir
        - Para materiales de uso médico, consulta con un especialista
        """)

# Verificación de parámetros y recomendaciones
st.sidebar.title("Verificación y Recomendaciones")

# Verificar proporciones
area_base = 3.14159 * (diametro_base/2)**2
area_cilindro = 3.14159 * (diametro_cilindro/2)**2
proporcion = area_cilindro / area_base

if proporcion > 0.5:
    st.sidebar.warning("⚠️ El cilindro ocupa más del 50% de la base. Considera reducir su diámetro.")
else:
    st.sidebar.success("✅ Proporción cilindro-base adecuada.")

# Verificar diámetro orificios
if diametro_orificios < 1.0:
    st.sidebar.warning("⚠️ Orificios muy pequeños pueden ser difíciles de imprimir con precisión.")
elif diametro_orificios > 3.0:
    st.sidebar.warning("⚠️ Orificios muy grandes pueden comprometer la integridad estructural.")
else:
    st.sidebar.success("✅ Diámetro de orificios adecuado.")

# Verificar número de orificios
if incluir_agujas_cilindro and num_orificios_cilindro > 8:
    st.sidebar.warning("⚠️ Un número elevado de orificios en el cilindro puede debilitar su estructura.")

# Mostrar imagen de ejemplo según el tipo de distribución seleccionado
patron_images = {
    "Equidistante": "https://via.placeholder.com/200x200?text=Distribucion+Equidistante",
    "Rectangular": "https://via.placeholder.com/200x200?text=Distribucion+Rectangular",
    "Circular": "https://via.placeholder.com/200x200?text=Distribucion+Circular",
    "Hexagonal": "https://via.placeholder.com/200x200?text=Distribucion+Hexagonal"
}

# Instrucciones generales
st.sidebar.subheader("Flujo de trabajo recomendado")
st.sidebar.markdown("""
1. Ajusta los parámetros según las necesidades médicas
2. Genera el código
3. Verifica el modelo en FreeCAD
4. Realiza ajustes si es necesario
5. Exporta a STL
6. Imprime con material biocompatible
""")

# Información sobre el proyecto
st.sidebar.info("Esta herramienta está diseñada para estandarizar el modelado de templates de braquiterapia para tratamiento de cáncer de cérvix.")


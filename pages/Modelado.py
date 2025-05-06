import streamlit as st

st.title("Generador de cilindro con punta tipo tampón (FreeCAD)")

# Parámetros en centímetros
diametro_cm = st.slider("Diámetro (cm)", min_value=1.0, max_value=12.0, value=3.0, step=0.1)
longitud_cm = st.slider("Longitud total (cm)", min_value=2.0, max_value=20.0, value=5.0, step=0.1)

# Convertir a milímetros
diametro_mm = round(diametro_cm * 10, 2)
longitud_mm = round(longitud_cm * 10, 2)

# Definir la longitud del cuerpo y la altura de la punta redondeada (20% del total)
altura_punta = round(longitud_mm * 0.2, 2)
altura_cuerpo = round(longitud_mm - altura_punta, 2)

# Generar código FreeCAD
codigo = f"""import FreeCAD as App
import Part

doc = App.newDocument()

# Parámetros
diametro = {diametro_mm}
radio = diametro / 2
altura_total = {longitud_mm}
altura_cuerpo = {altura_cuerpo}
altura_punta = {altura_punta}

# Crear cuerpo cilíndrico
cuerpo = Part.makeCylinder(radio, altura_cuerpo)

# Crear perfil de la punta redondeada (arco)
p1 = App.Vector(0, 0, altura_cuerpo)
p2 = App.Vector(radio, 0, altura_cuerpo)
p3 = App.Vector(0, 0, altura_total)

arco = Part.Arc(p1, p2, p3).toShape()
linea = Part.makeLine(App.Vector(0, 0, altura_cuerpo), App.Vector(0, 0, altura_total))
perfil = Part.Wire([linea, arco])

# Revolver perfil para crear la punta
punta = perfil.revolve(App.Vector(0, 0, 0), App.Vector(0, 0, 1), 360)

# Unir cilindro y punta
solido = cuerpo.fuse(punta)

# Mostrar en FreeCAD
Part.show(solido)
App.ActiveDocument.recompute()
"""

# Mostrar el código
st.code(codigo, language="python")

# Botón de descarga
st.download_button(
    label="Descargar código FreeCAD (.py)",
    data=codigo,
    file_name="cilindro_punta_redondeada.py",
    mime="text/x-python"
)

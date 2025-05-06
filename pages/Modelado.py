import streamlit as st

st.title("Generador de cilindro con punta redondeada (FreeCAD)")

# Parámetros de entrada
diametro = st.slider("Diámetro (mm)", min_value=5, max_value=50, value=10)
longitud = st.slider("Longitud total (mm)", min_value=10, max_value=100, value=50)

# Generar el código de FreeCAD como texto
codigo = f"""import FreeCAD as App
import Part

# Parámetros
diametro = {diametro}
longitud = {longitud}
radio = diametro / 2
cuerpo_cilindro = longitud - radio

# Crear cilindro
cilindro = Part.makeCylinder(radio, cuerpo_cilindro)

# Crear media esfera
esfera = Part.makeSphere(radio)
corte_plano = Part.makeBox(diametro, diametro, radio)
corte_plano.translate(App.Vector(-radio, -radio, 0))
media_esfera = esfera.cut(corte_plano)

# Posicionar la media esfera en el extremo del cilindro
media_esfera.translate(App.Vector(0, 0, cuerpo_cilindro))

# Unir ambas partes
solido_final = cilindro.fuse(media_esfera)

# Mostrar en FreeCAD
Part.show(solido_final)
App.ActiveDocument.recompute()
"""

# Mostrar el código en la app
st.code(codigo, language="python")

# Descargar como archivo .py
st.download_button(
    label="Descargar código FreeCAD (.py)",
    data=codigo,
    file_name="cilindro_redondeado.py",
    mime="text/x-python"
)

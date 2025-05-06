import streamlit as st
import cadquery as cq
import tempfile
import os

# Parámetros desde la interfaz de usuario
st.title("Cilindro con punta redondeada (tipo tampón)")

diametro = st.slider("Diámetro (mm)", min_value=5, max_value=50, value=10)
longitud = st.slider("Longitud total (mm)", min_value=10, max_value=100, value=50)

# Cálculos derivados
radio = diametro / 2
cuerpo_cilindro = longitud - radio  # parte recta del cilindro, excluyendo la punta

# Modelado con CadQuery
modelo = (
    cq.Workplane("XY")
    .circle(radio)
    .extrude(cuerpo_cilindro)  # cuerpo del cilindro
    .faces(">Z")
    .workplane()
    .sphere(radio)  # esfera completa
    .cutBlind(-radio)  # dejar solo la mitad de la esfera (punta redondeada)
)

# Exportar a STL
with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp:
    cq.exporters.export(modelo, tmp.name)
    st.download_button("Descargar STL", tmp.read(), file_name="cilindro_redondeado.stl")

st.success("Modelo generado con éxito.")

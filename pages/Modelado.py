import streamlit as st

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

# Generar código FreeCAD
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

# Vista
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

# Instrucciones de uso
st.subheader("Cómo usar este código en FreeCAD")
st.markdown("""
1. Copie el código generado
2. Abra FreeCAD
3. Vaya a Vista → Paneles → Consola Python
4. Pegue el código en la consola y presione Enter
5. El objeto se creará automáticamente en su documento

**Nota:** Si recibe un error relacionado con `Gui`, simplemente elimine o comente las líneas que contienen `Gui` ya que esas son solo para mejorar la visualización.
""")

# Botón de descarga
st.download_button(
    label="Descargar código FreeCAD (.py)",
    data=codigo,
    file_name="cilindro_punta_redondeada.py",
    mime="text/x-python"
)

# Visualización esquemática
st.subheader("Vista previa esquemática")
col1, col2 = st.columns([1, 2])
with col1:
    st.write("Dimensiones:")
    st.write(f"- Diámetro: {diametro_mm} mm")
    st.write(f"- Altura total: {longitud_mm} mm")
    st.write(f"- Altura del cuerpo: {altura_cuerpo} mm")
    st.write(f"- Altura de la punta: {altura_punta} mm")

with col2:
    # Código ASCII art básico para representar el cilindro
    altura_ascii = min(10, int(longitud_cm))
    diametro_ascii = min(20, int(diametro_cm * 3))
    
    ascii_art = ""
    # Punta redondeada
    for i in range(int(altura_ascii * prop_punta/100)):
        espacios = abs(int(diametro_ascii/2) - i)
        ascii_art += " " * espacios + "o" * (diametro_ascii - espacios*2) + "\n"
    
    # Cuerpo cilíndrico
    for i in range(int(altura_ascii * (100-prop_punta)/100)):
        ascii_art += " " + "|" + "-" * (diametro_ascii-4) + "|" + "\n"
    
    st.text(ascii_art)

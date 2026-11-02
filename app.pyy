
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
import math

# ----------------------------------
# PARKAI - KI EINPARKASSISTENT
# Version 0.1
# ----------------------------------

st.set_page_config(
    page_title="ParkAI",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 ParkAI")
st.subheader("Dein intelligenter Einparkassistent")

st.write(
    "Analysiere Parklücken und simuliere "
    "dein Einparkmanöver."
)

# ----------------------------------
# SEITENLEISTE
# ----------------------------------

st.sidebar.header("🚘 Fahrzeug")

car_length = st.sidebar.slider(
    "Fahrzeuglänge (m)",
    3.0, 6.0, 4.5, 0.1
)

car_width = st.sidebar.slider(
    "Fahrzeugbreite (m)",
    1.5, 2.5, 1.8, 0.1
)

turning_radius = st.sidebar.slider(
    "Wendekreisradius (m)",
    3.0, 8.0, 5.0, 0.1
)

st.sidebar.header("📏 Parklücke")

parking_length = st.sidebar.slider(
    "Länge der Parklücke (m)",
    4.0, 10.0, 6.0, 0.1
)

parking_width = st.sidebar.slider(
    "Breite der Parklücke (m)",
    1.8, 4.0, 2.5, 0.1
)

# ----------------------------------
# PARKLÜCKE ANALYSIEREN
# ----------------------------------

st.header("1. Parklücke analysieren")

fits = (
    parking_length >= car_length + 1.0
    and parking_width >= car_width + 0.2
)

if fits:
    st.success(
        "Die Parklücke ist anhand der eingegebenen "
        "Maße groß genug für das Fahrzeug."
    )
else:
    st.error(
        "Die Parklücke ist anhand der eingegebenen "
        "Maße zu klein."
    )

st.caption(
    "Dies ist eine vereinfachte geometrische "
    "Schätzung, keine Garantie für sicheres Einparken."
)

# ----------------------------------
# FOTO UPLOAD
# ----------------------------------

st.header("2. Parklücke fotografieren")

uploaded_file = st.file_uploader(
    "Lade ein Foto deiner Parklücke hoch",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Dein hochgeladenes Bild",
        use_container_width=True
    )

    if st.button("🤖 Foto mit KI analysieren"):

        with st.spinner("KI analysiert das Bild..."):

            try:
                model = YOLO("yolo26n.pt")

                results = model.predict(
                    image,
                    conf=0.35,
                    verbose=False
                )

                annotated_image = results[0].plot()

                annotated_image = annotated_image[:, :, ::-1]

                st.image(
                    annotated_image,
                    caption="KI-Objekterkennung",
                    use_container_width=True
                )

                detected_classes = [
                    results[0].names[int(cls)]
                    for cls in results[0].boxes.cls
                ]

                car_count = detected_classes.count("car")

                st.success(
                    f"KI hat {car_count} Autos erkannt."
                )

                st.write(
                    "Erkannte Objekte:",
                    ", ".join(detected_classes)
                    if detected_classes
                    else "Keine Objekte erkannt"
                )

            except Exception as error:
                st.error(
                    f"Fehler bei der KI-Analyse: {error}"
                )

# ----------------------------------
# EINPARK-SIMULATION
# ----------------------------------

st.header("3. Einpark-Simulation")

st.write(
    "Hier wird eine vereinfachte "
    "Parklücken-Situation dargestellt."
)

def draw_parking():

    fig, ax = plt.subplots(figsize=(10, 5))

    # Parklücke
    ax.add_patch(
        plt.Rectangle(
            (0, 0),
            parking_length,
            parking_width,
            fill=False,
            linewidth=2
        )
    )

    # Eigenes Fahrzeug
    car_x = (
        parking_length - car_length
    ) / 2

    car_y = (
        parking_width - car_width
    ) / 2

    ax.add_patch(
        plt.Rectangle(
            (car_x, car_y),
            car_length,
            car_width,
            color="royalblue",
            alpha=0.8
        )
    )

    ax.text(
        parking_length / 2,
        parking_width / 2,
        "DEIN AUTO",
        color="white",
        ha="center",
        va="center"
    )

    ax.set_xlim(-1, parking_length + 1)
    ax.set_ylim(-1, parking_width + 1)

    ax.set_aspect("equal")
    ax.set_xlabel("Meter")
    ax.set_ylabel("Meter")
    ax.set_title("ParkAI - Parklücken-Simulation")
    ax.grid(True, alpha=0.2)

    return fig


if fits:

    st.pyplot(draw_parking())

else:

    st.warning(
        "Simulation nicht möglich: "
        "Das Fahrzeug ist größer als die Parklücke."
    )

# ----------------------------------
# EINPARK-ANLEITUNG
# ----------------------------------

st.header("4. Einpark-Anleitung")

if fits:

    st.write("Ein vereinfachter Einparkplan:")

    st.info("1. Positioniere das Fahrzeug vor der Parklücke.")

    st.info("2. Beginne mit dem Einparkmanöver.")

    st.info("3. Korrigiere deine Position.")

    st.info("4. Richte das Fahrzeug innerhalb der Lücke aus.")

else:

    st.error(
        "Ein Einparkmanöver wird nicht empfohlen."
    )

# ----------------------------------
# FOOTER
# ----------------------------------

st.divider()

st.caption(
    "ParkAI | Informatikprojekt | Version 0.1"
)

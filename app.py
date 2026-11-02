import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

# ------------------------------------------------------------
# ParkAI – Lern- und Simulationsprojekt
# ------------------------------------------------------------

st.set_page_config(
    page_title="ParkAI – Parkassistent",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 ParkAI – Dein KI-Parkassistent")
st.write(
    "Dieses Projekt zeigt, wie Computer Vision und einfache Geometrie "
    "beim Lernen des Parkens helfen können."
)

st.warning(
    "Wichtig: ParkAI ist eine Lernsimulation. Die Hinweise und Spiegelbilder "
    "sind nicht für das echte Fahren geeignet und ersetzen weder Fahrschule "
    "noch die direkte Kontrolle der Umgebung."
)

# ------------------------------------------------------------
# Seitenleiste: Fahrzeug und Parklücke
# ------------------------------------------------------------

st.sidebar.header("⚙️ Fahrzeugdaten")

car_length = st.sidebar.slider(
    "Fahrzeuglänge (m)", min_value=3.0, max_value=6.0,
    value=4.4, step=0.1
)
car_width = st.sidebar.slider(
    "Fahrzeugbreite (m)", min_value=1.5, max_value=2.5,
    value=1.8, step=0.1
)
turn_radius = st.sidebar.slider(
    "Wendekreis-Durchmesser (m)", min_value=8.0, max_value=14.0,
    value=10.5, step=0.5
)

st.sidebar.header("🅿️ Parklücke")

gap_length = st.sidebar.slider(
    "Länge der Parklücke (m)", min_value=4.0, max_value=10.0,
    value=6.0, step=0.1
)
gap_width = st.sidebar.slider(
    "Breite der Parklücke (m)", min_value=1.8, max_value=4.0,
    value=2.5, step=0.1
)

# ------------------------------------------------------------
# Platzprüfung
# ------------------------------------------------------------

st.header("1. Passt das Fahrzeug in die Parklücke?")

required_length = car_length + 1.0
required_width = car_width + 0.2

length_ok = gap_length >= required_length
width_ok = gap_width >= required_width
fits = length_ok and width_ok

col1, col2, col3 = st.columns(3)
col1.metric("Parklückenlänge", f"{gap_length:.1f} m")
col2.metric("Benötigte Länge (vereinfachte Annahme)", f"{required_length:.1f} m")
col3.metric("Parklückenbreite", f"{gap_width:.1f} m")

if fits:
    st.success("Nach dieser vereinfachten Maßprüfung ist genügend Platz vorhanden.")
else:
    st.error("Nach dieser vereinfachten Maßprüfung ist die Parklücke zu klein.")

st.caption(
    "Die Prüfung berücksichtigt nur vereinfachte Maße und keine realen "
    "Lenk-, Sicherheits- oder Rangierbedingungen."
)

# ------------------------------------------------------------
# Parklücke visualisieren
# ------------------------------------------------------------

st.header("2. Parklücke visualisieren")

fig, ax = plt.subplots(figsize=(10, 4))

# Parklückenbegrenzungen
ax.plot([0, gap_length], [0, 0], linewidth=3)
ax.plot([0, gap_length], [gap_width, gap_width], linewidth=3)
ax.plot([0, 0], [0, gap_width], linestyle="--")
ax.plot([gap_length, gap_length], [0, gap_width], linestyle="--")

# Fahrzeug als Rechteck in der Mitte (nur schematisch)
car_x = max(0, (gap_length - car_length) / 2)
car_y = max(0, (gap_width - car_width) / 2)

if car_length <= gap_length and car_width <= gap_width:
    car = plt.Rectangle(
        (car_x, car_y),
        car_length,
        car_width,
        fill=False,
        linewidth=2,
        hatch="//"
    )
    ax.add_patch(car)
    ax.text(
        car_x + car_length / 2,
        car_y + car_width / 2,
        "Auto (schematisch)",
        ha="center",
        va="center"
    )
else:
    ax.text(
        gap_length / 2,
        gap_width / 2,
        "Fahrzeug ist größer als die Lücke",
        ha="center",
        va="center"
    )

ax.set_xlim(-0.5, gap_length + 0.5)
ax.set_ylim(-0.5, gap_width + 0.5)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("Länge (m)")
ax.set_ylabel("Breite (m)")
ax.set_title("Vereinfachte Draufsicht")
ax.grid(True, alpha=0.3)

st.pyplot(fig)
plt.close(fig)

# ------------------------------------------------------------
# KI-Fahrzeugerkennung
# ------------------------------------------------------------

st.header("3. Fahrzeug auf einem Foto erkennen")

uploaded_file = st.file_uploader(
    "Lade ein Bild mit Fahrzeugen hoch",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_container_width=True)

    if st.button("🔍 Fahrzeuge mit KI erkennen"):
        with st.spinner("KI wird geladen und analysiert das Bild ..."):
            try:
                model = YOLO("yolo26n.pt")
                results = model.predict(image, conf=0.35, verbose=False)

                annotated = results[0].plot()
                annotated_rgb = annotated[:, :, ::-1]
                st.image(
                    annotated_rgb,
                    caption="Erkennungsergebnis",
                    use_container_width=True
                )

                names = results[0].names
                boxes = results[0].boxes
                vehicle_classes = {"car", "truck", "bus", "motorcycle"}
                detected = []

                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        class_name = names.get(class_id, str(class_id))
                        if class_name in vehicle_classes:
                            detected.append(class_name)

                st.write(f"Erkannte Fahrzeuge: **{len(detected)}**")
                if detected:
                    st.write("Klassen: " + ", ".join(detected))
                else:
                    st.info(
                        "Keine Fahrzeuge erkannt. Das kann an Perspektive, "
                        "Bildqualität oder Modellgrenzen liegen."
                    )

            except Exception as error:
                st.error("Die KI-Erkennung konnte nicht ausgeführt werden.")
                st.code(str(error))
                st.info(
                    "Prüfe, ob die Pakete in requirements.txt installiert sind "
                    "und starte die App nach einem Commit neu."
                )

# ------------------------------------------------------------
# Parkphasen als Lernsimulation
# ------------------------------------------------------------

st.header("4. Parkphasen – Lernsimulation")

st.write(
    "Wähle eine Phase aus. Die Hinweise beschreiben allgemeine Beobachtungen "
    "und keine verbindlichen Lenk- oder Fahranweisungen."
)

phases = [
    {
        "title": "Phase 1: Ausgangsposition",
        "text": (
            "In der Simulation steht das Fahrzeug zunächst parallel zur "
            "Parklücke. Achte auf eine nachvollziehbare Ausgangsposition "
            "und darauf, wo sich die Begrenzungen der Lücke befinden."
        ),
        "mirror": (
            "Beobachtung: Im Außenspiegel sind die seitlichen Begrenzungen "
            "und das benachbarte Fahrzeug zu sehen. Die Darstellung hängt "
            "von Spiegel-Einstellung und Sitzposition ab."
        ),
    },
    {
        "title": "Phase 2: Rückwärtsbewegung (simuliert)",
        "text": (
            "Die Simulation zeigt eine langsame Rückwärtsbewegung. "
            "Beobachte, wie sich die Position des Fahrzeugs relativ zur "
            "Parklücke verändert."
        ),
        "mirror": (
            "Beobachtung: Das Bild im Spiegel verändert sich während der "
            "Bewegung. Ein einzelner Spiegel-Merkpunkt ist nicht bei jedem "
            "Fahrzeug gleich."
        ),
    },
    {
        "title": "Phase 3: Fahrzeugwinkel beobachten",
        "text": (
            "Das Fahrzeug steht in der Simulation schräg zur Lücke. "
            "Hier lässt sich untersuchen, wie Fahrzeugwinkel und Abstand "
            "zu den Begrenzungen zusammenhängen."
        ),
        "mirror": (
            "Beobachtung: Achte in der Simulation auf die relative Lage "
            "von Bordstein, Fahrzeugheck und Nachbarfahrzeug."
        ),
    },
    {
        "title": "Phase 4: Ausrichten (simuliert)",
        "text": (
            "In dieser Phase wird die Fahrzeugposition schrittweise "
            "gerader dargestellt. Vergleiche die Ausrichtung mit den "
            "Parklückenlinien."
        ),
        "mirror": (
            "Beobachtung: Die Abstände können links und rechts unterschiedlich "
            "sein. Spiegel liefern nur einen Ausschnitt der Umgebung."
        ),
    },
    {
        "title": "Phase 5: Endposition kontrollieren",
        "text": (
            "Die Simulation zeigt das Fahrzeug parallel innerhalb der Lücke. "
            "Prüfe grafisch, ob es innerhalb der markierten Begrenzungen liegt."
        ),
        "mirror": (
            "Beobachtung: Für eine reale Situation wären mehrere direkte "
            "Umgebungskontrollen nötig; die Simulation kann diese nicht ersetzen."
        ),
    },
]

phase_choice = st.selectbox(
    "Simulationsphase auswählen",
    options=list(range(len(phases))),
    format_func=lambda i: phases[i]["title"]
)

selected_phase = phases[phase_choice]
st.subheader(selected_phase["title"])
st.progress((phase_choice + 1) / len(phases))
st.write(selected_phase["text"])

with st.expander("🪞 Spiegel-Beobachtung (nur Simulation)", expanded=True):
    st.write(selected_phase["mirror"])

st.caption(
    "Es gibt keine universelle Regel wie „wenn das Nachbarauto an einer "
    "bestimmten Spiegelstelle erscheint, dann lenke genau so“. Spiegel, "
    "Fahrzeuggeometrie und Sitzposition unterscheiden sich."
)

# ------------------------------------------------------------
# Abschluss
# ------------------------------------------------------------

st.divider()
st.header("5. Projektidee und nächste Schritte")
st.write(
    "ParkAI verbindet eine einfache geometrische Platzprüfung, eine "
    "Bildanalyse mit YOLO und eine interaktive Lernsimulation. "
    "Die Ergebnisse sind bewusst vereinfacht."
)

st.info(
    "Nur als Lernprojekt verwenden. Nicht während des Fahrens bedienen. "
    "Echte Fahrmanöver ausschließlich im Rahmen einer Fahrausbildung "
    "mit einer geeigneten Aufsicht durchführen."
)

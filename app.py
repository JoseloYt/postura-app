import cv2
import numpy as np
import math
import time

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

from ultralytics import YOLO
import av  # Librería PyAV, necesaria para manejar VideoFrame

# -------------------------------------------------------------------
# 1) Parámetros globales y carga de modelo
# -------------------------------------------------------------------

MODEL_PATH = 'best.pt'
CONF_THRESHOLD = 0.3

# Umbral de curvatura (grados entre segmentos)
MAX_CURVE_ANGLE = 25  
ANGLE_LUMBAR = 15

# Duración (en segundos) de la pantalla de “¡Hora de tomar un descanso!”
ALERT_DURATION = 5*60 #5 minutos

@st.cache_resource(show_spinner=False)
def load_model(path):
    return YOLO(path)

model = load_model(MODEL_PATH)


# -------------------------------------------------------------------
# 2) Funciones de utilidad para calcular ángulo y curvatura
# -------------------------------------------------------------------

def angle_between(p0, p1, p2):
    """
    Calcula el ángulo en grados formado en p1 por los puntos p0->p1->p2.
    """
    v1 = np.array(p0) - np.array(p1)
    v2 = np.array(p2) - np.array(p1)
    dot = v1.dot(v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0.0
    cosang = np.clip(dot / norm, -1.0, 1.0)
    return math.degrees(math.acos(cosang))


def draw_and_evaluate_curve(frame, kpts):
    """
    Dibuja la curva de la columna vertebral usando 4 keypoints (cervical, torácico, lumbar, sacro),
    luego evalúa la curvatura en los segmentos torácico y lumbar y muestra advertencias si exceden el umbral.
    - frame: la imagen BGR en la que dibujar.
    - kpts: lista de (x, y, v) donde v es la confianza.
    """
    # Filtrar puntos válidos por umbral de confianza
    pts = [(int(x), int(y)) for x, y, v in kpts if v > CONF_THRESHOLD]
    if len(pts) < 4:
        return

    # Se asume que kpts está ordenado: 0=cervical, 1=torácico, 2=lumbar, último=sacro
    cervical = pts[0]
    thoracic = pts[1]
    lumbar = pts[2]
    sacral = pts[-1]
    spine = [cervical, thoracic, lumbar, sacral]

    # 1) Dibujar polilínea de la columna
    cv2.polylines(frame, [np.array(spine, np.int32)], False, (0, 255, 255), 2)

    # 2) Calcular ángulos en torácico y lumbar
    angle_thor = angle_between(cervical, thoracic, lumbar)
    angle_lum = angle_between(thoracic, lumbar, sacral)

    # 3) Curvatura = |180 - ángulo|
    curve_thor = abs(180 - angle_thor)
    curve_lum = abs(180 - angle_lum)

    # 4) Mostrar advertencias en la esquina superior izquierda (tamaño pequeño)
    y0 = 30

    if curve_thor > MAX_CURVE_ANGLE:
        cv2.putText(
            frame,
            f'AD espalda alta encorvada',
            (10, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,              # tamaño de fuente pequeño
            (0, 0, 255),
            2                 # grosor pequeño
        )
        y0 += 30

    if curve_lum > ANGLE_LUMBAR:
        cv2.putText(
            frame,
            f'AD espalda baja encorvada:',
            (10, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )


# -------------------------------------------------------------------
# 3) Clase para procesar cada frame recibido desde la cámara
# -------------------------------------------------------------------

class PoseProcessor(VideoProcessorBase):
    def __init__(self, break_interval_seconds: float):
        """
        break_interval_seconds: intervalo (en segundos) para recordar pausa.
                                Si es 0, se desactiva el recordatorio.
        """
        self.break_interval = break_interval_seconds
        self.last_break_time = time.time()
        self.alerting = False
        self.alert_start = 0

    def recv(self, frame):
        """
        Este método se llama para cada frame de vídeo que llega desde la webcam.
        Recibe un objeto “av.VideoFrame”, lo convertimos a numpy array, procesamos,
        y devolvemos otro VideoFrame ya con las marcas.
        """
        # 1) Convertir VideoFrame (av) a np.ndarray (BGR)
        img = frame.to_ndarray(format="bgr24")

        # 2) Verificar tiempo de pausa (solo si break_interval > 0)
        if self.break_interval > 0:
            current_time = time.time()
            # Si aún no estamos en "alerting" y ya pasó el intervalo -> comenzamos pausa
            if (not self.alerting) and ((current_time - self.last_break_time) > self.break_interval):
                self.alerting = True
                self.alert_start = current_time

            # Si estamos en estado alerting, dibujamos el texto en esquina superior izquierda
            if self.alerting:
                cv2.putText(
                    img,
                    "Hora de tomar un descanso!",
                    (50, img.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,                # tamaño de fuente original
                    (0, 0, 255),
                    3                   # grosor de línea original
                )

                # Si ya pasó ALERT_DURATION, terminamos la alerta
                if (current_time - self.alert_start) > ALERT_DURATION:
                    self.last_break_time = current_time
                    self.alerting = False

        # 3) Inferencia de pose con YOLOv8
        results = model(img, stream=True, conf=CONF_THRESHOLD)
        for res in results:
            # Dibujar bounding boxes
            for box in res.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Dibujar keypoints y evaluar curvatura
            if hasattr(res, 'keypoints') and res.keypoints is not None:
                kpts_data = res.keypoints.data  # shape (N, K, 3)
                for kpt_set in kpts_data:
                    # Dibujar todos los keypoints
                    for x, y, v in kpt_set:
                        if v > CONF_THRESHOLD:
                            cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)

                    # Dibujar polilínea y evaluar curva
                    draw_and_evaluate_curve(img, kpt_set)

        # 4) Devolver el frame modificado
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# -------------------------------------------------------------------
# 4) Interfaz principal de Streamlit
# -------------------------------------------------------------------

st.set_page_config(page_title="Detección de Curvatura Postural", layout="wide")
st.title("🧍‍♂️ Detección de Curvatura Postural en Tiempo Real")

st.write("""
Esta aplicación usa **YOLOv8** para estimación de pose y detecta curvaturas
en la columna vertebral. Además, recuerda al usuario hacer pausas
cada cierto tiempo (configurable).
""")

# ----------------------------------------------------------
# Sidebar: controles de pausa
# ----------------------------------------------------------

st.sidebar.header("⚙️ Configuración de pausas")

# Input para intervalo en minutos (por defecto 120 minutos = 2 horas)
break_minutes = st.sidebar.number_input(
    label="Intervalo de pausa (minutos). Pon 0 para desactivar:",
    min_value=0,
    max_value=1440,  # hasta 24 horas
    value=120,       # defecto 120 minutos
    step=5
)

if break_minutes == 0:
    st.sidebar.info("Recordatorio de pausas DESACTIVADO.")
else:
    st.sidebar.info(f"Recordatorio activo cada {break_minutes:.0f} minutos.")

# Convertimos a segundos
BREAK_INTERVAL = break_minutes * 60

# ----------------------------------------------------------
# Iniciar el stream de la webcam con nuestro PoseProcessor
# ----------------------------------------------------------

webrtc_ctx = webrtc_streamer(
    key="pose-curvature",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=lambda: PoseProcessor(break_interval_seconds=BREAK_INTERVAL),
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if not webrtc_ctx.state.playing:
    st.warning("Por favor, permite el acceso a tu cámara en el navegador para iniciar la detección.")

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import pandas as pd
import time
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import io
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.set_page_config(page_title="Evaluación de Estabilidad Postural", layout="wide")
st.title("Analizador de Control Postural y Prevención de Caídas")
st.error("NOTA DE SEGURIDAD: Realice esta prueba cerca de un apoyo firme.")

if "records" not in st.session_state:
    st.session_state.records = []
if "patient_name" not in st.session_state:
    st.session_state.patient_name = ""
if "cronometro_activo" not in st.session_state:
    st.session_state.cronometro_activo = False
if "tiempo_inicio" not in st.session_state:
    st.session_state.tiempo_inicio = None

st.session_state.patient_name = st.text_input("Nombre del Paciente:", st.session_state.patient_name, placeholder="Ej. Juan Pérez")

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calcular_angulo(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return abs(np.degrees(np.arctan2(dy, dx)))

class PostureProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.com_history = []
        self.ang_hombros = 0.0
        self.ang_caderas = 0.0
        self.sway_x = 0.0
        self.estado = "EN ESPERA"
        self.color_com = (0, 255, 0)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            ls = [int(landmarks[11].x * w), int(landmarks[11].y * h)]
            rs = [int(landmarks[12].x * w), int(landmarks[12].y * h)]
            lh = [int(landmarks[23].x * w), int(landmarks[23].y * h)]
            rh = [int(landmarks[24].x * w), int(landmarks[24].y * h)]
            
            self.ang_hombros = calcular_angulo(ls, rs)
            self.ang_caderas = calcular_angulo(lh, rh)
            
            com_x = (lh[0] + rh[0]) // 2
            com_y = (lh[1] + rh[1]) // 2
            self.com_history.append(com_x)
            if len(self.com_history) > 30:
                self.com_history.pop(0)
            
            self.sway_x = np.std(self.com_history) if len(self.com_history) > 5 else 0.0
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if self.ang_hombros < 3.0 and self.ang_caderas < 3.0 and self.sway_x < 4.0:
                self.estado, self.color_com = "SEGURO", (0, 255, 0)
            elif self.ang_hombros >= 5.0 or self.ang_caderas >= 5.0 or self.sway_x > 6.0:
                self.estado, self.color_com = "RIESGO DE CAÍDA", (255, 0, 0)
            else:
                self.estado, self.color_com = "INESTABILIDAD LEVE", (255, 255, 0)

            cv2.circle(img, (com_x, com_y), 12, self.color_com, -1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

col_cam, col_data = st.columns(2)

with col_cam:
    ctx = webrtc_streamer(
        key="posture-evaluation",
        video_processor_factory=PostureProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False}
    )
    status_text = st.empty()

with col_data:
    st.subheader("Control del Test (30 Segundos)")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Iniciar Test de 30s", use_container_width=True):
            st.session_state.cronometro_activo = True
            st.session_state.tiempo_inicio = time.time()
            st.session_state.records = []
    with col_btn2:
        if st.button("Reiniciar", use_container_width=True):
            st.session_state.cronometro_activo = False
            st.session_state.tiempo_inicio = None
            st.session_state.records = []
            
    txt_tiempo = st.empty()
    st.markdown("---")
    st.subheader("Monitoreo de Estabilidad")
    m_hombros = st.empty()
    m_caderas = st.empty()
    m_sway = st.empty()

if ctx.video_processor:
    processor = ctx.video_processor
    m_hombros.metric("Hombros", f"{processor.ang_hombros:.1f}°")
    m_caderas.metric("Caderas", f"{processor.ang_caderas:.1f}°")
    m_sway.metric("Oscilación", f"{processor.sway_x:.1f} px")
    
    if processor.estado == "SEGURO":
        status_text.success("POSTURA SEGURA")
    elif processor.estado == "RIESGO DE CAÍDA":
        status_text.error("¡RIESGO DE CAÍDA! SUJETESE.")
    else:
        status_text.warning("ADVERTENCIA: Oscilación detectada.")

tiempo_restante = 30.0
if st.session_state.cronometro_activo and st.session_state.tiempo_inicio:
    tiempo_transcurrido = time.time() - st.session_state.tiempo_inicio
    tiempo_restante = max(0.0, 30.0 - tiempo_transcurrido)
    txt_tiempo.markdown(f"### Tiempo Restante: **{tiempo_restante:.1f}s**")
    
    if ctx.video_processor and int(tiempo_restante * 2) % 10 == 0:
        t_label = f"{30 - int(tiempo_restante)}s"
        if not any(r[0] == t_label for r in st.session_state.records):
            st.session_state.records.append([
                t_label, 
                datetime.now().strftime("%H:%M:%S"), 
                round(ctx.video_processor.ang_hombros, 1), 
                round(ctx.video_processor.ang_caderas, 1), 
                round(ctx.video_processor.sway_x, 1), 
                ctx.video_processor.estado
            ])
            
    if tiempo_restante == 0:
        st.session_state.cronometro_activo = False
else:
      txt_tiempo.markdown("### Test en espera. Presione Iniciar.")
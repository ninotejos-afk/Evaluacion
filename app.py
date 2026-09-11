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
st.set_page_config(page_title="Evaluación de Estabilidad Postural", layout="wide")
st.title("n Analizador de Control Postural y Prevención de Caídas")
st.error("nn NOTA DE SEGURIDAD: Realice esta prueba cerca de un apoyo firme.")
if "records" not in st.session_state: st.session_state.records = []
if "patient_name" not in st.session_state: st.session_state.patient_name = ""
if "cronometro_activo" not in st.session_state: st.session_state.cronometro_activo = False
if "tiempo_inicio" not in st.session_state: st.session_state.tiempo_inicio = None
st.session_state.patient_name = st.text_input("Nombre del Paciente:", st.session_state.patien
t_name, placeholder="Ej. Juan Pérez")
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
def calcular_angulo(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return abs(np.degrees(np.arctan2(dy, dx)))
col_cam, col_data = st.columns(2)
with col_cam:
    FRAME_WINDOW = st.image([])
    status_text = st.empty()
with col_data:
    st.subheader("nn Control del Test (30 Segundos)")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("nn Iniciar Test de 30s", use_container_width=True):
            st.session_state.cronometro_activo = True
            st.session_state.tiempo_inicio = time.time()
            st.session_state.records = []
    with col_btn2:
        if st.button("nn Reiniciar", use_container_width=True):
            st.session_state.cronometro_activo = False
            st.session_state.tiempo_inicio = None
            st.session_state.records = []
    txt_tiempo = st.empty()
    st.markdown("---")
    st.subheader("n Monitoreo de Estabilidad")
    m_hombros = st.empty()
    m_caderas = st.empty()
    m_sway = st.empty()
cap = cv2.VideoCapture(0)
com_history = []
with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        tiempo_restante = 30.0
        if st.session_state.cronometro_activo and st.session_state.tiempo_inicio:
            tiempo_transcurrido = time.time() - st.session_state.tiempo_inicio
            tiempo_restante = max(0.0, 30.0 - tiempo_transcurrido)
            txt_tiempo.markdown(f"### n Tiempo Restante: **{tiempo_restante:.1f}s**")
            if tiempo_restante == 0: st.session_state.cronometro_activo = False
        else:
            txt_tiempo.markdown("### n Test en espera. Presione Iniciar.")
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            ls = [int(landmarks[11].x * w), int(landmarks[11].y * h)]
            rs = [int(landmarks[12].x * w), int(landmarks[12].y * h)]
            lh = [int(landmarks[23].x * w), int(landmarks[23].y * h)]
            rh = [int(landmarks[24].x * w), int(landmarks[24].y * h)]
            ang_hombros = calcular_angulo(ls, rs)
            ang_caderas = calcular_angulo(lh, rh)
            com_x = (lh[0] + rh[0]) // 2
            com_y = (lh[1] + rh[1]) // 2
            com_history.append(com_x)
            if len(com_history) > 30: com_history.pop(0)
            sway_x = np.std(com_history) if len(com_history) > 5 else 0.0
            mp_drawing.draw_landmarks(rgb_frame, results.pose_landmarks, mp_pose.POSE_CONNECT
IONS)
            if ang_hombros < 3.0 and ang_caderas < 3.0 and sway_x < 4.0:
                estado, color_com = "SEGURO", (0, 255, 0)
                status_text.success("n POSTURA SEGURA")
            elif ang_hombros >= 5.0 or ang_caderas >= 5.0 or sway_x > 6.0:
                estado, color_com = "RIESGO DE CAÍDA", (255, 0, 0)
                status_text.error("n ¡RIESGO DE CAÍDA! SUJETESE.")
            else:
                estado, color_com = "INESTABILIDAD LEVE", (255, 255, 0)
                status_text.warning("nn ADVERTENCIA: Oscilación detectada.")
            cv2.circle(rgb_frame, (com_x, com_y), 12, color_com, -1)
            m_hombros.metric("Hombros", f"{ang_hombros:.1f}°")
            m_caderas.metric("Caderas", f"{ang_caderas:.1f}°")
            m_sway.metric("Oscilación", f"{sway_x:.1f} px")
            if st.session_state.cronometro_activo and int(tiempo_restante * 2) % 10 == 0:
                t_label = f"{30 - int(tiempo_restante)}s"
                if not any(r[0] == t_label for r in st.session_state.records):
                    st.session_state.records.append([t_label, datetime.now().strftime("%H:%M:
%S"), round(ang_hombros, 1), round(ang_caderas, 1), round(sway_x, 1), estado])
FRAME_WINDOW.image(rgb_frame)
cap.release()
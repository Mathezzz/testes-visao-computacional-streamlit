# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 21:50:01 2024

@author: ander
"""

import streamlit as st
import cv2
from PIL import Image
import numpy as np

# Carrega o classificador Haar Cascade para detecção de rostos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
olho_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

st.title('Detecção de Rostos com OpenCV e Streamlit')
st.text('Essa versão está usando uma versão com câmera estática pois ainda estou entendendo\n' +
        'como fazer o stream de video no ambiente de deploy')

# Caixa de seleção para o modelo de detecção
lista_modelos = ['Rosto', 'Olhos']
select_modelo = st.multiselect("Selecione o modelo de Detecção", lista_modelos)

col1, col2 = st.columns(2)


# Exibe o frame com a detecção no lado direito
with col1:
    frame_placeholder_imagem = st.empty()
# Placeholder para manter a exibição no mesmo local
with col2:
    frame_placeholder_deteccao = st.empty()


def processar_frame(frame):
    img = frame.copy()
    
    
    # Converte para escala de cinza e realiza a detecção de rosto
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(35, 35))
    olhos = olho_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    # Desenha retângulos ao redor dos rostos detectados
    if 'Rosto' in select_modelo:    
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
    # Desenha círculos ao redor dos olhos
    if 'Olhos' in select_modelo:
        for (x2,y2,w2,h2) in olhos:
            centro_do_olho = (x2 + w2//2,y2 + h2//2)
            raio_do_olho = int(round((w2 + h2)*0.25))
            cv2.circle(frame, centro_do_olho, raio_do_olho, (0, 0, 255), 2)
    
    # Converte o frame para RGB para exibir no Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    
    frame_placeholder_imagem.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    frame_placeholder_deteccao.image(frame_rgb, caption="Vídeo com Detecção de Rosto", use_column_width=True)

    
img = st.camera_input("Tire uma foto")
if img is not None:
    # To read image file buffer with OpenCV:
    bytes_data = img.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    processar_frame(cv2_img)
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

st.title('Detecção de Rostos com OpenCV e Streamlit')

col1, col2 = st.columns(2)

enable = st.checkbox("Habilitar camera")
ativar_haar = st.checkbox("Habilitar detecção com Haar Cascade")

with col1:
    picture = st.camera_input("Tirar uma foto", disabled=not enable)

if picture and ativar_haar:
    # Converte a imagem do formato Streamlit para uma imagem OpenCV
    img = Image.open(picture)
    img_array = np.array(img.convert("RGB"))  # Converte para RGB
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Detecta os rostos na imagem
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Desenha retângulos ao redor dos rostos detectados
    for (x, y, w, h) in faces:
        cv2.rectangle(img_array, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    with col2:
        # Exibe a imagem com os rostos detectados
        st.image(img_array, caption="Imagem com Detecção de Rosto", use_column_width=True)

        # Exibe o número de rostos detectados
        st.write(f"Número de rostos detectados: {len(faces)}")
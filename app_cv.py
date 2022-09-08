import cv2
import streamlit as st
from PIL import Image
import numpy as np
from skimage import morphology, io, color, feature, filters # instalar biblioteca pip install scikit-image



def brilho_imagem(imagem, resultado):
    img_brilho = cv2.convertScaleAbs(imagem, beta=resultado)
    return img_brilho


def borra_imagem(imagem, resultado):
    img_borrada = cv2.GaussianBlur(imagem, (7, 7), resultado)
    return img_borrada


def melhora_detalhe(imagem):
    img_melhorada = cv2.detailEnhance(imagem, sigma_s=34, sigma_r=0.5)
    return img_melhorada

def escala_cinza(imagem):
    img_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    return img_cinza

def principal():
    st.title("OpenCV Data App")
    st.subheader(
        "Esse aplicativo web permite integrar processamento de imagens com OpenCV")
    st.text("Streamlit com OpenCV")

    #img = cv2.imread("imagens/NVR02.png")
    arquivo_imagem = st.file_uploader(
        'Enve sua Imagem', type=["jpg", 'png', 'jpeg'])

    taxa_borrao = st.sidebar.slider("Borrão", min_value=0.2, max_value=3.5)
    qtd_brilho = st.sidebar.slider("Brilho", min_value=-50, max_value=50, value=0)
    filtro_aprimoramento = st.sidebar.checkbox("Melhoras Detalhes da imagem")
    img_cinza = st.sidebar.checkbox("Converter para escala de cinza")
    img_erosao = st.sidebar.checkbox("Filtro Erosão")
    img_dilatacao = st.sidebar.checkbox("Filtro Dilatacao")
    img_edge = st.sidebar.checkbox("Filtro Edge")
    
    if not arquivo_imagem:
        return None
    
    imagem_original = Image.open(arquivo_imagem)
    imagem_original = np.array(imagem_original)

    imagem_processada = borra_imagem(imagem_original, taxa_borrao)
    imagem_processada = brilho_imagem(imagem_processada, qtd_brilho)

    if filtro_aprimoramento:#codigo de qaundo o checkbox é marcado
        imagem_processada = melhora_detalhe(imagem_processada)
        
    if img_cinza:#codigo de qaundo o checkbox é marcado
        imagem_processada = escala_cinza(imagem_processada)
    
    if img_erosao:#codigo de qaundo o checkbox é marcado
        imagem_processada = morphology.erosion(imagem_processada)

    if img_dilatacao:#codigo de qaundo o checkbox é marcado
        imagem_processada = morphology.dilation(imagem_processada)
        
    if img_edge:#codigo de qaundo o checkbox é marcado
        imagem_processada = filters.sobel(imagem_processada)

    st.text("Imagem Original")

    st.image(imagem_original)

    st.text("Imagem Processada")

    st.image(imagem_processada)


if __name__ == '__main__':
    principal()

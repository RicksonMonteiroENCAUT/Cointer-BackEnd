# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 22:33:33 2020

@author: Rickson Gomes Monteiro
"""
#Importação das bibliotecas
import os
import requests
import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, request, jsonify
#Carregamento do modelo pré-treinado
with open("modelo\modelo.json", "r") as f:
    model_json = f.read()

model = tf.keras.models.model_from_json(model_json)
model.load_weights("modelo\melhor_peso.best.hdf5")
#model.summary()

# Criação da API em Flask
app = Flask(__name__)

# Função para classificação de imagens
@app.route("/<string:img_name>", methods = ["POST", "GET"])
def classify_image(img_name):
    results=[]
    money=0
    upload_dir = "uploads/"
    classes = [0.01, 0.01, 0.05, 0.1, 0.25]
    read = cv2.imread(upload_dir + img_name)
    try:
        imagens_cortadas=image_segmentation(read)
        for img in imagens_cortadas:
            image=cv2.resize(img, (128,128))/255.0
            results.append(np.argmax(model.predict(image.reshape(1,128,128,3))))
        for i in results:
            money += classes[i]
        return jsonify({"Total ": money})
    except:
        image=cv2.resize(read, (128,128))/255.0
        return jsonify({"Total ": classes[np.argmax(model.predict(image.reshape(1,128,128,3)))]})
    else:
        return jsonify({"Sorry! ": 'We have a problem'})
        
  

#Função de Segmentação
def image_segmentation(image):
    imagens_cortadas=[]
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(7,7),cv2.BORDER_DEFAULT)
    all_circs=cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT,
                               0.9, 120, param1=50,
                               param2= 30, minRadius=80,
                               maxRadius=150)
    all_circs=np.uint16(np.around(all_circs))
    for corte in (all_circs[0]):
        crop= image[corte[1]-corte[2]:corte[1]+corte[2], corte[0]-corte[2]:corte[0]+corte[2]]
        imagens_cortadas.append(crop)
    return imagens_cortadas

# Iniciando a aplicação Flask
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

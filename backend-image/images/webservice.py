import os
from os import path
import tensorflow as tf
import keras as keras
from PIL import Image
import numpy as np
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, Query, applications
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import StreamingResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from .transferencia_estilo import run_style_transfer

app = FastAPI()

UPLOAD_DIR = "images/"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/images/", tags=["Endpoints"])
async def trata_image(
    conteudo_img: UploadFile = File(...),
    estilo_img: UploadFile = File(...)
):
    caminho_conteudo_img = os.path.join(UPLOAD_DIR, "conteudo_image.jpg")
    caminho_estilo_img = os.path.join(UPLOAD_DIR, "estilo_image.jpg")

    with open(caminho_conteudo_img, "wb") as f:
        f.write(await conteudo_img.read())
    
    with open(caminho_estilo_img, "wb") as f:
        f.write(await estilo_img.read())

    result_imagem = run_style_transfer(caminho_conteudo_img, caminho_estilo_img)

    caminho_imagem_resultado = os.path.join(UPLOAD_DIR, "imagem_resultado.jpg")
    Image.fromarray(result_imagem).save(caminho_imagem_resultado)
    
    return FileResponse(caminho_imagem_resultado)
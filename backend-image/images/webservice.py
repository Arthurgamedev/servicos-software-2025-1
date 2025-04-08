import os
from os import path
import tensorflow as tf
import keras as keras
from PIL import Image
import numpy as np
from fastapi import FastAPI, File, UploadFile, Query, applications
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import StreamingResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from .transferencia_estilo import run_style_transfer

app = FastAPI()

@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def index():
    return "/docs"

@app.post("/imagens", tags=["Endpoints"])
async def trata_image(
    image_file1: UploadFile = File(...),
    image_file2: UploadFile = File(...)
    ):
    
    prediction = run_style_transfer(image_file1, image_file2, iterations=1000, content_weight=1e3, style_weight=1e-2)    

    return JSONResponse(
        content = prediction
    )


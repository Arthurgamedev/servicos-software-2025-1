from PIL import Image
import io
import gradio as gr
import requests

url="http://backend-image:9001/images/"

def envia(imagem1,imagem2, url=url):

    img1 = Image.open(imagem1)
    img2 = Image.open(imagem2)
    # Converte a imagem PIL para bytes
    imagem1_bytes = io.BytesIO()
    img1.save(imagem1_bytes, format="JPEG")
    imagem1_bytes.seek(0)

    imagem2_bytes = io.BytesIO()
    img2.save(imagem2_bytes, format="JPEG")
    imagem2_bytes.seek(0)

    files = {"imagem_conteudo": ("conteudo_image.jpg", imagem1_bytes, "image/jpeg"),
             "imagem_estilo": ("estilo_image.jpg", imagem2_bytes, "image/jpeg")}

    r = requests.post(url, files=files)

    if r.status_code == 200:
        image_data = r.content
        imagem_bytes = io.BytesIO(image_data)
        # Cria uma imagem PIL a partir dos bytes
        image_criada = gr.Image(value = imagem_bytes, label="Imagem Resultado:")
        return image_criada
    else:
        print("Erro ao enviar a imagem:", r.status_code)
        image_criada = None
    

ui = gr.Interface(fn=envia, inputs=[gr.Image(label="Imagem Conteudo:", type="pil"), 
                                    gr.Image(label="Imagem Estilo:",type="pil")],
                                    outputs=gr.Image())
ui.title="Transferência de Estilo"

if __name__ == "__main__":
    ui.launch(server_name="0.0.0.0",server_port=7860, show_api=False)

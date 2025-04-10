from PIL import Image
import io
import gradio as gr
import requests

url="http://backend-image:9001/processar"

def pil_para_file(imagem_pil, nome="imagem.png"):
    buffer = io.BytesIO()
    imagem_pil.save(buffer, format='PNG')
    buffer.seek(0)
    return (nome, buffer, 'image/png')

def envia(imagem1,imagem2, url=url):

    img1 = imagem1
    img2 = imagem2
    
    files = {"imagem_conteudo": pil_para_file(img1), "imagem_estilo": pil_para_file(img2)}
    
    r = requests.post(url, files=files)

    if r.status_code == 200:
        # Cria uma imagem PIL a partir dos bytes
        buffer1 = io.BytesIO(r.content)
        imagem_criada = Image.open(buffer1)
        return imagem_criada
    else:
        print("Erro ao enviar a imagem:", r.status_code)
        imagem_criada = None
    
ui = gr.Interface(fn=envia, inputs=[gr.Image(label="Imagem Conteudo:", type="pil"), 
                                    gr.Image(label="Imagem Estilo:",type="pil")],
                            outputs=[gr.Image()])
ui.title="Transferência de Estilo"

if __name__ == "__main__":
    ui.launch(server_name="0.0.0.0",server_port=7860, show_api=False)

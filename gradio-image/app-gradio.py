import gradio as gr
import requests

def envia(imagem1,imagem2):
    url="http://backend-image:8081/imagens/"
    with open(imagem1,imagem2,'rb') as f:
        r = requests.post(url, files={["image_file1","image_file2"]:f})
    return r.content


ui = gr.Interface(fn=envia, inputs=[gr.Image(label="Primeira Imagem:", type="filepath"), 
                                    gr.Image(label="Segunda Imagem:",type="filepath")],
                                    outputs=gr.Image())

if __name__ == "__main__":
    ui.launch(server_name="0.0.0.0",server_port=7860, show_api=False)

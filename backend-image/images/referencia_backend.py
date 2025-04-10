from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
import io
import torch
from torchvision import models, transforms
from torch.autograd import Variable

app = FastAPI()

# Carregar o modelo pré-treinado VGG19
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg = models.vgg19(pretrained=True).features.to(device).eval()

# Funções auxiliares para processamento de imagem
def image_loader(image, imsize=512):
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    return image

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

# Função de transferência de estilo
def style_transfer(content_img, style_img, num_steps=1000, style_weight=500, content_weight=500):
    content_img = image_loader(content_img)
    style_img = image_loader(style_img)
    input_img = content_img.clone()

    input_img = input_img.requires_grad_(True)

    optimizer = torch.optim.LBFGS([input_img])

    for step in range(num_steps):
        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            content_loss = content_weight * torch.nn.functional.mse_loss(input_img, content_img)
            style_loss = 0
            for c in range(style_img.size(1)):  # Iterar sobre os canais
                gram_s = gram_matrix(style_img[:, c:c+1, :, :])  # Adicionar dimensão extra
                gram_g = gram_matrix(input_img[:, c:c+1, :, :])  # Adicionar dimensão extra
                style_loss += torch.nn.functional.mse_loss(gram_g, gram_s)
            loss = content_loss + style_weight * style_loss
            loss.backward()
            return loss
        optimizer.step(closure)
    input_img.data.clamp_(0, 1)
    return imshow(input_img)

@app.post("/processar")
async def processar(imagem_conteudo: UploadFile = File(...), imagem_estilo: UploadFile = File(...)):
    conteudo = Image.open(imagem_conteudo.file)
    estilo = Image.open(imagem_estilo.file)
    resultado = style_transfer(conteudo, estilo)
    buffer = io.BytesIO()
    resultado.save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")

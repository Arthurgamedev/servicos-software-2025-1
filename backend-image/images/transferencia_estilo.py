# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
from torchvision import transforms
import tensorflow as tf
from tensorflow.python.keras import models
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K
import io

# Função para carregar imagem e transformar em um array do TensorFlow
def load_img_from_bytes(image_bytes):
    max_dim = 512
    img = Image.open(io.BytesIO(image_bytes))
    long = max(img.size)
    scale = max_dim / long
    img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), Image.BICUBIC)
    img = tf.keras.utils.img_to_array(img)
 
    # Inclui eixo dos exemplos
    img = np.expand_dims(img, axis=0)
    return img
 
# Carrega e processa imagem para VGG19
def load_and_process_img(image_bytes):
    img = load_img_from_bytes(image_bytes)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img
 
def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)

    assert len(x.shape) == 3, ("Entrada deve ser uma imagem de"
                               "dimensão [1, altura, largura, canal]")
    if len(x.shape) != 3:
        raise ValueError("Entrada inválida para transformar")

    # Realiza o inverso da normalização das imagens
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x

# Camada de conteúdo
content_layers = ['block5_conv2']

# Camdas de estilo (lista com os nomes das camadas utilizadas)
style_layers = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

import tensorflow as tf
from tensorflow.keras import layers, models

def get_model(style_layers, content_layers):

    # Carrega a VGG19 pré-treinada
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    # Determina camada de entrada
    model_input = vgg.input

    # Determina saídas que correspondem às camadas de conteúdo e de estilo
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]

    # Combina saídas de estilo e conteúdo
    model_outputs = style_outputs + content_outputs

    # Cria o modelo final
    model = models.Model(inputs=model_input, outputs=model_outputs)

    return model

# Função de custo de conteúdo
def get_content_loss(base_content, target):
    JC = tf.reduce_mean(tf.square(base_content - target))/2
    return JC

def gram_matrix(input_tensor):
    # Redimensiona tensor para para ter 2 eixos (linhas*colunas e canais)
    channels = int(input_tensor.shape[-1])
    A = tf.reshape(input_tensor, [-1, channels])
    gram = tf.matmul(A, A, transpose_a=True)
    return gram

# Função de custo de estilo de uma camada
def get_style_loss(base_style, gram_target):
    # base_style = saídas das camadas de "estilo" referentes à imagem gerada
    # gram_target = matriz de Gram calculada para cada camada de "estilo" referente à imagem de estilo

    # Dimensões da camada de estilo
    height, width, channels = base_style.get_shape().as_list()

    # Matriz de Gram
    gram_style = gram_matrix(base_style)

    # Função de custo de estilo para uma camada
    JS = tf.reduce_mean(tf.square(gram_style - gram_target)) / (4. * (channels * width * height)**2)

    return JS

# Cálculo das representações de conteúdo e de estilo

def get_feature_representations(model, content_path, style_path):
    
    # Usando a função load_and_process_img carrega as imagens de conteùdo e de estilo
    content_image = load_and_process_img(content_path)
    style_image = load_and_process_img(style_path)

    # Usando a RNA VGG19 recebida no objeto "model", calcula as caracteríticas das imagens de conteúdo e de estilo
    style_outputs = model(style_image)
    content_outputs = model(content_image)

    # Obtém as representações de conteúdo e de estilo
    style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
    content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]

    return style_features, content_features

def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    
    style_weight, content_weight = loss_weights

    # Calcula saídas das camadas intermediárias da imagem incial
    model_outputs = model(init_image)

    # Define características de estilo e conteúdo
    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]

    # Inicializa custos de estilo e conteúdo
    style_score = 0
    content_score = 0

    # Calcula custo das camadas de estilo e soma a contribuição de cada uma
    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)

    # Acumula custo de conteúdo de todas as camadas e soma contribuições
    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer* get_content_loss(comb_content[0], target_content)

    # Multiplica custo de conteúdo e de estilo pelos pesos de ponderaçãp
    style_score *= style_weight
    content_score *= content_weight

    # Calcula custo total
    loss = style_score + content_score

    return loss, style_score, content_score

def compute_grads(cfg):
    # Calcula custo total no contexto de GradientTape
    with tf.GradientTape() as G:
        all_loss = compute_loss(**cfg)

    # Calcula gradientes da função de custo em relação aos pixels da imagem gerada
    total_loss = all_loss[0]
    grad = G.gradient(total_loss, cfg['init_image'])

    # retorna gradiente e custo
    return grad, all_loss

# Função principal de transferência de estilo
def run_style_transfer(content_path, style_path, num_iterations=100, content_weight=1e3, style_weight=1e-2):
    # Definição das camadas de conteúdo e estilo
    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)
 
    model = get_model(style_layers, content_layers)
    for layer in model.layers:
        layer.trainable = False
 
    # Carregar as imagens
    content_image = load_and_process_img(content_path)
    style_image = load_and_process_img(style_path)
 
    # Calcular características de estilo e conteúdo
    style_outputs = model(style_image)
    content_outputs = model(content_image)
 
    style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
    content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
 
    # Calcular matrizes de Gram
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]
 
    # Inicializar imagem gerada
    init_image = tf.Variable(content_image, dtype=tf.float32)
 
    # Otimizador Adam
    opt = tf.optimizers.Adam(learning_rate=5.0)
 
    # Processamento de treinamento
    best_loss, best_img = float('inf'), None
    loss_weights = (style_weight, content_weight)
    cfg = {'model': model, 'loss_weights': loss_weights, 'init_image': init_image, 'gram_style_features': gram_style_features, 'content_features': content_features}
 
    # Inicializa os valores de normalização para a imagem gerada
    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means
 
    for i in range(num_iterations):
        grads, all_loss = compute_grads(cfg)
 
        loss, style_score, content_score = all_loss
        opt.apply_gradients([(grads, init_image)])
 
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)
        init_image.assign(clipped)
 
        if loss < best_loss:
            best_loss = loss
            best_img = deprocess_img(init_image.numpy())
 
    return best_img
 
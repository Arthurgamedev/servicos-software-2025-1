
# Etapa para copiar recursos do Swagger UI
FROM swaggerapi/swagger-ui:v5.9.1 AS swagger-ui

# Base da imagem para Python
FROM python:3.10

# Definir diretório de trabalho
WORKDIR /images

# Copiar arquivos locais para o contêiner
COPY . .

# Copiar arquivos específicos do Swagger UI
COPY --from=swagger-ui /usr/share/nginx/html/swagger-ui.css swagger-ui-assets/swagger-ui.css
COPY --from=swagger-ui /usr/share/nginx/html/swagger-ui-bundle.js swagger-ui-assets/swagger-ui-bundle.js

# Instalar dependências do projeto
RUN pip install --no-cache-dir --upgrade -r /images/requirements.txt

# Instalar bibliotecas adicionais
RUN python3 -m pip install tf-nightly keras Pillow

# Expor a porta do servidor
EXPOSE 9001

# Executar o servidor da aplicação
CMD ["gunicorn", "--bind", "0.0.0.0:9001", "--workers", "1", "--timeout", "0", "images.webservice:app", "-k", "uvicorn.workers.UvicornWorker"]
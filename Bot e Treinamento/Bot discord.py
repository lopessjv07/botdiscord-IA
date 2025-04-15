import nextcord
import os
import cv2
import pytesseract
from nextcord.ext import commands
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.models import load_model

intents = nextcord.Intents.default()
intents.messages = True
intents.message_content = True
client = commands.Bot(command_prefix='!', intents=intents)

# Carregar o modelo treinado para classificação de imagens ilícitas
model = load_model('C:/Users/Seu_usuario/Downloads/meu_modelo.h5')

# Lista de palavras proibidas para texto ilícito (tanto no chat quanto nas imagens)
palavras_proibidas = ['palavras ilicitas']

# Função para classificar imagem como lícita ou ilícita usando modelo treinado
def classificar_imagem(caminho_imagem):
    img = load_img(caminho_imagem, target_size=(64, 64))  # Carrega e redimensiona a imagem
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0  # Normaliza a imagem

    resultado = model.predict(img)[0][0]
    return "lícita" if resultado > 0.5 else "ilícita"

# Função para verificar se uma mensagem contém palavras ilícitas
def verificar_palavras_ilicitas(texto):
    for palavra in palavras_proibidas:
        if palavra in texto.lower():
            return True
    return False

# Função para extrair texto de uma imagem usando OCR (Tesseract)
def extrair_texto_imagem(caminho_imagem):
    # Configurar o caminho do Tesseract
    pytesseract.pytesseract.tesseract_cmd = "C:/Users/Seu_usuario/AppData/Local/Programs/Tesseract-OCR/tesseract.exe"
    
    # Ler e processar a imagem para melhorar a extração de texto
    imagem = cv2.imread(caminho_imagem)
    if imagem is None:
        return None
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    imagem_binarizada = cv2.adaptiveThreshold(imagem_cinza, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    imagem_suave = cv2.medianBlur(imagem_binarizada, 3)

    # Extrair texto usando Tesseract
    texto = pytesseract.image_to_string(imagem_suave, config='--psm 3', lang='por')
    return texto

# Mostra quando o bot está funcionando
@client.event
async def on_ready():
    print(f'Bot {client.user} está funcionando!')

# Evento para monitorar as mensagens do servidor
@client.event
async def on_message(message):
    # Ignorar mensagens enviadas pelo próprio bot
    if message.author == client.user:
        return

    # Verificar se a mensagem contém anexos (imagens)
    if message.attachments:
        for attachment in message.attachments:
            if attachment.filename.endswith(('.png', '.jpg', '.jpeg')):
                caminho_imagem = f'./{attachment.filename}'
                await attachment.save(caminho_imagem)
                print(f"Imagem {attachment.filename} recebida e salva.")

                # Classificar a imagem como lícita ou ilícita
                resultado_imagem = classificar_imagem(caminho_imagem)
                if resultado_imagem == "ilícita":
                    await message.delete()
                    await message.channel.send("⚠️ Imagem removida por ser considerada ilícita!")
                else:
                    # Extrair o texto da imagem e verificar se contém palavras ilícitas
                    texto_extraido = extrair_texto_imagem(caminho_imagem)
                    if texto_extraido and verificar_palavras_ilicitas(texto_extraido):
                        await message.delete()
                        await message.channel.send(f"⚠️ {message.author.mention}, imagem removida devido a texto ilícito detectado!")
                    else:
                        print("Imagem lícita e sem texto ilícito.")
                
                # Remover a imagem após o processamento
                os.remove(caminho_imagem)

    # Verificar se a mensagem de texto contém palavras ilícitas
    if verificar_palavras_ilicitas(message.content):
        await message.delete()
        await message.channel.send(f"⚠️ {message.author.mention}, sem palavras ilícitas no chat!")

    # Continuar processando outros comandos do bot, se houver
    await client.process_commands(message)

# Rodar o bot
TOKEN = 'Seu_token_do_servidor_discord'
client.run(TOKEN)

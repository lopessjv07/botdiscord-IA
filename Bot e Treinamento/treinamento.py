import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2  # Regularização L2
import matplotlib.pyplot as plt

# Desativa o aviso de erro se não tiver GPU
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 1. Pré-processamento de dados

# Gerador de dados de treinamento
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Gerador de dados pra validação ou teste
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Carregar dados de treinamento
train_generator = train_datagen.flow_from_directory(
    'C:/Users/Seu_usuario/Documents/dataset/train',  # Caminho do diretório de treinamento
    target_size=(64, 64),  # Tamanho das imagens
    batch_size=32,
    class_mode='binary'  # Classificação binária 0 ou 1
)

# Carregar dados de validação
validation_generator = test_datagen.flow_from_directory(
    'C:/Users/Seu_Ususario/Documents/dataset/test',  # Caminho do diretório de validação 
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# 2. Construção do modelo da rede neural

model = Sequential()

# Camada convolucional com regularização L2
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu', kernel_regularizer=l2(0.01)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

# Camada densa com regularização L2 e Dropout
model.add(Dense(units=128, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))  # Aumentado para 50% de Dropout

# Camada de saída para classificação binária
model.add(Dense(units=1, activation='sigmoid'))

# Compilar o modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 3. Treinar o modelo

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),  # Número de imagens de treinamento dividido pelo batch_size
    epochs= 128,  # Número de épocas
    validation_data=validation_generator,
    validation_steps=len(validation_generator)  # Número de imagens de validação dividido pelo batch_size
)

# 4. Avaliação do modelo

test_loss, test_accuracy = model.evaluate(validation_generator)
print(f'Accuracy no teste: {test_accuracy}')

# 5. Salvar o modelo treinado 
model.save('C:/Users/Seu_usuario/Downloads/meu_modelo.h5')

# 6. Mostra os gráficos de Precisão (acurácia) ou perda 

# Gráfico de perda
plt.plot(history.history['loss'], label='Treinamento')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Perda durante o treinamento')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.legend()
plt.show()

# Gráfico de acurácia(precisão)
plt.plot(history.history['accuracy'], label='Treinamento')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Acurácia durante o treinamento')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()
plt.show()

# Função de classificação da imagem
def classificar_imagem(caminho_imagem):
    img = load_img(caminho_imagem, target_size=(64, 64))  # Carregar a imagem e redimensionar
    img = img_to_array(img)  # Converter para array
    img = np.expand_dims(img, axis=0)  # Adicionar a dimensão do batch
    img /= 255.0  # Normalizar a imagem

    resultado = model.predict(img)[0][0]
    
    # Fala se é licita ou ilicita 
    if resultado > 0.5:
        print('Imagem Classificada como: Lícita (Classe 1)')
    else:
        print('Imagem Classificada como: Ilícita (Classe 0)')
 
# Onde vai salvar o modelo treinado

model.save('C:/Users/Seu_usuario/Downloads/meu_modelo.h5')

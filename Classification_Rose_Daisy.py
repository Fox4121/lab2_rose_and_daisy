import os, cv2, random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
from random import shuffle

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


# Константи
TEST_SIZE = 0.5
RANDOM_STATE = 2018
BATCH_SIZE = 64
NO_EPOCHS = 20
NUM_CLASSES = 2
SAMPLE_SIZE = 20000
TRAIN_FOLDER = './train/'
TEST_FOLDER = './test/'
IMG_SIZE = 224


# Функція кодування назви у one-hot
def label_flower_image_one_hot_encoder(img):
    flower = img.split('.')[-3]
    if flower == 'daisy':
        return [1, 0]
    elif flower == 'rose':
        return [0, 1]


# Обробка даних
def process_data(data_image_list, DATA_FOLDER, isTrain=True):
    data_df = []
    for img in tqdm(data_image_list):
        path = os.path.join(DATA_FOLDER, img)
        if (isTrain):
            label = label_flower_image_one_hot_encoder(img)
        else:
            label = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data_df.append([np.array(img), np.array(label)])
    shuffle(data_df)
    return data_df


# Візуалізація кількості зображень
def plot_image_list_count(data_image_list):
    labels = []
    for img in data_image_list:
        labels.append(img.split('.')[-3])
    sns.countplot(labels)
    plt.title('Rose and Daisy')



# Підгрузка списків зображень
train_image_list = os.listdir("./train/")[0:SAMPLE_SIZE]
test_image_list = os.listdir("./test/")

# Візуалізація кількості
plot_image_list_count(train_image_list)

# Обробка train
train = process_data(train_image_list, TRAIN_FOLDER)


# Показати зображення
def show_images(data, isTest=False):
    f, ax = plt.subplots(5, 5, figsize=(15, 15))
    for i, data in enumerate(data[:25]):
        img_num = data[1]
        img_data = data[0]
        label = np.argmax(img_num)
        if label == 1:
            str_label = 'Rose'
        elif label == 0:
            str_label = 'Daisy'
        if (isTest):
            str_label = "None"
        ax[i // 5, i % 5].imshow(img_data)
        ax[i // 5, i % 5].axis('off')
        ax[i // 5, i % 5].set_title("Label: {}".format(str_label))
    plt.show()


show_images(train)

# Обробка test
test = process_data(test_image_list, TEST_FOLDER, False)
def show_images(data, model=None, isTest=False):
    f, ax = plt.subplots(5, 5, figsize=(15, 15))
    for i in range(min(25, len(data))):
        if isTest:
            img_data = data[i]
        else:
            img_data = data[i][0]

        if isTest and model is not None:
            img_input = img_data / 255.0
            img_input = np.expand_dims(img_input, axis=0)
            pred = model.predict(img_input)
            label = np.argmax(pred)
        else:
            label = np.argmax(data[i][1])

        str_label = 'Rose' if label == 1 else 'Daisy'

        ax[i // 5, i % 5].imshow(img_data)
        ax[i // 5, i % 5].axis('off')
        ax[i // 5, i % 5].set_title(f"Predicted: {str_label}")
    plt.show()


# Формування масивів
X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y = np.array([i[1] for i in train])

# Побудова моделі
model = Sequential()
model.add(ResNet50(include_top=False, pooling='max', weights='imagenet'))
model.add(Dense(NUM_CLASSES, activation='softmax'))
model.layers[0].trainable = True

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Навчання
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

train_model = model.fit(X_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=NO_EPOCHS,
                        verbose=1,
                        validation_data=(X_val, y_val))


# Побудова графіків
def plot_accuracy_and_loss(train_model):
    hist = train_model.history
    acc = hist['accuracy']
    val_acc = hist['val_accuracy']
    loss = hist['loss']
    val_loss = hist['val_loss']
    epochs = range(len(acc))
    f, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].plot(epochs, acc, 'g', label='Training accuracy')
    ax[0].plot(epochs, val_acc, 'r', label='Validation accuracy')
    ax[0].set_title('Training and validation accuracy')
    ax[0].legend()
    ax[1].plot(epochs, loss, 'g', label='Training loss')
    ax[1].plot(epochs, val_loss, 'r', label='Validation loss')
    ax[1].set_title('Training and validation loss')
    ax[1].legend()
    plt.show()


plot_accuracy_and_loss(train_model)

# Оцінка
score = model.evaluate(X_val, y_val, verbose=0)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])

# Прогнозування
predicted_classes = np.argmax(model.predict(X_val), axis=1)
y_true = np.argmax(y_val, axis=1)

# Класифікаційний звіт
target_names = ["Daisy", "Rose"]
print(classification_report(y_true, predicted_classes, target_names=target_names))

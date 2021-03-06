import os

import numpy as np
from flask import Flask, render_template, request, send_from_directory
from keras.layers import Conv2D, Dense, Flatten, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras

app = Flask(__name__)

if __name__ == '__main__':
    app.run()


@app.route('/icons/<filename>')
def getUrl(filename):
    return send_from_directory(
        os.path.join(
            "E:/projects/comvistugas2/templates/icons"
        ), filename
    )


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/upload-image', methods=["GET", "POST"])
def upload():
    ROOT_PATH = "E:/projects/comvistugas2/tmp/dataset"
    CLASS_MODE = 'categorical'
    COLOR_MODE = 'grayscale'

    BATCH_SIZE = 10
    TARGET_SIZE = (32, 32)

    dataset_generator = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=0.2,
        validation_split=0.2,
        zoom_range=0.2
    )

    train_set = dataset_generator.flow_from_directory(
        ROOT_PATH,
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        color_mode=COLOR_MODE,
        class_mode=CLASS_MODE,
        subset='training'
    )

    upload = request.files["image"]

    pathImage = "E:/projects/comvistugas2/uploads"
    imageUploadName = "original_image.jpg"

    fileNameSplit = upload.filename.split('.')
    image.filename = imageUploadName

    upload.save(os.path.join(pathImage, imageUploadName))

    img = image.load_img(pathImage + "/" + imageUploadName, target_size=TARGET_SIZE, color_mode="grayscale")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    model = keras.models.load_model("comvismodel.h5")

    images = np.vstack([x])
    classes = model.predict(images)
    predicted_index = np.argmax(classes)

    print(classes)
    print(predicted_index)

    result = ""

    baseUrl = ""
    if classes[0][predicted_index] > .5:
        if predicted_index == train_set.class_indices["masker"]:
            print('Masker')
            result = "Masker"
        elif predicted_index == train_set.class_indices["nomasker"]:
            result = "Tidak Pakai Masker"
            print('Tidak Pakai Masker')
        else:
            result = "Masih gak yakin"
            print("Index out of range")
    else:
        print("Tidak yakin dengan gambar apa yang dikirim")
        result = "Masih gak yakin"

    return render_template("index.html", result=result)


@app.route('/training-data')
def traindata():
    ROOT_PATH = "E:/projects/comvistugas2/tmp/dataset"
    CLASS_MODE = 'categorical'
    COLOR_MODE = 'grayscale'

    BATCH_SIZE = 15
    TARGET_SIZE = (48, 48)

    EPOCH = 30

    dataset_generator = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.3,
        rotation_range=0.3,
        validation_split=0.2,
        zoom_range=0.2
    )

    validation_generator = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2
    )

    train_set = dataset_generator.flow_from_directory(
        ROOT_PATH,
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        color_mode=COLOR_MODE,
        class_mode=CLASS_MODE,
        subset='training'
    )

    validation_set = validation_generator.flow_from_directory(
        ROOT_PATH,
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        color_mode=COLOR_MODE,
        class_mode=CLASS_MODE,
        subset='validation',
        shuffle=False
    )

    train_set.class_indices

    model = Sequential([
        Conv2D(64, (3, 3), input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 1), activation='relu'),
        Dropout(0.2),
        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(2, activation='softmax')
    ])

    optimizer = SGD(learning_rate=0.1)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(
        train_set,
        epochs=EPOCH,
        validation_data=validation_set
    )

    model.save("modelcomvisbaru.h5")



from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications import VGG16
import matplotlib.pyplot as plt
from keras import optimizers
from time import sleep as sl
from keras import layers
from keras import models
import os, shutil
import numpy as np

# --- GPU eğitim ---

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ----------- veri ön işleme (veri klasör yollarını kodda tanımlama) -----------


train_dir = os.path.join("train") # eğitim seti

validation_dir = os.path.join("validation") # doğrulama seti



train_S_dir = os.path.join(train_dir,"S") # sağlıklı

train_K_dir = os.path.join(train_dir,"K") # hastalıklı

validation_S_dir = os.path.join(validation_dir,"S") 

validation_K_dir = os.path.join(validation_dir,"K") 



# --- öneğitimli CNN ---

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

# --- CNN 'i modele dahil etmek ve bağzı katmanları dondurmak ---

model = models.Sequential()

model.add(conv_base)

model.add(layers.Flatten())

model.add(layers.Dense(256, activation="relu"))

model.add(layers.Dense(1,activation="sigmoid"))

conv_base.trainable = True

set_trainable = True

for layer in conv_base.layers: # sadece son 3 katman öğrenecek geri kalan katmanları donduruyoruz.

	if layer.name == "block5_conv1":

		set_trainable = True

	if set_trainable:

		layer.trainable = True 

	else:

		layer.trainable = False


# --- veri seti çeşitlendirme , hazır hale getirme. --- 

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')


test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(

        train_dir,

        target_size=(150, 150),
        batch_size=20,

        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

# --- eğitim ---

history = model.fit_generator(
      train_generator,
      steps_per_epoch=175,
      epochs=100,
      validation_data=validation_generator,
      validation_steps=75,
      verbose=2)

model.save('demir.h5')

# ----------- sonucları görselleştirme (modelin verimi) -----------

def smooth_curve(points, factor=0.8): # eğitim kayıplarını düzleştirme, düzeltme

	smoothed_points = []

	for point in points:

		if smoothed_points:

			previous = smoothed_points[-1]

			smoothed_points.append(previous * factor * point * (1- factor))

		else:

			smoothed_points.append(point)

	return smoothed_points

acc = history.history["acc"]

val_acc = history.history["val_acc"]

loss = history.history["loss"]

val_loss = history.history["val_loss"]

epochs = range(1, len(acc) + 1)

plt.plot(epochs, smooth_curve(acc), "bo", label="Eğitim başarımı")

plt.plot(epochs, smooth_curve(val_acc), "b", label="Doğrulama başarımı")

plt.title("Eğitim ve doğrulama başarımı")

plt.legend()

plt.figure()

plt.plot(epochs, smooth_curve(loss), "bo", label="Eğitim kaybı")

plt.plot(epochs, smooth_curve(val_loss), "b", label="Doğrulama kaybı")

plt.title("Eğitim ve doğrulama kaybı")

plt.legend()

plt.show()

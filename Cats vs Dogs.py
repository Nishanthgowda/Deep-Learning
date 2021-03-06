import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import matplotlib.pyplot as plt
import numpy as np

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

url="http://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
zip_dir = tf.keras.utils.get_file('cats_and_dogs_filterted.zip',origin=url,extract=True)

zip_dir_base = os.path.dirname(zip_dir)
!find $zip_dir_base -type d -print

base_dir = os.path.join(os.path.dirname(zip_dir), 'cats_and_dogs_filtered')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures

num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

print("the no.of cats images for train:",num_cats_tr)
print("the no.of dogs images for train:",num_dogs_tr)

print("the no.of cats images for validation:",num_cats_val)
print("the no.of dogs images for train:",num_dogs_val)

print("the total no.of images for training:",total_train)
print("the total no.of images for validation:",total_val)

BT_S = 100
IMG_SHAPE = 150

train_image_generator = ImageDataGenerator(
                                            rescale=1./255,
                                            horizontal_flip=True,
                                            zoom_range=0.5,
                                           fill_mode='nearest',
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           shear_range=0.2)

train_data_gen = train_image_generator.flow_from_directory(batch_size=BT_S,shuffle=True,directory=train_dir,target_size=(IMG_SHAPE,IMG_SHAPE),class_mode='binary')

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

valid_image_generator = ImageDataGenerator(rescale=1./255)
valid_data_gen = valid_image_generator.flow_from_directory(batch_size=BT_S,directory=validation_dir,shuffle=True,class_mode='binary',target_size=(IMG_SHAPE,IMG_SHAPE))

sample_training_images, _ = next(train_data_gen)

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()

plotImages(sample_training_images[:5])

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32,(3,3),activation="relu",input_shape=(150,150,3)),
                             tf.keras.layers.MaxPooling2D(2,2),
                             
                             tf.keras.layers.Conv2D(64,(3,3),activation="relu"),
                             tf.keras.layers.MaxPooling2D(2,2),
                             tf.keras.layers.Conv2D(128,(3,3),activation="relu"),
                             tf.keras.layers.MaxPooling2D(2,2),
                             
                             tf.keras.layers.Conv2D(128,(3,3),activation="relu"),
                             tf.keras.layers.MaxPooling2D(2,2),
                             
                             tf.keras.layers.Dropout(0.5),
                             tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(512,activation="relu"),
                             tf.keras.layers.Dense(2,activation="softmax")
                             ])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

history=model.fit_generator(
      train_data_gen,
      steps_per_epoch=int(np.ceil(total_train/float(BT_S))),
      epochs=100,
      validation_data=valid_data_gen,
      validation_steps=int(np.ceil(total_val/float(BT_S)))
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(100)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('./foo.png')
plt.show()

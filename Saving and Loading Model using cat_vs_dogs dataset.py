#install packages

!pip install -U tensorflow_hub
!pip install -U tensorflow_datasets

import time
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
import tensorflow_hub as hub
from tensorflow.keras import layers

(train_set,validation_set),ds = tfds.load('cats_vs_dogs',with_info=True,as_supervised=True,split=['train[:80%]','train[80%:]'])

def formate_image(image,label):
  image = tf.image.resize(image,(224,224))/255.
  return image,label

num_examples = ds.splits['train'].num_examples

IMA_SHP = 224
BT_SZ = 32

train_batches = train_set.cache().shuffle(num_examples//4).map(formate_image).batch(BT_SZ).prefetch(1)
validation_batches = validation_set.cache().map(formate_image).batch(BT_SZ).prefetch(1)

url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feacher_extractor = hub.KerasLayer(url,input_shape=(IMA_SHP,IMA_SHP,3))

feacher_extractor.trainable=False

model = tf.keras.Sequential([
                             feacher_extractor,
                             layers.Dense(2)
])
model.summary()

model.compile(
    optimizer='adam',
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics = ['accuracy']
)

EPOCHS = 3
history = model.fit(train_batches,
                    epochs=EPOCHS,
                    validation_data=validation_batches)

class_names = np.array(ds.features['label'].names)
class_names

image_batch,label_batch = next(iter(train_batches.take(3)))
image_batch = image_batch.numpy()
label_batch = label_batch.numpy()

predicted_batch = model.predict(image_batch)
predicted_batch = tf.squeeze(predicted_batch).numpy()
predicted_ids = np.argmax(predicted_batch,axis=-1)
predicted_class_names = class_names[predicted_ids]
predicted_class_names

print("Labels: ", label_batch)
print("Predicted labels: ", predicted_ids)

plt.figure(figsize=(10,9))
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  color = "blue" if predicted_ids[n] == label_batch[n] else "red"
  plt.title(predicted_class_names[n].title(), color=color)
  plt.axis('off')
_ = plt.suptitle("Model predictions (blue: correct, red: incorrect)")

#Save as Keras .h5 mode

t = time.time()

export_path_name = "./Nishanth.h5"
print(export_path_name)

model.save(export_path_name)



# Load the Keras .h5 Model

reloaded = tf.keras.models.load_model(export_path_name,
                                      # `custom_objects` tells keras how to load a `hub.KerasLayer`
                                      custom_objects={'KerasLayer': hub.KerasLayer})
reloaded.summary()

#We can check that the reloaded model and the previous model give the same result
result_batch = model.predict(image_batch)
reloaded_result_batch = reloaded.predict(image_batch)

#The difference in output should be zero:
(abs(reloaded_result_batch - result_batch)).max()

EPOCHS = 3
history = reloaded.fit(train_batches,
                    epochs=EPOCHS,
                    validation_data=validation_batches)

#Export as SavedModel

# Exporting a whole model to the TensorFlow SavedModel formate
t = time.time()

export_path_sm = "./{}".format(int(t))
print(export_path_sm)

tf.saved_model.save(model, export_path_sm)

#Load SavedModel

#Loading SavedModel and use it to make predictions
reloaded_sm = tf.saved_model.load(export_path_sm)

reloaded_result_batch_rm = reloaded_sm(image_batch,training=False).numpy()

(abs(result_batch - reloaded_result_batch_rm)).max()

#Loading the SavedModel as a Keras Model

#Loading the SavedModel as a Keras Model
t = time.time()
export_path_sm = "./.h5".format(int(t))
print(export_path_sm)

tf.saved_model.save(model,export_path_sm)

reload_sm_keras = tf.keras.models.load_model(
    export_path_sm,
    custom_objects={"KerasLayer":hub.KerasLayer}
)
reload_sm_keras.summary()

result_batch = model.predict(image_batch)
reload_sm_keras_result_batch = reload_sm_keras.predict(image_batch)

(abs(result_batch - reload_sm_keras_result_batch)).max()

#Download your model

#Download model to local Disk
!zip -r model.zip {export_path_sm}

try:
  from google.colab import files
  files.download('./model.zip')
except ImportError:
  pass


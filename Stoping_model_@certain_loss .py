import tensorflow as tf
import matplotlib.pyplot as plt

fhminst = tf.keras.datasets.fashion_mnist

(train_image,train_label),(test_image,test_label) = fhminst.load_data()

class myCallback(tf.keras.callbacks.Callback):
  def epoch_on_ha(self,epoch,logs={}):
    if(logs.get('loss')<0.4):
      print("you reached 60% accuracy so stop")
      self.model.stop_trainig = True

import numpy as np
np.set_printoptions(linewidth=200)
plt.imshow(train_image[0])
print(train_label[0])
print(train_image[0])

train_image = train_image / 255.0
test_image  = test_image / 255.0

model = tf.keras.Sequential([
                             tf.keras.layers.Flatten(input_shape=(28,28)),
                             tf.keras.layers.Dense(512,activation=tf.nn.relu),
                             tf.keras.layers.Dense(256,activation=tf.nn.relu),
                             tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer="Adam",
              loss = "sparse_categorical_crossentropy",
              metrics = ['accuracy'])

callback = myCallback()
model.fit(train_image,train_label,epochs=10,callbacks=[callback])

model.evaluate(test_image,test_label)

classification = model.predict(test_image)
print(classification[0])

print(test_label[0])


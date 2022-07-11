import tensorflow as tf
import gradio as gr

print(tf.__version__)

tf.get_logger().setLevel('ERROR')
(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

import matplotlib.pyplot as plt
flg, ax = plt.subplots(nrows=2, ncols=5, figsize=(10, 10), tight_layout=True)

n = 0
for i in range(2):
  for j in range(5):
    ax[i][j].imshow(x_train[n], cmap=plt.cm.binary)
    n += 1

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28,28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)

_, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(test_acc)

predictions = model.predict(x_test)

plt.imshow(x_test[0], cmap=plt.cm.binary)
pass

import numpy as np

np.argmax(predictions[0])

def recognize_digit(img):
  img = img.reshape(1, 28, 28)
  prediction = model.predict(img).tolist()[0]
  return {str(i): prediction[i] for i in range(10)}

label = gr.outputs.Label(num_top_classes=4)
interface = gr.Interface(fn=recognize_digit, inputs='sketchpad', outputs=label, live=False, title='Digit Recognizer')

interface.launch(share=True)

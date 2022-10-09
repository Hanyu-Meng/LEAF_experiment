import numpy as np
import tensorflow as tf
from leaf_audio import models
from example import data
import tensorflow_datasets as tfds
import config as CONF



datasets, info = tfds.load('speech_commands', with_info=True)
datasets = data.prepare(datasets, batch_size=256)
num_classes = info.features['label'].num_classes
model1 = tf.keras.models.load_model('Hanyu_model/emotion_recognition_LEAF')
encoder = model1.layers[1]
pooling = model1.layers[2]
dense = model1.layers[3]

model2 = tf.keras.models.load_model('Hanyu_model/emotion_recognition_not_trainable_LEAF')

leaf_not_trained = model2.layers[0]

model = models.AudioClassifier(num_classes,frontend=leaf_not_trained,encoder=encoder)
model._encoder = encoder
model._frontend = leaf_not_trained
model._pool = pooling
model._head = dense

# model = tf.keras.Sequential()
# model.add(leaf)
# model.add(encoder)
# model.add(pooling)
# model.add(dense)
loss = tf.keras.losses

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


model.compile(loss=loss_fn,
              optimizer=tf.keras.optimizers.Adam(CONF.BACKEND_LEARNING_RATE),
              metrics=[CONF.METRIC])

score = model.evaluate(datasets['test'], verbose = 0)
model.summary()
# score = model.evaluate(datasets['test'], verbose=0)

print("score is ", score)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
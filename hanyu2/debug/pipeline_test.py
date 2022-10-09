import tensorflow as tf
import tensorflow_datasets as tfds
from leaf_audio import models
from example import data
import nurupo2/config as CONF
# Load data set
dataset, info = tfds.load("voxforge", split = 'train', shuffle_files=True,data_dir="/home/lecai/Data/voxforge",
                          with_info=True)
print("LOADED")
print(dataset)

datasets = data.prepare(dataset, batch_size=CONF.batch_size)

num_classes = info.features['label'].num_classes

model = models.AudioClassifier(num_outputs=num_classes, **kwargs)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

metric = 'sparse_categorical_accuracy'

model.compile(loss=loss_fn,
              optimizer=tf.keras.optimizers.Adam(learning_rate),
              metrics=[metric])

model.fit(datasets['train'],
            validation_data=datasets['eval'],
            batch_size=None,
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=[model_checkpoint_callback])
# Display the model's architecture
model.summary()

# Save the entire model as a SavedModel.
model.save('saved_model/LEAF_language_ID')

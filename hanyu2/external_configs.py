"""Makes some classes and function gin configurable."""

import gin.tf.external_configurables
import tensorflow as tf
import tensorflow_addons.layers as tfa_layers

configurables = {
    'tf.keras.layers': (
        tf.keras.layers.Conv1D,
        tf.keras.layers.Conv1DTranspose,
        tf.keras.layers.Conv2D,
        tf.keras.layers.Conv2DTranspose,
        tf.keras.layers.Dense,
        tf.keras.layers.Flatten,
        tf.keras.layers.Reshape,
        tf.keras.layers.MaxPooling2D,
        tf.keras.layers.GlobalMaxPooling2D,
        tf.keras.layers.BatchNormalization,
        tf.keras.layers.LayerNormalization,
    ),
    'tfa.layers': (
        tfa_layers.GroupNormalization,
        tfa_layers.InstanceNormalization,
    ),
    'tf.keras.losses': (
        tf.keras.losses.BinaryCrossentropy,
        tf.keras.losses.CategoricalCrossentropy,
        tf.keras.losses.MeanSquaredError,
        tf.keras.losses.MeanAbsoluteError,
    ),
    'tf.keras.regularizers': (
        tf.keras.regularizers.L1,
        tf.keras.regularizers.L2,
        tf.keras.regularizers.L1L2,
    ),
    'tf.keras.applications': (
        tf.keras.applications.EfficientNetB0,
        tf.keras.applications.EfficientNetB1,
        tf.keras.applications.EfficientNetB3,
        tf.keras.applications.MobileNetV2,
        tf.keras.applications.ResNet50,
    ),
    'tf.keras.initializers': (
        tf.keras.initializers.Constant,
    )
}


for module in configurables:
  for v in configurables[module]:
    gin.config.external_configurable(v, module=module)

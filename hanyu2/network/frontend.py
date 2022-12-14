# LEAF frontend
from typing import Callable, Optional
import gin
import network.convolution as convolution
import network.initializers as initializers
import network.pooling as pooling
import network.postprocessing as postprocessing
import tensorflow.compat.v2 as tf
import tensorflow_addons as tfa
import config as CONF

_TensorCallable = Callable[[tf.Tensor], tf.Tensor]
_Initializer = tf.keras.initializers.Initializer

gin.external_configurable(tf.keras.regularizers.l1_l2,
                          module='tf.keras.regularizers')


class SquaredModulus(tf.keras.layers.Layer):
    """Squared modulus layer.

    Returns a keras layer that implements a squared modulus operator.
    To implement the squared modulus of C complex-valued channels, the expected
    input dimension is N*1*W*(2*C) where channels role alternates between
    real and imaginary part.
    The way the squared modulus is computed is real ** 2 + imag ** 2 as follows:
    - squared operator on real and imag
    - average pooling to compute (real ** 2 + imag ** 2) / 2
    - multiply by 2

    Attributes:
    pool: average-pooling function over the channel dimensions
    """

    def __init__(self):
        super().__init__(name='squared_modulus')
        self._pool = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2)

    def call(self, x):
        x = tf.transpose(x, perm=[0, 2, 1])
        output = 2 * self._pool(x**2)
        return tf.transpose(output, perm=[0, 2, 1])


@gin.configurable
class Leaf(tf.keras.models.Model):
  """
  LEAF process: Gabor filterbank --> Squaremodule --> Gaussian LPF --> sPCEN
  """
  def __init__(
      self,
      learn_pooling: bool = CONF.Gaussian,
      learn_filters: bool = CONF.Gabor,
      conv1d_cls=convolution.GaborConv1D,
      activation=SquaredModulus(),
      pooling_cls=pooling.GaussianLowpass,
      n_filters: int = 40, # 40 filters for the first layer
      sample_rate: int = 16000, # fs = 16kHz
      window_len: float = 25., # 25ms frame
      window_stride: float = 10., # 10ms shift
      # compression layer: PCEN
      compression_fn: _TensorCallable = postprocessing.PCENLayer(
          alpha=0.96,
          smooth_coef=0.04,
          delta=2.0,
          floor=1e-12,
          trainable = CONF.PCEN,
          learn_smooth_coef = CONF.sPCEN,
          per_channel_smooth_coef= CONF.sPCEN_pre_channel),
      preemp: bool = False, # no pre-emphasize
      preemp_init: _Initializer = initializers.PreempInit(),
      complex_conv_init: _Initializer = initializers.GaborInit(
          sample_rate=16000, min_freq=60.0, max_freq=7800.0), # initial setting of gabor filters
      pooling_init: _Initializer = tf.keras.initializers.Constant(0.4),
      regularizer_fn: Optional[tf.keras.regularizers.Regularizer] = None,
      mean_var_norm: bool = False,
      spec_augment: bool = False,
      name='leaf'):
    super().__init__(name=name)
    window_size = int(sample_rate * window_len // 1000 + 1)
    window_stride = int(sample_rate * window_stride // 1000)
    if preemp:
      self._preemp_conv = tf.keras.layers.Conv1D(
          filters=1,
          kernel_size=2,
          strides=1,
          padding='SAME',
          use_bias=False,
          input_shape=(None, None, 1),
          kernel_initializer=preemp_init,
          kernel_regularizer=regularizer_fn if learn_filters else None,
          name='tfbanks_preemp',
          trainable=learn_filters)

    self._complex_conv = conv1d_cls(
        filters=2 * n_filters,
        kernel_size=window_size,
        strides=1,
        padding='SAME',
        use_bias=False,
        input_shape=(None, None, 1),
        kernel_initializer=complex_conv_init,
        kernel_regularizer=regularizer_fn if learn_filters else None,
        name='tfbanks_complex_conv',
        trainable=learn_filters)

    self._activation = activation
    self._pooling = pooling_cls(
        kernel_size=window_size,
        strides=window_stride,
        padding='SAME',
        use_bias=False,
        kernel_initializer=pooling_init,
        kernel_regularizer=regularizer_fn if learn_pooling else None,
        trainable=learn_pooling)

    self._instance_norm = None
    if mean_var_norm:
      self._instance_norm = tfa.layers.InstanceNormalization(
          axis=2,
          epsilon=1e-6,
          center=True,
          scale=True,
          beta_initializer='zeros',
          gamma_initializer='ones',
          name='tfbanks_instancenorm')

    self._compress_fn = compression_fn if compression_fn else tf.identity
    self._spec_augment_fn = postprocessing.SpecAugment(
    ) if spec_augment else tf.identity

    self._preemp = preemp

  def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
    """Computes the Leaf representation of a batch of waveforms.

    Args:
      inputs: input --> [batch_size, num_samples] or [batch_size, num_samples, 1].
      training: training mode, controls whether SpecAugment is applied or not.

    Returns:
      Leaf features --> [batch_size, time_frames, freq_bins].
    """
    # Inputs should be [B, W] or [B, W, C]
    outputs = inputs[:, :, tf.newaxis] if inputs.shape.ndims < 3 else inputs

    if self._preemp:
        outputs = self._preemp_conv(outputs)

    outputs = self._complex_conv(outputs)
    outputs = self._activation(outputs)
    outputs = self._pooling(outputs)
    outputs = tf.maximum(outputs, 1e-5)
    outputs = self._compress_fn(outputs)
    if self._instance_norm is not None:
        outputs = self._instance_norm(outputs)
    if training:
        outputs = self._spec_augment_fn(outputs)
    return outputs


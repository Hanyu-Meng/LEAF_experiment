"""Initializer classes for each layer of the learnable frontend."""

import network.filterbank as filterbank
import network.utils as utils
import numpy as np
import tensorflow.compat.v2 as tf


class PreempInit(tf.keras.initializers.Initializer):
  """Keras initializer for the pre-emphasis.

  Returns a Tensor to initialize the pre-emphasis layer of a Leaf instance.

  Attributes:
    alpha: parameter that controls how much high frequencies are emphasized by
      the following formula output[n] = input[n] - alpha*input[n-1] with 0 <
      alpha < 1 (higher alpha boosts high frequencies)
  """

  def __init__(self, alpha=0.97):
    self.alpha = alpha

  def __call__(self, shape, dtype=None):
    assert shape == (
        2, 1, 1), 'Cannot initialize preemp layer of size {}'.format(shape)
    preemp_arr = np.zeros(shape)
    preemp_arr[0, 0, 0] = -self.alpha
    preemp_arr[1, 0, 0] = 1
    return tf.convert_to_tensor(preemp_arr, dtype=dtype)

  def get_config(self):
    return self.__dict__


class GaborInit(tf.keras.initializers.Initializer):
  """Keras initializer for the complex-valued convolution.

  Returns a Tensor to initialize the complex-valued convolution layer of a
  Leaf instance with Gabor filters designed to match the
  frequency response of standard mel-filterbanks.

  If the shape has rank 2, this is a complex convolution with filters only
  parametrized by center frequency and FWHM, so we initialize accordingly.
  In this case, we define the window len as 401 (default value), as it is not
  used for initialization.
  """

  def __init__(self, **kwargs):
    kwargs.pop('n_filters', None)
    self._kwargs = kwargs

  def __call__(self, shape, dtype=None):
    n_filters = shape[0] if len(shape) == 2 else shape[-1] // 2
    window_len = 401 if len(shape) == 2 else shape[0]
    gabor_filters = filterbank.Gabor(
        n_filters=n_filters, window_len=window_len, **self._kwargs)
    if len(shape) == 2:
      return gabor_filters.gabor_params_from_mels
    else:
      even_indices = tf.range(shape[2], delta=2)
      odd_indices = tf.range(start=1, limit=shape[2], delta=2)
      filters = gabor_filters.gabor_filters
      filters_real_and_imag = tf.dynamic_stitch(
          [even_indices, odd_indices],
          [tf.math.real(filters), tf.math.imag(filters)])
      return tf.transpose(filters_real_and_imag[:, tf.newaxis, :], [2, 1, 0])

  def get_config(self):
    return self._kwargs



class LowpassInit(tf.keras.initializers.Initializer):
  """Keras initializer for the lowpass filter.

  Returns a Tensor to initialize the complex-valued convolution layer of a
  TDFbanks instance with a window function.

  Attributes:
    nfilters: number of filters
    sample_rate: sampling rate of the input signal (samples/s)
    window_len: window size in ms
    window_type: a WindowType
  """

  def __init__(
      self,
      sample_rate: int = 16000,
      window_len: float = 25.,
      window_type: utils.WindowType = utils.WindowType.SQUARED_HANNING):
    self.sample_rate = sample_rate
    self.window_len = window_len
    self.window_type = window_type

  def __call__(self, shape, dtype=None):
    lowpass_arr = np.zeros(shape)
    lowpass_filter = utils.window(
        self.window_type, int(self.sample_rate * self.window_len // 1000 + 1))
    for channel_idx in range(lowpass_arr.shape[2]):
      if lowpass_arr.ndim == 3:
        lowpass_arr[:, 0, channel_idx] = lowpass_filter
      else:
        lowpass_arr[0, :, channel_idx, 0] = lowpass_filter
    return tf.convert_to_tensor(lowpass_arr, dtype=dtype)

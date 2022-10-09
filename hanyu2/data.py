"""Preprocess the input data."""

import functools
from typing import Dict, Mapping
import config as CONF
import gin
import tensorflow as tf


AUTOTUNE = tf.data.experimental.AUTOTUNE

def db_to_linear(samples):
  return 10.0 ** (samples / 20.0)


@gin.configurable
def loudness_normalization(samples: tf.Tensor,
                           target_db: float = 15.0,
                           max_gain_db: float = 30.0):
  """Normalizes the loudness of the input signal."""
  std = tf.math.reduce_std(samples) + 1e-9
  gain = tf.minimum(db_to_linear(max_gain_db), db_to_linear(target_db) / std)
  return gain * samples


@gin.configurable
def align(samples: tf.Tensor, seq_len: int = CONF.align_seq_len):
  pad_length = tf.maximum(seq_len - tf.size(samples), 0)
  return tf.image.random_crop(tf.pad(samples, [[0, pad_length]]), [seq_len])


def preprocess(inputs: Mapping[str, tf.Tensor],
               transform_fns=(align, loudness_normalization)):
  """Sequentially applies the transformations to the waveform."""
  audio = tf.cast(inputs['audio'], tf.float32) / tf.int16.max
  # audio = tf.cast(inputs['audio'], tf.float32)
  audio = tf.cast(inputs['audio'], tf.int16)
  for transform_fn in transform_fns:
    audio = transform_fn(audio)
  # return audio, inputs['instrument']['family']
  return audio, inputs['label']
  # return audio, inputs['pitch']
  # return audio, inputs['speaker_id']


@gin.configurable
def prepare(datasets: Mapping[str, tf.data.Dataset],
            transform_fns=(align, loudness_normalization),
            batch_size: int = 256) -> Dict[str, tf.data.Dataset]:
  """Prepares the datasets for training and evaluation."""
  valid = 'valid' if 'valid' in datasets else 'test'
  test = 'test'
  # split = 'train_clean100'
  # x_data = datasets[split]
  # train_sample_num = 25740
  # valid_train_num = 2799
  # x_train, x_valid = tf.split(
  #     x_data,
  #     num_or_size_splits=[train_sample_num, valid_train_num],
  #     axis=0
  # )
  # y_train, y_valid = tf.split(
  #   self.y,
  #   [train_sample, test_sample, valid_train],
  #   axis=0
  # )
  # ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(256).shuffle(1000000)  # 封装 dataset数据集格式
  #
  # ds = ds.map(functools.partial(preprocess, transform_fns=transform_fns),
  #                num_parallel_calls=AUTOTUNE)
  result = {}
  for split, key in ('train', 'train'), (valid, 'valid'), (test, 'test'):
    ds = datasets[split]
    ds = ds.map(functools.partial(preprocess, transform_fns=transform_fns),
                num_parallel_calls=AUTOTUNE)
    result[key] = ds.batch(batch_size).prefetch(AUTOTUNE)

  return result

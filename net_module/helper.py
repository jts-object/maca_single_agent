# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A library of helpers for use with SamplingDecoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf 
import six
from tensorflow.contrib.seq2seq import Helper
from tensorflow.contrib.seq2seq.python.ops import decoder


__all__ = [
    "GreedyEmbeddingHelper",
]

_transpose_batch_time = decoder._transpose_batch_time  # pylint: disable=protected-access

class GreedyEmbeddingHelper(Helper):
  """A helper for use during inference.

  Uses the argmax of the output (treated as logits) and passes the
  result through an embedding layer to get the next input.
  """

  def __init__(self, memory, start_tokens, end_token):
    """Initializer.

    Args:
      embedding: A callable that takes a vector tensor of `ids` (argmax ids),
        or the `params` argument for `embedding_lookup`. The returned tensor
        will be passed to the decoder input.
      start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
      end_token: `int32` scalar, the token that marks end of decoding.

    Raises:
      ValueError: if `start_tokens` is not a 1D tensor or `end_token` is not a
        scalar.
    """
    # if callable(embedding):
    #   self._embedding_fn = embedding
    # else:
    #   self._embedding_fn = (
    #       lambda ids: tf.nn.embedding_lookup(embedding, ids))

    self._start_tokens = tf.convert_to_tensor(
        start_tokens, dtype=tf.int32, name="start_tokens")
    self._end_token = tf.convert_to_tensor(
        end_token, dtype=tf.int32, name="end_token")
    if self._start_tokens.get_shape().ndims != 1:
      raise ValueError("start_tokens must be a vector")
    self._batch_size = tf.size(start_tokens)
    if self._end_token.get_shape().ndims != 0:
      raise ValueError("end_token must be a scalar")
    self._memory = memory
    batch, _, embedding_size = self._memory.shape.as_list()
    self._start_inputs = tf.zeros([batch, embedding_size], dtype=tf.float32)
    # self._start_inputs = self._embedding_fn(self._start_tokens)

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def sample_ids_shape(self):
    return tf.TensorShape([])

  @property
  def sample_ids_dtype(self):
    return tf.int32

  def initialize(self, name=None):
    finished = tf.tile([False], [self._batch_size])
    return (finished, self._start_inputs)

  def sample(self, time, outputs, state, name=None):
    """sample for GreedyEmbeddingHelper."""
    del time  # unused by sample_fn
    # Outputs are logits, use argmax to get the most probable id
    if not isinstance(outputs, tf.Tensor):
      raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                      type(outputs))

    # 根据alignments得到最后的输出，因此和指针保持一致。
    sample_ids = tf.argmax(state.alignments, axis=-1, output_type=tf.int32)
    return sample_ids

  def next_inputs(self, time, outputs, state, sample_ids, name=None):
    """next_inputs_fn for GreedyEmbeddingHelper."""
    del time, outputs  # unused by next_inputs_fn

    finished = tf.equal(sample_ids, self._end_token)
    all_finished = tf.reduce_all(finished)

    index_ = tf.stack([tf.range(self._memory.shape.as_list()[0]), sample_ids], axis=1)
    next_inputs = tf.cond(
        all_finished,
        # If we're finished, the next_inputs value doesn't matter
        lambda: self._start_inputs,
        lambda: tf.gather_nd(self._memory, index_)
        )
    
    return (finished, next_inputs, state)


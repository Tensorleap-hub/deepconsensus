# Copyright (c) 2021, Google Inc.
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# 3. Neither the name of Google Inc. nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""TF2 + tf.keras implementations of networks for DeepConsensus."""

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import ml_collections
import tensorflow as tf

from deepconsensus.models import data_providers
from deepconsensus.models import encoder_stack
from official.nlp.modeling import layers
from deepconsensus.utils import dc_constants


@tf.keras.utils.register_keras_serializable(package="DeepConsensus")
class ModifiedOnDeviceEmbedding(layers.OnDeviceEmbedding):
  """Subclass of OnDeviceEmbedding, init similar to EmbeddingSharedWeights."""

  def __init__(self, vocab_size, embedding_width, **kwargs):
    super().__init__(
        vocab_size=vocab_size,
        hidden_size=embedding_width,
        initializer=tf.random_normal_initializer(
            mean=0.0, stddev=embedding_width**-0.5
        ),
        scale_factor=embedding_width**0.5,
        **kwargs,
    )
    self._vocab_size = vocab_size
    self._embedding_width = embedding_width

  def call(self, inputs):
    embeddings = super().call(inputs)
    # Zero out embeddings where input == 0:
    mask = tf.cast(tf.not_equal(inputs, 0), embeddings.dtype)
    embeddings *= tf.expand_dims(mask, -1)
    return embeddings

  def get_config(self):
    # We must return a *plain* dictionary with JSON-serializable values
    config = super().get_config()
    config.update({
        'vocab_size': self._vocab_size,
        'embedding_width': self._embedding_width,
    })
    return config

  @classmethod
  def from_config(cls, config):
    # Keras calls this to reconstruct the layer from the config dict
    return cls(**config)


# pylint: disable=invalid-name
def FullyConnectedNet(params: ml_collections.ConfigDict) -> tf.keras.Model:
  """Fully connected neural network architecture."""

  inputs = tf.keras.Input(
      shape=(params.hidden_size, params.max_length, params.num_channels)
  )
  l2_reg = tf.keras.regularizers.l2
  net = inputs
  net = tf.keras.layers.Flatten()(net)
  for i in range(len(params.fc_size)):
    net = tf.keras.layers.Dense(
        units=params.fc_size[i],
        activation=tf.nn.relu,
        kernel_regularizer=l2_reg(params.l2),
    )(net)
    net = tf.keras.layers.Dropout(rate=params.fc_dropout)(net)

  net = tf.keras.layers.Dense(
      units=params.max_length * dc_constants.SEQ_VOCAB_SIZE
  )(net)
  net = tf.keras.layers.Reshape(
      (params.max_length, dc_constants.SEQ_VOCAB_SIZE)
  )(net)
  net = tf.keras.layers.Softmax(axis=-1)(net)
  outputs = net
  return tf.keras.Model(inputs=inputs, outputs=outputs)


def get_conv_sub_model(
    conv_model,
) -> Tuple[
    Callable[..., tf.Tensor], Callable[[tf.keras.Model], tf.keras.Model]
]:
  """Returns a predefined convolutional architecture."""
  if conv_model == 'resnet50':
    return (
        tf.keras.applications.ResNet50V2,
        tf.keras.applications.resnet_v2.preprocess_input,
    )
  elif conv_model == 'resnet101':
    return (
        tf.keras.applications.ResNet101V2,
        tf.keras.applications.resnet_v2.preprocess_input,
    )
  elif conv_model == 'resnet152':
    return (
        tf.keras.applications.ResNet152V2,
        tf.keras.applications.resnet_v2.preprocess_input,
    )
  else:
    raise NotImplementedError(f'conv model "{conv_model}" not found')


# pylint: disable=invalid-name
class ConvNet(tf.keras.Model):
  """Convolutional neural network architecture."""

  def __init__(self, params: ml_collections.ConfigDict, **kwargs):
    super().__init__(**kwargs)
    # Convert ConfigDict to a plain dictionary for serialization:
    self.params = dict(params)

    self.resnet_input_shape = (self.params['hidden_size'], 
                               self.params['max_length'], 
                               3)
    self.dimensions = (self.params['max_length'] * dc_constants.SEQ_VOCAB_SIZE)

    model_cls, self.conv_preprocess = get_conv_sub_model(
        self.params['conv_model']
    )
    self.model = model_cls(
        include_top=False,
        weights=None,
        input_shape=self.resnet_input_shape,
        pooling='avg',
    )
    self.use_sn = self.params.get('use_sn', False)
    self.max_length = self.params['max_length']

    self.layer_dense = tf.keras.layers.Dense(units=self.dimensions)

  def call(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
    input_rows, _, sn_rows = tf.split(inputs, [3, 1, 1], 3)

    cn_input = self.conv_preprocess(input_rows)
    net = self.model(cn_input, training=training)

    if self.use_sn:
      logging.info('Using SN Values')
      sn_rows = tf.image.crop_to_bounding_box(sn_rows, 0, 0, 4, self.max_length)
      sn_rows = tf.keras.layers.Flatten()(sn_rows)
      net = tf.keras.layers.Flatten()(net)
      net = tf.concat([net, sn_rows], 1)
    else:
      net = tf.keras.layers.Flatten()(net)

    net = self.layer_dense(net)
    net = tf.keras.layers.Reshape(
        (self.max_length, dc_constants.SEQ_VOCAB_SIZE)
    )(net)
    net = tf.keras.layers.Softmax(axis=-1)(net)
    return net

  def get_config(self):
    # Return only JSON-serializable data
    config = super().get_config()
    config.update({
        'params': self.params,  # now a plain dict, not a ConfigDict
        'resnet_input_shape': self.resnet_input_shape,
        'dimensions': self.dimensions,
        'use_sn': self.use_sn,
        'max_length': self.max_length,
    })
    return config

  @classmethod
  def from_config(cls, config):
    # Re-create a ml_collections.ConfigDict only if you want:
    # Or simply pass the plain dict if your code no longer needs ConfigDict.
    params = ml_collections.ConfigDict(config['params'])
    # Rebuild the object
    obj = cls(params)
    # If you want to restore any other fields:
    obj.resnet_input_shape = tuple(config['resnet_input_shape'])
    obj.dimensions = config['dimensions']
    obj.use_sn = config['use_sn']
    obj.max_length = config['max_length']
    return obj

class EncoderOnlyTransformer(tf.keras.Model):
  """Modified encoder-only transformer for DeepConsensus."""

  def __init__(
      self,
      params: ml_collections.ConfigDict,
      name: Optional[str] = None,
      **kwargs,
  ):
    super().__init__(**kwargs)
    # Convert params to plain dict for JSON-serialization
    self.params = dict(params)

    if self.params['add_pos_encoding']:
      self.position_embedding = layers.RelativePositionEmbedding(
          hidden_size=self.params['hidden_size']
      )
    self.encoder_stack = encoder_stack.EncoderStack(self.params)
    self.fc1 = tf.keras.layers.Dense(
        units=dc_constants.SEQ_VOCAB_SIZE,
        activation=None,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
    )
    self.softmax = tf.keras.layers.Softmax()

  def call(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
    # [ ... same logic as before ... ]
    with tf.name_scope('Transformer'):
      intermediate_outputs_dict = self.get_intermediate_outputs(
          inputs, training=training
      )
      logits = intermediate_outputs_dict['logits']
      preds = self.softmax(logits)
      return preds

  def get_intermediate_outputs(
      self, inputs: tf.Tensor, training: bool
  ) -> Dict[str, tf.Tensor]:
    # [ ... same logic as before ... ]
    # Make sure any custom ops are straightforward
    # for Keras to track in a static graph.
    inputs = tf.squeeze(inputs, -1)
    inputs = tf.transpose(inputs, [0, 2, 1])
    all_zeros = tf.reduce_sum(tf.zeros_like(inputs), -1)
    attention_bias = tf.expand_dims(tf.expand_dims(all_zeros, 1), 1)
    return self.encode(inputs, attention_bias, training)

  def encode(
      self, inputs: tf.Tensor, attention_bias: tf.Tensor, training: bool
  ) -> Dict[str, tf.Tensor]:
    with tf.name_scope('encode'):
      encoder_inputs = inputs
      if self.params['add_pos_encoding'] and (encoder_inputs.shape[2] % 2 != 0):
        empty_row = tf.zeros(
            shape=(encoder_inputs.shape[0], encoder_inputs.shape[1], 1)
        )
        encoder_inputs = tf.concat([encoder_inputs, empty_row], axis=-1)
        assert self.params['hidden_size'] == encoder_inputs.shape[2]

      inputs_padding = tf.reduce_sum(tf.zeros_like(encoder_inputs), -1)
      attention_bias = tf.cast(attention_bias, dc_constants.TF_DATA_TYPE)

      if self.params['add_pos_encoding']:
        pos_encoding = self.position_embedding(inputs=encoder_inputs)
        pos_encoding = tf.cast(pos_encoding, dc_constants.TF_DATA_TYPE)
        encoder_inputs += pos_encoding

      if training:
        encoder_inputs = tf.nn.dropout(
            encoder_inputs, rate=self.params['layer_postprocess_dropout']
        )

      encoder_outputs_dict = self.encoder_stack(
          encoder_inputs, attention_bias, inputs_padding, training=training
      )

      encoder_outputs = self.fc1(encoder_outputs_dict['final_output'])
      encoder_outputs_dict['logits'] = encoder_outputs
      return encoder_outputs_dict

  def decode(
      self,
      encoder_outputs: tf.Tensor,
      attention_bias: tf.Tensor,
      training: bool,
  ) -> tf.Tensor:
    raise NotImplementedError

  def predict(self, encoder_inputs: tf.Tensor) -> tf.Tensor:
    return self.call(encoder_inputs, training=False)

  def get_config(self):
    # Must return a JSON-serializable dictionary
    config = super().get_config()
    config.update({
        'params': self.params,  # now a plain dict
    })
    return config

  @classmethod
  def from_config(cls, config):
    # Rebuild the model. If you want to return a ml_collections.ConfigDict:
    params = ml_collections.ConfigDict(config['params'])
    return cls(params)


class EncoderOnlyLearnedValuesTransformer(EncoderOnlyTransformer):
  """Modified transformer that learns embeddings for the bases."""

  def __init__(
      self, params: ml_collections.ConfigDict, name: Optional[str] = None
  ):
    # Convert to dict so it's JSON-serializable
    params = dict(params)  
    super().__init__(params, name=name)

    # We re-wrap self.params as a dict if not already done in the parent
    self.params = params

    if self.params.get('use_bases', False):
      self.bases_embedding_layer = ModifiedOnDeviceEmbedding(
          vocab_size=dc_constants.SEQ_VOCAB_SIZE,
          embedding_width=self.params['per_base_hidden_size'],
          name='bases_embedding',
      )
    if self.params.get('use_pw', False):
      pw_vocab_size = self.params['PW_MAX'] + 1
      self.pw_embedding_layer = ModifiedOnDeviceEmbedding(
          vocab_size=pw_vocab_size,
          embedding_width=self.params['pw_hidden_size'],
          name='pw_embedding',
      )
    if self.params.get('use_ip', False):
      ip_vocab_size = self.params['IP_MAX'] + 1
      self.ip_embedding_layer = ModifiedOnDeviceEmbedding(
          vocab_size=ip_vocab_size,
          embedding_width=self.params['ip_hidden_size'],
          name='ip_embedding',
      )
    if self.params.get('use_ccs_bq', False):
      ccs_bq_scores_vocab_size = self.params['CCS_BQ_MAX']
      self.ccs_base_quality_scores_embedding_layer = ModifiedOnDeviceEmbedding(
          vocab_size=ccs_bq_scores_vocab_size,
          embedding_width=self.params['ccs_bq_hidden_size'],
          name='ccs_base_quality_scores_embedding',
      )
    if self.params.get('use_sn', False):
      sn_vocab_size = self.params['SN_MAX'] + 1
      self.sn_embedding_layer = ModifiedOnDeviceEmbedding(
          vocab_size=sn_vocab_size,
          embedding_width=self.params['sn_hidden_size'],
          name='sn_embedding',
      )
    if self.params.get('use_strand', False):
      strand_vocab_size = self.params['STRAND_MAX'] + 1
      self.strand_embedding_layer = ModifiedOnDeviceEmbedding(
          vocab_size=strand_vocab_size,
          embedding_width=self.params['strand_hidden_size'],
          name='strand_embedding',
      )

    if self.params.get('condense_transformer_input', False):
      logging.info('Condensing input.')
      self.transformer_input_condenser = tf.keras.layers.Dense(
          units=self.params['transformer_input_size'],
          activation=None,
          use_bias=False,
          kernel_initializer='glorot_uniform',
          bias_initializer='zeros',
      )

  def encode(
      self, inputs: tf.Tensor, attention_bias: tf.Tensor, training: bool
  ) -> Dict[str, tf.Tensor]:
    # [ ... same logic as before ... ]
    embedded_inputs = []
    (
        base_indices,
        pw_indices,
        ip_indices,
        strand_indices,
        ccs_indices,
        ccs_bq_indices,
        sn_indices,
    ) = data_providers.get_indices(
        self.params['max_passes'],
        self.params['use_ccs_bq'],
    )

    if self.params.get('use_bases', False):
      for i in range(*base_indices):
        embedded = self.bases_embedding_layer(tf.cast(inputs[:, :, i], tf.int32))
        embedded_inputs.append(embedded)
    # [ ... same pattern for pw, ip, ccs, ccs_bq, sn, strand ... ]

    embedded_inputs = tf.concat(embedded_inputs, axis=-1)
    embedded_inputs = tf.cast(embedded_inputs, dc_constants.TF_DATA_TYPE)

    if self.params.get('condense_transformer_input', False):
      transformer_input = self.transformer_input_condenser(embedded_inputs)
    else:
      transformer_input = embedded_inputs

    return super(EncoderOnlyLearnedValuesTransformer, self).encode(
        transformer_input, attention_bias, training
    )

  def get_config(self):
    config = super().get_config()  # captures 'params' from parent
    # If you have extra fields unique to this subclass, add them here:
    return config

  @classmethod
  def from_config(cls, config):
    return cls(config['params'])

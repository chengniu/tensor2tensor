# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
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
"""Transformer model from "Attention Is All You Need".

The Transformer model consists of an encoder and a decoder. Both are stacks
of self-attention layers followed by feed-forward layers. This model yields
good results on a number of problems, especially in NLP and machine translation.

See "Attention Is All You Need" (https://arxiv.org/abs/1706.03762) for the full
description of the model and the results obtained with its early version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range  # pylint: disable=redefined-builtin

from tensor2tensor.data_generators import librispeech
from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import beam_search
from tensor2tensor.utils import expert_utils
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf

from tensorflow.python.util import nest


@registry.register_model
class TransformerWithContext(t2t_model.T2TModel):
  """Attention net.  See file docstring."""

  def __init__(self, *args, **kwargs):
    super(Transformer, self).__init__(*args, **kwargs)
    self.attention_weights = dict()  # For visualizing attention heads.

  def encode(self, inputs_context, inputs, target_space, hparams, features=None, losses=None):
    """Encode transformer inputs.

    Args:
      inputs_context: contextual input [batch_size, input_length, hidden_dim]
      inputs: Transformer inputs [batch_size, input_length, input_height,
        hidden_dim] which will be flattened along the two spatial dimensions.
      target_space: scalar, target space ID.
      hparams: hyperparameters for model.
      features: optionally pass the entire features dictionary as well.
        This is needed now for "packed" datasets.
      losses: optional list onto which to append extra training losses

    Returns:
      Tuple of:
          encoder_output: Encoder representation.
              [batch_size, input_length, hidden_dim]
          encoder_output_context: Contextual encoder representation
              [batch_size, input_length, hidden_dim]
          encoder_decoder_attention_bias: Bias and mask weights for
              encoder-decoder attention. [batch_size, input_length]
    """
    inputs_context = common_layers.flatten4d3d(inputs_context)

    inputs = common_layers.flatten4d3d(inputs)

    encoder_input_context, self_attention_bias_context, encoder_decoder_attention_bias_context = (
        transformer_prepare_encoder(
            inputs_context, target_space, hparams, features=features))

    encoder_input_context = tf.nn.dropout(encoder_input_context,
                                  1.0 - hparams.layer_prepostprocess_dropout)

    encoder_input, self_attention_bias, encoder_decoder_attention_bias = (
        transformer_prepare_encoder(
            inputs, target_space, hparams, features=features))

    encoder_input = tf.nn.dropout(encoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)

    encoder_output_context_0 = transformer_encoder(
        encoder_input_context,
        self_attention_bias_context,
        hparams,
        hparams,
        nonpadding=features_to_nonpadding(features, "inputs"),
        save_weights_to=self.attention_weights,
        losses=losses)

    encoder_output = transformer_encoder(
        encoder_input,
        self_attention_bias,
        hparams,
        nonpadding=features_to_nonpadding(features, "inputs"),
        save_weights_to=self.attention_weights,
        losses=losses)

    encoder_output_context = transformer_decoder(
        encoder_output,
        encoder_output_context_0,
        encoder_decoder_attention_bias,
        encoder_decoder_attention_bias_context,
        hparams)

    return encoder_output_context, encoder_output, encoder_decoder_attention_bias

  def decode(self,
             decoder_input,
             encoder_output,
             encoder_decoder_attention_bias,
             decoder_self_attention_bias,
             hparams,
             cache=None,
             decode_loop_step=None,
             nonpadding=None,
             losses=None):
    """Decode Transformer outputs from encoder representation.

    Args:
      decoder_input: inputs to bottom of the model.
          [batch_size, decoder_length, hidden_dim]
      encoder_output: Encoder representation.
          [batch_size, input_length, hidden_dim]
      encoder_decoder_attention_bias: Bias and mask weights for
          encoder-decoder attention. [batch_size, input_length]
      decoder_self_attention_bias: Bias and mask weights for decoder
          self-attention. [batch_size, decoder_length]
      hparams: hyperparameters for model.
      cache: dict, containing tensors which are the results of previous
          attentions, used for fast decoding.
      decode_loop_step: An integer, step number of the decoding loop.
          Only used for inference on TPU.
      nonpadding: optional Tensor with shape [batch_size, decoder_length]
      losses: optional list onto which to append extra training losses

    Returns:
      Final decoder representation. [batch_size, decoder_length, hidden_dim]
    """
    decoder_input = tf.nn.dropout(decoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)

    decoder_output = transformer_decoder(
        decoder_input,
        encoder_output,
        decoder_self_attention_bias,
        encoder_decoder_attention_bias,
        hparams,
        cache=cache,
        decode_loop_step=decode_loop_step,
        nonpadding=nonpadding,
        save_weights_to=self.attention_weights,
        losses=losses)

    if (common_layers.is_on_tpu() and
        hparams.mode == tf.estimator.ModeKeys.TRAIN):
      # TPU does not react kindly to extra dimensions.
      # TODO(noam): remove this once TPU is more forgiving of extra dims.
      return decoder_output
    else:
      # Expand since t2t expects 4d tensors.
      return tf.expand_dims(decoder_output, axis=2)

  def cat_and_compress(self, decoder_output_context, decoder_output, decoder_self_attention_bias, hparams):
      """cantenant decoder_output_context and decoder_output_context at the
         last dimenstion, and then compress.

          Args:
            decoder_output_context:  [batch_size, decoder_length, hidden_dim]
            decoder_output:  [batch_size, input_length, hidden_dim]
            decoder_self_attention_bias: Bias and mask weights for decoder
                self-attention. [batch_size, decoder_length]\

          Returns:
            Final decoder representaiton. [batch_size, decoder_length, hidden_dim]
          """
      with tf.variable_scope("cat_and_compress"):
        x = tf.concat([decoder_output_context, decoder_output], 2)
        original_shape = tf.shape(x)
        hidden_dim = original_shape[2] / 2
        pad_remover = None
        if hparams.use_pad_remover:
          pad_remover = expert_utils.PadRemover(common_attention.attention_bias_to_padding(decoder_self_attention_bias))
          x = tf.reshape(x, tf.concat([[-1], tf.shape(x)[2:]], axis=0))
          x = tf.expand_dims(pad_remover.remove(x), axis=0)
        conv_output = common_layers.conv_hidden_relu(x, 1, hidden_dim)
        if pad_remover:
          conv_output = tf.reshape(pad_remover.restore(tf.squeeze(conv_output, axis=0)), original_shape)
          return conv_output

  def body(self, features):
    """Transformer main model_fn.

    Args:
      features: Map of features to the model. Should contain the following:
          "inputs": Transformer inputs [batch_size, input_length, hidden_dim]
          "targets": Target decoder outputs.
              [batch_size, decoder_length, hidden_dim]
          "target_space_id": A scalar int from data_generators.problem.SpaceID.

    Returns:
      Final decoder representation. [batch_size, decoder_length, hidden_dim]
    """
    hparams = self._hparams

    losses = []

    if self.has_input:
      inputs_context = features["inputs_context"]
      inputs = features["inputs"]
      target_space = features["target_space_id"]
      encoder_output_context, encoder_output, encoder_decoder_attention_bias = self.encode(
          inputs_context, inputs, target_space, hparams, features=features, losses=losses)
    else:
        encoder_output_context, encoder_output, encoder_decoder_attention_bias = (None, None, None)

    targets = features["targets"]
    targets_shape = common_layers.shape_list(targets)
    targets = common_layers.flatten4d3d(targets)

    decoder_input, decoder_self_attention_bias = transformer_prepare_decoder(
        targets, hparams, features=features)

    decoder_output_context = self.decode(
        decoder_input_context,
        encoder_output,
        encoder_decoder_attention_bias,
        decoder_self_attention_bias,
        hparams,
        nonpadding=features_to_nonpadding(features, "targets"),
        losses=losses)

    decoder_output = self.decode(
        decoder_input,
        encoder_output,
        encoder_decoder_attention_bias,
        decoder_self_attention_bias,
        hparams,
        nonpadding=features_to_nonpadding(features, "targets"),
        losses=losses)

    decoder_output = self.cat_and_compress(decoder_output_context, decoder_output)

    expected_attentions = features.get("expected_attentions")
    if expected_attentions is not None:
      attention_loss = common_attention.encoder_decoder_attention_loss(
          expected_attentions, self.attention_weights,
          hparams.expected_attention_loss_type,
          hparams.expected_attention_loss_multiplier)
      return decoder_output, {"attention_loss": attention_loss}

    ret = tf.reshape(decoder_output, targets_shape)
    if losses:
      return ret, {"extra_loss": tf.add_n(losses)}
    else:
      return ret

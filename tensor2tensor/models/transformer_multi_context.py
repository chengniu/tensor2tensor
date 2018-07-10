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
class TransformerMultiContext(t2t_model.T2TModel):

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

        inputs = common_layers.flatten4d3d(inputs)
        encoder_input, self_attention_bias, encoder_decoder_attention_bias = (
            transformer_prepare_encoder(
                inputs, target_space, hparams, features=features)
        )


def context_guide_encoder(encoder_input,
                          encoder_input_context,
                          encoder_self_attention_bias,
                          hparams,
                          name='encoder',
                          nonpadding=None,
                          save_weights_to=None,
                          losses=None,
                          make_image_summary=True):
        x = encoder_input
        z = encoder_input_context
        attention_dropout_broadcast_dims = (
            common_layers.comma_separated_string_to_integer_list(
                getattr(hparams, "attention_dropout_boradcast_dims", "")))
        with tf.variable_scope(name):
            if nonpadding is not None:
                padding = 1.0 - nonpadding
            else:
                padding = common_attention.attention_bias_to_padding(
                    encoder_self_attention_bias)
                nonpadding = 1.0 - padding
            if hparams.use_pad_remover and not common_layers.is_on_tpu():
                pad_remover = expert_utils.PadRemover(padding)
            for layer in range(hparams.num_encoder_layers or hparams.num_hidden_layers):
                with tf.variable_scope('layer_%d'% layer ):
                    with tf.variable_scope('self_attention'):
                        # y1 and y2 are two parts for guided self-attention
                        # because the self-attention is iteratively,
                        # the guidance info will be added into self-att.
                        y1 = common_attention.multihead_attention(
                            common_layers.layer_preprocess(x, hparams),
                            None,
                            encoder_self_attention_bias,
                            hparams.attention_key_channels or hparams.hidden_size,
                            hparams.attention_value_channels or hparams.hidden_size,
                            hparams.hidden_size,
                            hparams.num_heads,
                            hparams.attention_dropout,
                            attention_type=hparams.self_attention_type,
                            save_weights_to=save_weights_to,
                            max_relative_position=hparams.max_relative_position,
                            make_image_summary=make_image_summary,
                            dropout_broadcast_dims=attention_dropout_broadcast_dims,
                            max_length=hparams.get("max_length"),
                            vars_3d=hparams.get("attention_variables_3d"))

                        y2 = common_attention.multihead_attention(
                            common_layers.layer_preprocess(x, hparams),
                            common_layers.layer_preprocess(z, hparams),

                        )


        return


def transformer_prepare_encoder(inputs, target_space, hparams, features=None):
    """Prepare one shard of the model for the encoder.

    Args:
      inputs: a Tensor.
      target_space: a Tensor.
      hparams: run hyperparameters
      features: optionally pass the entire features dictionary as well.
        This is needed now for "packed" datasets.

    Returns:
      encoder_input: a Tensor, bottom of encoder stack
      encoder_self_attention_bias: a bias tensor for use in encoder self-attention
      encoder_decoder_attention_bias: a bias tensor for use in encoder-decoder
        attention
    """
    ishape_static = inputs.shape.as_list()
    encoder_input = inputs
    if features and "inputs_segmentation" in features:
        # Packed dataset.  Keep the examples from seeing each other.
        inputs_segmentation = features["inputs_segmentation"]
        inputs_position = features["inputs_position"]
        targets_segmentation = features["targets_segmentation"]
        encoder_self_attention_bias = common_attention.attention_bias_same_segment(
            inputs_segmentation, inputs_segmentation)
        encoder_decoder_attention_bias = (
            common_attention.attention_bias_same_segment(targets_segmentation,
                                                         inputs_segmentation))
    else:
        # Usual case - not a packed dataset.
        encoder_padding = common_attention.embedding_to_padding(encoder_input)
        ignore_padding = common_attention.attention_bias_ignore_padding(
            encoder_padding)
        encoder_self_attention_bias = ignore_padding
        encoder_decoder_attention_bias = ignore_padding
        inputs_position = None
    if hparams.proximity_bias:
        encoder_self_attention_bias += common_attention.attention_bias_proximal(
            common_layers.shape_list(inputs)[1])
    if hparams.get("use_target_space_embedding", True):
        # Append target_space_id embedding to inputs.
        emb_target_space = common_layers.embedding(
            target_space,
            32,
            ishape_static[-1],
            name="target_space_embedding",
            dtype=tf.bfloat16
            if hparams.activation_dtype == "bfloat16" else tf.float32)
        emb_target_space = tf.reshape(emb_target_space, [1, 1, -1])
        encoder_input += emb_target_space
    if hparams.pos == "timing":
        if inputs_position is not None:
            encoder_input = common_attention.add_timing_signal_1d_given_position(
                encoder_input, inputs_position)
        else:
            encoder_input = common_attention.add_timing_signal_1d(encoder_input)
    elif hparams.pos == "emb":
        encoder_input = common_attention.add_positional_embedding(
            encoder_input, hparams.max_length, "inputs_positional_embedding",
            inputs_position)
    if hparams.activation_dtype == "bfloat16":
        encoder_self_attention_bias = tf.cast(encoder_self_attention_bias,
                                              tf.bfloat16)
        encoder_decoder_attention_bias = tf.cast(encoder_decoder_attention_bias,
                                                 tf.bfloat16)
    return (encoder_input, encoder_self_attention_bias,
            encoder_decoder_attention_bias)
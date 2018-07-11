# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
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

"""Data generators for translation data-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Dependency imports

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

import tensorflow as tf

FLAGS = tf.flags.FLAGS

# End-of-sentence marker.
EOS = text_encoder.EOS_ID

# End-of-sentence marker.
EOS = text_encoder.EOS_ID

# This is far from being the real WMT17 task - only toyset here
# you need to register to get UN data and CWT data. Also, by convention,
# this is EN to ZH - use translate_enzh_wmt8k_rev for ZH to EN task
_ZHZH_TRAIN_DATASETS = [
    [
        "http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz",
        ("data/train.en", "data/train.zh", 'data/train.cxt')
    ],
]
_ZHZH_TEST_DATASETS = [
    [
        "http://data.statmt.org/wmt16/translation-task/dev.tgz",
        ("data/dev.en", "data/dev.zh", "data/dev.cxt")
    ],
]


@registry.register_problem
class ZH_to_ZH_With_Context(translate.TranslateProblem):
  """Problem spec for Zh-Zh translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  @property
  def source_vocab_name(self):
    return "tokens.vocab.%d" % self.approx_vocab_size

  @property
  def target_vocab_name(self):
    return "tokens.vocab.%d" % self.approx_vocab_size

  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    return _ZHZH_TRAIN_DATASETS if train else _ZHZH_TEST_DATASETS

  def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
    symbolizer_vocab = generator_utils.get_or_generate_vocab(
        data_dir,
        tmp_dir,
        self.source_vocab_name,
        self.approx_vocab_size,
        _ZHZH_TRAIN_DATASETS,
        file_byte_budget=1e8)
    train = dataset_split == problem.DatasetSplit.TRAIN
    datasets = _ZHZH_TRAIN_DATASETS if train else _ZHZH_TEST_DATASETS
    tag = "train" if train else "dev"
    data_path = translate.compile_data_with_context(tmp_dir, datasets, "mydata_enzh_tok_%s" % tag)
    return text_problems.text2text_generate_encoded(
        text_problems.text2text_cxt_iterator(data_path + ".lang1",
                                             data_path + ".lang2",
                                             data_path + ".cxt"),
        symbolizer_vocab, symbolizer_vocab)

  def feature_encoders(self, data_dir):
    vocab_filename = os.path.join(data_dir, self.source_vocab_name)
    encoder = text_encoder.SubwordTextEncoder(vocab_filename)
    return {
        "inputs": encoder,
        "targets": encoder,
    }
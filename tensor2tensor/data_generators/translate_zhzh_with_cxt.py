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

# In this problem, inputs, targets and
@registry.register_problem
class MydataZhzhTokensCxt_32k(text_problems.QuestionAndContext2TextProblem):
  """Problem spec for Zh-Zh translation."""

  _URL = ''
  _DEV_SET = 'data/train'
  _TRAINING_SET = 'data/dev'
  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    return _ZHZH_TRAIN_DATASETS if train else _ZHZH_TEST_DATASETS

  # the original target feature encoder of qa problem is not appropriate
  def feature_encoders(self, data_dir):
    encoders = super(MydataZhzhTokensCxt_32k, self).feature_encoders(data_dir)
    encoders['targets'] = encoders['inputs']
    return encoders
    
  def is_generate_per_split(self):
    return True

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    url = self._URL
    filename = (self._TRAINING_SET if dataset_split ==
                problem.DatasetSplit.TRAIN else self._DEV_SET)
    ret_dict = self._read_in_files(tmp_dir, filename)
    # here in samples, the context is also text lines
    # todo: Now there's no division between multi-turn of conversation
    return ret_dict
    
  def _read_in_files(self, tmp_dir, filename):

    q_file = os.path.join(tmp_dir, filename + '.en')
    a_file = os.path.join(tmp_dir, filename + '.zh')
    cxt_file = os.path.join(tmp_dir, filename + '.cxt')
    # import pdb
    # pdb.set_trace()
    return text_problems.text2text_cxt_iterator(q_file, a_file, cxt_file)
    
    
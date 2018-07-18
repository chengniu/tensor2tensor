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

"""data generation for chatbot task with context."""

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
		return 2 ** 15  # 32768
	
	def source_data_files(self, dataset_split):
		train = dataset_split == problem.DatasetSplit.TRAIN
		return _ZHZH_TRAIN_DATASETS if train else _ZHZH_TEST_DATASETS
	
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
	
	# In this task, context is a list of chat sentences.
	def generate_text_for_vocab(self, data_dir, tmp_dir):
		for i, sample in enumerate(
			self.generate_samples(data_dir, tmp_dir, problem.DatasetSplit.TRAIN)):
			yield sample["inputs"]
			yield sample["targets"]
			if isinstance(sample['context'], list):
				yield " ".join(sample["context"])
			else:
				yield sample['context']
			
			if self.max_samples_for_vocab and (i+1) >= self.max_samples_for_vocab:
				break

	def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
		generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
		encoder = self.get_or_create_vocab(data_dir, tmp_dir)
		generator = text2text_generate_encoded(generator, encoder,
		                                       has_inputs=self.has_inputs)
		return generator

	def example_reading_spec(self):
		data_fields, data_items_to_decoder = super(MydataZhzhTokensCxt_32k,
		                                           self).example_reading_spec()
		for ix in range(5):
			data_fields["context_{}".format(str(ix))] = tf.VarLenFeature(tf.int64)
		return (data_fields, data_items_to_decoder)
		
		
def text2text_generate_encoded(sample_generator,
                               vocab,
                               targets_vocab=None,
                               has_inputs=True):
	"""Encode Text2Text samples from the generator with the vocab."""
	targets_vocab = targets_vocab or vocab
	
	for sample in sample_generator:
		if has_inputs:
			sample["inputs"] = vocab.encode(sample["inputs"])
			sample["inputs"].append(text_encoder.EOS_ID)
		sample["targets"] = targets_vocab.encode(sample["targets"])
		sample["targets"].append(text_encoder.EOS_ID)
		
		# create lots of context with key: context_{ix}
		for ix, sent in enumerate(sample['context']):
			context = vocab.encode(sent)
			context.append(text_encoder.EOS_ID)
			sample['context_{}'.format(str(ix))] = context
			
		if isinstance(sample['context'], list):
			sample["context"] = vocab.encode(" ".join(sample['context']))
		else:
			sample["context"] = vocab.encode(sample['context'])
		sample['context'].append(text_encoder.EOS_ID)
		yield sample

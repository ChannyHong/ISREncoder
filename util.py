# Copyright 2020 Superb AI, Inc.
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

'''
Authors: Channy Hong, Jaeyeon Lee, Jung Kwon Lee.
Description: Useful functions and class definitions.
'''

import numpy as np
import random
import os



LABEL_MAP = {
  "entailment": 0,
  "neutral": 1,
  "contradiction": 2,
}

language_dict = {
  'English': 'en',
  'French': 'fr',
  'Spanish': 'es',
  'German': 'de',
  'Greek': 'el',
  'Bulgarian': 'bg',
  'Russian': 'ru',
  'Turkish': 'tr',
  'Arabic': 'ar',
  'Vietnamese': 'vi',
  'Thai': 'th',
  'Chinese': 'zh',
  'Hindi': 'hi',
  'Swahili': 'sw',
  'Urdu': 'ur',
}



def parse_languages_into_abbreviation_list(languages):
  return [language_dict[language] for language in languages.split(',')]

def create_language_reference(train_language_abbreviations):
  language_reference = {}
  for i, language_abbreviation in enumerate(train_language_abbreviations):
    language_reference[language_abbreviation] = i
  return language_reference

def convert_to_onehots(num_train_languages, labels):
  onehots = []
  for label in labels:
    label_onehot = [0] * num_train_languages
    label_onehot[label] = 1
    onehots.append(label_onehot)
  return onehots

def create_random_labels(num_train_languages, batch_size):
  labels = []
  for _ in range(batch_size):
    labels.append(random.randrange(num_train_languages))
  return labels

def create_xhat_alphas(batch_size):
  xhat_alphas = []
  for _ in range(batch_size):
    xhat_alpha = random.uniform(0, 1)
    xhat_alphas.append(xhat_alpha)
  return xhat_alphas

def get_mc_minibatch(train_examples, step_num, batch_size, language_reference):
  start_index = (step_num-1)*batch_size
  end_index = step_num*batch_size

  indices = range(start_index, end_index)
  sentences = [train_examples[i].sentence for i in indices]
  languages = [language_reference[train_examples[i].language] for i in indices]
  return sentences, languages

def get_xnli_minibatch(train_examples, step_num, batch_size, language_reference):
  start_index = (step_num-1)*batch_size
  end_index = step_num*batch_size

  indices = range(start_index, end_index)
  premise_vectors = [train_examples[i].sentence1 for i in indices]
  hypothesis_vectors = [train_examples[i].sentence2 for i in indices]
  labels = [train_examples[i].label for i in indices]
  languages = [language_reference[train_examples[i].language] for i in indices]
  return premise_vectors, hypothesis_vectors, labels, languages

def convert_to_singles_from_pairs(train_example_in_pairs):
  train_examples = []
  for train_example_in_pair in train_example_in_pairs:
    train_examples.append(InputSentence(sentence=train_example_in_pair.sentence1, language=train_example_in_pair.language))
    train_examples.append(InputSentence(sentence=train_example_in_pair.sentence2, language=train_example_in_pair.language))
  return train_examples

def get_mc_train_examples(data_dir, train_language_abbreviations):
  train_examples = []

  for language_abbreviation in train_language_abbreviations:

    loaded_examples = np.load(os.path.join(data_dir, "mc_%s.npy" % language_abbreviation), allow_pickle=True)

    for example in loaded_examples:
      train_examples.append(InputSentence(sentence=example, language=language_abbreviation))
  
  return train_examples

def get_xnli_train_examples(data_dir, train_language_abbreviations):
  train_examples = []

  for language_abbreviation in train_language_abbreviations:

    loaded_examples = np.load(os.path.join(data_dir, "bse_%s.npy" % language_abbreviation), allow_pickle=True)

    for example in loaded_examples:
      train_examples.append(InputSentencePair(sentence1=example[0], sentence2=example[1], label=example[2], language=language_abbreviation))
  
  return train_examples

def get_xnli_dev_examples(data_dir, language_abbreviations, in_pairs=True):
  dev_examples = []

  loaded_examples = np.load(os.path.join(data_dir, "DEV.npy"), allow_pickle=True)

  if in_pairs:
    for example in loaded_examples:
      if example[3] in language_abbreviations:
      	dev_examples.append(InputSentencePair(sentence1=example[0], sentence2=example[1], label=example[2], language=example[3]))
  else:
    for example in loaded_examples:
      if example[3] in language_abbreviations:
	    dev_examples.append(InputSentence(sentence=example[0], language=example[3]))
	    dev_examples.append(InputSentence(sentence=example[1], language=example[3]))
  
  return dev_examples

def get_xnli_dev_examples_by_language(data_dir, language_abbreviations):
  dev_examples_by_lang_dict = {}

  dev_example_in_pairs = get_xnli_dev_examples(data_dir, language_abbreviations, True)
  
  for language_abbreviation in language_abbreviations:
    dev_examples_by_lang = []
    for dev_example_in_pair in dev_example_in_pairs:
      if dev_example_in_pair.language == language_abbreviation:
        dev_examples_by_lang.append(dev_example_in_pair)
    dev_examples_by_lang_dict[language_abbreviation] = dev_examples_by_lang

  return dev_examples_by_lang_dict



# A single training/eval/test sentence for simple sequence classification.
class Minibatch(object):
  def __init__(self, examples, num_train_languages, language_reference, with_ISR):
    num_examples = len(examples)

    self.prem_sentences = [example.sentence1 for example in examples]
    self.hyp_sentences = [example.sentence2 for example in examples]

    original_labels = [language_reference[example.language] for example in examples]
    self.original_label_onehots = convert_to_onehots(num_train_languages, original_labels)

    target_labels = create_random_labels(num_train_languages, num_examples)
    self.target_label_onehots = convert_to_onehots(num_train_languages, target_labels)

    self.xhat_alphas = create_xhat_alphas(num_examples)

    self.nli_labels = None
    if with_ISR:
      self.nli_labels = [example.label for example in examples]

# A single training/eval/test sentence.
class InputSentence(object):
  def __init__(self, sentence, language):
    self.sentence = sentence
    self.language = language

# A single training/eval/test sentence pair.
class InputSentencePair(object):
  def __init__(self, sentence1, sentence2, language, label=None):
    self.sentence1 = sentence1
    self.sentence2 = sentence2
    self.label = label
    self.language = language



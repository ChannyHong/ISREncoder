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
Description: Script for caching NLI examples / monolingual corpus into BERT sentence embeddings.
'''

import tensorflow as tf
import numpy as np
import subprocess
import csv
import os
import datetime
import string
from bert_serving.client import BertClient

import util

flags = tf.flags
FLAGS = flags.FLAGS

## I/O parameters
flags.DEFINE_string("data_dir", None, "The input data dir. Should contain the .tsv files (or other data files).")
flags.DEFINE_string("data_type", None, "The input data type, either one of 'mnli', 'dev', 'test', or 'mc'.")
flags.DEFINE_string("language", None, "The language of training data to convert into BSE.")
flags.DEFINE_string("output_dir", None, "The output directory where the cache file and config file will be written to.")

## bert-as-service parameters
flags.DEFINE_string("bert_dir", None, "The directory where BERT model, vocab file, and config files reside.")
flags.DEFINE_bool("do_lower_case", False, "Whether to use cased or uncased tokenization; should be cased tokenization for our use since we used BERT multilingual cased.")
flags.DEFINE_integer( "max_seq_length", 128, "The maximum number of tokens per example (sentence).")



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


# A single training/test example for simple sequence classification.
class InputExample(object):
  def __init__(self, input_a, input_b, label):
    self.input_a = input_a
    self.input_b = input_b
    self.label = label

# Reads a tab separated value file.
def read_tsv(input_file, quotechar=None):
  with tf.gfile.Open(input_file, "r") as f:
    reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
    lines = []
    for line in reader:
      lines.append(line)
    return lines

def get_train_examples_in_bse(data_dir, train_language, bert_service_client):
  raw_premise_list = []
  raw_hypothesis_list = []
  label_list = []

  lines = read_tsv(os.path.join(data_dir, "multinli.train.%s.tsv" % language_dict[train_language]))

  for (i, line) in enumerate(lines):
    if i == 0:
      continue
    raw_premise_list.append(util.convert_to_unicode(line[0]))
    raw_hypothesis_list.append(util.convert_to_unicode(line[1]))
    
    label_raw = util.convert_to_unicode(line[2])
    if label_raw == util.convert_to_unicode("contradictory"):
      label_raw = util.convert_to_unicode("contradiction")

    label_list.append(LABEL_MAP[label_raw])
  
  bse_premise_array = bert_service_client.encode(raw_premise_list)
  bse_hypothesis_array = bert_service_client.encode(raw_hypothesis_list)

  return np.array(zip(bse_premise_array, bse_hypothesis_array, label_list))

def get_devtest_examples_in_bse(data_dir, data_type, bert_service_client):
  raw_premise_list = []
  raw_hypothesis_list = []
  label_list = []
  language_list = []

  lines = read_tsv(os.path.join(data_dir, "xnli.{}.tsv".format(data_type)))

  for (i, line) in enumerate(lines):
    if i == 0:
      continue
    raw_premise_list.append(util.convert_to_unicode(line[6]))
    raw_hypothesis_list.append(util.convert_to_unicode(line[7]))
    
    label_raw = util.convert_to_unicode(line[1])
    if label_raw == util.convert_to_unicode("contradictory"):
      label_raw = util.convert_to_unicode("contradiction")

    label_list.append(LABEL_MAP[label_raw])

    language_list.append(util.convert_to_unicode(line[0]))
  
  bse_premise_array = bert_service_client.encode(raw_premise_list)
  bse_hypothesis_array = bert_service_client.encode(raw_hypothesis_list)

  return np.array(zip(bse_premise_array, bse_hypothesis_array, label_list, language_list))

def get_mc_sentences_in_bse(data_dir, language, bert_service_client):
  # So the entire train_examples is: npdarray: [... [[... 768d ...], [... 768d ...], int_label] ...]

  fr = open(os.path.join(data_dir, "mc_%s.txt" % language_dict[language]), "r")

  raw_sentences = []

  for sentence in fr:
    raw_sentences.append(sentence)
  
  bse_sentences = bert_service_client.encode(raw_sentences)

  return np.array(bse_sentences)



def main():
  os.system("mkdir -p {}".format(os.path.join(FLAGS.output_dir)))

  # Start bert-serving-server; make sure the server supports Python >= 3.5 and Tensorflow >=1.10
  cased_arg = "-cased_tokenization"

  start_bertservice_command = "bert-serving-start -model_dir={} -max_seq_len={} {}".format(FLAGS.bert_dir, FLAGS.max_seq_length, cased_arg)

  bert_server_process = subprocess.Popen(start_bertservice_command.split())

  # Start bert_serving client
  bert_service_client = BertClient()

  # Convert input data into BSE
  if FLAGS.data_type == "mnli":
    cache_filename = os.path.join(FLAGS.output_dir, "bse_{}".format(FLAGS.language))
    examples = get_train_examples_in_bse(FLAGS.data_dir, FLAGS.language, bert_service_client)
    np.save(cache_filename, examples)

  elif FLAGS.data_type == "dev" or FLAGS.data_type == "test":
    cache_filename = os.path.join(FLAGS.output_dir, "{}".format(string.upper(FLAGS.data_type)))
    examples = get_devtest_examples_in_bse(FLAGS.data_dir, FLAGS.data_type, bert_service_client)
    np.save(cache_filename, examples)

  elif FLAGS.data_type == "mc":
    cache_filename = os.path.join(FLAGS.output_dir, "mc_{}".format(FLAGS.language))
    examples = get_mc_sentences_in_bse(FLAGS.data_dir, FLAGS.language, bert_service_client)
    np.save(cache_filename, examples)

  else:
    print("Invalid data type (must be one of 'mnli', 'dev', 'test' or 'mc').")

  # Kill bert_server_process
  bert_server_process.kill()



##############################
##### The Program Driver #####
##############################

if __name__ == "__main__":
  main()


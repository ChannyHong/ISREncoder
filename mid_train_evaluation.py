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
Description: Mid-train evaluation functions for ISR Encoder training.
'''

import tensorflow as tf
import util



#############################################
##### EVALUATE DISCRIMINATOR CLASSIFIER #####
#############################################

def evaluate_Discriminator_classifier(to_eval_examples, language_reference, num_train_languages, sess, original_sentences_tensor, original_label_onehots_tensor, target_label_onehots_tensor, original_sentences_cls_predictions_tensor, generated_sentences_cls_predictions_tensor):

  eval_batch_size = 32

  num_original_sentences_cls_correct = 0
  num_generated_sentences_cls_correct = 0

  num_examples = len(to_eval_examples)
  num_examples_seen = 0

  num_eval_steps = int(num_examples / eval_batch_size) # Omit remainders

  for step_num in range(1, num_eval_steps+1):

    indices = range((step_num-1)*eval_batch_size, step_num*eval_batch_size)

    to_eval_original_sentences = [to_eval_examples[i].sentence for i in indices]
    to_eval_original_labels = [language_reference[to_eval_examples[i].language] for i in indices]
    to_eval_original_label_onehots = util.convert_to_onehots(num_train_languages, to_eval_original_labels)
    to_eval_target_labels = util.create_random_labels(num_train_languages, eval_batch_size)
    to_eval_target_label_onehots = util.convert_to_onehots(num_train_languages, to_eval_target_labels)

    to_eval_feed_dict = {original_sentences_tensor: to_eval_original_sentences, original_label_onehots_tensor: to_eval_original_label_onehots, target_label_onehots_tensor: to_eval_target_label_onehots}

    original_sentences_cls_predictions, generated_sentences_cls_predictions = sess.run([original_sentences_cls_predictions_tensor, generated_sentences_cls_predictions_tensor], feed_dict=to_eval_feed_dict)

    for i in range(eval_batch_size):
      if to_eval_original_labels[i] == original_sentences_cls_predictions[i]:
        num_original_sentences_cls_correct += 1 
      if to_eval_target_labels[i] == generated_sentences_cls_predictions[i]:
        num_generated_sentences_cls_correct += 1 

    num_examples_seen += eval_batch_size
  
  original_sentences_cls_accuracy = float(num_original_sentences_cls_correct) / float(num_examples_seen)
  generated_sentences_cls_accuracy = float(num_generated_sentences_cls_correct) / float(num_examples_seen)

  return original_sentences_cls_accuracy, generated_sentences_cls_accuracy



####################################
##### EVALUATE NLI TASK SOLVER #####
####################################

def evaluate_nli_task_solver(to_eval_examples, train_language_abbreviations, language_reference, num_train_languages, mid_train_eval_nli_target_language_abbreviation, original_sentences_tensor, original_label_onehots_tensor, target_label_onehots_tensor, sess, output_sentences_tensor, mid_train_eval_nli_model_path):

  # language_reference = {'en': 0, 'de': 1, ...}
  eval_batch_size = 32
  evaluation_results = {}

  # Set up NLI solver graph
  model_path = mid_train_eval_nli_model_path

  classifier_graph = tf.Graph()
  nli_sess = tf.Session(graph=classifier_graph)

  with nli_sess.as_default():
    with classifier_graph.as_default(): 

      imported_graph = tf.train.import_meta_graph("{}.meta".format(model_path))
      imported_graph.restore(nli_sess, model_path)

  premise_x = classifier_graph.get_tensor_by_name("premise_x:0")
  hypothesis_x = classifier_graph.get_tensor_by_name("hypothesis_x:0")
  predictions_tensor = classifier_graph.get_tensor_by_name("predictions_tensor:0")

  # Iterate over each language used in training ISR
  for current_language_abbreviation in train_language_abbreviations:

    to_eval_examples_by_language = []
    for to_eval_example in to_eval_examples:
      if to_eval_example.language == current_language_abbreviation:
        to_eval_examples_by_language.append(to_eval_example)

    num_correct = 0
    num_examples = len(to_eval_examples_by_language)
    num_examples_seen = 0

    num_eval_steps = int(num_examples / eval_batch_size) # Omit remainders

    for step_num in range(1, num_eval_steps+1):

      indices = range((step_num-1)*eval_batch_size, step_num*eval_batch_size)

      to_eval_original_premises = [to_eval_examples_by_language[i].sentence1 for i in indices]
      to_eval_original_hypotheses = [to_eval_examples_by_language[i].sentence2 for i in indices]
      to_eval_original_nli_labels = [to_eval_examples_by_language[i].label for i in indices]
      to_eval_original_language_labels = [language_reference[to_eval_examples_by_language[i].language] for i in indices]
      to_eval_original_language_label_onehots = util.convert_to_onehots(num_train_languages, to_eval_original_language_labels)

      to_eval_target_label = [language_reference[mid_train_eval_nli_target_language_abbreviation]] * eval_batch_size
      to_eval_target_label_onehots = util.convert_to_onehots(num_train_languages, to_eval_target_label)

      to_eval_premise_feed_dict = {original_sentences_tensor: to_eval_original_premises, original_label_onehots_tensor: to_eval_original_language_label_onehots, target_label_onehots_tensor: to_eval_target_label_onehots}
      to_eval_hypotheses_feed_dict = {original_sentences_tensor: to_eval_original_hypotheses, original_label_onehots_tensor: to_eval_original_language_label_onehots, target_label_onehots_tensor: to_eval_target_label_onehots}

      output_premises = sess.run(output_sentences_tensor, feed_dict=to_eval_premise_feed_dict)

      output_hypotheses = sess.run(output_sentences_tensor, feed_dict=to_eval_hypotheses_feed_dict)
    
      to_eval_feed_dict = {premise_x: output_premises, hypothesis_x: output_hypotheses}

      predictions = nli_sess.run(predictions_tensor, feed_dict=to_eval_feed_dict)

      for i in range(eval_batch_size):
        if to_eval_original_nli_labels[i] == predictions[i]:
          num_correct += 1

      num_examples_seen += eval_batch_size

    evaluation_result_by_language = float(num_correct) / float(num_examples_seen)
    evaluation_results[current_language_abbreviation] = evaluation_result_by_language

  nli_sess.close()

  return evaluation_results
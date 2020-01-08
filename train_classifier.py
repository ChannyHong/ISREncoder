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
Description: The NLI classifier training script (on top of fixed ISR Encoder).
'''

import tensorflow as tf
import random
import json
import os

import util
import classifier_model

flags = tf.flags
FLAGS = flags.FLAGS

# Input parameters
flags.DEFINE_string("data_dir", None, "The directory to the XNLI datasets and cache files. Should contain the .tsv files for both training and evaluation examples (and also the BSE cache files, if applicable).")
flags.DEFINE_string("isr_encoder_dir", None, "The pathway to folder containing ISR models.")
flags.DEFINE_string("isr_encoder_name", None, "The name of exact model to run classifier training with.")
flags.DEFINE_string("output_dir", None, "The output directory where the classifier models' checkpoints will be written.")

## Specification
flags.DEFINE_string("xnli_train_languages", "English", "The language(s) in which XNLI training will be performed on.")
flags.DEFINE_integer("embedding_size", 768, "The dimension of our BERT-M sentence embedding and ISR.")

# Hyperparameters
flags.DEFINE_integer("train_batch_size", 32, "The batch size for training the Generator.")
flags.DEFINE_float("dropout_rate", 0.2, "The dropout rate for post MLP layer.")
flags.DEFINE_float("learning_rate", 0.00001, "The initial learning rate for Adam.")
flags.DEFINE_float("beta1", 0.5, "The beta1 value for Adam.")
flags.DEFINE_float("beta2", 0.999, "The beta2 value for Adam.")

# Training duration parameters
flags.DEFINE_integer("num_train_epochs", 3, "Total number of training epochs to perform.")
flags.DEFINE_integer("save_checkpoints_steps", 5000, "How often (in steps) to save the model checkpoint.")

# Logging parameters
flags.DEFINE_bool("log_losses", True, "Toggle to log loss summaries using Tensorboard.")
flags.DEFINE_bool("do_mid_train_eval", True, "Whether to run evaluation mid training (and after training).")

# Mid train evaluation parameters
flags.DEFINE_string("mid_train_xnli_eval_languages", "English", "The languages in which XNLI evaluation will be run on.")
flags.DEFINE_integer("run_mid_train_eval_steps", 2000, "How often (in steps) to run evaluations.")
flags.DEFINE_integer("mid_train_eval_batch_size", 32, "The eval batch size for our mid training evaluation.")



def evaluate_model(sess, isr_sess, original_sentences_tensor, original_label_onehots_tensor,  isr_sentences_tensor, premise_x, hypothesis_x, predictions_tensor, examples_to_eval, language_reference):
  num_examples = len(examples_to_eval)
  num_examples_seen = 0
  num_correct = 0

  num_eval_steps = int(num_examples / FLAGS.mid_train_eval_batch_size) # Omit remainders

  for step_num in range(1, num_eval_steps+1):

    minibatch_bse_premise_vectors = None
    minibatch_bse_hypothesis_vectors = None
    minibatch_labels = None
    minibatch_languages = None

    minibatch_bse_premise_vectors, minibatch_bse_hypothesis_vectors, minibatch_labels, minibatch_languages = util.get_xnli_minibatch(examples_to_eval, step_num, FLAGS.mid_train_eval_batch_size, language_reference)

    minibatch_language_onehots = util.convert_to_onehots(len(language_reference), minibatch_languages)

    #### GET ISR SENTENCES #####

    # feed_dicts
    get_premise_isr_feed_dict = None
    get_hypothesis_isr_feed_dict = None

    get_premise_isr_feed_dict = {original_sentences_tensor: minibatch_bse_premise_vectors, original_label_onehots_tensor: minibatch_language_onehots}
    get_hypothesis_isr_feed_dict = {original_sentences_tensor: minibatch_bse_hypothesis_vectors, original_label_onehots_tensor: minibatch_language_onehots} 


    # forward pass through Generator's Encoder
    minibatch_isr_premise_sentences = isr_sess.run(isr_sentences_tensor, feed_dict=get_premise_isr_feed_dict)
    minibatch_isr_hypothesis_sentences = isr_sess.run(isr_sentences_tensor, feed_dict=get_hypothesis_isr_feed_dict)




    feed_dict = {premise_x: minibatch_isr_premise_sentences, hypothesis_x: minibatch_isr_hypothesis_sentences}

    predictions = sess.run(predictions_tensor, feed_dict=feed_dict)



    for i in range(eval_batch_size):
      if minibatch_labels[i] == predictions[i]:
        num_correct += 1

    num_examples_seen += eval_batch_size

  return float(num_correct) / float(num_examples_seen)







def main():


  ##############################################################
  #                                                             
  #                FETCH TRAIN AND DEV EXAMPLES
  #                                                             
  ##############################################################

  # Get language_reference; the languages used in training ISR, which decides the original_label_onehots_tensor shape that Encoder will expect
  language_reference_file = open(os.path.join(FLAGS.isr_encoder_dir, "language_reference.json"), 'r')
  language_reference = json.load(language_reference_file)

  # Get train examples; either from raw_dataset or from bse_caches
  train_language_abbreviations = util.parse_languages_into_abbreviation_list(FLAGS.xnli_train_languages)
  train_examples = util.get_xnli_train_examples(FLAGS.data_dir, train_language_abbreviations)

  random.shuffle(train_examples)

  # Get dev examples; either from raw_dataset or from bse_caches
  eval_language_abbreviations = util.parse_languages_into_abbreviation_list(FLAGS.mid_train_xnli_eval_languages)
  dev_examples_by_lang_dict = util.get_xnli_dev_examples_by_language(FLAGS.data_dir, eval_language_abbreviations)



  ##############################################################
  #                                                             
  #                      LOAD ISR ENCODER
  #                                                             
  ##############################################################


  # Set up graph to extract ISR sentences
  isr_encoder_graph = tf.Graph()
  isr_sess = tf.Session(graph=isr_encoder_graph)

  with isr_sess.as_default():
    with isr_encoder_graph.as_default(): 
      isr_model = "{}/{}".format(FLAGS.isr_encoder_dir, FLAGS.isr_encoder_name)
      imported_graph = tf.train.import_meta_graph("{}.meta".format(isr_model))
      imported_graph.restore(isr_sess, isr_model)

  # Placeholder tensors to pass in real data at each training step
  original_sentences_tensor = isr_encoder_graph.get_tensor_by_name("original_sentences_tensor:0")
  original_label_onehots_tensor = isr_encoder_graph.get_tensor_by_name("original_label_onehots_tensor:0")

  ### HERE IT IS LADIES AND GENTLEMEN! ###
  isr_sentences_tensor = isr_encoder_graph.get_tensor_by_name("forward_isr_sentences_tensor:0")



  ##############################################################
  #                                                             
  #                   BUILD XNLI CLASSIFIER
  #                                                             
  ##############################################################

  ##### PREPROCESS INPUT SENTENCES #####

  premise_x = tf.placeholder(tf.float32, [None, FLAGS.embedding_size]) 
  hypothesis_x = tf.placeholder(tf.float32, [None, FLAGS.embedding_size])
  y = tf.placeholder(tf.int32, [None])
  keep_rate = 1.0 - FLAGS.dropout_rate
  input_layer = tf.concat([premise_x, hypothesis_x], 1)

  classifier = classifier_model.Classifier(input_layer.shape[-1].value, 3) # 3 for one of either entailment/contradiction/neutral

  logits, predictions_tensor = classifier(input_layer, keep_rate)

  ##### LOSS AND OPTIMIZER #####

  total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))

  cls_vars = [cls_var for cls_var in tf.trainable_variables() if cls_var.name.startswith('cls')]

  cls_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta1=FLAGS.beta1, beta2=FLAGS.beta2).minimize(total_loss, var_list=cls_vars)


  ### LOSS TENSORBOARD ###
  tf.summary.scalar('LOSS 1: classifier_loss', total_loss, collections=['loss'])

  ### EVALUATION ACCURACY TENSORBOARD ###

  # DISCRIMINIATOR CLASSIFIFER #

  train_accuracy_tensor = tf.Variable(0.)

  tf.summary.scalar("EVAL TRAIN: train_accuracy_tensor", train_accuracy_tensor, collections=['eval_accuracy'])

  dev_accuracy_tensors_dict = {}
  for eval_language_abbreviation in eval_language_abbreviations:
    dev_accuracy_by_lang_tensor = tf.Variable(0.)

    dev_accuracy_tensors_dict[eval_language_abbreviation] = dev_accuracy_by_lang_tensor
    tf.summary.scalar("EVAL DEV: {} eval_accuracy".format(eval_language_abbreviation), dev_accuracy_by_lang_tensor, collections=['eval_accuracy'])



  ##### SESSION CONFIGURATION #####
  merged_loss_summary = tf.summary.merge_all(key='loss')
  merged_eval_accuracy_summaries = tf.summary.merge_all(key='eval_accuracy')
  sess = tf.InteractiveSession() ## make this worth it by adding summaries...
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver()

  os.system("mkdir -p {}".format(os.path.join(FLAGS.output_dir, "logs")))
  train_writer = tf.summary.FileWriter(os.path.join(FLAGS.output_dir, "logs"), sess.graph)


  ##############################################################
  #                                                             
  #                          TRAINING
  #                                                             
  ##############################################################

  # The remainder train examples from not cleanly divisible batch size will be omitted from training
  global_step = 0
  num_train_steps_per_epoch = int(len(train_examples) / FLAGS.train_batch_size)

  for epoch_num in range(1, FLAGS.num_train_epochs+1):

    for step_num in range(1, num_train_steps_per_epoch+1):


      ##############################################################
      #                  Process Training Minibatch
      ##############################################################

      minibatch_bse_premise_vectors, minibatch_bse_hypothesis_vectors, minibatch_labels, minibatch_languages = get_xnli_minibatch(train_examples, step_num, FLAGS.train_batch_size, language_reference)

      ##
      minibatch_language_onehots = util.convert_to_onehots(len(language_reference), minibatch_languages)
      ##


      ##############################################################
      #                      Get ISR Sentences
      ##############################################################

      #### GET ISR SENTENCES #####

      get_premise_isr_feed_dict = {original_sentences_tensor: minibatch_bse_premise_vectors, original_label_onehots_tensor: minibatch_language_onehots}
      get_hypothesis_isr_feed_dict = {original_sentences_tensor: minibatch_bse_hypothesis_vectors, original_label_onehots_tensor: minibatch_language_onehots} 



      # forward pass through Generator's Encoder
      minibatch_isr_premise_sentences = isr_sess.run(isr_sentences_tensor, feed_dict=get_premise_isr_feed_dict)
      minibatch_isr_hypothesis_sentences = isr_sess.run(isr_sentences_tensor, feed_dict=get_hypothesis_isr_feed_dict)

      
      ##############################################################
      #             Perform Gradient Descent On Classifier
      ##############################################################

      ##### RUN TRAINING WITH ISR SENTENCES #####
      cls_feed_dict = {premise_x: minibatch_isr_premise_sentences, hypothesis_x: minibatch_isr_hypothesis_sentences, y: minibatch_labels}

      _, current_loss, loss_summary, input_layer_val = sess.run([cls_optimizer, total_loss, merged_loss_summary, input_layer], feed_dict=cls_feed_dict)

      # Update loss information to Tensorboard
      train_writer.add_summary(loss_summary, global_step=global_step)


      ##############################################################
      #                      Save Checkpoints
      ##############################################################

      # Save checkpoint at every save_checkpoint_steps or at end of epoch
      if (step_num % FLAGS.save_checkpoints_steps == 0) or (step_num == num_train_steps_per_epoch):
        saved_path = saver.save(sess, os.path.join(FLAGS.output_dir, "classifier-{}-{}".format(epoch_num, step_num)))

        
      ##############################################################
      #                  Run Mid Train Evaluation
      ##############################################################

      if FLAGS.do_mid_train_eval:
        # Run evaluation at every run_mid_train_eval_steps or at very beginning of training or at end of epoch
        if (step_num % FLAGS.run_mid_train_eval_steps == 0) or ((epoch_num == 1) and (step_num == 1)) or (step_num == num_train_steps_per_epoch):

          # --------------------------------------------- #
          ###### Run Evaluation On Training Examples ######
          # --------------------------------------------- #
          train_examples_to_eval = None
          if epoch_num == 1: 
            train_examples_to_eval = train_examples[:step_num*FLAGS.train_batch_size] # Only evaluate up to data trained on
          else:
            train_examples_to_eval = train_examples

          train_accuracy = evaluate_model(sess, isr_sess, original_sentences_tensor, original_label_onehots_tensor, isr_sentences_tensor, premise_x, hypothesis_x, predictions_tensor, train_examples_to_eval, language_reference)

          # Update Tensorboard summary tensor
          sess.run(train_accuracy_tensor.assign(train_accuracy))

          # ---------------------------------------------------------- #
          ###### Run Evaluation On Dev Examples For Each Language ######
          # ---------------------------------------------------------- #
          for eval_language_abbreviation in eval_language_abbreviations:
            # Run evaluation on dev set
            dev_examples_to_eval = dev_examples_by_lang_dict[eval_language_abbreviation] 
            
            dev_accuracy_by_lang = evaluate_model(sess, isr_sess, original_sentences_tensor, original_label_onehots_tensor, isr_sentences_tensor, premise_x, hypothesis_x, predictions_tensor, dev_examples_to_eval, language_reference)

            # Update Tensorboard summary tensor for each eval language
            sess.run(dev_accuracy_tensors_dict[eval_language_abbreviation].assign(dev_accuracy_by_lang))


          # ------------------------------------------ #
          ###### Update Summary With Eval Results ######
          # ------------------------------------------ #
          eval_accuracy_summaries = sess.run(merged_eval_accuracy_summaries)
          train_writer.add_summary(eval_accuracy_summaries, global_step=global_step)

      # Increment global_step every training step
      global_step += 1

  ##############################################################
  #                                                             
  #                         EXIT PROGRAM
  #                                                             
  ##############################################################

  # Close main tensorflow session and isr_sess
  isr_sess.close()
  sess.close()





##############################################################
#                   The Program Driver
##############################################################

if __name__ == "__main__":
  main()









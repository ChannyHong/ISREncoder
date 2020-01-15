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
Description: The main ISR Encoder training script.
'''

import tensorflow as tf
import numpy as np
import subprocess
import csv
import os
import random
import math
import json

import util
import gan_models
import mid_train_evaluation

flags = tf.flags
FLAGS = flags.FLAGS

## Input parameters
flags.DEFINE_string("data_dir", None, "The input data dir. Should contain the .tsv files (or other data files).")
flags.DEFINE_string("output_dir", None, "The output directory where the models' checkpoints will be written.")

## Specification
flags.DEFINE_string("train_languages", None, "The languages to train the model in ',' separated form (i.e. 'train_laugages=English,Chinese,Spanish,German,Arabic,Urdu'); used to decide the length of class vector fed to Encoder and Decoder.")
flags.DEFINE_integer("embedding_size", 768, "The dimension of our BERT-M sentence embedding and ISR.")

## Hyperparameters
flags.DEFINE_integer("train_batch_size", 32, "The batch size for training the Generator.")
flags.DEFINE_integer("Dis_Gen_train_ratio", 5, "The number of steps to train the Discriminator per step of training for Generator.")
flags.DEFINE_float("Dis_learning_rate", 5e-5, "The initial learning rate for the Discriminator (for Adam).")
flags.DEFINE_float("Gen_learning_rate", 5e-5, "The initial learning rate for the Generator (for Adam).")
flags.DEFINE_float("lambda_Dis_cls", 1., "The weight value of classification loss within total Discriminator loss.")
flags.DEFINE_float("lambda_Dis_gp", 1., "The weight value of gradient penalty within total Discriminator loss.")
flags.DEFINE_float("lambda_Gen_cls", 10., "The weight value of classification loss within total Generator loss.")
flags.DEFINE_float("lambda_Gen_rec", 1., "The weight value of reconstruction loss within total Generator loss.")
flags.DEFINE_float("lambda_Gen_isr", 1., "The weight value of ISR loss within total Generator loss.")
flags.DEFINE_float("beta1", 0.5, "The beta1 value for Adam.")
flags.DEFINE_float("beta2", 0.999, "The beta2 value for Adam.")

# Training duration parameters
flags.DEFINE_integer("num_train_epochs", 3, "Total number of training epochs to perform.")
flags.DEFINE_integer("save_checkpoints_steps", 5000, "How often (in steps) to save the model checkpoint.")

# Logging parameters
flags.DEFINE_bool("log_losses", True, "Toggle to log loss summaries using Tensorboard.")
flags.DEFINE_bool("do_mid_train_eval", False, "Whether to run evaluation mid training (and after training) and log evaluation summaries using Tensorboard.")

# Mid train evaluation parameters
flags.DEFINE_integer("run_mid_train_eval_steps", 2000, "How often (in steps) to run evaluations.")
flags.DEFINE_string("mid_train_eval_nli_target_language", "English", "The language into which premises and hypotheses will be decoded into before NLI task of the mid train evaluation.")
flags.DEFINE_string("mid_train_eval_nli_model_path", None, "The pathway to model trained with specified mid_train_eval_nli_target_language.")



def batch_mean(tensor):
  # input tensor is of shape=(?,1) -> [[1.013], [2.231], ...] <- coming from straight up src's
  # OR of shape=(?,) -> [0.221, 0.312, 0.125, 0.123 ....] <- coming from classification_measure & gradient_penalty_measure
  return tf.reduce_mean(tensor, axis=None) # if axis=0, the reduced mean's of src inputs will have a single [] around it... (the actual numerical value is the same as when axis=None)
  # returned tensor is of shape=() -> 0.323

def gradient_penalty_measure(xhat_sentences_input_tensor, xhat_output_tensor):
  xhat_gradient = tf.gradients(xhat_output_tensor, xhat_sentences_input_tensor)[0]
  xhat_gradient_norm = tf.norm(tf.layers.flatten(xhat_gradient), axis=1) # L2 norm
  gradient_penalty = tf.square(xhat_gradient_norm - 1.)
  return gradient_penalty
  # returned tensor is of shape=(?,) -> [0.221, 0.312, 0.125, 0.123 ....]

def classification_measure(output_tensor, correct_label_onehots_tensor):
  return tf.nn.softmax_cross_entropy_with_logits_v2(labels=correct_label_onehots_tensor, logits=output_tensor)
  # returned tensor is of shape=(?,) -> [0.221, 0.312, 0.125, 0.123 ....]

def difference_measure(measure_type, sentences_tensor_1, sentences_tensor_2):
  return tf.reduce_sum(tf.square(sentences_tensor_1-sentences_tensor_2), axis=1)



def main():

  # Parse training languages.
  train_language_abbreviations = util.parse_languages_into_abbreviation_list(FLAGS.train_languages)
  num_train_languages = len(train_language_abbreviations)

  # Save language reference into a json file for future reference: language abbreviation as key and 0-indexed number as value (i.e. {'en': 0, 'es': 1}).
  language_reference = util.create_language_reference(train_language_abbreviations) 
  language_reference_file = open(os.path.join(FLAGS.output_dir, "language_reference.json"), 'w')
  json.dump(language_reference, language_reference_file)

  ##############################
  ##### GET TRAIN EXAMPLES #####
  ##############################
  train_examples = util.get_mc_train_examples(FLAGS.data_dir, train_language_abbreviations)

  random.shuffle(train_examples)

  # The remainder train examples from not cleanly divisible batch size will be omitted from training
  num_train_steps_per_epoch = int(len(train_examples) / FLAGS.train_batch_size)

  ############################
  ##### GET DEV EXAMPLES #####
  ############################
  if FLAGS.do_mid_train_eval:
    dev_examples = util.get_xnli_dev_examples(FLAGS.data_dir, in_pairs=False)
    dev_example_in_pairs = util.get_xnli_dev_examples(FLAGS.data_dir, in_pairs=True)

  ###############################
  ##### PLACEHOLDER TENSORS #####
  ###############################

  # Placeholder tensors to pass in real data at each training step
  original_sentences_tensor = tf.placeholder(tf.float32, [None, FLAGS.bse_size], name="original_sentences_tensor")
  original_label_onehots_tensor = tf.placeholder(tf.float32, [None, num_train_languages], name="original_label_onehots_tensor")
  target_label_onehots_tensor = tf.placeholder(tf.float32, [None, num_train_languages])
  xhat_alphas_tensor = tf.placeholder(tf.float32, [None])
  xhat_alphas_reshaped = tf.reshape(xhat_alphas_tensor, [-1,1]) # reshape for broadcasting scalar multiplier xhat_alphas

  ###################################################
  ##### INITIALIZE GENERATOR AND DISCRIMINATORS #####
  ###################################################
  Gen = gan_models.Generator(train_isr=True, embedding_size=FLAGS.embedding_size, num_train_languages=num_train_languages)
  Dis = gan_models.Discriminator(embedding_size=FLAGS.embedding_size, num_train_languages=num_train_languages)

  ###########################################################
  ##### RUN TENSORS THROUGH GENERATOR AND DISCRIMINATOR #####
  ###########################################################

  # Generator forward direction 
  isr_sentences_tensor, generated_sentences_tensor = Gen("forward", original_sentences_tensor, original_label_onehots_tensor, target_label_onehots_tensor)

  # Generator backward direction 
  backward_isr_sentences_tensor, reconstructed_sentences_tensor = Gen("backward", generated_sentences_tensor, target_label_onehots_tensor, original_label_onehots_tensor)

  # Discriminator on original sentences
  original_sentences_cls, original_sentences_src, original_sentences_cls_predictions_tensor = Dis(original_sentences_tensor)

  # Discriminator on generated sentences
  generated_sentences_cls, generated_sentences_src, generated_sentences_cls_predictions_tensor = Dis(generated_sentences_tensor)

  # Discriminator on xhat sentences for gradient penalty
  xhat_sentences_tensor = (xhat_alphas_reshaped * original_sentences_tensor) + ((1.-xhat_alphas_reshaped) * generated_sentences_tensor)
  _, xhat_sentences_src, _ = Dis(xhat_sentences_tensor)



  ##############################################################
  #                                                             
  #                 DISCRIMINATOR LOSS FUNCTION
  #                                                             
  ##############################################################

  # DiscriminatorSrc's Adversarial Loss : loss_DisSrc_adv
  # The negative of the entire WGAN formula, which is the likelihood of DiscriminatorSrc classifying real sentences (BSE) as 'real' MINUS likelihood of DiscriminatorSrc classifying generated sentences as 'real' MINUS the gradient penalty.

  # DiscriminatorCls's Classification Loss : loss_real_cls
  # The negative log of the likelihood of DiscriminatorCls classifying real sentences (BSE) to their correct class.

  # Discriminator's Total Loss : loss_Dis_total
  # Sum of loss_DisSrc_adv and loss_real_cls with their respective weights (lambda).

  ##### LOSSES AND OPTIMIZER #####

  loss_DisSrc_real = -batch_mean(original_sentences_src)
  loss_DisSrc_generated = batch_mean(generated_sentences_src)
  loss_DisSrc_gp = batch_mean(gradient_penalty_measure(xhat_sentences_tensor, xhat_sentences_src))
  lambda_Dis_gp = FLAGS.lambda_Dis_gp

  loss_real_cls = batch_mean(classification_measure(original_sentences_cls, original_label_onehots_tensor))
  lambda_Dis_cls = FLAGS.lambda_Dis_cls

  loss_Dis_total = (loss_DisSrc_real + loss_DisSrc_generated + (lambda_Dis_gp * loss_DisSrc_gp)) + (lambda_Dis_cls * loss_real_cls)

  Dis_vars = [Dis_var for Dis_var in tf.trainable_variables() if Dis_var.name.startswith('Dis')]

  Dis_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.Dis_learning_rate, beta1=FLAGS.beta1, beta2=FLAGS.beta2).minimize(loss_Dis_total, var_list=Dis_vars)

  ### TENSORBOARD ###
  if FLAGS.log_losses:
    tf.summary.scalar('LOSS 1: loss_Dis_total', loss_Dis_total, collections=['loss'])
    tf.summary.scalar('LOSS 2: loss_DisSrc_real', loss_DisSrc_real, collections=['loss'])
    tf.summary.scalar('LOSS 3: loss_DisSrc_generated', loss_DisSrc_generated, collections=['loss'])
    tf.summary.scalar('LOSS 4: loss_DisSrc_gp', loss_DisSrc_gp, collections=['loss'])
    tf.summary.scalar('LOSS 5: loss_real_cls', loss_real_cls, collections=['loss'])



  ##############################################################
  #                                                             
  #                   GENERATOR LOSS FUNCTION
  #                                                             
  ##############################################################

  # Generator's Adversarial Loss : loss_Gen_adv
  # The negative of the likelihoood of DiscriminatorSrc classifying generated sentences as 'real'.

  # Generator's Classification Loss : loss_gen_cls
  # The negative log of the likelihood of DiscriminatorCls classifying generated sentences to their correct class.

  # Generator's Reconstruction Loss : loss_Gen_rec
  # The difference measure between original sentence and reconstructed (two-fold generated) sentence.

  # Generator's ISR Loss : loss_Gen_isr
  # The difference measure between forward ISR (original sentence passed through the Encoder) and backward ISR (generated sentence passed through the encoder).

  # Generator's Total Loss : loss_Gen_total
  # Sum of loss_Gen_adv and loss_real_cls with their respective weights (lambda).

  ##### LOSSES AND OPTIMIZER #####
  
  loss_Gen_adv = -batch_mean(generated_sentences_src)
  
  loss_gen_cls = batch_mean(classification_measure(generated_sentences_cls, target_label_onehots_tensor))
  lambda_Gen_cls = FLAGS.lambda_Gen_cls

  loss_Gen_rec = batch_mean(difference_measure(FLAGS.difference_measure, original_sentences_tensor, reconstructed_sentences_tensor)) # ADD batch_mean LATER ON
  lambda_Gen_rec = FLAGS.lambda_Gen_rec

  loss_Gen_isr = batch_mean(difference_measure(FLAGS.difference_measure, isr_sentences_tensor, backward_isr_sentences_tensor))
  lambda_Gen_isr = FLAGS.lambda_Gen_isr

  loss_Gen_total = (loss_Gen_adv) + (lambda_Gen_cls * loss_gen_cls) + (lambda_Gen_rec * loss_Gen_rec) + (lambda_Gen_isr * loss_Gen_isr)
  
  Gen_vars = [Gen_var for Gen_var in tf.trainable_variables() if Gen_var.name.startswith('Gen')]

  Gen_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.Gen_learning_rate, beta1=FLAGS.beta1, beta2=FLAGS.beta2).minimize(loss_Gen_total, var_list=Gen_vars)

  ### TENSORBOARD ###
  if FLAGS.log_losses:
    tf.summary.scalar('LOSS 6: loss_Gen_total', loss_Gen_total, collections=['loss'])
    tf.summary.scalar('LOSS 7: loss_Gen_adv', loss_Gen_adv, collections=['loss'])
    tf.summary.scalar('LOSS 8: loss_gen_cls', loss_gen_cls, collections=['loss'])
    tf.summary.scalar('LOSS 9: loss_Gen_rec', loss_Gen_rec, collections=['loss'])
    tf.summary.scalar('LOSS 10: loss_Gen_isr', loss_Gen_isr, collections=['loss'])



  ################################################
  ### EVALUATION ACCURACY TENSORBOARD ###

  # DISCRIMINIATOR CLASSIFIFER #

  original_sentences_cls_accuracy_tensor = tf.Variable(0.)
  generated_sentences_cls_accuracy_tensor = tf.Variable(0.)

  tf.summary.scalar("EVAL 1: original_sentences_cls_accuracy_tensor", original_sentences_cls_accuracy_tensor, collections=['eval_accuracy'])
  tf.summary.scalar("EVAL 2: generated_sentences_cls_accuracy_tensor", generated_sentences_cls_accuracy_tensor, collections=['eval_accuracy'])

  # NLI TASK SOLVERS #

  mid_train_eval_nli_target_language_abbreviation = util.language_dict[FLAGS.mid_train_eval_nli_target_language]

  # GENERATED SENTENCES 
  generated_nli_task_eval_accuracy_dict = {}
  for train_language_abbreviation in train_language_abbreviations:
    generated_nli_task_eval_accuracy = tf.Variable(0.)
    generated_nli_task_eval_accuracy_dict[train_language_abbreviation] = generated_nli_task_eval_accuracy
    tf.summary.scalar("EVAL NLI: nli_task_accuracy_generated_{}-{}".format(train_language_abbreviation, mid_train_eval_nli_target_language_abbreviation), generated_nli_task_eval_accuracy, collections=['eval_accuracy'])

  # RECONSTRUCTED SENTENCES 
  reconstructed_nli_task_eval_accuracy_dict = {}
  for train_language_abbreviation in train_language_abbreviations:
    reconstructed_nli_task_eval_accuracy = tf.Variable(0.)
    reconstructed_nli_task_eval_accuracy_dict[train_language_abbreviation] = reconstructed_nli_task_eval_accuracy
    tf.summary.scalar("EVAL NLI: nli_task_accuracy_reconstructed_{}-{}".format(train_language_abbreviation, mid_train_eval_nli_target_language_abbreviation), reconstructed_nli_task_eval_accuracy, collections=['eval_accuracy'])

  ################################################



  # Start training by running session...
  merged_loss_summaries = tf.summary.merge_all(key='loss')
  merged_eval_accuracy_summaries = tf.summary.merge_all(key='eval_accuracy')
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver()

  os.system("mkdir -p {}".format(os.path.join(FLAGS.output_dir, "logs")))
  train_writer = tf.summary.FileWriter(os.path.join(FLAGS.output_dir, "logs"), sess.graph)

  # Losses to log
  loss_summaries = None
  global_step = 0

  ##############################################################
  #                                                             
  #                        START TRAINING
  #                                                             
  ##############################################################
  for epoch_num in range(1, FLAGS.num_train_epochs+1):

    # Shuffle train_examples every epoch
    random.shuffle(train_examples)

    for step_num in range(1, num_train_steps_per_epoch+1):

      ######################################
      ##### PREPROCESS INPUT SENTENCES #####
      ######################################

      # minibatch_bse_sentences : [[0.233, -0.146, ..., 0.256, -0.876], [...], ..., [...], [...]]; with each entry being of BSE dimensions (768-dim)
      # minibatch_language_labels : [0, 3, 2, 4, 2, 3, 1, 0, 2, ... , 3, 2, 0, 0, 1]; given five train languages
      
      original_sentences, original_labels = util.get_mc_minibatch(train_examples, step_num, FLAGS.train_batch_size, language_reference)

      original_label_onehots = util.convert_to_onehots(num_train_languages, original_labels)

      target_labels = util.create_random_labels(num_train_languages, FLAGS.train_batch_size)
      target_label_onehots = util.convert_to_onehots(num_train_languages, target_labels)

      xhat_alphas = util.create_xhat_alphas(FLAGS.train_batch_size)
      
      # Organize the dictionary to feed in for the placeholders of the model
      feed_dict = {original_sentences_tensor: original_sentences, original_label_onehots_tensor: original_label_onehots, target_label_onehots_tensor: target_label_onehots, xhat_alphas_tensor: xhat_alphas}

      ##########################################
      ##### TRAIN DISCRIMINATOR (ONE STEP) #####
      ##########################################
      
      if (step_num % (FLAGS.Dis_Gen_train_ratio+1)) != 0:
        _, loss_summaries = sess.run([Dis_optimizer, merged_loss_summaries], feed_dict=feed_dict)

      #############################################################
      ##### TRAIN GENERATOR (ONE STEP AFTER EVERY X DIS STEP) #####
      #############################################################

      else:
        _, loss_summaries = sess.run([Gen_optimizer, merged_loss_summaries], feed_dict=feed_dict)

      # Update loss information to Tensorboard
      train_writer.add_summary(loss_summaries, global_step=global_step)

      ##############################
      ##### SAVING CHECKPOINTS #####
      ##############################

      # Save checkpoint at every save_checkpoint_steps or at end of epoch
      if (step_num % FLAGS.save_checkpoints_steps == 0) or (step_num == num_train_steps_per_epoch):
        saved_path = saver.save(sess, os.path.join(FLAGS.output_dir, "isr_encoder-{}-{}".format(epoch_num, step_num)))



      ##############################################################
      #                                                             
      #                    MID-TRAIN EVALUATIONS
      #                                                             
      ##############################################################

      # Run evaluation at every run_mid_train_eval_steps or at very beginning of training or at end of every epoch
      if FLAGS.do_mid_train_eval:
        if (step_num % FLAGS.run_mid_train_eval_steps == 0) or ((epoch_num == 1) and (step_num == 1)) or (step_num == num_train_steps_per_epoch):
          
          ####################################
          ##### DISCRIMINATOR CLASSIFIER #####
          ####################################

          to_eval_examples = dev_examples

          original_sentences_cls_accuracy, generated_sentences_cls_accuracy = mid_train_evaluation.evaluate_Discriminator_classifier(to_eval_examples, language_reference, num_train_languages, sess, original_sentences_tensor, original_label_onehots_tensor, target_label_onehots_tensor, original_sentences_cls_predictions_tensor, generated_sentences_cls_predictions_tensor)

          sess.run(original_sentences_cls_accuracy_tensor.assign(original_sentences_cls_accuracy))
          sess.run(generated_sentences_cls_accuracy_tensor.assign(generated_sentences_cls_accuracy))

          ###############################################
          ##### NLI TASK SOLVERS #####
          ###############################################

          ### GENERATED SENTENCES ###

          generated_to_eval_examples = dev_example_in_pairs
            
          generated_sentences_nli_task_solver_accuracy = mid_train_evaluation.evaluate_nli_task_solver(generated_to_eval_examples, train_language_abbreviations, language_reference, num_train_languages, mid_train_eval_nli_target_language_abbreviation, original_sentences_tensor, original_label_onehots_tensor, target_label_onehots_tensor, sess, generated_sentences_tensor, FLAGS.mid_train_eval_nli_model_path)

          for train_language_abbreviation in train_language_abbreviations:
            sess.run(generated_nli_task_eval_accuracy_dict[train_language_abbreviation].assign(generated_sentences_nli_task_solver_accuracy[train_language_abbreviation]))

          ### RECONSTRUCTED SENTENCES ###

          reconstructed_to_eval_examples = dev_example_in_pairs
            
          reconstructed_sentences_nli_task_solver_accuracy = mid_train_evaluation.evaluate_nli_task_solver(reconstructed_to_eval_examples, train_language_abbreviations, language_reference, num_train_languages, mid_train_eval_nli_target_language_abbreviation, original_sentences_tensor, original_label_onehots_tensor, target_label_onehots_tensor, sess, reconstructed_sentences_tensor, FLAGS.mid_train_eval_nli_model_path)

          for train_language_abbreviation in train_language_abbreviations:
            sess.run(reconstructed_nli_task_eval_accuracy_dict[train_language_abbreviation].assign(reconstructed_sentences_nli_task_solver_accuracy[train_language_abbreviation]))

          ############################################
          ##### UPDATE SUMMARY WITH EVAL RESULTS #####
          ############################################
          eval_accuracy_summaries = sess.run(merged_eval_accuracy_summaries)
          train_writer.add_summary(eval_accuracy_summaries, global_step=global_step)

      # Increment global step
      global_step += 1

  # Close main tensorflow session
  sess.close()



##############################
##### The Program Driver #####
##############################

if __name__ == "__main__":
  main()
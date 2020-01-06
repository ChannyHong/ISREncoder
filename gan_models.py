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
Description: The Discriminator and Generator models of our ISR Encoder training framework.
'''

import tensorflow as tf
import tensorflow.contrib as tf_contrib


def weight_initializer():
  return tf_contrib.layers.xavier_initializer()


##############################################################
#                                                             
#                        DISCRIMINATOR
#                                                              
##############################################################

class Discriminator():
  def __init__(self, embedding_size, num_train_languages):

    ##### CONFIGURATIONS #####

    bse_size = embedding_size
    hidden_size = embedding_size

    ##### WEIGHTS AND BIASES #####

    self.W_1 = tf.get_variable(name="Dis_W_1", shape=[bse_size, hidden_size*2], initializer=weight_initializer())
    self.b_1 = tf.get_variable(name="Dis_b_1", shape=[hidden_size*2], initializer=tf.zeros_initializer())

    self.W_2 = tf.get_variable(name="Dis_W_2", shape=[hidden_size*2, hidden_size*4], initializer=weight_initializer())
    self.b_2 = tf.get_variable(name="Dis_b_2", shape=[hidden_size*4], initializer=tf.zeros_initializer())

    self.W_3 = tf.get_variable(name="Dis_W_3", shape=[hidden_size*4, hidden_size*4], initializer=weight_initializer())
    self.b_3 = tf.get_variable(name="Dis_b_3", shape=[hidden_size*4], initializer=tf.zeros_initializer())

    self.W_4 = tf.get_variable(name="Dis_W_4", shape=[hidden_size*4, hidden_size*2], initializer=weight_initializer())
    self.b_4 = tf.get_variable(name="Dis_b_4", shape=[hidden_size*2], initializer=tf.zeros_initializer())

    self.W_5 = tf.get_variable(name="Dis_W_5", shape=[hidden_size*2, hidden_size], initializer=weight_initializer())
    self.b_5 = tf.get_variable(name="Dis_b_5", shape=[hidden_size], initializer=tf.zeros_initializer())

    self.W_cls = tf.get_variable(name="Dis_W_cls", shape=[hidden_size, num_train_languages], initializer=weight_initializer())
    self.b_cls = tf.get_variable(name="Dis_b_cls", shape=[num_train_languages], initializer=tf.zeros_initializer())

    self.W_src = tf.get_variable(name="Dis_W_src", shape=[hidden_size, 1], initializer=weight_initializer())
    self.b_src = tf.get_variable(name="Dis_b_src", shape=[1], initializer=tf.zeros_initializer())


  def __call__(self, sentences_input_tensor):
    
    ##### HIDDEN LAYERS #####

    h_1 = tf.nn.tanh(tf.matmul(sentences_input_tensor, self.W_1) + self.b_1)
    h_2 = tf.nn.tanh(tf.matmul(h_1, self.W_2) + self.b_2)
    h_3 = tf.nn.tanh(tf.matmul(h_2, self.W_3) + self.b_3)
    h_4 = tf.nn.tanh(tf.matmul(h_3, self.W_4) + self.b_4)
    h_5 = tf.nn.tanh(tf.matmul(h_4, self.W_5) + self.b_5)

    ##### OUTPUT TENSORS #####

    output_tensor_cls = tf.matmul(h_5, self.W_cls) + self.b_cls
    output_tensor_src = tf.matmul(h_5, self.W_src) + self.b_src

    predictions_tensor = tf.argmax(input=output_tensor_cls, axis=1)

    return output_tensor_cls, output_tensor_src, predictions_tensor


##############################################################
#                                                             
#                         GENERATOR
#                                                             
##############################################################

class Generator():
  def __init__(self, train_isr, embedding_size, num_train_languages):

    ##### CONFIGURATIONS #####

    bse_size = embedding_size
    Enc_hidden_size = embedding_size
    isr_size = embedding_size
    Dec_hidden_size = embedding_size


    ##### WEIGHTS AND BIASES #####

    ### Encoder ###

    # Upsampling
    self.W_Enc_Down_1 = tf.get_variable(name="Gen_W_Enc_Down_1", shape=[bse_size+num_train_languages, Enc_hidden_size*2], initializer=weight_initializer())
    self.b_Enc_Down_1 = tf.get_variable(name="Gen_b_Enc_Down_1", shape=[Enc_hidden_size*2], initializer=tf.zeros_initializer())

    self.W_Enc_Down_2 = tf.get_variable(name="Gen_W_Enc_Down_2", shape=[Enc_hidden_size*2, Enc_hidden_size*4], initializer=weight_initializer())
    self.b_Enc_Down_2 = tf.get_variable(name="Gen_b_Enc_Down_2", shape=[Enc_hidden_size*4], initializer=tf.zeros_initializer())

    # Bottleneck
    self.W_Enc_BN_1 = tf.get_variable(name="Gen_W_Enc_BN_1", shape=[Enc_hidden_size*4, Enc_hidden_size*4], initializer=weight_initializer())
    self.b_Enc_BN_1 = tf.get_variable(name="Gen_b_Enc_BN_1", shape=[Enc_hidden_size*4], initializer=tf.zeros_initializer())

    self.W_Enc_BN_2 = tf.get_variable(name="Gen_W_Enc_BN_2", shape=[Enc_hidden_size*4, Enc_hidden_size*4], initializer=weight_initializer())
    self.b_Enc_BN_2 = tf.get_variable(name="Gen_b_Enc_BN_2", shape=[Enc_hidden_size*4], initializer=tf.zeros_initializer())

    self.W_Enc_BN_3 = tf.get_variable(name="Gen_W_Enc_BN_3", shape=[Enc_hidden_size*4, Enc_hidden_size*4], initializer=weight_initializer())
    self.b_Enc_BN_3 = tf.get_variable(name="Gen_b_Enc_BN_3", shape=[Enc_hidden_size*4], initializer=tf.zeros_initializer())

    self.W_Enc_BN_4 = tf.get_variable(name="Gen_W_Enc_BN_4", shape=[Enc_hidden_size*4, Enc_hidden_size*4], initializer=weight_initializer())
    self.b_Enc_BN_4 = tf.get_variable(name="Gen_b_Enc_BN_4", shape=[Enc_hidden_size*4], initializer=tf.zeros_initializer())

    # Downsampling
    self.W_Enc_Up_1 = tf.get_variable(name="Gen_W_Enc_Up_1", shape=[Enc_hidden_size*4, Enc_hidden_size*2], initializer=weight_initializer())
    self.b_Enc_Up_1 = tf.get_variable(name="Gen_b_Enc_Up_1", shape=[Enc_hidden_size*2], initializer=tf.zeros_initializer())

    self.W_Enc_Up_2 = tf.get_variable(name="Gen_W_Enc_Up_2", shape=[Enc_hidden_size*2, isr_size], initializer=weight_initializer())
    self.b_Enc_Up_2 = tf.get_variable(name="Gen_b_Enc_Up_2", shape=[isr_size], initializer=tf.zeros_initializer())


    if train_isr:

      ### Decoder ###

      # Upsampling
      self.W_Dec_Down_1 = tf.get_variable(name="Gen_W_Dec_Down_1", shape=[isr_size+num_train_languages, Dec_hidden_size*2], initializer=weight_initializer())
      self.b_Dec_Down_1 = tf.get_variable(name="Gen_b_Dec_Down_1", shape=[Dec_hidden_size*2], initializer=tf.zeros_initializer())

      self.W_Dec_Down_2 = tf.get_variable(name="Gen_W_Dec_Down_2", shape=[Dec_hidden_size*2, Dec_hidden_size*4], initializer=weight_initializer())
      self.b_Dec_Down_2 = tf.get_variable(name="Gen_b_Dec_Down_2", shape=[Dec_hidden_size*4], initializer=tf.zeros_initializer())

      # Bottleneck
      self.W_Dec_BN_1 = tf.get_variable(name="Gen_W_Dec_BN_1", shape=[Dec_hidden_size*4, Dec_hidden_size*4], initializer=weight_initializer())
      self.b_Dec_BN_1 = tf.get_variable(name="Gen_b_Dec_BN_1", shape=[Dec_hidden_size*4], initializer=tf.zeros_initializer())

      self.W_Dec_BN_2 = tf.get_variable(name="Gen_W_Dec_BN_2", shape=[Dec_hidden_size*4, Dec_hidden_size*4], initializer=weight_initializer())
      self.b_Dec_BN_2 = tf.get_variable(name="Gen_b_Dec_BN_2", shape=[Dec_hidden_size*4], initializer=tf.zeros_initializer())

      self.W_Dec_BN_3 = tf.get_variable(name="Gen_W_Dec_BN_3", shape=[Dec_hidden_size*4, Dec_hidden_size*4], initializer=weight_initializer())
      self.b_Dec_BN_3 = tf.get_variable(name="Gen_b_Dec_BN_3", shape=[Dec_hidden_size*4], initializer=tf.zeros_initializer())

      self.W_Dec_BN_4 = tf.get_variable(name="Gen_W_Dec_BN_4", shape=[Dec_hidden_size*4, Dec_hidden_size*4], initializer=weight_initializer())
      self.b_Dec_BN_4 = tf.get_variable(name="Gen_b_Dec_BN_4", shape=[Dec_hidden_size*4], initializer=tf.zeros_initializer())

      # Downsampling
      self.W_Dec_Up_1 = tf.get_variable(name="Gen_W_Dec_Up_1", shape=[Dec_hidden_size*4, Dec_hidden_size*2], initializer=weight_initializer())
      self.b_Dec_Up_1 = tf.get_variable(name="Gen_b_Dec_Up_1", shape=[Dec_hidden_size*2], initializer=tf.zeros_initializer())

      self.W_Dec_Up_2 = tf.get_variable(name="Gen_W_Dec_Up_2", shape=[Dec_hidden_size*2, bse_size], initializer=weight_initializer())
      self.b_Dec_Up_2 = tf.get_variable(name="Gen_b_Dec_Up_2", shape=[bse_size], initializer=tf.zeros_initializer())



  def __call__(self, direction, original_sentences_tensor, original_label_onehots_tensor, target_label_onehots_tensor=None):
    ##### LAYERS #####

    # Tensor for input sentence
    # self.original_sentences_tensor = tf.placeholder(tf.float32, [None, bse_size])
    # self.original_label_onehots_tensor = tf.placeholder(tf.float32, [None, num_train_languages])
    

    forward_Encoder_input_tensor = tf.concat([original_sentences_tensor, original_label_onehots_tensor], axis=1)

    # FORWARD DIRECTION
    # Forward: Encoder hidden layers
    h_Enc_1 = tf.nn.tanh(tf.matmul(forward_Encoder_input_tensor, self.W_Dec_Down_1) + self.b_Dec_Down_1)
    h_Enc_2 = tf.nn.tanh(tf.matmul(h_Enc_1, self.W_Dec_Down_2) + self.b_Dec_Down_2)
    h_Enc_3 = tf.nn.tanh(tf.matmul(h_Enc_2, self.W_Dec_BN_1) + self.b_Dec_BN_1)
    h_Enc_4 = tf.nn.tanh(tf.matmul(h_Enc_3, self.W_Dec_BN_2) + self.b_Dec_BN_2)
    h_Enc_5 = tf.nn.tanh(tf.matmul(h_Enc_4, self.W_Dec_BN_3) + self.b_Dec_BN_3)
    h_Enc_6 = tf.nn.tanh(tf.matmul(h_Enc_5, self.W_Dec_BN_4) + self.b_Dec_BN_4)
    h_Enc_7 = tf.nn.tanh(tf.matmul(h_Enc_6, self.W_Dec_Up_1) + self.b_Dec_Up_1)

    # Forward: Interlingual Semantic Representations (ISR) 'sentences'
    isr_sentences_tensor = tf.add(tf.matmul(h_Enc_7, self.W_Dec_Up_2), self.b_Dec_Up_2, name="{}_isr_sentences_tensor".format(direction))

    generated_sentences_tensor = None

    if not (target_label_onehots_tensor is None):
      forward_Decoder_input_tensor = tf.concat([isr_sentences_tensor, target_label_onehots_tensor], axis=1)

      # Forward: Decoder hidden layers
      h_Dec_1 = tf.nn.tanh(tf.matmul(forward_Decoder_input_tensor, self.W_Dec_Down_1) + self.b_Dec_Down_1)
      h_Dec_2 = tf.nn.tanh(tf.matmul(h_Dec_1, self.W_Dec_Down_2) + self.b_Dec_Down_2)
      h_Dec_3 = tf.nn.tanh(tf.matmul(h_Dec_2, self.W_Dec_BN_1) + self.b_Dec_BN_1)
      h_Dec_4 = tf.nn.tanh(tf.matmul(h_Dec_3, self.W_Dec_BN_2) + self.b_Dec_BN_2)
      h_Dec_5 = tf.nn.tanh(tf.matmul(h_Dec_4, self.W_Dec_BN_3) + self.b_Dec_BN_3)
      h_Dec_6 = tf.nn.tanh(tf.matmul(h_Dec_5, self.W_Dec_BN_4) + self.b_Dec_BN_4)
      h_Dec_7 = tf.nn.tanh(tf.matmul(h_Dec_6, self.W_Dec_Up_1) + self.b_Dec_Up_1)

      # Tensor for output generated sentence
      generated_sentences_tensor = tf.matmul(h_Dec_7, self.W_Dec_Up_2) + self.b_Dec_Up_2

    return isr_sentences_tensor, generated_sentences_tensor










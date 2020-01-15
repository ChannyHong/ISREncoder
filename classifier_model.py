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
Description: The NLI classifier model (to be trained on top of fixed ISR Encoder). 
    Single hidden layer of size 768, outputs a single 3-way confidence score.
'''

import tensorflow as tf
import tensorflow.contrib as tf_contrib



def weight_initializer():
  return tf_contrib.layers.xavier_initializer()

def batch_norm(layer):
  return tf.layers.batch_normalization(layer)



##############################################################
#                                                             
#                      CLASSIFIER MODEL
#                                                              
##############################################################

class Classifier():
  def __init__(self, input_layer_dim, num_labels):

    ##### WEIGHTS AND BIASES #####
    
    self.W_mlp_1 = tf.get_variable(name="cls_W_mlp_1", shape=[input_layer_dim, 768], initializer=weight_initializer())
    self.b_mlp_1 = tf.get_variable(name="cls_b_mlp_1", shape=[768], initializer=tf.zeros_initializer())

    self.W_cls = tf.get_variable(name="cls_W_cls", shape=[768, num_labels], initializer=weight_initializer())
    self.b_cls = tf.get_variable(name="cls_b_cls", shape=[num_labels], initializer=tf.zeros_initializer())

  def __call__(self, input_layer, keep_rate):

    ##### LAYERS #####
    input_layer_norm = batch_norm(input_layer)

    # MLP layer
    h_mlp_1 = tf.nn.relu(tf.matmul(input_layer_norm, self.W_mlp_1) + self.b_mlp_1)
    
    # Dropout applied to classifier
    h_drop = tf.nn.dropout(h_mlp_1, keep_prob=keep_rate)

    # Get prediction
    logits = tf.matmul(h_drop, self.W_cls) + self.b_cls
    predictions_tensor = tf.argmax(input=logits, axis=1)

    return logits, predictions_tensor
# coding=utf-8
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors.
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

import array
import json
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "DeepLearningExamples", "TensorFlow", "LanguageModeling", "BERT"))
sys.path.insert(0, os.getcwd())

import modeling
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.tools.graph_transforms import TransformGraph
from google.protobuf.json_format import MessageToDict

def save_model(fname, sess, graph=None):
  def save(fname, graph_def):
    pass
    with tf.Graph().as_default() as g:
        tf.import_graph_def(graph_def, name='')
        graph_def = g.as_graph_def(add_shapes=True)
    tf.train.write_graph(graph_def, ".", fname, as_text=False)
  
  if graph == None:
    graph_def = sess.graph_def
  else:
    graph_def = graph.as_graph_def(add_shapes=True)
  
  #tmp = tf.get_default_graph().get_operations()
  #messages = [MessageToDict(o.node_def) for o in tmp] 
  input_nodes = ['IteratorGetNext:0', 'IteratorGetNext:1', 'IteratorGetNext:2']
  output_nodes =  ['logits']
  graph_def = graph_util.convert_variables_to_constants(
      sess=sess,
      input_graph_def=graph_def,
      output_node_names=output_nodes)
  graph_def = graph_util.remove_training_nodes(graph_def, protected_nodes=output_nodes)
  graph_def = optimize_for_inference_lib.optimize_for_inference(graph_def, [], output_nodes, dtypes.bfloat16.as_datatype_enum)

  transforms = [
    'remove_nodes(op=Identity, op=StopGradient)',
    'fold_batch_norms',
    'fold_old_batch_norms',
  ]
  graph_def = TransformGraph(graph_def, input_nodes, output_nodes, transforms)
  save("build/data/bert_tf_v1_1_large_fp32_384_v2/model_one_hot_true_bfloat16_final.pb", graph_def)

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings,
            compute_type=tf.bfloat16)
    final_hidden = model.get_sequence_output()

    final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
    batch_size = final_hidden_shape[0]
    seq_length = final_hidden_shape[1]
    hidden_size = final_hidden_shape[2]

    output_weights = tf.get_variable(
            "cls/squad/output_weights", [2, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02),
            dtype=tf.bfloat16)

    output_bias = tf.get_variable(
            "cls/squad/output_bias", [2], initializer=tf.zeros_initializer(),
            dtype=tf.bfloat16)

    final_hidden_matrix = tf.reshape(final_hidden, [batch_size * seq_length, hidden_size])
    logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)

    logits = tf.reshape(logits, [batch_size, seq_length, 2], name="logits")
    return logits

# One-time 
def print_checkpoint(save_path, sess):
  reader = tf.train.load_checkpoint(save_path)
  shapes = reader.get_variable_to_shape_map()
  dtypes = reader.get_variable_to_dtype_map()
  print(f"Checkpoint at '{save_path}':")
  vars = {}
  for key in shapes:
    print(f"  (key='{key}', shape={shapes[key]}, dtype={dtypes[key].name}") 
    #print(f"  (key='{key}', shape={shapes[key]}, dtype={dtypes[key].name}, "
    #      f"value={reader.get_tensor(key)})")
    #vars[key] = tf.Variable(reader.get_tensor(key), dtype=tf.bfloat16)
    vars[key] = reader.get_tensor(key)
  #init_op = tf.initialize_variables(vars)#tf.global_variables_initializer()
  #sess.run(init_op)
  #tf.train.Saver(var_list=vars).save(sess=sess, save_path="build/data/bert_tf_v1_1_large_fp32_384_v2/model.ckpt-bf16", write_meta_graph=False)
  return vars.values()

def main():
    bert_config = modeling.BertConfig.from_json_file("bert_config.json")
    #init_checkpoint="build/data/bert_tf_v1_1_large_fp32_384_v2/model.ckpt-5474"
    init_checkpoint="build/data/bert_tf_v1_1_large_fp32_384_v2/model.ckpt-bf16"
    
    input_ids = tf.placeholder(tf.int32, shape=(None,384), name="input_ids")
    input_mask = tf.placeholder(tf.int32, shape=(None,384), name="input_mask")
    segment_ids = tf.placeholder(tf.int32, shape=(None,384), name="segment_ids")

    logits = create_model(
        bert_config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        use_one_hot_embeddings=True)
    
    # one time operation to recreate bf16 checkpoint
    """ 
    with tf.compat.v1.Session() as sess:
        bfloat16_vars = print_checkpoint(init_checkpoint, sess)
    
    tvars = bfloat16_vars
    """
    tvars = tf.compat.v1.trainable_variables()
    initialized_variable_names = {}
    (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    
    tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)
    predictions = {
        "logits": logits
    }
    output_spec = tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.PREDICT, predictions=predictions)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        save_model("bert_large_nv.pb", sess)

if __name__ == "__main__":
    main()

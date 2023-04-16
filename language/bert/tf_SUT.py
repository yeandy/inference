# coding=utf-8
# Copyright 2021 Arm Limited and affiliates.
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
import os
import time
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "DeepLearningExamples", "TensorFlow", "LanguageModeling", "BERT"))
sys.path.insert(0, os.getcwd())

import mlperf_loadgen as lg
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from squad_QSL import get_squad_QSL

class BERT_TF_SUT():
    def __init__(self, args):
        if args.tpu or args.tpu_v2:
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver('local')
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
 
        print("Loading TF model...")
        '''
        infer_config = tf.compat.v1.ConfigProto()
        infer_config.intra_op_parallelism_threads = int(os.environ['TF_INTRA_OP_PARALLELISM_THREADS']) \
                if 'TF_INTRA_OP_PARALLELISM_THREADS' in os.environ else os.cpu_count()
        infer_config.inter_op_parallelism_threads = int(os.environ['TF_INTER_OP_PARALLELISM_THREADS']) \
                if 'TF_INTER_OP_PARALLELISM_THREADS' in os.environ else os.cpu_count()
        infer_config.use_per_session_threads = 1
        self.sess = tf.compat.v1.Session(config=infer_config)
        model_file = os.environ.get('ML_MODEL_FILE_WITH_PATH', 'build/data/bert_tf_v1_1_large_fp32_384_v2/model.pb')
        with gfile.FastGFile(model_file, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            self.sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
        '''
        #'''
        OUTPUT_SAVED_MODEL_DIR = args.saved_model_path
        # Prepare input and outputs of model
        if args.tpu:
            graph = tf.compat.v1.Graph()
            self.sess = tf.compat.v1.Session(graph=graph)#,config=infer_config)
            with graph.as_default():
                tf.compat.v1.saved_model.loader.load(self.sess,
                                [tf.compat.v1.saved_model.tag_constants.SERVING, tf.compat.v1.saved_model.tag_constants.TPU],
                                OUTPUT_SAVED_MODEL_DIR)
                tf.compat.v1.tpu.initialize_system()
        elif args.tpu_v2:
            with tf.device('/TPU:0'):
                self.model = tf.saved_model.load(export_dir=OUTPUT_SAVED_MODEL_DIR, tags=['serve'])
                self.inference_func = self.model.signatures['serving_default']
        else:
            graph = tf.compat.v1.Graph()
            self.sess = tf.compat.v1.Session(graph=graph)#,config=infer_config)
            with graph.as_default():
                tf.compat.v1.saved_model.loader.load(self.sess,
                        [tf.compat.v1.saved_model.tag_constants.SERVING],
                        OUTPUT_SAVED_MODEL_DIR)        
        #'''
        #'''
        if args.batch_size:
            self.batch_size = args.batch_size
        else:
            self.batch_size = 1
        print("Batch size: ", self.batch_size)
        # warmup
        input_ids   = np.array([[0]*384]* self.batch_size)
        input_mask  = np.array([[1]*384]* self.batch_size)
        segment_ids = np.array([[0, 1]*192]*self.batch_size)
        if args.tpu:
            feeds = {
            'input_ids:0':   input_ids,
            'input_mask:0':  input_mask,
            'segment_ids:0': segment_ids
            }
            warmup_res = self.sess.run(["logits:0"], feed_dict=feeds)
        elif args.tpu_v2:
            if args.quant_inputs:
                feeds = {
                    'input_ids':   tf.convert_to_tensor(input_ids, dtype=tf.int64),
                    'attention_mask':  tf.convert_to_tensor(input_mask, dtype=tf.int64),
                    'token_type_ids': tf.convert_to_tensor(segment_ids, dtype=tf.int64)
                }
            else:
                feeds = {
                    'input_ids':   tf.convert_to_tensor(input_ids, dtype=tf.int64),
                    'input_mask':  tf.convert_to_tensor(input_mask, dtype=tf.int64),
                    'segment_ids': tf.convert_to_tensor(segment_ids, dtype=tf.int64)
                }
            with tf.device('/TPU:0'):
                warmup_res = self.inference_func(**feeds)
        #print(warmup_res)
        #'''
        print("Constructing SUT...")
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        print("Finished constructing SUT.")
        self.args = args
        self.qsl = get_squad_QSL(args.max_examples)

    def issue_queries(self, query_samples):
        print("!!!!", len(query_samples))
        """
	all_input_ids = []
        all_input_masks = []
        all_segment_ids = []

        # loop through samples individually
        for i in range(len(query_samples)):
            eval_features = self.qsl.get_features(query_samples[i].index)
            input_ids   = np.array([eval_features.input_ids])
            input_mask  = np.array([eval_features.input_mask])
            segment_ids = np.array([eval_features.segment_ids])
            all_input_ids.append(input_ids)
            all_input_masks.append(input_mask)
        # batch predict
        all_feeds = {
            'input_ids:0':   np.vstack(all_input_ids),
            'input_mask:0':  np.vstack(all_input_masks),
            'segment_ids:0': np.vstack(all_segment_ids)
        }
        s = time.time()
        #with tf.profiler.experimental.Profile('bert_profile'):
        batch_result = self.sess.run(["logits:0"], feed_dict=all_feeds)
        #print((time.time() - s) * 1000,  "ms")
        responses = []
        for i in range(len(query_samples)):
            result = batch_result[0][i]
            logits = [float(x) for x in result[0].flat]
            response_array = array.array("B", np.array(logits).astype(np.float32).tobytes())
            bi = response_array.buffer_info()
            response = lg.QuerySampleResponse(query_samples[i].id, bi[0], bi[1])
            responses.append(response)
            #lg.QuerySamplesComplete([response])
        lg.QuerySamplesComplete(responses)
        #'''
	
        """
        idx = 0
        while idx < len(query_samples):
            
            all_input_ids = []
            all_input_masks = []
            all_segment_ids = []

            # loop through samples individually
            for id in range(idx, min(idx + self.batch_size, len(query_samples))):
                #print("id: ", id)
                eval_features = self.qsl.get_features(query_samples[id].index)
                input_ids   = np.array([eval_features.input_ids])
                input_mask  = np.array([eval_features.input_mask])
                segment_ids = np.array([eval_features.segment_ids])
                all_input_ids.append(input_ids)
                all_input_masks.append(input_mask)
                all_segment_ids.append(segment_ids)
            if self.args.tpu:
                all_feeds = {
                'input_ids:0':   np.vstack(all_input_ids),
                'input_mask:0':  np.vstack(all_input_masks),
                'segment_ids:0': np.vstack(all_segment_ids)
                }
            elif self.args.tpu_v2:
                if self.args.quant_inputs:
                    all_feeds = {
                        'input_ids':   tf.convert_to_tensor(input_ids, dtype=tf.int64),
                        'attention_mask':  tf.convert_to_tensor(input_mask, dtype=tf.int64),
                        'token_type_ids': tf.convert_to_tensor(segment_ids, dtype=tf.int64)
                    }
                else:
                    all_feeds = {
                        'input_ids':   tf.convert_to_tensor(input_ids, dtype=tf.int64),
                        'input_mask':  tf.convert_to_tensor(input_mask, dtype=tf.int64),
                        'segment_ids': tf.convert_to_tensor(segment_ids, dtype=tf.int64)
                    }
            s = time.time()
            if self.args.tpu:
                batch_result = self.sess.run(["logits:0"], feed_dict=all_feeds)
            elif self.args.tpu_v2:
                with tf.device('/TPU:0'):
                    batch_result = self.inference_func(**all_feeds)
            
            if self.args.quant_inputs:
                batch_result = np.stack([batch_result['start_logits'], batch_result['end_logits']], axis=-1)
            responses = []
            for i, id in zip(range(0, self.batch_size), range(idx, min(idx + self.batch_size, len(query_samples)))):
                #print(i, id)
                result = batch_result[0][i]
                logits = [float(x) for x in result[0].flat]
                response_array = array.array("B", np.array(logits).astype(np.float32).tobytes())
                bi = response_array.buffer_info()
                response = lg.QuerySampleResponse(query_samples[id].id, bi[0], bi[1])
                responses.append(response)
            lg.QuerySamplesComplete(responses)


            idx = idx + self.batch_size
            #print("idx: ", idx)


    def flush_queries(self):
        pass

    def __del__(self):
        print("Finished destroying SUT.")

def get_tf_sut(args):
    return BERT_TF_SUT(args)

import os
import time
import numpy as np
import array
import jax
import torch
from jax.config import config
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from transformers import FlaxAutoModelForCausalLM, AutoTokenizer
import mlperf_loadgen as lg
from dataset import Dataset

#"""
# beam
gen_kwargs = {
    "early_stopping": True,
    "max_new_tokens": 128,
    "min_new_tokens": 30,
    "num_beams": 4,
}
#"""
"""
# greedy
gen_kwargs = {
    "early_stopping": False,
    "max_new_tokens": 128,
    "min_new_tokens": 30,
    "num_beams": 1,
}
"""

class SUT_base():
    def __init__(self, model_path, dtype, dataset_path, max_examples, do_init=False, from_pt=False, bf16_weights=False):
        # TODO : Pass model file name to init instead of args
        print("Loading JAX model...")
        self.model_name = "EleutherAI/gpt-j-6B"
        self.dataset_path = dataset_path
        self.model_path = model_path
        # dtype of activations
        if dtype == 'bfloat16':
            self.dtype = jax.numpy.bfloat16
            print("BF16 activations")
        elif dtype == 'float16':
            self.dtype = jax.numpy.float16
        else:
            self.dtype = jax.numpy.float32
            jax.config.update('jax_default_matmul_precision', 'float32')

        print("do_init", do_init, "from_pt", from_pt, "bf16_weights", bf16_weights)
        self.model, self.params = FlaxAutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=self.dtype,
            from_pt=from_pt,
            _do_init=do_init if not from_pt else True
        )
        print("finish from_pretrained")
        if bf16_weights:
            self.params = self.model.to_bf16(self.params)
            print("finish bf16 cast")
            print(self.params)
        else:
            self.params = self.model.to_fp32(self.params)
            print("finish fp32 cast")
            print(self.params)
        n_devices = len(jax.devices())
        print("n devices", n_devices, jax.devices())
        # Use a simple sharding scheme to just fit the model.
        devices = mesh_utils.create_device_mesh((n_devices, 1))
        sharding = PositionalSharding(devices)

        def put_sharded(v):
              return jax.device_put(v, sharding.reshape(1, n_devices))
        
        self.params["transformer"]["h"] = jax.tree_util.tree_map(
                    put_sharded, self.params["transformer"]["h"]
                    )
        self.params["lm_head"] = jax.device_put(
                    self.params["lm_head"], sharding.replicate(axis=0, keepdims=True)
                    )
        self.params["transformer"]["ln_f"] = jax.device_put(
                    self.params["transformer"]["ln_f"], sharding.replicate(axis=0, keepdims=True)
                    )
        self.params["transformer"]["wte"] = jax.device_put(
                    self.params["transformer"]["wte"], sharding.replicate(axis=0, keepdims=True)
                    )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            model_max_length=1919,
            padding_side="left",
            use_fast=False,)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.data_object = Dataset(
            self.dataset_path, total_count_override=max_examples, framework="jax")
        
        self.qsl = lg.ConstructQSL(self.data_object.count, self.data_object.perf_count,
                                   self.data_object.LoadSamplesToRam, self.data_object.UnloadSamplesFromRam)

        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)

        # warmup
        print("Compiling model. Generation args", gen_kwargs)
        from functools import partial
        @partial(jax.jit, static_argnums=[2, 3, 4, 5])
        def generator(input_batch, params, early_stopping, max_new_tokens, min_new_tokens, num_beams):
            return self.model.generate(**input_batch, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens, num_beams=num_beams, early_stopping=early_stopping, pad_token_id=self.tokenizer.eos_token_id, params=params)

        self.generator_compiled = generator
        attention_mask = self.data_object.source_encoded_attn_masks[0]
        input_id = self.data_object.source_encoded_input_ids[0]
        input_batch={'input_ids':input_id,'attention_mask':attention_mask}

        print("Running first inference")
        s = time.time()
        out = self.generator_compiled(input_batch=input_batch,params=self.params, early_stopping=gen_kwargs['early_stopping'], max_new_tokens=gen_kwargs['max_new_tokens'], min_new_tokens=gen_kwargs['min_new_tokens'], num_beams=gen_kwargs['num_beams'])
        print("First time ", time.time() - s)
        print("Running second inference")
        s = time.time()
        out = self.generator_compiled(input_batch=input_batch,params=self.params, early_stopping=gen_kwargs['early_stopping'], max_new_tokens=gen_kwargs['max_new_tokens'], min_new_tokens=gen_kwargs['min_new_tokens'], num_beams=gen_kwargs['num_beams'])
        out = np.array(out.sequences)
        print("Second time ", time.time() - s)



    def issue_queries(self, query_samples):
        print("Number of Samples in query_samples : ", len(query_samples))

        total_samples_done = 0
        list_prompts_tokens = []
        list_prompts_attn_masks = []

        #jax.profiler.start_trace("/tmp/tensorboard")
        for i in range(len(query_samples)):
            index = query_samples[i].index
            input_ids_tensor = self.data_object.source_encoded_input_ids[index]
            input_masks_tensor = self.data_object.source_encoded_attn_masks[index]
            s = time.time()
            pred_output_batch = np.array(self.inference_call(
                input_ids_tensor, input_masks_tensor))
            print(time.time()-s)
            response_array = array.array("B", pred_output_batch[0].tobytes())
            bi = response_array.buffer_info()
            response = [lg.QuerySampleResponse(
                query_samples[i].id, bi[0], bi[1])]
            lg.QuerySamplesComplete(response)
            if i % 5 == 0:
                print("Completed : ", i)
        #jax.profiler.stop_trace()

    def inference_call(self, input_ids_tensor, input_masks_tensor):
        ''' Common for all scenarios '''

        input_batch = dict()
        input_batch['input_ids'] = input_ids_tensor
        input_batch['attention_mask'] = input_masks_tensor

        output_batch = self.generator_compiled(
            input_batch=input_batch,params=self.params,early_stopping=gen_kwargs['early_stopping'], max_new_tokens=gen_kwargs['max_new_tokens'], min_new_tokens=gen_kwargs['min_new_tokens'], num_beams=gen_kwargs['num_beams']).sequences

        input_batch_lengths = [x.shape[0]
                               for x in input_batch["input_ids"]]

        output_batch_lengths = [x.shape[0] for x in output_batch]

        output_batch_truncated = []
        for data, source_len in zip(output_batch, input_batch_lengths):
            output_batch_truncated.append(data[source_len:])

        output_batch_truncated = jax.numpy.stack(output_batch_truncated)
        return output_batch_truncated

    def flush_queries(self):
        pass

    def __del__(self):
        print("Finished destroying SUT.")


class SUT_Offline(SUT_base):
    def __init__(self, model_path, dtype, dataset_path, max_examples, do_init, from_pt, bf16_weights):
        SUT_base.__init__(self, model_path, dtype, dataset_path, max_examples, do_init, from_pt, bf16_weights)
    '''IssueQuery and inference methods implemented in Base class'''


class SUT_Server(SUT_base):
    def __init__(self, model_path, dtype, dataset_path, max_examples, do_init, from_pt, bf16_weights):

        SUT_base.__init__(self, model_path, dtype, dataset_path, max_examples, do_init, from_pt, bf16_weights)
        self.total_samples_done = 0
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        print("SUT Server")

    def issue_queries(self, query_samples):

        index = query_samples[0].index
        input_ids_tensor = self.data_object.source_encoded_input_ids[index]
        input_masks_tensor = self.data_object.source_encoded_attn_masks[index]
        #s = time.time()
        pred_output_batch = np.array(self.inference_call(
            input_ids_tensor, input_masks_tensor))
        #print(time.time()-s)
        response_array = array.array("B", pred_output_batch.tobytes())
        bi = response_array.buffer_info()
        responses = [lg.QuerySampleResponse(query_samples[0].id, bi[0], bi[1])]
        lg.QuerySamplesComplete(responses)
        self.total_samples_done += 1
        if self.total_samples_done % 5 == 0:
            print("Completed : ", self.total_samples_done)


class SUT_SingleStream(SUT_base):
    def __init__(self, model_path, dtype, dataset_path, max_examples, do_init, from_pt, bf16_weights):
        SUT_base.__init__(self, model_path, dtype, dataset_path, max_examples, do_init, from_pt, bf16_weights)
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        self.total_samples_done = 0

    def issue_queries(self, query_samples):

        index = query_samples[0].index
        input_ids_tensor = self.data_object.source_encoded_input_ids[index]
        input_masks_tensor = self.data_object.source_encoded_attn_masks[index]
        s = time.time()
        pred_output_batch = np.array(self.inference_call(
            input_ids_tensor, input_masks_tensor))
        print(time.time()-s)
        response_array = array.array("B", pred_output_batch.tobytes())
        bi = response_array.buffer_info()
        responses = [lg.QuerySampleResponse(query_samples[0].id, bi[0], bi[1])]
        lg.QuerySamplesComplete(responses)
        self.total_samples_done += 1
        if self.total_samples_done % 5 == 0:
            print("Completed : ", self.total_samples_done)


def get_SUT(model_path, scenario, dtype, dataset_path, max_examples, do_init=False, from_pt=False, bf16_weights=False):
    if scenario == "Offline":
        return SUT_Offline(model_path, dtype, dataset_path, max_examples, do_init, from_pt, bf16_weights)
    elif scenario == "Server":
        return SUT_Server(model_path, dtype, dataset_path, max_examples, do_init, from_pt, bf16_weights)
    elif scenario == "SingleStream":
        return SUT_SingleStream(model_path, dtype, dataset_path, max_examples, do_init, from_pt, bf16_weights)

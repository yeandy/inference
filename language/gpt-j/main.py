import subprocess
import mlperf_loadgen as lg
import argparse
import os

import sys
from backend import get_SUT as get_torch_SUT
from jax_backend import get_SUT as get_jax_SUT
sys.path.insert(0, os.getcwd())


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=["tf", "pytorch", "onnxruntime", "tf_estimator", "jax"], default="pytorch", help="Backend")
    parser.add_argument("--scenario", choices=["SingleStream", "Offline",
                        "Server", "MultiStream"], default="Offline", help="Scenario")
    parser.add_argument("--model-path", default="EleutherAI/gpt-j-6B", help="")
    parser.add_argument(
        "--dataset-path", default="./data/cnn_eval.json", help="")
    parser.add_argument("--accuracy", action="store_true",
                        help="enable accuracy pass")
    parser.add_argument("--dtype", default="float32", help="data type of the model, choose from float16, bfloat16 and float32")
    parser.add_argument("--quantized", action="store_true",
                        help="use quantized model (only valid for onnxruntime backend)")
    parser.add_argument("--profile", action="store_true",
                        help="enable profiling (only valid for onnxruntime backend)")
    parser.add_argument("--gpu", action="store_true",
                        help="use GPU instead of CPU for the inference")
    parser.add_argument(
        "--mlperf_conf", default="mlperf.conf", help="mlperf rules config")
    parser.add_argument("--user_conf", default="user.conf",
                        help="user config for user LoadGen settings such as target QPS")
    parser.add_argument("--max_examples", type=int, default=13368,
                        help="Maximum number of examples to consider (not limited by default)")
    parser.add_argument("--jax_do_init", action="store_true", help="If True, initialize model's weights automatically")
    parser.add_argument("--jax_from_pt", action="store_true", help="If True, load JAX model from torch checkpoint")
    parser.add_argument("--jax_bf16_weights", action="store_true", help="If True, set data type of weights to bf16")
    args = parser.parse_args()
    return args


scenario_map = {
    "SingleStream": lg.TestScenario.SingleStream,
    "Offline": lg.TestScenario.Offline,
    "Server": lg.TestScenario.Server,
    "MultiStream": lg.TestScenario.MultiStream
}


def main():
    args = get_args()

    if args.backend == "pytorch":
        sut = get_torch_SUT(
            model_path=args.model_path,
            scenario=args.scenario,
            dtype=args.dtype,
            dataset_path=args.dataset_path,
            max_examples=args.max_examples,
            use_gpu=args.gpu,
        )
    elif args.backend == "jax":
        sut = get_jax_SUT(
            model_path=args.model_path,
            scenario=args.scenario,
            dtype=args.dtype,
            dataset_path=args.dataset_path,
            max_examples=args.max_examples,
            do_init=args.jax_do_init,
            from_pt=args.jax_from_pt,
            bf16_weights=args.jax_bf16_weights,
        )
    else:
        raise Exception(f'Implementation for {args.backend} is not defined')

    settings = lg.TestSettings()
    settings.scenario = scenario_map[args.scenario]
    # Need to update the conf
    settings.FromConfig(args.mlperf_conf, "gptj", args.scenario)
    settings.FromConfig(args.user_conf, "gptj", args.scenario)

    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
    else:
        settings.mode = lg.TestMode.PerformanceOnly
    log_path = os.environ.get("LOG_PATH")
    if not log_path:
        log_path = "build/logs"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = log_path
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    log_settings.enable_trace = True

    lg.StartTestWithLogSettings(sut.sut, sut.qsl, settings, log_settings)
    print("Test Done!")

    print("Destroying SUT...")
    lg.DestroySUT(sut.sut)

    print("Destroying QSL...")
    lg.DestroyQSL(sut.qsl)


if __name__ == "__main__":
    main()

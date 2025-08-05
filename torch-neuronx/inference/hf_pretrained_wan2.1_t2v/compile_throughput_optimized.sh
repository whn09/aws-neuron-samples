#!/bin/bash

export PYTHONPATH=`pwd`:$PYTHONPATH

echo "compiling text encoder"
python neuron_wan2_1_t2v/compile_text_encoder.py \
--compiled_models_dir "compile_workdir_throughput_optimized"

echo "compiling transformer"
python neuron_wan2_1_t2v/compile_transformer_throughput_optimized.py \
--compiled_models_dir "compile_workdir_throughput_optimized"

echo "compiling decoder"
python neuron_wan2_1_t2v/compile_decoder.py \
--compiled_models_dir "compile_workdir_throughput_optimized"
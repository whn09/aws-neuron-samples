#!/bin/bash

# source /opt/aws_neuronx_venv_pytorch_2_7_nxd_inference/bin/activate

export PYTHONPATH=`pwd`:$PYTHONPATH

# echo "compiling text encoder"
# python neuron_wan2.1_t2v/compile_text_encoder.py \
# --compiled_models_dir "compile_workdir_latency_optimized"

# echo "compiling transformer"
# python neuron_wan2.1_t2v/compile_transformer_latency_optimized.py \
# --compiled_models_dir "compile_workdir_latency_optimized"

echo "compiling decoder"
python neuron_wan2.1_t2v/compile_decoder.py \
--compiled_models_dir "compile_workdir_latency_optimized"sh 
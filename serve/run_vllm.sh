#!/usr/bin/env bash
# Launch vLLM (NVIDIA Thor container) serving the Qwen3-VL FP8 checkpoint, exposing an
# OpenAI-compatible API for the FP8-speed recon vs the transformers bf16 baseline.
#
# Uses the pre-staged NVIDIA image (vLLM 0.19.0, cu130, sm_110 kernels). The FP8 quant
# is auto-detected from the checkpoint config. Flags marked TUNE may need adjusting on
# Jetson's unified memory (vLLM assumes discrete VRAM):
#   --gpu-memory-utilization : fraction of (unified) memory for weights+KV cache
#   --max-model-len          : caps KV cache; our prompt is ~1 image + short text + 128 out
#
# Launch under thor-job (long-running service):
#   thor-job start vga-vllm -- serve/run_vllm.sh
# then GET http://127.0.0.1:8078/health once it's up.
set -uo pipefail

MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3-VL-8B-Instruct-FP8}"
PORT="${PORT:-8078}"
IMAGE="${IMAGE:-ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor}"

[ -d "$MODEL_DIR" ] || { echo "ERROR: model dir not found: $MODEL_DIR" >&2; exit 2; }

# Triton kernels fail with "unspecified launch failure" on Thor sm_110/CUDA13, but the
# FP8 CUTLASS path and native torch kernels work (bf16 transformers runs fine). Strategy:
# route every op through non-Triton paths while KEEPING FP8 CUTLASS.
#   --enforce-eager               : no torch.compile/inductor (kills the compile Triton path)
#   custom_ops ["none","+quant_fp8"] : disable all custom ops (native torch, incl. rotary)
#                                      EXCEPT the FP8 quant/CUTLASS path that we need + works
#   VLLM_ATTENTION_BACKEND        : FLASH_ATTN (CUDA flash-attn, not Triton); iterate if needed
# All three overridable via env so we can iterate without editing this file.
CUSTOM_OPS="${CUSTOM_OPS:-[\"none\",\"+quant_fp8\"]}"
ATTN="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
EXTRA="${EXTRA:-}"
COMPILE_CFG="{\"custom_ops\":${CUSTOM_OPS}}"
echo "[run_vllm] attn=$ATTN custom_ops=$CUSTOM_OPS extra='$EXTRA'"

exec docker run --rm --runtime nvidia --network host \
  -v "$MODEL_DIR":/model:ro \
  -e VLLM_ATTENTION_BACKEND="$ATTN" \
  -e CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}" \
  --name vga-vllm \
  "$IMAGE" \
  bash -lc "vllm serve /model \
      --served-model-name qwen3vl-fp8 \
      --port ${PORT} \
      --max-model-len 8192 \
      --gpu-memory-utilization 0.5 \
      --enforce-eager \
      --trust-remote-code \
      --compilation-config '${COMPILE_CFG}' ${EXTRA}"

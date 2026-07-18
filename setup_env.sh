#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# setup_env.sh — Hardware optimization for 6GB VRAM laptops
# 
# Source this before running autonomous.py or model_router.py:
#   source setup_env.sh
#
# Or add to ~/.bashrc / ~/.zshrc for permanent effect.
# These settings force GPU acceleration and prevent VRAM OOM.
# ═══════════════════════════════════════════════════════════════

# Force model execution onto GPU tensor cores
export CUDA_VISIBLE_DEVICES=0
export GGML_CUDA_FORCE_MMQ=1

# Critical: prevent parallel requests from splitting KV cache
# On 6GB VRAM, parallel requests cause immediate OOM
export OLLAMA_NUM_PARALLEL=1

# Recommended: enable flash attention if supported
export OLLAMA_FLASH_ATTENTION=1

# VRAM budget (adjust for your card)
# RTX 4050 = 6GB → set to 5.0 (leave 1GB for OS/desktop)
# RTX 4060 = 8GB → set to 7.0
# RTX 4070 = 12GB → set to 11.0
export VRAM_BUDGET_GB="${VRAM_BUDGET_GB:-5.0}"

# Allow system RAM fallback for models that don't fit in VRAM
# (slower but prevents crashes)
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-300}"  # 5 min default

echo "╔════════════════════════════════════════════════════╗"
echo "║  Flock Environment Configured                     ║"
echo "║                                                    ║"
echo "║  GPU: CUDA device 0 (forced MMQ)                   ║"
echo "║  Parallel: DISABLED (OLLAMA_NUM_PARALLEL=1)        ║"
echo "║  Flash Attention: ENABLED                          ║"
echo "║  VRAM Budget: ${VRAM_BUDGET_GB} GB                          ║"
echo "║                                                    ║"
echo "║  Iterator will pin permanently.                    ║"
echo "║  Specialists evict immediately after use.          ║"
echo "╚════════════════════════════════════════════════════╝"

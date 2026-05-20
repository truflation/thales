#!/bin/bash
# Vast.ai box setup for Phase 1.5 hierarchical housing (and future GPU work).
# Run this ON the Vast box, not locally.
#
# Assumes you've already rsynced the trufonomics-models/ directory to ~/.

set -e

cd ~/trufonomics-models

# 1) Install uv (if not already there)
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env || export PATH="$HOME/.local/bin:$PATH"
fi

# 2) Create venv + install base deps
echo "Setting up venv + base deps (this includes JAX-CPU)..."
uv sync

# 3) Upgrade JAX to CUDA build
echo "Installing JAX with CUDA support..."
uv pip install --upgrade "jax[cuda12]" "jaxlib"

# 4) Verify GPU is visible
echo
echo "─── JAX device check ───"
JAX_PLATFORMS=cuda uv run python -c "
import jax
print('JAX devices:', jax.devices())
print('Backend:', jax.default_backend())
"

# 5) Run the synthetic recovery test on GPU
echo
echo "─── Phase 1.5 synthetic recovery on GPU ───"
JAX_PLATFORMS=cuda uv run python scripts/demo_hierarchical_housing.py

# 6) Run the test suite (slow tests included)
echo
echo "─── Full Phase 1.5 test suite on GPU ───"
JAX_PLATFORMS=cuda uv run pytest tests/test_hierarchical_housing_archetype.py -v -m slow

echo
echo "═══════════════════════════════════════════════════════════════"
echo "  Vast box ready for Phase 1.5 production work."
echo "  See docs/vast_phase_1_5_runbook.md for next steps."
echo "═══════════════════════════════════════════════════════════════"

#!/usr/bin/env python3
"""Run after `fluxatlas generate`. All model hyperparameters are fixed in source."""
from pathlib import Path
from fluxatlas.study import study
study(Path(__file__).resolve().parents[1])

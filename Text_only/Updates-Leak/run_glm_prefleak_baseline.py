#!/usr/bin/env python3
"""Run the shared Updates-Leak-style baseline on frozen GLM transcripts."""

from run_llava_prefleak_baseline import main


if __name__ == "__main__":
    main(default_model_family="glm")

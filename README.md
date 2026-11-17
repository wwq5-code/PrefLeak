# PrefLeak

PrefLeak is a text-only black-box attack for inferring the direction of a
private visual-preference update.  It compares generated text from a base VLM
and its adapted version, calibrates the attack on counterfactual shadow
updates, and selects stable textual probes without querying the target with
images.

This repository contains source code and public prompt configurations only.
Datasets, model checkpoints, adapters, generated responses, experimental
results, and the paper are intentionally excluded.

## Repository layout

- `Text_only/`: VLM preference-update, probing, attack, evaluation, and
  robustness code.
- `Text_only/GLM/`: GLM-4.1V experiment runners.
- `Text_only/PPSD/` and `Text_only/PARA/`: external visual-preference dataset
  protocols.
- `Text_only/Query_budget_eva/`: query-budget evaluation scripts.
- `Text_only/Updates-Leak/` and `Text_only/dissecting_dist_inf/`: adapted
  black-box baselines.
- `On_CIFAR100/`: the preliminary CLIP retrieval experiment.
- `scripts/`: dataset download helpers.

## Environment

The experiments require Python, a CUDA-enabled PyTorch installation, and
sufficient GPU memory for the selected VLM.  From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r Text_only/requirements.txt
git clone --depth 1 https://github.com/huggingface/trl.git trl
python -m pip install -e ./trl
export PYTHONPATH="$PWD"
```

Model checkpoints are not distributed.  Pass either a local checkpoint path
or a Hugging Face model identifier with `--model`.

## Prepare data

Data are not included.  The download helpers accept the destination as their
first argument:

```bash
bash scripts/download_coco2017.sh ./data/coco2017
bash scripts/download_ppsd.sh ./data/PPSD
```

Build the visual-comparison manifests with the dataset-specific builders:

```bash
python Text_only/build_visual_preferences.py --help
python -m Text_only.PPSD.build_ppsd_preferences --help
python -m Text_only.PARA.build_para_preferences --help
```

For PPSD and PARA, either pass the source directory to the builder or set
`PREFLEAK_PPSD_ROOT` and `PREFLEAK_PARA_ROOT`, respectively.

## Run PrefLeak

The main runners expose restart-safe phases and write all generated artifacts
under the selected output directory.  Use `--help` to inspect dataset- and
model-specific options.

```bash
# COCO with LLaVA-OneVision
python Text_only/run_joint_vlm_tuning.py all --model /path/to/llava-checkpoint
python Text_only/run_joint_vlm_experiment.py all \
  --model /path/to/llava-checkpoint --jobs 4
python Text_only/run_remote_probe_experiment.py all \
  --model /path/to/llava-checkpoint

# COCO with GLM-4.1V
python Text_only/GLM/run_glm_tuning.py all --model /path/to/glm-checkpoint
python Text_only/GLM/run_glm_experiment.py all \
  --model /path/to/glm-checkpoint --jobs 3
python Text_only/GLM/run_remote_glm_experiment.py all \
  --model /path/to/glm-checkpoint

# PPSD and PARA
python -m Text_only.PPSD.run_ppsd_experiment --help
python -m Text_only.GLM.PPSD.run_glm_ppsd --help
python -m Text_only.PARA.run_para_llava_experiment --help
python -m Text_only.GLM.PARA.run_glm_para_experiment --help
```

Run the deterministic unit tests without downloading datasets or models:

```bash
python -m unittest \
  Text_only.test_remote_attack \
  Text_only.test_shared_response_resampling \
  Text_only.test_validation -v
```

## Notes

- Shadow calibration must be completed and frozen before target querying.
- Output directories are resumable; completed adapters and generations are
  reused.
- Dataset licenses and model access requirements remain those of their
  original providers.


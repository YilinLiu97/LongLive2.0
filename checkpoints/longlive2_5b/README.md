---
license: apache-2.0
pipeline_tag: text-to-video
tags:
  - text-to-video
  - video-generation
  - diffusion
  - long-video
  - longlive2
  - wan2.2
---

<p align="center">
  <img src="https://github.com/wileewang/LongLive2.0/blob/release-clean-merge/assets/longlive2/logo.png?raw=true" alt="LongLive2.0 logo" width="100%">
</p>

# LongLive2.0 5B Checkpoints

This repository hosts LongLive2.0 5B checkpoints for inference with
the LongLive2.0 release code:

https://github.com/wileewang/LongLive2.0

The checkpoint package supports two inference layouts:

- **Merged generator checkpoint (recommended)**: the AR-trained base generator
  and DMD-distilled LoRA adapter are already merged, so inference only loads one
  `generator_ckpt`.
- **Base generator + LoRA checkpoint**: the release code can also load the base
  generator first, attach LoRA modules, and then load the LoRA weights. This is
  useful for debugging or for users who want to inspect the adapter separately.

Use only one layout at a time. If you use the merged checkpoint, do not configure
a separate `lora_ckpt` or `adapter` section, otherwise the LoRA adapter would be
applied a second time.

## Installation

```bash
git clone https://github.com/wileewang/LongLive2.0.git
cd LongLive2.0

conda create -n longlive2 python=3.10 -y
conda activate longlive2
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

The released LongLive2.0 checkpoint is sufficient for standard inference. You
only need to download the original Wan2.2-TI2V-5B components if you want to run
training, initialize from the original Wan weights, or use code paths that
explicitly load the base Wan model files:

```bash
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B \
  --local-dir wan_models/Wan2.2-TI2V-5B
```

Download this checkpoint repository:

```bash
huggingface-cli download Perflow-Shuai/longlive_2.0_5B_tmp_20260507 \
  --local-dir checkpoints/longlive2_5b
```

## Configure Inference

Edit `configs/inference.yaml`:

### Option A: Merged Checkpoint (Recommended)

```yaml
checkpoints:
  generator_ckpt: checkpoints/longlive2_5b/merged_generator.pt

data:
  data_path: /path/to/inference_prompts

output_folder: videos/longlive2
num_samples: 1

inference:
  sampling_steps: 4
  sink_size: 8
  guidance_scale: 1.0
  multi_shot_sink: true
  multi_shot_rope_offset: 8
```

Replace `merged_generator.pt` with the actual merged checkpoint filename in this
repository. If your local config was copied from a base+LoRA setup, remove
`checkpoints.lora_ckpt` and the top-level `adapter` section before running
inference.

### Option B: Base Generator + LoRA

```yaml
checkpoints:
  generator_ckpt: checkpoints/longlive2_5b/generator.pt
  lora_ckpt: checkpoints/longlive2_5b/lora.pt

adapter:
  type: lora
  rank: 128
  alpha: 128
  dropout: 0.0
  verbose: true

data:
  data_path: /path/to/inference_prompts

output_folder: videos/longlive2
num_samples: 1

inference:
  sampling_steps: 4
  sink_size: 8
  guidance_scale: 1.0
  multi_shot_sink: true
  multi_shot_rope_offset: 8
```

This layout should reproduce the merged checkpoint behavior, but it keeps the
adapter explicit at runtime.

## Prompt Folder

`data.data_path` is passed to `MultiTextConcatDataset` in `inference.py`. It can
be either:

- a `.txt` file, where each line is one single-shot prompt; or
- a directory of multi-shot prompt folders.

For a directory input, the code supports both of the following layouts. The
direct caption-root layout is the simplest:

```text
inference_prompts/
  robot_lab_demo/
    0.json
    1.json
    2.json
    shot_durations.txt
```

It also supports a dataset root with an outer `caption/` folder:

```text
inference_prompts/
  caption/
    robot_lab_demo/
      0.json
      1.json
      2.json
      shot_durations.txt
```

Each JSON file contains:

```json
{
  "caption": "A compact silver robot with one blue optic explores a clean robotics lab."
}
```

`shot_durations.txt` is optional. If provided, each number is the number of
temporal chunks assigned to the corresponding caption, for example:

```text
2 2 4
```

## Run

Single node, 8 GPUs:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 inference.py \
  --config_path configs/inference.yaml
```

Single GPU:

```bash
python inference.py --config_path configs/inference.yaml
```

Outputs are written to `output_folder`.

## Notes

- For the merged checkpoint, standard inference only needs
  `checkpoints.generator_ckpt`.
- For the base+LoRA layout, set both `checkpoints.generator_ckpt` and
  `checkpoints.lora_ckpt`, and keep the `adapter` section.
- Do not mix the two layouts. A merged checkpoint should not be used together
  with `lora_ckpt` or `adapter`.
- `inference.sampling_steps` controls the number of denoising steps.
- `inference.multi_shot_sink` enables the multi-shot attention sink.
- `inference.multi_shot_rope_offset` controls the multi-shot RoPE offset.
- For NVFP4 inference, use the separate NVFP4 config and setup instructions in
  the LongLive2.0 documentation.

## Citation

Citation will be updated after the paper is released.

```bibtex
@article{longlive2,
  title   = {LongLive2.0: An NVFP4 Parallel Infrastructure for Long Video Generation},
  author  = {TODO},
  journal = {TODO},
  year    = {2026}
}
```

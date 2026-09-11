# SFT smoke runner

This runner uses TRL's `SFTTrainer` and the `allenai/Olmo-3-1025-7B` tokenizer/configuration.

Install the Python dependencies with `requirements.txt`. The CUDA-enabled
PyTorch package is supplied by the image and is intentionally excluded from
this file.

```bash
source /venv/main/bin/activate
uv pip install -r sft_experiment/requirements.txt
```

The smoke config downloads only the Olmo 3 base configuration and tokenizer, then
constructs a much smaller model from the same architecture. It uses synthetic
conversation data so the full training and metric path can be tested without
downloading 7B weights.

Run it from `/workspace` with:

```bash
source /venv/main/bin/activate
python sft_experiment/train_sft.py \
  --config sft_experiment/configs/smoke.yaml
```

The run writes one metric file per epoch. Each file contains smoothed loss,
hard-label NLL, token accuracy, entropy, and optional SFT-to-base KL divergence
for both train and validation data.

Each epoch checkpoint keeps `model.safetensors`, `optimizer.pt`, `scheduler.pt`,
`rng_state.pth`, and `trainer_state.json`. They are kept as separate files so
`SFTTrainer` can inspect or resume the checkpoint without duplicating the large
model weights into a combined `.pt` file.

For the H100/H200 run, use `configs/olmo3_7b.yaml`, set the output directory to
the target machine, and run it once per label-smoothing value:

```bash
python sft_experiment/train_sft.py \
  --config sft_experiment/configs/olmo3_7b.yaml \
  --label-smoothing 0.05
```

When `--output-dir` is omitted, the script appends the label-smoothing value
to the configured directory, for example `olmo3-7b-ls000`, `olmo3-7b-ls005`,
and `olmo3-7b-ls010`.

The custom chat template adds `{% generation %}` markers because the published
Olmo 3 template does not expose assistant token masks to TRL's
`assistant_only_loss` mode.

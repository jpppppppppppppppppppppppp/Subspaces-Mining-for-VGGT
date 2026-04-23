## Start

```
conda create -n YOUR_ENV_NAME python=3.11.0 -y
conda activate YOUR_ENV_NAME
pip install -r requirements.txt
pip install -e .
```

After installing `peft==0.17.0`, replace the corresponding files in your environment with our modified versions from the local `peft/` folder.

Copy the following files to:

`/YOUR_ENV_PATH/lib/python3.11/site-packages/peft/tuners/lora/`

and overwrite the original files:

- `config.py`: adds `apply_svd_lora`, `pattern_svd_lora`, and `use_predefined_space` to `LoraConfig`.
- `layer.py`: initializes the customized LoRA layers.
- `model.py`: adds the training support required by this project.

## Finetuning

Before finetuning, you need to prepare the dataset and update the training configuration accordingly. We provide example scripts and dataloaders to help you.

Make sure to update the dataset paths in the training config files:

- `training/config/default.yaml`.

Then launch training with:

```
cd training

CUDA_VISIBLE_DEVICES=... /YOUR_ENV_PATH/bin/torchrun launch.py \
    --seed ${SEED} \
    --config default \
    --lora \
    --use_predefined_space /PREDEFINED_SPACE.safetensors

or

CUDA_VISIBLE_DEVICES=... /YOUR_ENV_PATH/bin/torchrun launch.py \
    --seed ${SEED} \
    --config default \
    --lora \
    --apply_svd_lora \
    --pattern_svd_lora block
```

# Controller Experiments

This directory contains GRPO controller experiments and code.

## Files
- `train_controller.py`: entrypoint for training
- `grpo_trainer.py`: trainer implementation
- `controller_model.py`: model definitions
- `data_preparation.py`: dataset and preprocessing helpers
- `GRPO_CONTROLLER_DESIGN.md`: design notes
- `SUMMARY.md`: brief overview

## Usage
Run training with appropriate config/paths (example placeholder):
```bash
python experiments/controller/train_controller.py \
  --config configs/controller/example.yaml \
  --output_dir outputs/controller_run
```
Adjust arguments according to your environment and dataset locations.

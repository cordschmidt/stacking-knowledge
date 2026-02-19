import os
import yaml
from pathlib import Path

# 1. Define variations
prefix = "baseline"
learning_rates = [1e-3, 5e-4, 1e-4, 5e-5]
schedules = ["linear", "cosine", "cosine_with_min_lr", "constant"]

experiments = []
for lr in learning_rates:
    for sched in schedules:
        experiments.append({"lr": lr, "sched": sched})

# 2. Load base config to find the group
base_config_path = "conf/config.yaml"
with open(base_config_path, "r") as f:
    base_cfg = yaml.safe_load(f)

# Extract and sanitize group name (replace spaces with underscores)
#
group_name = base_cfg["experiment"].get("group", "default_group").replace(" ", "_")
output_base = Path("experiments") / group_name

# 3. Generate configs in nested subfolders
for i, params in enumerate(experiments):
    run_dir = output_base / f"run_{i}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Construct descriptive name as requested
    exp_name = f"{prefix}_lr_{params['lr']}_schedule_{params['sched']}"

    # Update config values
    run_cfg = base_cfg.copy()
    run_cfg["trainer"]["lr"] = params["lr"]
    run_cfg["trainer"]["lr_scheduler_type"] = params["sched"]
    run_cfg["experiment"]["name"] = exp_name

    # Save the specific config.yaml
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(run_cfg, f)

print(f"Generated {len(experiments)} configs in: {output_base}")
#!/usr/bin/env python3
import subprocess
import sys
import copy

BASE_COMMAND = [
    "python3", "finetune_contact.py",
    "hydra/launcher=basic",
    "agent=diffusion_contact",
    "task=franka_2cam_contact",
    "buffer_path=/home/labiiwa/dit-policy/my_data/pick_orange_hindsight/buf.pkl",
    "agent/features=resnet_gn",
    "agent.features.restore_path=/home/labiiwa/dit-policy/visual_features/resnet18/IN_1M_resnet18.pth",
    "trainer=bc_contact",
    "trainer.schedule_builder.schedule_kwargs.num_warmup_steps=500",
    "ac_chunk=8",
    "img_chunk=1",
    "batch_size=128",
    "num_workers=2",
    "train_transform=preproc",
    "max_iterations=8000",
    "eval_freq=500",
    "save_freq=500",
    "devices=1",
    "wandb.entity=aatxag-mondragon-university",
    "wandb.project=dit-policy",
]

RUNS_CONFIG = [
    {
        "exp_name": "pick_orange_hindsight",
        "task.train_buffer.n_test_trans": "800",
    }
]

def run_trainings():
    for i, config in enumerate(RUNS_CONFIG, 1):
        current_cmd = copy.deepcopy(BASE_COMMAND)
        for key, value in config.items():
            current_cmd.append(f"{key}={value}")

        print(f"\n{'='*60}")
        print(f"Ejecutando Run {i}/{len(RUNS_CONFIG)}")
        print(f"Configuración específica: {config}")
        print(f"{'='*60}\n")

        result = subprocess.run(current_cmd)

        if result.returncode != 0:
            print(f"\n[ERROR] El Run {i} falló con código {result.returncode}. Abortando.")
            sys.exit(result.returncode)

    print("\n¡Todas las ejecuciones se completaron con éxito!")

if __name__ == "__main__":
    run_trainings()
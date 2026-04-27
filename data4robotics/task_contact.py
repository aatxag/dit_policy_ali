# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch
import wandb

from data4robotics.task import DefaultTask, _build_data_loader


class BCTaskContact(DefaultTask):
    """BCTask variant that handles the optional 4-element contact batch."""

    def eval(self, trainer, global_step):
        losses = []
        action_l2, action_lsig = [], []

        for batch in self.test_loader:
            contact_point = None
            if len(batch) == 4:
                (imgs, obs), actions, mask, contact_point = batch
                contact_point = contact_point.to(trainer.device_id)
            else:
                (imgs, obs), actions, mask = batch

            imgs = {k: v.to(trainer.device_id) for k, v in imgs.items()}
            obs, actions, mask = [
                ar.to(trainer.device_id) for ar in (obs, actions, mask)
            ]

            with torch.no_grad():
                loss = trainer.training_step(batch, global_step)
                losses.append(loss.item())

                pred_actions = trainer.model.get_actions(
                    imgs, obs, contact_point=contact_point
                )

                l2_delta = torch.square(mask * (pred_actions - actions))
                l2_delta = l2_delta.sum((1, 2)) / mask.sum((1, 2))

                lsig = torch.logical_or(
                    torch.logical_and(actions > 0, pred_actions <= 0),
                    torch.logical_and(actions <= 0, pred_actions > 0),
                )
                lsig = (lsig.float() * mask).sum((1, 2)) / mask.sum((1, 2))

                action_l2.append(l2_delta.mean().item())
                action_lsig.append(lsig.mean().item())

        mean_val_loss = np.mean(losses)
        ac_l2, ac_lsig = np.mean(action_l2), np.mean(action_lsig)
        print(f"Step: {global_step}\tVal Loss: {mean_val_loss:.4f}")
        print(f"Step: {global_step}\tAC L2={ac_l2:.2f}\tAC LSig={ac_lsig:.2f}")

        if wandb.run is not None:
            wandb.log(
                {
                    "eval/task_loss": mean_val_loss,
                    "eval/action_l2": ac_l2,
                    "eval/action_lsig": ac_lsig,
                },
                step=global_step,
            )
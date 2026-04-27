# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from data4robotics.trainers.base import BaseTrainer


class BehaviorCloning(BaseTrainer):
    def training_step(self, batch, global_step):
        contact_point = None
        if len(batch) == 4:
            (imgs, obs), actions, mask, contact_point = batch
            contact_point = contact_point.to(self.device_id)
        else:
            (imgs, obs), actions, mask = batch

        imgs = {k: v.to(self.device_id) for k, v in imgs.items()}
        obs, actions, mask = [ar.to(self.device_id) for ar in (obs, actions, mask)]

        ac_flat = actions.reshape((actions.shape[0], -1))
        mask_flat = mask.reshape((mask.shape[0], -1))
        loss = self.model(imgs, obs, ac_flat, mask_flat, contact_point=contact_point)
        self.log("bc_loss", global_step, loss.item())
        if self.is_train:
            self.log("lr", global_step, self.lr)
        return loss

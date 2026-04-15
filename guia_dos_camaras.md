# Guía: Entrenar DiT-Policy con dos cámaras

## Resumen

Para entrenar dit-policy con dos cámaras necesitas cambios en **3 áreas**:

1. **Conversión de datos** → añadir `enc_cam_1` al pickle robobuf
2. **Configuración YAML** → indicar `n_cams: 2` en el task config
3. **Inferencia** → alimentar ambas imágenes al agente

Tu script de grabación (`recorder_node_two_cam.py`) ya está preparado — graba `cam0_256/` (wrist) y `cam1_256/` (externa). Solo falta convertir esos datos y configurar el entrenamiento.

---

## 1. Conversión de datos a formato robobuf

El formato robobuf es un pickle (`buf.pkl`) con esta estructura:

```
buf.pkl = [traj_0, traj_1, ..., traj_N]

traj_i = [(obs_0, action_0, reward_0), (obs_1, action_1, reward_1), ...]

obs_t = {
    "state":     np.float32 array (8,)     # [q_1..q_7, gripper]
    "enc_cam_0": np.ndarray (JPEG bytes)   # cámara wrist
    "enc_cam_1": np.ndarray (JPEG bytes)   # cámara externa  ← NUEVO
}
```

La convención de nombres es crítica: el framework busca claves `enc_cam_0`, `enc_cam_1`, etc.

### Uso del script de conversión

```bash
python convert_to_robobuf.py \
    --episodes_dir ~/dit_demos/pick_demo/episodes \
    --out_path ~/dit_demos/pick_demo/robobuf \
    --img_size 256 \
    --action_type joint_position
```

Esto genera `~/dit_demos/pick_demo/robobuf/buf.pkl`.

---

## 2. Configuración del entrenamiento

### 2.1 Fichero YAML del task

El framework `data4robotics` usa la variable `n_cams` en la configuración del task para saber cuántas cámaras cargar del pickle. Necesitas crear o modificar el YAML de tu task para indicar `n_cams: 2`.

Ejemplo: crea `experiments/task/franka_two_cam.yaml`:

```yaml
# experiments/task/franka_two_cam.yaml
_target_: data4robotics.tasks.bc_task.BCTask
buffer_path: /ruta/a/tu/robobuf/buf.pkl
n_cams: 2
train_transform: gpu_default
batch_size: ${batch_size}
num_workers: ${num_workers}
```

**Lo más importante es `n_cams: 2`.** El dataloader del framework iterará sobre `enc_cam_0` y `enc_cam_1` automáticamente.

### 2.2 Verificar la configuración del agente

El agente de difusión (DiT) con ResNet tokenizer ya soporta múltiples cámaras. Los features de cada cámara se extraen independientemente con el mismo backbone y luego se concatenan como tokens para el transformer.

No necesitas cambiar nada en `agent/diffusion.yaml` ni en `agent/features/resnet_gn.yaml` — el número de cámaras se propaga automáticamente desde el task.

### 2.3 Comando de entrenamiento

```bash
python finetune.py \
    exp_name=pick_two_cam \
    agent=diffusion \
    task=franka_two_cam \
    agent/features=resnet_gn \
    agent.features.restore_path=/ruta/a/IN_1M_resnet18.pth \
    trainer=bc_cos_sched \
    ac_chunk=100 \
    batch_size=32 \
    num_workers=4 \
    buffer_path=/ruta/a/tu/robobuf/buf.pkl
```

Si usas un task YAML que hereda de `end_effector_r6` o similar, simplemente
sobrescribe `n_cams`:

```bash
python finetune.py \
    exp_name=pick_two_cam \
    agent=diffusion \
    task=end_effector_r6 \
    task.n_cams=2 \
    task.buffer_path=/ruta/a/tu/robobuf/buf.pkl \
    agent/features=resnet_gn \
    agent.features.restore_path=/ruta/a/IN_1M_resnet18.pth \
    trainer=bc_cos_sched \
    ac_chunk=100
```

---

## 3. Inferencia con dos cámaras

En inferencia, debes pasar un diccionario de imágenes con las claves `cam_0` y `cam_1` (sin el prefijo `enc_`).

### Ejemplo de script de inferencia

```python
import torch
import cv2
import numpy as np
from omegaconf import OmegaConf
import hydra


def load_agent(checkpoint_path, agent_config_path):
    """Carga el agente entrenado desde un checkpoint."""
    agent_cfg = OmegaConf.load(agent_config_path)
    agent = hydra.utils.instantiate(agent_cfg)
    
    checkpoint = torch.load(checkpoint_path, map_location="cuda:0")
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()
    agent.cuda()
    return agent


def preprocess_image(bgr_img, size=(256, 256)):
    """Preprocesa una imagen BGR para el agente."""
    img = cv2.resize(bgr_img, size, interpolation=cv2.INTER_AREA)
    # BGR -> RGB, HWC -> CHW, normalizar a [0, 1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)  # (3, 256, 256)
    return img


def predict_action(agent, img_wrist, img_external, state, device="cuda:0"):
    """
    Predice la acción dados las dos imágenes y el estado.
    
    Args:
        agent: modelo entrenado
        img_wrist: imagen BGR de la cámara wrist
        img_external: imagen BGR de la cámara externa
        state: np.array (8,) [q_1..q_7, gripper]
    
    Returns:
        action: np.array con la acción predicha
    """
    # Preprocesar imágenes
    cam_0 = preprocess_image(img_wrist).unsqueeze(0).to(device)   # (1, 3, 256, 256)
    cam_1 = preprocess_image(img_external).unsqueeze(0).to(device)
    
    # Diccionario de imágenes — las claves deben coincidir con el entrenamiento
    imgs = {
        "cam_0": cam_0,
        "cam_1": cam_1,
    }
    
    # Estado
    obs_state = torch.from_numpy(state).float().unsqueeze(0).to(device)  # (1, 8)
    
    # Observación completa
    obs = (imgs, obs_state)
    
    with torch.no_grad():
        action = agent.predict(obs)
    
    return action.cpu().numpy().squeeze()
```

### Integración con ROS2 (nodo de inferencia)

```python
#!/usr/bin/env python3
"""Nodo de inferencia con dos cámaras para dit-policy."""

import numpy as np
import torch
import cv2
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray


class DitPolicyNode(Node):
    def __init__(self):
        super().__init__("dit_policy_node")
        
        self.bridge = CvBridge()
        self.device = "cuda:0"
        
        # Cargar agente
        self.agent = load_agent(
            checkpoint_path="bc_finetune/pick_two_cam/model_best.pth",
            agent_config_path="bc_finetune/pick_two_cam/agent_config.yaml",
        )
        
        # Últimos datos recibidos
        self.latest_wrist_img = None
        self.latest_ext_img = None
        self.latest_state = None
        
        # Subscripciones (las mismas que tu recorder)
        self.sub_wrist = self.create_subscription(
            Image, "/camera/camera_wrist/color/image_raw",
            self._cb_wrist, 1)
        self.sub_ext = self.create_subscription(
            Image, "/camera/camera_d405/color/image_raw",
            self._cb_ext, 1)
        self.sub_joints = self.create_subscription(
            JointState, "/joint_states",
            self._cb_joints, 1)
        
        # Publicador de acciones
        self.pub_action = self.create_publisher(
            Float64MultiArray, "/dit_policy/action", 1)
        
        # Timer de inferencia (10 Hz como la grabación)
        self.timer = self.create_timer(0.1, self._infer)
    
    def _cb_wrist(self, msg):
        self.latest_wrist_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
    
    def _cb_ext(self, msg):
        self.latest_ext_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
    
    def _cb_joints(self, msg):
        # Extraer q + gripper (adaptar a tu configuración)
        name_to_pos = dict(zip(msg.name, msg.position))
        arm_names = [f"fr3_joint{i}" for i in range(1, 8)]
        q = [name_to_pos[n] for n in arm_names]
        gripper = name_to_pos.get("fr3_finger_joint1", 0.0) + \
                  name_to_pos.get("fr3_finger_joint2", 0.0)
        self.latest_state = np.array(q + [gripper], dtype=np.float32)
    
    def _infer(self):
        if any(x is None for x in 
               [self.latest_wrist_img, self.latest_ext_img, self.latest_state]):
            return
        
        action = predict_action(
            self.agent,
            self.latest_wrist_img,
            self.latest_ext_img,
            self.latest_state,
            self.device,
        )
        
        msg = Float64MultiArray()
        msg.data = action.tolist()
        self.pub_action.publish(msg)
```

---

## 4. Verificación rápida

### Comprobar el buf.pkl generado

```python
import pickle
import numpy as np

with open("robobuf/buf.pkl", "rb") as f:
    buf = pickle.load(f)

print(f"Episodios: {len(buf)}")
print(f"Timesteps episodio 0: {len(buf[0])}")

obs, action, reward = buf[0][0]
print(f"State shape: {obs['state'].shape}")       # (8,)
print(f"Action shape: {action.shape}")             # (8,)
print(f"enc_cam_0 type: {type(obs['enc_cam_0'])}")  # ndarray (JPEG)
print(f"enc_cam_1 type: {type(obs['enc_cam_1'])}")  # ndarray (JPEG)
print(f"Claves obs: {list(obs.keys())}")           # ['state', 'enc_cam_0', 'enc_cam_1']

# Decodificar y visualizar una imagen
import cv2
img0 = cv2.imdecode(obs['enc_cam_0'], cv2.IMREAD_COLOR)
img1 = cv2.imdecode(obs['enc_cam_1'], cv2.IMREAD_COLOR)
print(f"Imagen cam0: {img0.shape}")  # (256, 256, 3)
print(f"Imagen cam1: {img1.shape}")  # (256, 256, 3)
```

---

## Resumen de cambios

| Componente | Cambio necesario |
|-----------|-----------------|
| Grabación (recorder) | Ya está listo — graba cam0 y cam1 |
| Conversión (robobuf) | Usar `convert_to_robobuf.py` — añade `enc_cam_0` + `enc_cam_1` |
| Config YAML (task) | Añadir `n_cams: 2` |
| Config YAML (agent) | Sin cambios — soporta N cámaras automáticamente |
| finetune.py | Sin cambios |
| Inferencia | Pasar diccionario con `cam_0` y `cam_1` al agente |

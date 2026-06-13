import os
import torch
import numpy as np
import pybullet as p
import pybullet_data
import cv2
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from transformers import AutoProcessor, AutoModelForVision2Seq
from src.model import SmolVLA

class FirstGuessSim:
    def __init__(self, model_path="robotmodel/models/checkpoints/latest.pt"):
        # 1. Connect PyBullet (Headless)
        self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Color mapping
        self.color_map = {
            "red": [1, 0, 0, 1], "green": [0, 1, 0, 1], "blue": [0, 0, 1, 1],
            "yellow": [1, 1, 0, 1], "purple": [1, 0, 1, 1], "cyan": [0, 1, 1, 1]
        }
        
        # Environment Setup (Matches sim_viz.py)
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        
        self.mat_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.001]),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.001], rgbaColor=[0.95, 0.95, 0.95, 1]),
            basePosition=[0.4, 0, 0.001]
        )
        
        self.gripper_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.04, 0.02]),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.04, 0.02], rgbaColor=[0.2, 0.2, 0.2, 1]),
            basePosition=[0.3, 0, 0.2]
        )

        # 2. Load Models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Loading Models...")
        self.processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
        self.vlm_base = AutoModelForVision2Seq.from_pretrained(
            "HuggingFaceTB/SmolVLM-256M-Instruct", torch_dtype=torch.bfloat16
        ).to(self.device).eval()
        
        self.vla_model = SmolVLA().to(self.device)
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint["model_state"]
            new_state_dict = {k[10:] if k.startswith("_orig_mod.") else k: v for k, v in state_dict.items()}
            self.vla_model.load_state_dict(new_state_dict)
            print(f"Loaded model from {model_path}")
        else:
            print("Warning: Model checkpoint not found. Using base weights.")
        self.vla_model.eval()

    def generate_first_guess(self, output_path="viz/first_guess.png"):
        # 1. Randomize Scene
        b_color = random.choice(list(self.color_map.keys()))
        t_color = random.choice([c for c in self.color_map.keys() if c != b_color])
        block_pos = [random.uniform(0.3, 0.45), random.uniform(-0.2, 0.2), 0.02]
        bucket_pos = [random.uniform(0.5, 0.65), random.uniform(-0.25, 0.25), 0.01]
        
        p.createMultiBody(
            baseMass=0.1,
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.015, 0.015, 0.015], rgbaColor=self.color_map[b_color]),
            basePosition=block_pos
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.06, 0.06, 0.01], rgbaColor=self.color_map[t_color]),
            basePosition=bucket_pos
        )
        
        instruction = f"put the {b_color} cube in the {t_color} bucket"
        print(f"Task: {instruction}")

        # Start Position (Randomized)
        actual_pos = np.array([random.uniform(0.2, 0.4), random.uniform(-0.2, 0.2), random.uniform(0.2, 0.4)])
        p.resetBasePositionAndOrientation(self.gripper_id, actual_pos, [0, 0, 0, 1])

        # 2. Get Observation (Matches sim_viz.py)
        obs_view_matrix = p.computeViewMatrix([-0.1, 0.4, 0.4], [0.4, 0.0, 0.1], [0, 0, 1])
        obs_proj_matrix = p.computeProjectionMatrixFOV(50, 1.0, 0.1, 10.0)
        (_, _, rgb, _, _) = p.getCameraImage(224, 224, obs_view_matrix, obs_proj_matrix, renderer=p.ER_TINY_RENDERER)
        rgb = np.reshape(rgb, (224, 224, 4))[:, :, :3]
        
        inputs = self.processor(images=[rgb], size={"longest_edge": 512}, return_tensors="pt").to(self.device, dtype=torch.bfloat16)
        with torch.no_grad():
            pixel_values = inputs.pixel_values.view(-1, 3, 512, 512)
            vision_outputs = self.vlm_base.model.vision_model(pixel_values=pixel_values)
            v_tokens = self.vlm_base.model.connector(vision_outputs.last_hidden_state)
            vision_emb = v_tokens.cpu().to(torch.float32).numpy().flatten()
            vision_emb = torch.tensor(vision_emb).unsqueeze(0).to(self.device)

        # 3. Predict Trajectory
        # State Norm (Matches sim_viz.py)
        norm_state = np.array([(actual_pos[0]-0.4)/0.2, actual_pos[1]/0.4, (actual_pos[2]-0.25)/0.25], dtype=np.float32)
        state_tensor = torch.tensor([[norm_state[0], norm_state[1], norm_state[2], 1.0]], dtype=torch.float32).to(self.device)
        input_ids = self.processor.tokenizer(instruction, return_tensors="pt", padding='max_length', max_length=32, truncation=True).input_ids.to(self.device)

        with torch.no_grad():
            pred_traj = self.vla_model.predict_action(vision_emb, state_tensor, input_ids, num_steps=16)
            trajectory = pred_traj.view(16, 4).cpu().numpy()

        # 4. De-normalize to physical meters
        Q01 = np.array([-0.0290, -0.0449, -0.0303], dtype=np.float32)
        Q99 = np.array([0.0273, 0.0456, 0.0522], dtype=np.float32)
        range_act = Q99 - Q01
        
        pred_pos_phys = (trajectory[:, :3] + 1.0) / 2.0 * range_act + Q01
        px = actual_pos[0] + np.cumsum(pred_pos_phys[:, 0])
        py = actual_pos[1] + np.cumsum(pred_pos_phys[:, 1])
        pz = actual_pos[2] + np.cumsum(pred_pos_phys[:, 2])

        # 5. Plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(px, py, pz, 'r-o', label='Predicted First Guess', markersize=4)
        ax.scatter(actual_pos[0], actual_pos[1], actual_pos[2], color='green', s=100, label='Start')
        ax.scatter(block_pos[0], block_pos[1], block_pos[2], color=self.color_map[b_color], s=50, label='Target Cube')
        ax.scatter(bucket_pos[0], bucket_pos[1], bucket_pos[2], color=self.color_map[t_color], s=150, alpha=0.3, label='Target Bucket')
        
        ax.set_xlabel('X (Forward)')
        ax.set_ylabel('Y (Left)')
        ax.set_zlabel('Z (Up)')
        ax.set_title(f'Model First Guess: {instruction}')
        ax.legend()
        
        os.makedirs("viz", exist_ok=True)
        plt.savefig(output_path)
        print(f"First guess visualization saved to {output_path}")
        p.disconnect()

if __name__ == "__main__":
    guess = FirstGuessSim()
    guess.generate_first_guess()

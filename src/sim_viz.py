import pybullet as p
import pybullet_data
import numpy as np
import cv2
import torch
import pandas as pd
import os
import random
from src.model import SmolVLA
from transformers import AutoTokenizer

class FloatingSim:
    def __init__(self, model_path=None):
        # Connect headlessly
        self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Color mapping for randomization
        self.color_map = {
            "red": [1, 0, 0, 1],
            "green": [0, 1, 0, 1],
            "blue": [0, 0, 1, 1],
            "yellow": [1, 1, 0, 1],
            "purple": [1, 0, 1, 1],
            "cyan": [0, 1, 1, 1]
        }
        
        # Initial Environment Setup
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        
        # Table (Static)
        self.table_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.3]),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.3], rgbaColor=[0.8, 0.8, 0.8, 1]),
            basePosition=[0.6, 0, 0.3]
        )
        
        # Create Floating Gripper (Simple Box)
        gripper_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.04, 0.02], rgbaColor=[0.2, 0.2, 0.2, 1])
        gripper_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.04, 0.02])
        self.gripper_id = p.createMultiBody(
            baseMass=0, # Kinematic/Floating
            baseCollisionShapeIndex=gripper_collision,
            baseVisualShapeIndex=gripper_visual,
            basePosition=[0.5, 0, 0.7]
        )
        
        # Dynamic Object Placeholders
        self.block_id = None
        self.bucket_id = None
        self.grasp_constraint = None
        
        # Load Models
        from transformers import AutoProcessor, AutoModelForVision2Seq
        print("Loading SmolVLM for real-time observations...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
        self.tokenizer = self.processor.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.vlm_base = AutoModelForVision2Seq.from_pretrained(
            "HuggingFaceTB/SmolVLM-256M-Instruct", 
            torch_dtype=torch.bfloat16
        ).to(self.device).eval()

        self.vla_model = None
        if model_path and os.path.exists(model_path):
            print(f"Loading VLA model from {model_path}...")
            self.vla_model = SmolVLA()
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint["model_state"]
            # Fix state dict mapping for compiled models
            new_state_dict = {k[10:] if k.startswith("_orig_mod.") else k: v for k, v in state_dict.items()}
            self.vla_model.load_state_dict(new_state_dict)
            self.vla_model.eval()
            self.vla_model.to(self.device)
        else:
            print(f"Warning: Model not found at {model_path}. Simulation will fail during inference.")

    def randomize_environment(self):
        """Randomizes colors, positions, and lighting for a new task."""
        b_color = random.choice(list(self.color_map.keys()))
        t_color = random.choice([c for c in self.color_map.keys() if c != b_color])
        
        block_pos = [random.uniform(0.45, 0.55), random.uniform(-0.15, 0.15), 0.62]
        bucket_pos = [random.uniform(0.65, 0.75), random.uniform(-0.25, 0.25), 0.61]
        
        if self.block_id is not None: p.removeBody(self.block_id)
        if self.bucket_id is not None: p.removeBody(self.bucket_id)
        if self.grasp_constraint is not None:
            p.removeConstraint(self.grasp_constraint)
            self.grasp_constraint = None
            
        self.block_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.015, 0.015, 0.015]),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.015, 0.015, 0.015], rgbaColor=self.color_map[b_color]),
            basePosition=block_pos
        )
        p.changeDynamics(self.block_id, -1, lateralFriction=2.0, rollingFriction=0.1)
        
        self.bucket_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.06, 0.06, 0.02]),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.06, 0.06, 0.02], rgbaColor=self.color_map[t_color]),
            basePosition=bucket_pos
        )
        
        instruction = f"put the {b_color} cube in the {t_color} bucket"
        return instruction, block_pos

    def get_observation(self):
        """Captures a 224x224 image from the observation camera (aligned to Bridge V2 dataset)."""
        # Common BridgeData camera pose: Side-ish view looking at center table
        obs_view_matrix = p.computeViewMatrix(
            cameraEyePosition=[0.3, -0.6, 0.8], # Closer and lower to match WidowX dataset style
            cameraTargetPosition=[0.5, 0.0, 0.5],
            cameraUpVector=[0, 0, 1]
        )
        obs_proj_matrix = p.computeProjectionMatrixFOV(fov=50, aspect=1.0, nearVal=0.1, farVal=10.0)
        
        (_, _, rgb, _, _) = p.getCameraImage(224, 224, obs_view_matrix, obs_proj_matrix, renderer=p.ER_TINY_RENDERER)
        rgb = np.reshape(rgb, (224, 224, 4))[:, :, :3]
        
        inputs = self.processor(images=[rgb], size={"longest_edge": 512}, return_tensors="pt").to(self.device, dtype=torch.bfloat16)
        with torch.no_grad():
            pixel_values = inputs.pixel_values.view(-1, inputs.pixel_values.shape[-3], inputs.pixel_values.shape[-2], inputs.pixel_values.shape[-1])
            vision_outputs = self.vlm_base.model.vision_model(pixel_values=pixel_values)
            v_tokens = self.vlm_base.model.connector(vision_outputs.last_hidden_state)
            vision_emb = v_tokens.cpu().to(torch.float32).numpy().flatten()
        
        # Need to return as tensor with batch dim for model
        return torch.tensor(vision_emb, dtype=torch.float32).unsqueeze(0).to(self.device)

    def run_simulation(self, output_video="viz/final_trajectory.mp4"):
        if self.vla_model is None:
            print("Error: VLA model not loaded.")
            return

        instruction, block_start_pos = self.randomize_environment()
        print(f"Executing Randomized Task: {instruction}")
        
        os.makedirs(os.path.dirname(output_video), exist_ok=True)
        width, height = 640, 480
        out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))
        
        # Studio Recording Camera
        rec_view_matrix = p.computeViewMatrix(
            cameraEyePosition=[1.2, -0.8, 1.0],
            cameraTargetPosition=[0.5, 0, 0.5],
            cameraUpVector=[0, 0, 1]
        )
        rec_proj_matrix = p.computeProjectionMatrixFOV(fov=45, aspect=float(width)/height, nearVal=0.1, farVal=100.0)

        input_ids = self.tokenizer(instruction, return_tensors="pt", padding='max_length', max_length=32, truncation=True).input_ids.to(self.device)

        # Start Position
        actual_pos = np.array([block_start_pos[0], block_start_pos[1], block_start_pos[2] + 0.1])
        p.resetBasePositionAndOrientation(self.gripper_id, actual_pos, [0, 0, 0, 1])
        
        # De-normalization stats (Exact high-precision)
        ACTION_MEAN = np.array([0.00043198, 0.00029432, 0.00088205], dtype=np.float32)
        ACTION_STD  = np.array([0.01033815, 0.01570009, 0.01481095], dtype=np.float32)

        for cycle in range(6):
            print(f"  Cycle {cycle+1}/6: Predicting...", end="", flush=True)
            vision_emb = self.get_observation()
            state_tensor = torch.tensor([[actual_pos[0], actual_pos[1], actual_pos[2], 1.0]], dtype=torch.float32).to(self.device)

            with torch.no_grad():
                # Use Euler solver for CFM inference
                pred_traj = self.vla_model.predict_action(vision_emb, state_tensor, input_ids, num_steps=16)
                trajectory = pred_traj.view(16, 4).cpu().numpy()
            print(" Done.", flush=True)

            for i in range(len(trajectory)):
                # De-normalize
                delta = (trajectory[i][:3] * ACTION_STD) + ACTION_MEAN
                gripper_val = trajectory[i][3] # 0 closed, 1 open
                
                actual_pos = actual_pos + delta
                p.resetBasePositionAndOrientation(self.gripper_id, actual_pos, [0, 0, 0, 1])
                
                # Grasp Logic
                if gripper_val < 0.5: # Requesting Closed
                    if self.grasp_constraint is None:
                        block_pos = p.getBasePositionAndOrientation(self.block_id)[0]
                        dist = np.linalg.norm(np.array(block_pos) - actual_pos)
                        if dist < 0.05:
                            # Create weld constraint
                            rel_pos = np.array(block_pos) - actual_pos
                            self.grasp_constraint = p.createConstraint(
                                self.gripper_id, -1, self.block_id, -1, 
                                p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], rel_pos
                            )
                else: # Requesting Open
                    if self.grasp_constraint is not None:
                        p.removeConstraint(self.grasp_constraint)
                        self.grasp_constraint = None
                
                for _ in range(2): # Step simulation
                    p.stepSimulation()
                    
                    # Record frame
                    (_, _, px, _, _) = p.getCameraImage(width, height, rec_view_matrix, rec_proj_matrix, renderer=p.ER_TINY_RENDERER)
                    rgb_array = cv2.cvtColor(np.reshape(px, (height, width, 4))[:, :, :3], cv2.COLOR_RGB2BGR)
                    out.write(rgb_array)
                
        out.release()
        print(f"Simulation saved to {output_video}")

if __name__ == "__main__":
    import sys
    sys.path.append(os.getcwd())
    
    sim = FloatingSim("robotmodel/models/checkpoints/latest.pt")
    
    for i in range(5):
        video_path = f"viz/sim_video_{i+1}.mp4"
        print(f"\n--- Generating Video {i+1}/5 ---")
        sim.run_simulation(video_path)

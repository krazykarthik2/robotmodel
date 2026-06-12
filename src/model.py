import torch
import torch.nn as nn
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import LoraConfig, get_peft_model

import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class FlowHead(nn.Module):
    def __init__(self, hidden_size, num_waypoints=16, state_dim=4, head_hidden_size=1024):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.state_dim = state_dim
        
        # Time Embedding (1 -> 128)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(128),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, 256)
        )
        
        # Action Projection (64 -> 256)
        self.action_proj = nn.Sequential(
            nn.Linear(num_waypoints * state_dim, 256),
            nn.GELU()
        )
        
        # Combined MLP
        # Input: hidden_size (576) + action_proj (256) + time_mlp (256) = 1088
        input_dim = hidden_size + 256 + 256
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, head_hidden_size),
            nn.GELU(),
            nn.Linear(head_hidden_size, head_hidden_size),
            nn.GELU(),
            nn.Linear(head_hidden_size, head_hidden_size),
            nn.GELU(),
            nn.Linear(head_hidden_size, num_waypoints * state_dim)
        )
        
    def forward(self, hidden_state, noisy_actions, tau):
        # hidden_state: [B, hidden_size]
        # noisy_actions: [B, 64]
        # tau: [B, 1]
        
        # Scale tau to [0, 1000] for standard sinusoidal embedding frequencies
        t_emb = self.time_mlp(tau.squeeze(-1) * 1000.0) # [B, 256]
        a_emb = self.action_proj(noisy_actions) # [B, 256]
        
        x = torch.cat([hidden_state, a_emb, t_emb], dim=-1)
        return self.mlp(x)

class SmolVLA(nn.Module):
    def __init__(self):
        super().__init__()
        # Load native SmolVLM-256M-Instruct base model
        self.vlm_model = AutoModelForVision2Seq.from_pretrained(
            "HuggingFaceTB/SmolVLM-256M-Instruct",
            torch_dtype=torch.bfloat16
        )
        
        # SmolVLM-256M text_model hidden size is 576
        self.hidden_size = self.vlm_model.config.text_config.hidden_size 
        
        # State Projection: 4 -> 1 token with increased capacity
        self.state_proj = nn.Sequential(
            nn.Linear(4, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, self.hidden_size)
        )
        
        # Flow Head (Replaces TrajectoryHead)
        self.flow_head = FlowHead(self.hidden_size)
        
        # Vision Projection: Map input embedding size (e.g. 768) to hidden_size (576)
        # This allows us to use different pre-extracted vision backbones (SigLIP, CLIP, etc.)
        self.vision_proj = nn.Linear(768, self.hidden_size)
        
        # LoRA Configuration
        self.apply_lora()
        
        # Ensure critical modules are trainable
        for param in self.state_proj.parameters():
            param.requires_grad = True
        for param in self.flow_head.parameters():
            param.requires_grad = True
        for param in self.vision_proj.parameters():
            param.requires_grad = True
            
        # Native Multi-modal Connector (Projector)
        for param in self.vlm_model.model.connector.parameters():
            param.requires_grad = True
            
        # Unfreeze last 4 layers of the language model
        layers = self.vlm_model.model.text_model.layers
        for i in range(len(layers) - 4, len(layers)):
            for param in layers[i].parameters():
                param.requires_grad = True

    def apply_lora(self):
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=None 
        )
        self.vlm_model.model.text_model = get_peft_model(self.vlm_model.model.text_model, lora_config)

    def forward(self, vision_embeddings, state, input_ids, noisy_actions=None, tau=None):
        # vision_embeddings: [B, D] where D might be 768 or N_tokens * 576
        # state: [B, 4]
        # input_ids: [B, N]
        
        batch_size = vision_embeddings.shape[0]
        
        # 1. Vision Tokens
        # If D is 768, project to [B, 1, 576]
        # If D is N*576, reshape to [B, N, 576]
        if vision_embeddings.shape[1] == 768:
            v_tokens = self.vision_proj(vision_embeddings.to(dtype=torch.float32)).unsqueeze(1).to(dtype=torch.bfloat16)
        else:
            n_vision_tokens = vision_embeddings.shape[1] // self.hidden_size
            v_tokens = vision_embeddings.view(batch_size, n_vision_tokens, self.hidden_size).to(dtype=torch.bfloat16)
        
        # 2. State Token
        s_tokens = self.state_proj(state.to(dtype=torch.float32)).unsqueeze(1).to(dtype=torch.bfloat16)
        
        # 3. Word Embeddings
        w_tokens = self.vlm_model.model.text_model.get_input_embeddings()(input_ids)
        
        # Sequence: [VISION] [STATE] [INSTRUCTIONS]
        tokens = torch.cat([v_tokens, s_tokens, w_tokens], dim=1)
        
        # Backbone Forward
        outputs = self.vlm_model.model.text_model(inputs_embeds=tokens)
        last_hidden_state = outputs.last_hidden_state # [B, Seq_Len, hidden_size]
        
        # Use the last token's hidden state for flow matching
        hidden_state = last_hidden_state[:, -1, :].to(dtype=torch.float32)
        
        if noisy_actions is not None and tau is not None:
            # Training mode: Predict velocity field
            velocity = self.flow_head(hidden_state, noisy_actions.to(dtype=torch.float32), tau.to(dtype=torch.float32))
            return velocity
        
        return hidden_state # Return hidden state for inference solver

    @torch.no_grad()
    def predict_action(self, vision_embeddings, state, input_ids, num_steps=8):
        """Inference using Euler integration for Conditional Flow Matching."""
        self.eval()
        device = vision_embeddings.device
        batch_size = vision_embeddings.shape[0]
        
        # 1. Get hidden state from VLM (only once)
        hidden_state = self.forward(vision_embeddings, state, input_ids)
        
        # 2. Initialize with Gaussian noise: x_0 ~ N(0, I)
        x = torch.randn(batch_size, 16 * 4, device=device)
        
        # 3. Euler integration from tau=0 to tau=1
        dt = 1.0 / num_steps
        for i in range(num_steps):
            tau = torch.ones(batch_size, 1, device=device) * (i * dt)
            velocity = self.flow_head(hidden_state, x, tau)
            x = x + velocity * dt
            
        return x

if __name__ == "__main__":
    # Test model initialization
    model = SmolVLA()
    print("Model initialized.")
    # Check trainable parameters
    trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"Number of trainable parameters: {len(trainable_params)}")

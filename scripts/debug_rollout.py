import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import transformers.utils
if not hasattr(transformers.utils, 'torch_compilable_check'):
    transformers.utils.torch_compilable_check = lambda f: f
import huggingface_hub
huggingface_hub.sync_bucket = lambda *args, **kwargs: None
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
import torch
from src.env_pybullet import Robot4DofEnv

env = Robot4DofEnv(gui=False, seed=0)
env.reset(task='pick', seed=0, prompt='pick the red cube')

policy = SmolVLAPolicy.from_pretrained('models/smolvla_lora')
policy.config.adapt_to_pi_aloha = False
tok = policy.model.vlm_with_expert.processor.tokenizer
enc = tok('pick the red cube', return_tensors='pt')

for step in range(8):
    obs = env.observe()
    batch = {
        'observation.images.camera1': torch.from_numpy(obs['observation.images.front']).permute(2,0,1).float().unsqueeze(0)/255.0,
        'observation.state': torch.from_numpy(obs['observation.state']).float().unsqueeze(0),
        'observation.language.tokens': enc['input_ids'],
        'observation.language.attention_mask': enc['attention_mask'].bool(),
    }
    act = policy.select_action(batch)
    a = act[0, :5].detach().numpy()
    obs, r, d, info = env.step(a)
    tcp = info['tcp']
    cube = info['cube']
    dist = float(((tcp - cube)**2).sum()**0.5)
    print("Step %d: TCP=[%.3f, %.3f, %.3f], Cube=[%.3f, %.3f, %.3f], dist=%.3f, act_g=%.2f, picked=%s" % (
        step, tcp[0], tcp[1], tcp[2], cube[0], cube[1], cube[2], dist, a[4], info['picked']
    ))
env.close()

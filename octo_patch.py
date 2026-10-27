import os
import sys

# Prevent network timeout checks for local huggingface cache
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# 1. IMPORT TRANSFORMERS FIRST
import transformers
from transformers import AutoConfig, FlaxT5EncoderModel

# 2. IMPORT JAX
import jax
import jax.numpy as jnp
import jax.tree_util
import jax.interpreters.xla

# Monkey patch missing or deprecated JAX internals for Octo
if not hasattr(jax.interpreters.xla, 'pytype_aval_mappings'):
    jax.interpreters.xla.pytype_aval_mappings = jax.core.pytype_aval_mappings
if not hasattr(jax.random, 'KeyArray'):
    jax.random.KeyArray = jax.Array
if not hasattr(jax, 'tree_map'):
    jax.tree_map = jax.tree_util.tree_map
if not hasattr(jax, 'tree_leaves'):
    jax.tree_leaves = jax.tree_util.tree_leaves

# 3. FIX PROTOBUF RUNTIME VERSION FOR TFDS
import google.protobuf
if not hasattr(google.protobuf, 'runtime_version'):
    class _MockDomain:
        PUBLIC = 1
        INTERNAL = 2
    class _MockRuntimeVersion:
        Domain = _MockDomain
        @staticmethod
        def ValidateProtobufRuntimeVersion(*args, **kwargs):
            return True
    google.protobuf.runtime_version = _MockRuntimeVersion()

# 4. IMPORT TENSORFLOW
try:
    import tensorflow as tf
    if hasattr(tf, "__version__"):
        parts = tf.__version__.split(".")
        if len(parts) >= 2 and parts[0] == "2" and int(parts[1]) < 18:
            tf.__version__ = "2.18.0"
except Exception:
    pass

# Patch LanguageTokenizer cleanly
try:
    import octo.model.components.tokenizers as _tok
    def _safe_setup(self):
        if self.encoder is not None:
            config = AutoConfig.from_pretrained(self.encoder)
            self.hf_model = FlaxT5EncoderModel(config).module
    _tok.LanguageTokenizer.setup = _safe_setup
except Exception:
    pass

# Patch Orbax deserialize_tree for FlaxT5EncoderModule_0 name mismatch
def _apply_orbax_patch():
    # Patch across all potential import locations in orbax.checkpoint
    modules_to_patch = []
    try:
        import orbax.checkpoint.utils as u1
        modules_to_patch.append(u1)
    except Exception:
        pass
    try:
        import orbax.checkpoint._src.tree.utils as u2
        modules_to_patch.append(u2)
    except Exception:
        pass

    for mod in modules_to_patch:
        if hasattr(mod, '_reconstruct_from_keypath'):
            _orig = getattr(mod, '_reconstruct_from_keypath')
            def make_patched(orig_fn):
                def _reconstruct_from_keypath_patched(entry, target):
                    # Handle dictionary key lookup safely
                    try:
                        res = entry
                        for key in target:
                            k_name = getattr(key, 'key', key)
                            if k_name in res:
                                res = res[k_name]
                            elif 'FlaxT5EncoderModule' in str(k_name) or k_name == 'FlaxT5EncoderModule_0':
                                continue
                            elif 'hf_model' in res:
                                res = res['hf_model']
                            else:
                                return entry
                        return res
                    except Exception:
                        return entry
                return _reconstruct_from_keypath_patched
            setattr(mod, '_reconstruct_from_keypath', make_patched(_orig))

_apply_orbax_patch()

_orig_clip = jnp.clip
def _patched_clip(a, a_min=None, a_max=None, *args, **kwargs):
    min_val = kwargs.pop('min', a_min)
    max_val = kwargs.pop('max', a_max)
    return _orig_clip(a, min=min_val, max=max_val, *args, **kwargs)
jnp.clip = _patched_clip

import octo.utils.typing
octo.utils.typing.PRNGKey = jax.Array

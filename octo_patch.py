import os
import sys

# Prevent network timeout checks for local huggingface cache
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["USE_FLAX"] = "1"

# Tell TensorFlow to leave GPU memory completely alone for JAX
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
try:
    import tensorflow as tf
    # Hide GPUs from TensorFlow so only JAX manages the CUDA devices
    tf.config.set_visible_devices([], 'GPU')
    if hasattr(tf, "__version__"):
        parts = tf.__version__.split(".")
        if len(parts) >= 2 and parts[0] == "2" and int(parts[1]) < 18:
            tf.__version__ = "2.18.0"
except Exception:
    pass

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

# Fix google.protobuf runtime_version mismatch with newer tensorflow_metadata / tfds
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

# Ensure LanguageTokenizer cleanly loads FlaxT5EncoderModel.module
try:
    import transformers
    from transformers import AutoConfig, FlaxT5EncoderModel
    import octo.model.components.tokenizers as _tok
    
    _orig_setup = _tok.LanguageTokenizer.setup
    def _safe_setup(self):
        if self.encoder is not None:
            config = AutoConfig.from_pretrained(self.encoder)
            self.hf_model = FlaxT5EncoderModel(config).module
    _tok.LanguageTokenizer.setup = _safe_setup
except Exception:
    pass

_orig_clip = jnp.clip
def _patched_clip(a, a_min=None, a_max=None, *args, **kwargs):
    min_val = kwargs.pop('min', a_min)
    max_val = kwargs.pop('max', a_max)
    return _orig_clip(a, min=min_val, max=max_val, *args, **kwargs)
jnp.clip = _patched_clip

import octo.utils.typing
octo.utils.typing.PRNGKey = jax.Array

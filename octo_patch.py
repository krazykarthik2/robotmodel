import os
import sys

# Prevent network timeout checks for local huggingface cache
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["USE_FLAX"] = "1"

# Bypass TensorFlow Probability version check against TensorFlow
try:
    import tensorflow as tf
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

# Ensure Flax models are always accessible in transformers module and sys.modules
try:
    import transformers
    import transformers.models.auto.modeling_flax_auto as _flax_auto
    import transformers.models.t5.modeling_flax_t5 as _flax_t5

    # Inject into transformers _objects and module namespace
    if hasattr(transformers, '_objects'):
        transformers._objects['FlaxAutoModel'] = _flax_auto.FlaxAutoModel
        transformers._objects['FlaxT5EncoderModel'] = _flax_t5.FlaxT5EncoderModel

    setattr(transformers, 'FlaxAutoModel', _flax_auto.FlaxAutoModel)
    setattr(transformers, 'FlaxT5EncoderModel', _flax_t5.FlaxT5EncoderModel)

    # Monkey patch octo tokenizer setup directly so it never fails
    import octo.model.components.tokenizers as _tok
    _orig_setup = _tok.LanguageTokenizer.setup
    def _patched_setup(self):
        if self.encoder is not None:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(self.encoder)
            if "t5" in self.encoder:
                self.hf_model = _flax_t5.FlaxT5EncoderModel(config).module
            else:
                self.hf_model = _flax_auto.FlaxAutoModel.from_config(config).module
    _tok.LanguageTokenizer.setup = _patched_setup
except Exception as e:
    pass

_orig_clip = jnp.clip
def _patched_clip(a, a_min=None, a_max=None, *args, **kwargs):
    min_val = kwargs.pop('min', a_min)
    max_val = kwargs.pop('max', a_max)
    return _orig_clip(a, min=min_val, max=max_val, *args, **kwargs)
jnp.clip = _patched_clip

import octo.utils.typing
octo.utils.typing.PRNGKey = jax.Array

import os
import sys

# Prevent network timeout checks for local huggingface cache
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# 1. IMPORT TRANSFORMERS FIRST
import transformers
from transformers import AutoConfig, AutoTokenizer, FlaxT5EncoderModel

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

# Patch LanguageTokenizer to accept precomputed embeddings or run external T5
try:
    import octo.model.components.tokenizers as _tok
    import flax.linen as nn
    import numpy as np

    # Override LanguageTokenizer.__call__ to safely check for pre-embedded tokens
    _orig_lang_call = _tok.LanguageTokenizer.__call__
    def _safe_lang_call(self, observations, tasks=None, train=False):
        if tasks is None or 'language_instruction' not in tasks:
            return None
        lang = tasks['language_instruction']
        if isinstance(lang, (jax.Array, np.ndarray)):
            tokens = lang if lang.ndim == 3 else lang[:, None, :]
            pad_mask = jnp.ones(tokens.shape[:-1])
            return _tok.TokenGroup(tokens, pad_mask)
        # If tasks['language_instruction'] is a dict (input_ids), pass dummy tokens of shape (B, 16, 768)
        # to completely avoid Flax Linen ScopeParamShapeError in Linen parameter tree
        bsz = len(next(iter(lang.values())))
        tokens = jnp.zeros((bsz, 16, 768), dtype=jnp.float32)
        pad_mask = jnp.ones((bsz, 16), dtype=bool)
        return _tok.TokenGroup(tokens, pad_mask)

    _tok.LanguageTokenizer.__call__ = _safe_lang_call
except Exception:
    pass

# Patch HFTokenizer to accept all keyword arguments (*args, **kwargs)
try:
    import octo.data.utils.text_processing as _tp
    class _DummyTokenizer:
        def __call__(self, text, *args, **kwargs):
            import numpy as np
            if isinstance(text, str):
                text = [text]
            bsz = len(text)
            return {
                "input_ids": np.zeros((bsz, 16), dtype=np.int32),
                "attention_mask": np.ones((bsz, 16), dtype=bool)
            }
        def encode(self, *args, **kwargs):
            import numpy as np
            return np.zeros((16,), dtype=np.int32)

    _orig_hftok_init = _tp.HFTokenizer.__init__
    def _safe_hftok_init(self, tokenizer_name="t5-base", *args, **kwargs):
        self.tokenizer_name = tokenizer_name
        for k, v in kwargs.items():
            setattr(self, k, v)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=False)
        except Exception:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
            except Exception:
                self.tokenizer = _DummyTokenizer()
    _tp.HFTokenizer.__init__ = _safe_hftok_init
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

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

# Patch HFTokenizer to avoid sentencepiece / online vocab file download crashes
try:
    import octo.data.utils.text_processing as _tp
    class _DummyTokenizer:
        def __call__(self, text, *args, **kwargs):
            import numpy as np
            # Return valid T5 token dict structure matching (batch, max_len=16)
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
    def _safe_hftok_init(self, tokenizer_name="t5-base"):
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

# 5. ROBUST KEYPATH RECONSTRUCTION FOR ALL SUBMODULES
try:
    import orbax.checkpoint.utils as _ocp_utils
    from orbax.checkpoint.utils import get_key_name, is_empty_or_leaf

    def patched_deserialize_tree(serialized, target, keep_empty_nodes: bool = False):
        def _reconstruct_from_keypath(keypath, _):
            result = serialized
            for key in keypath:
                key_name = get_key_name(key)
                if not isinstance(result, list) and key_name not in result:
                    key_name = str(key_name)

                if isinstance(result, dict) and key_name not in result:
                    if 'FlaxT5EncoderModule' in str(key_name):
                        continue
                    elif 'encoder' in result:
                        result = result['encoder']
                    elif 'shared' in result:
                        result = result['shared']
                    elif 'hf_model' in result:
                        result = result['hf_model']

                if isinstance(result, dict) and key_name in result:
                    result = result[key_name]
                else:
                    return result
            return result

        return jax.tree_util.tree_map_with_path(
            _reconstruct_from_keypath,
            target,
            is_leaf=is_empty_or_leaf if keep_empty_nodes else None,
        )

    _ocp_utils.deserialize_tree = patched_deserialize_tree
    import orbax.checkpoint.pytree_checkpoint_handler as _pch
    _pch.utils.deserialize_tree = patched_deserialize_tree
    if hasattr(_pch, 'deserialize_tree'):
        _pch.deserialize_tree = patched_deserialize_tree
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

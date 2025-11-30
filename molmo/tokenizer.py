# import ujson as json
import logging
import os
from os import environ
from pathlib import Path
from typing import List, Optional, Union, Dict

# Set HuggingFace cache directory BEFORE importing transformers
# This ensures environment variables are set before transformers reads them
if "HF_HOME" in os.environ:
    hf_home = os.environ["HF_HOME"]
    hf_hub_cache = str(Path(hf_home) / "hub")
    os.environ["HF_HUB_CACHE"] = hf_hub_cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache
    os.environ["TRANSFORMERS_CACHE"] = str(Path(hf_home) / "transformers")
    # Ensure directories exist
    Path(hf_home).mkdir(parents=True, exist_ok=True)
    Path(hf_hub_cache).mkdir(parents=True, exist_ok=True)
    (Path(hf_home) / "transformers").mkdir(parents=True, exist_ok=True)

from transformers import AutoTokenizer

from molmo.torch_util import get_local_rank, barrier
from molmo.util import is_url

try:
    from functools import cache
except ImportError:
    from functools import lru_cache as cache

# Special tokens, these should be present in any tokenizer we use since the preprocessor uses them
# Note: These match the tokens defined in molmo_hf/molmo/preprocessors/preprocessing_molmo.py
DEFAULT_IMAGE_PATCH_TOKEN = f"<im_patch>"
DEFAULT_IM_START_TOKEN = f"<im_start>"
DEFAULT_IM_END_TOKEN = f"<im_end>"
DEFAULT_IM_COL_TOKEN = f"<im_col>"
IMAGE_PROMPT = "<|image|>"

EXTRA_TOKENS = (DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN,
                DEFAULT_IM_COL_TOKEN, IMAGE_PROMPT)


class HfTokenizerWrapper:
    """Tokenizer wrapper

    This exists mostly for legacy reasons since we used to support other kinds of tokenizers
    with different API
    """
    def __init__(self, tokenizer, bos_token_id=None, adds_space=False):
        self.adds_space = adds_space
        self.tokenizer = tokenizer
        if bos_token_id is None:
            self.bos_token_id = tokenizer.bos_token_id
        else:
            self.bos_token_id = bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_id = -1

    def encode(self, x: str):
        return self.tokenizer.encode(x, add_special_tokens=False)

    def decode(self, x: List[int], truncate_at_eos=True):
        x = [int(t) for t in x]

        if self.eos_token_id == self.bos_token_id and (len(x) > 0 and x[0] == self.eos_token_id):
            # Assume an EOS at the start is functioning as BOS
            x = x[1:]

        if truncate_at_eos:
            # Follow seqio and automatically cut off at EOS
            try:
                eos_ix = x.index(self.eos_token_id)
                x = x[:eos_ix]
            except ValueError:
                pass
        else:
            # Keep our special tokens, but skip BOS/EOS
            x = [t for t in x if t != self.eos_token_id and t != self.bos_token_id]
        return self.tokenizer.decode(x)

    def vocab_size(self):
        return len(self.tokenizer)


def build_tokenizer(
    tokenizer_type, has_extra_token=True,
    tokenizer_dir="gs://mm-olmo/tokenizer",
    pad_tokenizer_to=None,
    memory_cache={}
) -> HfTokenizerWrapper:
    cache_key = (tokenizer_type, has_extra_token, pad_tokenizer_to)
    if cache_key in memory_cache:
        return memory_cache[cache_key]

    # Map legacy tokenizer IDs to current ones
    # If tokenizer_type is OLMoE-1B-7B-0924, try using MolmoE-1B-0924 instead
    # (they should use the same tokenizer)
    original_tokenizer_type = tokenizer_type
    fallback_tokenizer = None
    if "OLMoE-1B-7B-0924" in tokenizer_type:
        # Try MolmoE-1B-0924 as fallback (same tokenizer, but model exists on HF)
        fallback_tokenizer = tokenizer_type.replace("OLMoE-1B-7B-0924", "MolmoE-1B-0924")
        logging.info(f"Tokenizer {tokenizer_type} may not be recognized, will try {fallback_tokenizer} as fallback")

    # If tokenizer_dir is URL or None, use HF_HOME environment variable if available
    if tokenizer_dir is None or is_url(tokenizer_dir):
        # Use HF_HUB_CACHE if set, otherwise HF_HOME/hub, otherwise None
        cache_dir = os.environ.get("HF_HUB_CACHE")
        if not cache_dir and "HF_HOME" in os.environ:
            cache_dir = str(Path(os.environ["HF_HOME"]) / "hub")
            os.environ["HF_HUB_CACHE"] = cache_dir
            os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
        if cache_dir:
            # Ensure directory exists
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            logging.info(f"build_tokenizer: Using cache_dir={cache_dir} (from HF_HOME={os.environ.get('HF_HOME')})")
        else:
            logging.warning(f"build_tokenizer: cache_dir is None, HF_HOME={os.environ.get('HF_HOME')}, HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE')}")
    else:
        cache_dir = tokenizer_dir

    # Stop multiple processes on one node trying to download and cache the tokenizer
    # files, which seems to rarely cause an error
    if get_local_rank() == 0:
        # For Molmo models, use AutoProcessor instead of AutoTokenizer
        # AutoProcessor includes the tokenizer and works better with custom models
        tokenizer = None
        tokenizer_errors = []
        
        # Try using AutoProcessor first (works for MolmoE models)
        # This is the same approach as test_hf_molmoe.py
        try:
            from transformers import AutoProcessor
            # Try original tokenizer_type first, then fallback if available
            tokenizer_types_to_try = [tokenizer_type]
            if fallback_tokenizer:
                tokenizer_types_to_try.append(fallback_tokenizer)
            
            for try_type in tokenizer_types_to_try:
                if try_type is None:
                    continue
                try:
                    processor = AutoProcessor.from_pretrained(
                        try_type,
                        trust_remote_code=True,
                        torch_dtype="auto",
                        device_map="auto",
                    )
                    # Extract tokenizer from processor
                    tokenizer = processor.tokenizer
                    if try_type != tokenizer_type:
                        logging.info(f"Successfully loaded tokenizer via AutoProcessor from {try_type} (fallback from {tokenizer_type})")
                    else:
                        logging.info(f"Successfully loaded tokenizer via AutoProcessor from {tokenizer_type}")
                    break
                except Exception as e:
                    tokenizer_errors.append(f"AutoProcessor({try_type}): {str(e)}")
                    logging.debug(f"Failed to load via AutoProcessor from {try_type}: {e}")
                    continue
        except ImportError:
            logging.warning("AutoProcessor not available, falling back to AutoTokenizer")
        except Exception as e:
            tokenizer_errors.append(f"AutoProcessor import/usage: {str(e)}")
        
        # Fallback to AutoTokenizer if AutoProcessor failed
        if tokenizer is None:
            # Try original tokenizer_type first, then fallback if available
            tokenizer_types_to_try = [tokenizer_type]
            if fallback_tokenizer:
                tokenizer_types_to_try.append(fallback_tokenizer)
            
            for try_type in tokenizer_types_to_try:
                if try_type is None:
                    continue
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        try_type,
                        token=environ.get("HF_ACCESS_TOKEN"),
                        cache_dir=cache_dir,
                        trust_remote_code=True,  # Required for custom tokenizers
                    )
                    if try_type != tokenizer_type:
                        logging.info(f"Successfully loaded tokenizer from {try_type} (fallback from {tokenizer_type})")
                    break
                except Exception as e:
                    tokenizer_errors.append(f"AutoTokenizer({try_type}): {str(e)}")
                    logging.debug(f"Failed to load tokenizer from {try_type}: {e}")
                    continue
        
        if tokenizer is None:
            # Last attempt: try without trust_remote_code
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_type,
                    token=environ.get("HF_ACCESS_TOKEN"),
                    cache_dir=cache_dir,
                )
            except Exception as e:
                error_msg = f"Failed to load tokenizer '{tokenizer_type}'. Errors: {'; '.join(tokenizer_errors)}; Final attempt: {str(e)}"
                logging.error(error_msg)
                raise RuntimeError(error_msg) from e
    barrier()

    extra_tokens = list(EXTRA_TOKENS)
    if pad_tokenizer_to is not None:
        # Use trust_remote_code for custom tokenizers
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_type, 
                token=environ.get("HF_ACCESS_TOKEN"), 
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_type, 
                token=environ.get("HF_ACCESS_TOKEN"), 
                cache_dir=cache_dir,
            )
        assert len(tokenizer) <= pad_tokenizer_to
        n_extra_tokens = pad_tokenizer_to - len(tokenizer)
        # This handles a case where the LLM embedding matrix is larger than the vocab size
        # We need the extra tokens in `EXTRA_TOKENS` to be assigned id's higher than the embedding
        # matrix size, not the vocab size, since we will concat the embedding and matrix with
        # the special token embedding matrix, so we pad the vocab with additional special tokens
        if n_extra_tokens > 0:
            logging.info(f"Padding tokenizer with {n_extra_tokens} tokens")
            extra_tokens = [f"|<EXTRA_TOKENS_{i}>|" for i in range(n_extra_tokens)] + extra_tokens

    bos_token_id = None

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_type, additional_special_tokens=extra_tokens,
        token=environ.get("HF_ACCESS_TOKEN"),
        cache_dir=cache_dir,
    )
    if ("qwen2" in tokenizer_type.lower()) or ("olmo" in tokenizer_type.lower()):
        # These tokenizers do not have a BOS, and instead use EOS as a generic seperator token.
        # In this case we will use EOS as BOS
        assert tokenizer.bos_token_id is None
        bos_token_id = tokenizer.eos_token_id

    tok = HfTokenizerWrapper(tokenizer, bos_token_id=bos_token_id, adds_space=False)
    memory_cache[cache_key] = tok
    return tok


def get_special_token_ids(tokenizer):
    """
    Get special token IDs. This function is compatible with both HfTokenizerWrapper
    and standard transformers tokenizers.
    
    Note: This implementation is compatible with molmo_hf's preprocessing_molmo.py
    which uses add_special_tokens=False.
    """
    if isinstance(tokenizer, HfTokenizerWrapper):
        ids = tokenizer.encode("".join(EXTRA_TOKENS))
        if len(ids) == len(EXTRA_TOKENS) + 1:
            ids = ids[1:]
    else:
        # For standard transformers tokenizers, use add_special_tokens=False
        # to match molmo_hf's preprocessing_molmo.py implementation
        ids = tokenizer.encode("".join(EXTRA_TOKENS), add_special_tokens=False)
        # Fallback if that doesn't work
        if len(ids) != len(EXTRA_TOKENS):
            ids = tokenizer.encode(" ".join(EXTRA_TOKENS), add_special_tokens=False)

    assert len(ids) == len(EXTRA_TOKENS), f"Expected {len(EXTRA_TOKENS)} token IDs, got {len(ids)}"
    return {k: i for k, i in zip(EXTRA_TOKENS, ids)}


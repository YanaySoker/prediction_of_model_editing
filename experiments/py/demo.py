# New demo.py
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from baselines.ft import FTHyperParams, apply_ft_to_model
from rome import ROMEHyperParams, apply_rome_to_model, apply_rome_to_model_k_and_v
# from ...rome.rome_main import ROMEHyperParams, apply_rome_to_model, apply_rome_to_model_k_and_v
from util import nethook
from util.generate import generate_fast
from util.globals import *
from config import prefix

LAYER_IDX = [None]

def demo_model_editing(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    generation_prompts: List[str],
    alg_name: str = "ROME",
    pre_k_and_v = None,            # Yanay. 
) -> Tuple[AutoModelForCausalLM, Dict[str, torch.Tensor]]:
    """
    Applies the selected model editing algorithm. Generates text both before and after
    for comparison of model behavior. Returns the updated model and the original values of
    weights that were changed.
    """
    print("\n", requests)
    nethook.set_requires_grad(True, model)

    RewritingParamsClass, apply_method, hparams_prefix, hparams_suffix = load_alg(
        alg_name
    )
    params_name = (
        HPARAMS_DIR
        / hparams_prefix
        / f"{model.config._name_or_path.replace('/', '_')}{hparams_suffix}.json"
    )
    
    params_name = prefix + "hparams/ROME/gpt2-xl.json"
    hparams = RewritingParamsClass.from_json(params_name)
    print("layer index:", LAYER_IDX[0])
    hparams.layers=[LAYER_IDX[0]]   # New
    model_new, orig_weights = apply_method(
        model, tok, requests, hparams, return_orig_weights=True, pre_k_and_v=pre_k_and_v  # Yanay. origin: model, tok, requests, hparams, return_orig_weights=True
    )

    return model_new, orig_weights


# Yanay
def demo_model_editing_k_and_v(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    generation_prompts: List[str],
    alg_name: str = "ROME_k_v",
) -> Tuple[AutoModelForCausalLM, Dict[str, torch.Tensor]]:
    """
    Applies the selected model editing algorithm. Generates text both before and after
    for comparison of model behavior. Returns the updated model and the original values of
    weights that were changed.
    """
    
    nethook.set_requires_grad(True, model)
    
    alg_name = "ROME_k_v"
    RewritingParamsClass, apply_method, hparams_prefix, hparams_suffix = load_alg(
        alg_name
    )
#     params_name = (
#         HPARAMS_DIR
#         / hparams_prefix
#         / f"{model.config._name_or_path.replace('/', '_')}{hparams_suffix}.json"
#     )

    params_name = prefix + "hparams/ROME/gpt2-xl.json"
    hparams = RewritingParamsClass.from_json(params_name)
    print("layer index:", LAYER_IDX[0])
    hparams.layers=[LAYER_IDX[0]]   # New
    print("hparams:", hparams)

    model_new, orig_weights, k_and_v = apply_method(             # yanay. origin: model_new, orig_weights = apply_method(
        model, tok, requests, hparams, return_orig_weights=True
    )

    return orig_weights, k_and_v


def load_alg(alg_name):
    """
    Loads dependencies for the desired algorithm.
    Implementation is slightly awkward to prevent unnecessary imports on Colab.

    The return value is a tuple of the following:
    1. Class for storing hyperparameters
    2. Method for applying rewrites
    3. Location of parameters
    4. Predefined suffix for the param file
    """
    assert alg_name in [
        "FT",
        "FT-L",
        "FT-AttnEdit",
        "KN",
        "MEND",
        "MEND-CF",
        "MEND-zsRE",
        "KE",
        "KE-CF",
        "ROME",
        "ROME_k_v",
    ]

    if alg_name == "ROME":
        return ROMEHyperParams, apply_rome_to_model, "ROME", ""
    elif alg_name == "ROME_k_v":
        return ROMEHyperParams, apply_rome_to_model_k_and_v, "ROME", ""
    elif "FT" in alg_name:
        d = {
            "FT": (FTHyperParams, apply_ft_to_model, "FT", "_unconstr"),
            "FT-AttnEdit": (FTHyperParams, apply_ft_to_model, "FT", "_attn"),
            "FT-L": (FTHyperParams, apply_ft_to_model, "FT", "_constr"),
        }
        return d[alg_name]
    else:
        from baselines.efk import EFKHyperParams, EfkRewriteExecutor
        from baselines.kn import KNHyperParams, apply_kn_to_model
        from baselines.mend import MENDHyperParams, MendRewriteExecutor

        d = {
            "KN": (KNHyperParams, apply_kn_to_model, "KN", ""),
            "MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model, "MEND", ""),
            "KE": (EFKHyperParams, EfkRewriteExecutor().apply_to_model, "KE", ""),
            "MEND-CF": (
                MENDHyperParams,
                MendRewriteExecutor().apply_to_model,
                "MEND",
                "_CF",
            ),
            "MEND-zsRE": (
                MENDHyperParams,
                MendRewriteExecutor().apply_to_model,
                "MEND",
                "_zsRE",
            ),
            "KE-CF": (
                EFKHyperParams,
                EfkRewriteExecutor().apply_to_model,
                "MEND",
                "_CF",
            ),
        }
        return d[alg_name]

def print_loud(x, pad=3):
    """
    Prints a string with # box for emphasis.

    Example:
    ############################
    #                          #
    #  Applying ROME to model  #
    #                          #
    ############################
    """

    n = len(x)
    print()
    print("".join(["#" for _ in range(n + 2 * pad)]))
    print("#" + "".join([" " for _ in range(n + 2 * (pad - 1))]) + "#")
    print(
        "#"
        + "".join([" " for _ in range(pad - 1)])
        + x
        + "".join([" " for _ in range(pad - 1)])
        + "#"
    )
    print("#" + "".join([" " for _ in range(n + 2 * (pad - 1))]) + "#")
    print("".join(["#" for _ in range(n + 2 * pad)]))

class StopExecution(Exception):
    def _render_traceback_(self):
        pass

def stop_execution():
    raise StopExecution

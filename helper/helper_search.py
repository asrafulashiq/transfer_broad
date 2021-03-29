import os
from typing import List
import pandas as pd
import re
import parse
import pickle
import os, glob
import numpy as np
import copy
from loguru import logger
from omegaconf import OmegaConf


def extract_hp(file_name: str):
    # extract hp values from file_name: ..|x=1__y=5|
    match = re.search("HP(.*)HP", str(file_name)).group(1)
    out = {}
    if match:
        for each_m in match.split("__"):
            m_parse = parse.parse("{}_e_{}", each_m)
            out[m_parse[0]] = m_parse[1]
    return out


def get_aggr_score(df: pd.DataFrame, field: str) -> float:
    try:
        val = float(df.tail(1)[field])
    except KeyError:
        return None
    if np.isnan(val):
        return None
    return val


def best_score_from_files(file_list: List[str], field="acc_mean", mode="max"):
    score_hp = []
    for fname in file_list:
        hp = extract_hp(fname)
        path = os.path.join(fname, "version_0", "metrics.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path).dropna()
        score = get_aggr_score(df, field=field)
        if score is not None:
            score_hp.append((score, hp))
            logger.info(f"@ {hp} = {score}")

    if mode == "max":
        max_score, max_hp = max(score_hp, key=lambda x: x[0])
    else:
        max_score, max_hp = min(score_hp, key=lambda x: x[0])
    logger.info(
        f"\nBest hyperparameters with score {max_score}:\n{OmegaConf.to_yaml(max_hp)}"
    )
    return max_hp, max_score


def get_best_hp(base_model_path: str,
                suffixes: List,
                model="resnet50",
                field="acc_mean",
                mode="max"):

    file_list = [base_model_path + "" + fp + f"_{model}" for fp in suffixes]
    best_hp, best_score = best_score_from_files(file_list, field, mode=mode)
    return best_hp, best_score


def update_params(args, config):
    nargs = copy.deepcopy(args)
    vargs = vars(nargs)
    for key in config:
        val = config[key]
        if key in vargs:
            vargs[key] = type(vargs[key])(val)
    return nargs
import sys
import os

proj_root = os.path.abspath(os.curdir)
sys.path.append(proj_root)

from caafe import (
    CAAFEClassifier,
    RACAAFEClassifier,
)  # Automated Feature Engineering for tabular datasets
from tabpfn import (
    TabPFNClassifier,
)  # Fast Automated Machine Learning method for small tabular datasets
from sklearn.ensemble import RandomForestClassifier

import random
import torch
import itertools
from caafe import data
from functools import partial
from timeit import default_timer as timer
from caafe.caafe_evaluate import evaluate_dataset
from caafe.run_llm_code import run_llm_code
from caafe.metrics import auc_metric
from caafe.racaafe import get_data_samples
import numpy as np
import logging
import set_openai_key

from caafe.preprocessing import make_datasets_numeric

from typing import Tuple

import pandas as pd

INSTRUCTIONS = {
    "icl": {
        "query": "Convert this example into vector to look for useful examples: ",
        "key": "Convert this example into vector for retrieval: ",
    }
}

EMBED_MODEL = "BAAI/llm-embedder"
COLLECTION_NAME = "full_collection"
LIMITED_COLLECTION_NAME = "limited_collection"
DISTANCE_FUNC = "cosine"
# TODO: aanpassen
NUM_EXP_DATASETS = 1

KEEP_TOP_N = 3
MIN_FEAT_SCORE = 0

# TODO: should be 10
NUM_FE_ITS = 3

# Beware: these parameters will be 'exploded' into a cartesian product
PARAM_EXPERIENCES = [2, 6, 10]
PARAM_ABSTRACTION = ["example", "insight", "one_liner"]

# From random.org > Random Sequence Generator. Took the first five results.
# To reproduce:
# - Set sequence boundaries to [-4999, 5000]
# - Choose 'pregenerated randomization based on persistent identifier' with
# seed = 0 (switch to advanced mode to see this option)
# RANDOM_SEEDS = [-4685, 1081, -2665, 3527, 4844]
# CUR_SEED = -4685
# EXP_ITERATION = RANDOM_SEEDS.index(CUR_SEED) + 1

EXP_ITERATION = 1

def allocate_seen_unseen_data(datasets: list) -> Tuple[list, list]:
    all_idx = list(range(len(datasets)))
    exp_idx = sorted(random.sample(all_idx, NUM_EXP_DATASETS))
    new_idx = sorted(set(all_idx) - set(exp_idx))

    exp_ds = [datasets[i] for i in range(len(datasets)) if i in exp_idx]
    new_ds = [datasets[i] for i in range(len(datasets)) if i in new_idx]

    return exp_ds, new_ds


def get_performance_increments(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    ds,
    predictor,
    perf_metric,
    new_f_code,
) -> pd.DataFrame:

    iterations = []
    rocs = []
    delta_roc = []
    score = evaluate_dataset(
        df_train,
        df_test,
        "XX",
        ds[0],
        predictor,
        perf_metric,
        ds[4][-1],
        max_time=300,
        seed=None
    )
    iterations += [-1]
    rocs += [score["roc"]]
    delta_roc += [0]
    full_code = ""
    # Sort new_f_code based on the iteration nr of the engineered feature
    new_f_sorted = dict(
        sorted(new_f_code.items(), key=lambda item: item[1][0])
    )

    for feat in new_f_sorted.keys():

        df_train_it = df_train.copy()
        df_test_it = df_test.copy()

        full_code += new_f_sorted[feat][1]
        df_train_it = run_llm_code(
            full_code,
            df_train_it,
            convert_categorical_to_integer=not ds[0].startswith("kaggle"),
        )
        df_test_it = run_llm_code(
            full_code,
            df_test_it,
            convert_categorical_to_integer=not ds[0].startswith("kaggle"),
        )
        score = evaluate_dataset(
            df_train_it,
            df_test_it,
            "XX",
            ds[0],
            predictor,
            perf_metric,
            ds[4][-1],
            max_time=300,
            seed=None
        )
        iterations += [new_f_sorted[feat][0]]
        delta_roc += [score["roc"] - rocs[-1]]
        rocs += [score["roc"]]

    log_perf_incr = pd.DataFrame(
        {"Iteration": iterations, "ROC": rocs, "deltaROC": delta_roc}
    )
    log_perf_incr.insert(0, "DataName", ds[0])

    return log_perf_incr


def set_limited_collection(
    complete_collection,
    limited_collection,
    datasets: list[str],
    abstraction: str,
) -> None:
    new_docs = complete_collection.get(
        where={
            "$and": [
                {"dataset": {"$in": datasets}},
                {"exp_type": {"$eq": abstraction}},
            ]
        },
        include=["embeddings", "documents", "metadatas"],
    )

    existing_docs = limited_collection.get(include=[])
    if len(existing_docs["ids"]) > 0:
        limited_collection.delete(ids=existing_docs["ids"])

    if len(new_docs["ids"]) > 0:
        limited_collection.add(
            ids=new_docs["ids"],
            embeddings=new_docs["embeddings"],
            metadatas=new_docs["metadatas"],
            documents=new_docs["documents"],
        )


def set_log_feat_eng_exp(collection, exp_it) -> pd.DataFrame:
    docs = collection.get(include=["metadatas", "documents"])

    abstraction_level = [
        metadata["exp_type"] for metadata in docs["metadatas"]
    ]
    raw_implementation = [
        doc.replace(INSTRUCTIONS["icl"]["key"], "")
        for doc in docs["documents"]
    ]
    data_name = [metadata["dataset"] for metadata in docs["metadatas"]]

    log_feat_eng_exp = pd.DataFrame(
        {
            "DataName": data_name,
            "AbstractionLevel": abstraction_level,
            "RawImplementation": raw_implementation,
        }
    )

    log_feat_eng_exp.insert(0, "RepeatID", exp_it)

    log_feat_eng_exp.to_csv(
        os.path.join("logs", f"{EXP_ITERATION}_FeatEngExperiences.csv"),
        sep=";",
        index=False,
        mode="w",
        decimal=",",
    )


"""
Unittest:

f_importances = pd.DataFrame({'rwww': [0.4], 'aaa': [0.4], 'bbb': [0.5]})

new_f_code = {'rwww': [4, '44444'], 'aaa': [2, '22222'], 'bbb': [5, '55555']}

data = {
    'Iteration': [1, 2, 3, 4, 5],
    'ErrorID': [None, None, 1, None, None],  # Using None to represent NaN
    'Kept': ['False', 'True', np.nan, 'True', 'True'],
    'RawImplementation': [
        "\n# New feature 'diagonal-win'\n# Usefulness: ...",
        "22222",
        "errrrrr",
        "44444",
        "555555",
    ]
}
"""


def add_importance_fe_log(
    log_feat_eng: pd.DataFrame, new_f_code: dict, f_importances: pd.DataFrame
) -> pd.DataFrame:

    f_importances = f_importances.copy()
    f_importances = f_importances.T.reset_index()
    f_importances.columns = ["feat_name", "PermImpRank"]
    f_importances["PermImpRank"] = f_importances["PermImpRank"].rank(
        method="dense", ascending=False
    )

    feat_iterations = [value[0] for key, value in new_f_code.items()]
    feat_names = list(new_f_code.keys())
    feat_iterations = pd.DataFrame(
        {"Iteration": feat_iterations, "feat_name": feat_names}
    )

    feat_iterations = feat_iterations.merge(f_importances, on="feat_name")
    feat_iterations = feat_iterations.drop(columns=["feat_name"])

    log_feat_eng = log_feat_eng.merge(
        feat_iterations, on="Iteration", how="left"
    )

    log_feat_eng.loc[log_feat_eng["Kept"] == "False", "PermImpRank"] = -1

    return log_feat_eng


def get_log_dataset_general(datasets: list[list], predictor) -> pd.DataFrame:
    ds_general_logs = {
        "DataName": [],
        "BaselinePerf": [],
        "NumberObs": [],
        "NumberFeats": [],
        "NumberNumFeats": [],
        "NumberCatFeats": [],
        "RawDescription": [],
        "Samples": [],
    }

    """ 
    ds = [
        dataset_name (str),
        observation_without_label (numpy.ndarray),
        labels (numpy.ndarray),
        indices_of_categorical_features (list),
        feature_names_and_label_name (list),
        dataset_settings (dict),
        raw_dataset_description (str),
    ]
    """
    for ds in datasets:

        dataset, df_train, df_test, _, _ = data.get_data_split(
            ds, seed=None, set_clean_description=False
        )

        ds_general_logs["DataName"].append(ds[0])

        score = evaluate_dataset(
            df_train,
            df_test,
            "XX",
            ds[0],
            predictor,
            auc_metric,
            ds[4][-1],
            max_time=300,
            seed=None
        )

        ds_general_logs["BaselinePerf"].append(score["roc"])

        ds_general_logs["NumberObs"].append(ds[1].shape[0])

        ds_general_logs["NumberFeats"].append(ds[1].shape[1])

        # Number of numerical features in ds
        ds_general_logs["NumberNumFeats"].append(ds[1].shape[1] - len(ds[3]))

        ds_general_logs["NumberCatFeats"].append(len(ds[3]))

        ds_general_logs["RawDescription"].append(ds[-1])

        ds_general_logs["Samples"].append(get_data_samples(df_train))

    ds_general_logs = pd.DataFrame(ds_general_logs)

    return ds_general_logs


def get_log_set_ups(experiment_params: list) -> pd.DataFrame:
    abstraction_level = []
    experience_volume = []

    for comb in experiment_params:
        abstraction_level.append(comb[0])
        experience_volume.append(comb[1])

    log_set_ups = pd.DataFrame(
        {
            "AbstractionLevel": abstraction_level,
            "ExperienceVolume": experience_volume,
        }
    )

    log_set_ups = log_set_ups.reset_index(names="SetUpID")

    return log_set_ups


def set_general_logs(experiment_params: itertools.product, datasets) -> None:
    ### Setup Base Classifier
    clf_no_feat_eng = TabPFNClassifier(
        device=("cuda" if torch.cuda.is_available() else "cpu"),
        N_ensemble_configurations=4,
    )
    clf_no_feat_eng.fit = partial(clf_no_feat_eng.fit, overwrite_warning=True)

    # Should be outside of experiment loop
    dataset_general_logs = get_log_dataset_general(datasets, clf_no_feat_eng)

    dataset_general_logs.to_csv(
        os.path.join("logs", f"{EXP_ITERATION}_DataSetsGeneral.csv"),
        index=False,
        sep=";",
        mode="w",
        decimal=",",
    )

    log_set_ups = get_log_set_ups(experiment_params)

    log_set_ups.to_csv(
        os.path.join("logs", f"{EXP_ITERATION}_SetUps.csv"),
        index=False,
        sep=";",
        mode="w",
        decimal=",",
    )


def set_log_dataset_repeat(datasets, exp_it):
    log_datasets_repeat = {"DataName": [], "DescriptionSummary": []}

    for ds in datasets:
        log_datasets_repeat["DataName"].append(ds[0])
        log_datasets_repeat["DescriptionSummary"].append(ds[-1])

    log_datasets_repeat = pd.DataFrame(log_datasets_repeat)

    log_datasets_repeat.insert(0, "RepeatID", exp_it)

    log_datasets_repeat.to_csv(
        os.path.join("logs", f"{EXP_ITERATION}_DatasetsRepeat.csv"),
        index=False,
        sep=";",
        mode="w",
        decimal=",",
    )


def run_experiment_it(experiment_params, original_datasets):
    logging.info(f"Iteration: {EXP_ITERATION}")

    ### Setup Base Classifier
    clf_no_feat_eng = TabPFNClassifier(
        device=("cuda" if torch.cuda.is_available() else "cpu"),
        N_ensemble_configurations=4,
    )
    clf_no_feat_eng.fit = partial(clf_no_feat_eng.fit, overwrite_warning=True)

    ### Setup and Run RACAAFE
    racaafe_base = RACAAFEClassifier(
        embed_model=EMBED_MODEL,
        collection_name=COLLECTION_NAME,
        distance_func=DISTANCE_FUNC,
        base_classifier=clf_no_feat_eng,
        ephimeral_collection=False,
        overwrite_collection=True,
        # llm_model="gpt-4",
        iterations=NUM_FE_ITS,
    )

    it_datasets = original_datasets.copy()

    start_summary = timer()
    it_datasets = racaafe_base.set_summarized_descriptions(it_datasets)
    end_summary = timer()

    log_time = pd.DataFrame(
        {
            "label": ["generate_ds_summaries"],
            "DataName": [np.nan],
            "time": [end_summary - start_summary],
        }
    )
    log_time.to_csv(
        "logs/Time.csv",
        index=False,
        mode="a",
        sep=";",
        header=False,
        decimal=",",
    )

    set_log_dataset_repeat(it_datasets, EXP_ITERATION)

    exp_ds, new_ds = allocate_seen_unseen_data(it_datasets)

    logging.info(f'exp_ds:\n{[ds[0] for ds in exp_ds]}')
    logging.info(f'new_ds:\n{[ds[0] for ds in new_ds]}')

    # Get CAAFE baseline performance and feature engineering experiences
    seen_log_feat_eng, seen_log_errors = racaafe_base.gain_experience(
        exp_ds,
        None,
        use_experience=False,
        keep_top_n=KEEP_TOP_N,
        min_feat_score=MIN_FEAT_SCORE,
        keep_logs=True,
    )

    seen_log_feat_eng.insert(loc=0, column="SetUpID", value=-1)

    seen_log_errors.insert(loc=0, column="SetUpID", value=-1)

    added_insights = racaafe_base.get_experience_abstractions("insight", True)

    added_one_liners = racaafe_base.get_experience_abstractions(
        "one_liner", True
    )

    logging.info(f"The following insights were added:\n{added_insights}")

    logging.info(f"The following one liners were added:\n{added_one_liners}")

    set_log_feat_eng_exp(racaafe_base.collection, EXP_ITERATION)

    ### Setup and Run RACAAFE with limited collection
    racaafe_exp = RACAAFEClassifier(
        embed_model=EMBED_MODEL,
        collection_name=LIMITED_COLLECTION_NAME,
        distance_func=DISTANCE_FUNC,
        base_classifier=clf_no_feat_eng,
        ephimeral_collection=False,
        overwrite_collection=True,
        # llm_model="gpt-4",
        iterations=NUM_FE_ITS,
    )

    log_set_ups = pd.read_csv(os.path.join("logs", f"{EXP_ITERATION}_SetUps.csv"), sep=";")

    unseen_log_feat_eng = pd.DataFrame()
    unseen_log_errors = pd.DataFrame()
    log_perf_incr = pd.DataFrame()

    # param = (abstraction_level, num_experience_datasets)
    for param in experiment_params:

        logging.info(f"Evaluating param ({experiment_params.index(param)}/{len(experiment_params)}): {param}")

        set_up_id = log_set_ups.loc[
            (log_set_ups["AbstractionLevel"] == param[0])
            & (log_set_ups["ExperienceVolume"] == param[1])
        ]["SetUpID"].values[0]

        # Get the names of the first n datasets that were used for feature
        # feature engineering experiences
        used_exp_ds = [ds[0] for ds in exp_ds[: param[1]]]

        set_limited_collection(
            racaafe_base.collection,
            racaafe_exp.collection,
            used_exp_ds,
            param[0],
        )

        lim_col = racaafe_exp.collection.get(include=[])
        logging.info(f"Limited collection:\n{[i for i in lim_col['ids']]}")

        param_log_feat_eng = pd.DataFrame()
        param_log_errors = pd.DataFrame()
        param_log_perf_incr = pd.DataFrame()

        cnt = 1
        for ds in new_ds:
            logging.info(f"Evaluating dataset ({cnt}/{len(new_ds)}): {ds[0]}")
            cnt += 1

            ds, df_train, df_test, _, _ = data.get_data_split(
                ds, seed=None, set_clean_description=False
            )
            target_column_name = ds[4][-1]

            df_train, df_test = make_datasets_numeric(
                df_train, df_test, target_column_name
            )

            (
                code,
                prompt,
                messages,
                new_f_code,
                f_importances,
                ds_log_feat_eng,
                ds_log_errors,
            ) = racaafe_exp.generate_features(
                ds,
                df_train,
                param[0],
                use_experience=True,
                store_experience=False,
                keep_top_n=KEEP_TOP_N,
                min_feat_score=MIN_FEAT_SCORE,
                keep_logs=True,
            )

            param_log_feat_eng = pd.concat(
                [param_log_feat_eng, ds_log_feat_eng], ignore_index=True
            )

            param_log_errors = pd.concat(
                [param_log_errors, ds_log_errors], ignore_index=True
            )

            ds_log_perf_incr = get_performance_increments(
                df_train, df_test, ds, clf_no_feat_eng, auc_metric, new_f_code
            )

            param_log_perf_incr = pd.concat(
                [param_log_perf_incr, ds_log_perf_incr], ignore_index=True
            )

        param_log_perf_incr.insert(loc=0, column="SetUpID", value=set_up_id)

        log_perf_incr = pd.concat(
            [log_perf_incr, param_log_perf_incr], ignore_index=True
        )

        param_log_feat_eng.insert(loc=0, column="SetUpID", value=set_up_id)

        unseen_log_feat_eng = pd.concat(
            [unseen_log_feat_eng, param_log_feat_eng], ignore_index=True
        )

        param_log_errors.insert(loc=0, column="SetUpID", value=set_up_id)

        unseen_log_errors = pd.concat(
            [unseen_log_errors, param_log_errors], ignore_index=True
        )

    log_perf_incr.insert(loc=0, column="RepeatID", value=EXP_ITERATION)

    log_perf_incr.to_csv(
        os.path.join("logs", f"{EXP_ITERATION}_PerformanceIncrements.csv"),
        sep=";",
        index=False,
        mode="w",
        decimal=",",
    )

    it_log_feat_eng = pd.concat([seen_log_feat_eng, unseen_log_feat_eng])

    it_log_feat_eng.insert(loc=0, column="RepeatID", value=EXP_ITERATION)

    it_log_feat_eng.to_csv(
        os.path.join("logs", f"{EXP_ITERATION}_FeatEngOperations.csv"),
        sep=";",
        index=False,
        mode="w",
        decimal=",",
    )

    it_log_errors = pd.concat([seen_log_errors, unseen_log_errors])

    it_log_errors.insert(loc=0, column="RepeatID", value=EXP_ITERATION)

    it_log_errors.to_csv(
        os.path.join("logs", f"{EXP_ITERATION}_Errors.csv"),
        sep=";",
        index=False,
        mode="w",
        decimal=",",
    )

    logging.info(
        f"Iteration: {EXP_ITERATION}"
    )


if __name__ == "__main__":
    logging.basicConfig(
        filename=f"logs/exp_it_{EXP_ITERATION}.log",
        filemode="w",
        format="%(asctime)s\t%(levelname)s\t%(funcName)s:\n%(message)s\n",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    log_time = pd.DataFrame(
        {"label": ["create_file"], "DataName": [np.nan], "time": [np.nan]}
    )

    log_time.to_csv(
        os.path.join("logs", "Time.csv"),
        index=False,
        mode="w",
        sep=";",
        decimal=",",
    )

    start_exp_it = timer()

    # metric_used = tabular_metrics.auc_metric
    original_datasets = data.load_all_data()

    # TODO: fix
    original_datasets = original_datasets[:2]

    # Cartesian product of PARAM_ABSTRACTION, PARAM_EXPERIENCES
    experiment_params = list(
        itertools.product(PARAM_ABSTRACTION, PARAM_EXPERIENCES)
    )

    # TODO: remove
    # experiment_params = [("example", 2), ("insight", 2), ("one_liner", 2)]
    experiment_params = [("example", 1), ("insight", 1)]

    set_general_logs(experiment_params, original_datasets)

    run_experiment_it(experiment_params, original_datasets)

    end_exp_it = timer()

    log_time = pd.DataFrame(
        {
            "label": ["exp_it"],
            "DataName": [np.nan],
            "time": [end_exp_it - start_exp_it],
        }
    )

    log_time.to_csv(
        os.path.join("logs", "Time.csv"),
        index=False,
        mode="a",
        sep=";",
        decimal=",",
        header=False,
    )

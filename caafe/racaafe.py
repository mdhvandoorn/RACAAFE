import copy
import numpy as np
import pandas as pd
import re

import openai
from sklearn.model_selection import RepeatedKFold, StratifiedKFold
from sklearn.inspection import permutation_importance
from .caafe_evaluate import (
    evaluate_dataset,
)
from .run_llm_code import run_llm_code
from .preprocessing import (
    make_datasets_numeric,
    split_target_column,
)
from typing import Union


def get_examples(examples: str) -> str:
    return f"""
============================================================================
Given the same instructions, these feature engineering operations on other datasets 
resulted in the biggest predictive performance gain:

{examples}
============================================================================
    """


def get_insights(insights: str) -> str:
    return f"""
============================================================================
After analysing the most succesful previous feature engineering operations, 
given the same initial instructions but applied to other datasets, the 
following insights were extracted:

{insights}
============================================================================
    """


def extract_insight(experience_doc: str) -> str:
    prompt = f"""{experience_doc}

    From the examples above, what patterns can we observe about the relationship between dataset characteristics (enclosed between Context: ### and ###) and the best feature engineering operations (enclosed between Examples: ### and ###)? 
    Answer MUST be concise, critical, point-by-point, line-by-line, and brief. 
    (Only include relevant observations without unnecessary elaboration.)
    """

    messages = [
        {
            "role": "system",
            "content": "You are an expert datascientist assistant solving Kaggle problems. Answer as concisely as possible.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.5,
        max_tokens=500,
    )

    insight = completion["choices"][0]["message"]["content"]

    return insight


def get_example_ids(collection) -> list[str]:
    # Only retrieve ids
    col = collection.get(include=[])

    # Identify dataset_name-counter combinations with exp_type as 'insight'
    exclude_combinations = set()
    for doc_id in col["ids"]:
        parts = doc_id.split("---")
        dataset_name, exp_type, counter = parts[0], parts[1], parts[2]
        if exp_type == "insight":
            exclude_combinations.add((dataset_name, counter))

    # Collect strings where their dataset_name-counter combination is not
    # in exclude_combinations
    ex_doc_ids = []
    for doc_id in col["ids"]:
        parts = doc_id.split("---")
        dataset_name, exp_type, counter = parts[0], parts[1], parts[2]
        if (dataset_name, counter) not in exclude_combinations:
            ex_doc_ids.append(doc_id)

    return ex_doc_ids


def retrieve_experiences(
    collection, ds, samples, exp_type: str, n_results: int = 3
) -> Union[str, None]:
    """
    exp_type (str):
        Specifies the type of feature engineering experience to retrieve
        ('insight' or 'example').
    """

    query = [
        f"""
    Description of the dataset in 'df' (column dtypes might be inaccurate):
    {ds[-1]}

    Columns in 'df' (true feature dtypes listed here, categoricals encoded as int):
    {samples}
    """
    ]

    query_results = collection.query(
        query_texts=query,
        where={"exp_type": {"$eq": exp_type}},
        n_results=n_results,
    )

    # No experiences retrieved. Possibly empty data store.
    if len(query_results["documents"][0]) == 0:
        return None

    query_results = "\n\n".join(query_results["documents"][0])

    return query_results


def embed_document(embed_model, docs: list[str]) -> list[list[float]]:
    return embed_model.encode(docs)


def get_doc_ids(
    collection, docs: list, dataset_name: str, exp_type: str
) -> list[str]:
    # current_ids = [ID1, ID2, ...],
    # where IDi ::= datasetName---expType---uniqueInt(i)
    cur_ids = collection.get(
        where={
            "$and": [
                {"dataset": {"$eq": dataset_name}},
                {"exp_type": {"$eq": exp_type}},
            ]
        }
    )["ids"]

    suffix = list(range(len(cur_ids), len(cur_ids) + len(docs)))

    return [f"{dataset_name}---{exp_type}---{str(sfx)}" for sfx in suffix]


def save_experience_db(
    collection,
    docs: list[list[str]],
    doc_embeddings: list[list[float]],
    exp_type: str,
    dataset_name: str,
) -> None:

    docs_ids = get_doc_ids(collection, docs, dataset_name, exp_type)

    collection.add(
        documents=docs,
        embeddings=doc_embeddings,
        metadatas=[{"exp_type": exp_type, "dataset": dataset_name}]
        * len(docs),
        ids=docs_ids,
    )


def get_prompt(
    df,
    ds,
    fe_experiences,
    iterative=1,
    data_description_unparsed=None,
    samples=None,
    **kwargs,
):
    how_many = (
        "up to 10 useful columns. Generate as many features as useful for downstream classifier, but as few as necessary to reach good performance."
        if iterative == 1
        else "exactly one useful column"
    )
    return f"""
The dataframe `df` is loaded and in memory. Columns are also named attributes.
Description of the dataset in `df` (column dtypes might be inaccurate):
"{data_description_unparsed}"

Columns in `df` (true feature dtypes listed here, categoricals encoded as int):
{samples}
    
This code was written by an expert datascientist working to improve predictions. It is a snippet of code that adds new columns to the dataset.
Number of samples (rows) in training dataset: {int(len(df))}
    
This code generates additional columns that are useful for a downstream classification algorithm (such as XGBoost) predicting \"{ds[4][-1]}\".
Additional columns add new semantic information, that is they use real world knowledge on the dataset. They can e.g. be feature combinations, transformations, aggregations where the new column is a function of the existing columns.
The scale of columns and offset does not matter. Make sure all used columns exist. Follow the above description of columns closely and consider the datatypes and meanings of classes.
This code also drops columns, if these may be redundant and hurt the predictive performance of the downstream classifier (Feature selection). Dropping columns may help as the chance of overfitting is lower, especially if the dataset is small.
The classifier will be trained on the dataset with the generated columns and evaluated on a holdout set. The evaluation metric is accuracy. The best performing code will be selected.
Added columns can be used in other codeblocks, dropped columns are not available anymore.

Code formatting for each added column:
```python
# (Feature name and description)
# Usefulness: (Description why this adds useful real world knowledge to classify \"{ds[4][-1]}\" according to dataset description and attributes.)
# Input samples: (Three samples of the columns used in the following code, e.g. '{df.columns[0]}': {list(df.iloc[:3, 0].values)}, '{df.columns[1]}': {list(df.iloc[:3, 1].values)}, ...)
(Some pandas code using {df.columns[0]}', '{df.columns[1]}', ... to add a new column for each row in df)
```end

Code formatting for dropping columns:
```python
# Explanation why the column XX is dropped
df.drop(columns=['XX'], inplace=True)
```end
{fe_experiences}

Each codeblock generates {how_many} and can drop unused columns (Feature selection).
Each codeblock ends with ```end and starts with "```python"
Codeblock:
"""


def build_prompt_from_df(
    ds, df, collection, exp_type, samples, use_experience, iterative=1
):
    data_description_unparsed = ds[-1]
    feature_importance = {}  # xgb_eval(_obj)

    kwargs = {
        "data_description_unparsed": data_description_unparsed,
        "samples": samples,
        "feature_importance": {
            k: "%s" % float("%.2g" % feature_importance[k])
            for k in feature_importance
        },
    }

    fe_experiences = retrieve_experiences(collection, ds, samples, exp_type)

    if fe_experiences is None or use_experience == False:
        fe_experiences = ""
    elif exp_type == "example":
        fe_experiences = get_examples(fe_experiences)
    elif exp_type == "insight":
        fe_experiences = get_insights(fe_experiences)

    prompt = get_prompt(
        df,
        ds,
        fe_experiences,
        data_description_unparsed=data_description_unparsed,
        iterative=iterative,
        samples=samples,
    )

    return prompt


def get_data_samples(df: pd.DataFrame) -> str:
    samples = ""
    df_ = df.head(10)
    for i in list(df_):
        # show the list of values
        nan_freq = "%s" % float("%.2g" % (df[i].isna().mean() * 100))
        s = df_[i].tolist()
        if str(df[i].dtype) == "float64":
            s = [round(sample, 2) for sample in s]
        samples += f"{df_[i].name} ({df[i].dtype}): NaN-freq [{nan_freq}%], Samples {s}\n"

    return samples


def generate_features(
    ds,
    df,
    collection,
    exp_type,
    use_experience: bool = True,
    model="gpt-3.5-turbo",
    just_print_prompt=False,
    iterative=1,
    metric_used=None,
    iterative_method="logistic",
    display_method="markdown",
    n_splits=10,
    n_repeats=2,
    keep_logs=False,
):
    log_feat_eng, log_errors = None, None
    if keep_logs:
        log_feat_eng = pd.DataFrame(
            columns=["ErrorID", "Kept", "RawImplementation"]
        )
        log_errors = pd.DataFrame(columns=["ErrorType", "RawError"])

    def format_for_display(code):
        code = (
            code.replace("```python", "")
            .replace("```", "")
            .replace("<end>", "")
            .replace("end", "")
        )
        return code

    if display_method == "markdown":
        from IPython.display import display, Markdown

        display_method = lambda x: display(Markdown(x))
    else:

        display_method = print

    assert (
        iterative == 1 or metric_used is not None
    ), "metric_used must be set if iterative"

    samples = get_data_samples(df)

    prompt = build_prompt_from_df(
        ds,
        df,
        collection,
        exp_type,
        samples,
        use_experience,
        iterative=iterative,
    )

    if just_print_prompt:
        code, prompt = None, prompt
        return code, prompt, None

    def generate_code(messages):
        if model == "skip":
            return ""

        completion = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            # TODO:
            # Remove stop paramater since it probably stops ChatGPT from formulating
            # feature creation and feature selection operations in a single operation.
            stop=["```end"],
            temperature=0.5,
            max_tokens=500,
        )
        code = completion["choices"][0]["message"]["content"]

        code = (
            code.replace("```python", "")
            .replace("```", "")
            .replace("<end>", "")
        )

        return code

    def execute_and_evaluate_code_block(full_code, code):
        old_accs, old_rocs, accs, rocs = [], [], [], []

        ss = RepeatedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=0
        )

        for train_idx, valid_idx in ss.split(df):
            df_train, df_valid = df.iloc[train_idx], df.iloc[valid_idx]

            # Remove target column from df_train
            target_train = df_train[ds[4][-1]]
            target_valid = df_valid[ds[4][-1]]
            df_train = df_train.drop(columns=[ds[4][-1]])
            df_valid = df_valid.drop(columns=[ds[4][-1]])

            df_train_extended = copy.deepcopy(df_train)
            df_valid_extended = copy.deepcopy(df_valid)

            try:
                df_train = run_llm_code(
                    full_code,
                    df_train,
                    convert_categorical_to_integer=not ds[0].startswith(
                        "kaggle"
                    ),
                )
                df_valid = run_llm_code(
                    full_code,
                    df_valid,
                    convert_categorical_to_integer=not ds[0].startswith(
                        "kaggle"
                    ),
                )
                df_train_extended = run_llm_code(
                    full_code + "\n" + code,
                    df_train_extended,
                    convert_categorical_to_integer=not ds[0].startswith(
                        "kaggle"
                    ),
                )
                df_valid_extended = run_llm_code(
                    full_code + "\n" + code,
                    df_valid_extended,
                    convert_categorical_to_integer=not ds[0].startswith(
                        "kaggle"
                    ),
                )

            except Exception as e:
                display_method(f"Error in code execution. {type(e)} {e}")
                display_method(f"```python\n{format_for_display(code)}\n```\n")
                return e, None, None, None, None, None

            # Add target column back to df_train
            df_train[ds[4][-1]] = target_train
            df_valid[ds[4][-1]] = target_valid
            df_train_extended[ds[4][-1]] = target_train
            df_valid_extended[ds[4][-1]] = target_valid

            from contextlib import contextmanager
            import sys, os

            with open(os.devnull, "w") as devnull:
                old_stdout = sys.stdout
                sys.stdout = devnull
                try:
                    result_old = evaluate_dataset(
                        df_train=df_train,
                        df_test=df_valid,
                        prompt_id="XX",
                        name=ds[0],
                        method=iterative_method,
                        metric_used=metric_used,
                        seed=0,
                        target_name=ds[4][-1],
                    )

                    result_extended = evaluate_dataset(
                        df_train=df_train_extended,
                        df_test=df_valid_extended,
                        prompt_id="XX",
                        name=ds[0],
                        method=iterative_method,
                        metric_used=metric_used,
                        seed=0,
                        target_name=ds[4][-1],
                    )
                finally:
                    sys.stdout = old_stdout

            old_accs += [result_old["acc"]]
            old_rocs += [result_old["roc"]]
            accs += [result_extended["acc"]]
            rocs += [result_extended["roc"]]

            # TODO: fix what happens when multiple FE operations are conducted
            fe_op_name = list(
                set(df_train.columns) ^ set(df_train_extended.columns)
            )[0]

            if len(df_train_extended.columns) < len(df_train.columns):
                fe_op_name = f"remove----{fe_op_name}"

        return None, rocs, accs, old_rocs, old_accs, fe_op_name

    messages = [
        {
            "role": "system",
            "content": "You are an expert datascientist assistant solving Kaggle problems. You answer only by generating code. Answer as concisely as possible.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]
    display_method(f"*Dataset description:*\n {ds[-1]}")

    n_iter = iterative
    full_code = ""
    new_f_code = {}

    i = 0
    while i < n_iter:
        try:
            code = generate_code(messages)
        except Exception as e:
            display_method("Error in LLM API." + str(e))
            continue
        i = i + 1
        e, rocs, accs, old_rocs, old_accs, fe_op_name = (
            execute_and_evaluate_code_block(full_code, code)
        )
        if e is not None:
            messages += [
                {"role": "assistant", "content": code},
                {
                    "role": "user",
                    "content": f"""Code execution failed with error: {type(e)} {e}.\n Code: ```python{code}```\n Generate next feature (fixing error?):
                                ```python
                                """,
                },
            ]

            if keep_logs:
                new_error_log = pd.DataFrame(
                    {"ErrorType": [type(e)], "RawError": [e]}
                )
                log_errors = pd.concat(
                    [log_errors, new_error_log], ignore_index=True
                )
                error_id = log_errors.index[-1]
                ["ErrorID", "RawImplementation"]
                new_feat_eng_log = pd.DataFrame(
                    {
                        "ErrorID": [error_id],
                        "Kept": [np.nan],
                        "RawImplementation": [code],
                    }
                )
                log_feat_eng = pd.concat(
                    [log_feat_eng, new_feat_eng_log],
                    ignore_index=True,
                )

            continue

        # importances = get_leave_one_out_importance(
        #    df_train_extended,
        #    df_valid_extended,
        #    ds,
        #    iterative_method,
        #    metric_used,
        # )
        # """ROC Improvement by using each feature: {importances}"""

        improvement_roc = np.nanmean(rocs) - np.nanmean(old_rocs)
        improvement_acc = np.nanmean(accs) - np.nanmean(old_accs)

        add_feature = True
        add_feature_sentence = (
            "The code was executed and changes to ´df´ were kept."
        )
        if improvement_roc + improvement_acc <= 0:
            add_feature = False
            add_feature_sentence = f"The last code changes to ´df´ were discarded. (Improvement: {improvement_roc + improvement_acc})"

        display_method(
            "\n"
            + f"*Iteration {i}*\n"
            + f"```python\n{format_for_display(code)}\n```\n"
            + f"Performance before adding features ROC {np.nanmean(old_rocs):.3f}, ACC {np.nanmean(old_accs):.3f}.\n"
            + f"Performance after adding features ROC {np.nanmean(rocs):.3f}, ACC {np.nanmean(accs):.3f}.\n"
            + f"Improvement ROC {improvement_roc:.3f}, ACC {improvement_acc:.3f}.\n"
            + f"{add_feature_sentence}\n"
            + f"\n"
        )
        
        if len(code) > 10:
            messages += [
                {"role": "assistant", "content": code},
                {
                    "role": "user",
                    "content": f"""Performance after adding feature ROC {np.nanmean(rocs):.3f}, ACC {np.nanmean(accs):.3f}. {add_feature_sentence}
Next codeblock:
""",
                },
            ]
        if add_feature:
            full_code += code

            if fe_op_name is not None:
                new_f_code[fe_op_name] = [i, code]

        if keep_logs:
            new_feat_eng_log = pd.DataFrame(
                {
                    "ErrorID": [np.nan],
                    "Kept": [str(add_feature)],
                    "RawImplementation": [code],
                }
            )
            log_feat_eng = pd.concat(
                [log_feat_eng, new_feat_eng_log], ignore_index=True
            )

    return full_code, prompt, messages, new_f_code, log_feat_eng, log_errors


def get_feat_importances(
    full_code: str,
    new_f_code: dict,
    df: pd.DataFrame,
    ds,
    iterative_method,
    metric_used,
    samples,
) -> pd.DataFrame:
    df_train_final = run_llm_code(
        full_code,
        df,
    )

    df_train_final, _ = make_datasets_numeric(
        df_train_final, df_test=None, target_column=ds[4][-1]
    )

    # Get feature names without label
    feat_names = [c for c in df_train_final.columns if c != ds[4][-1]]

    df_train_final, y = split_target_column(df_train_final, ds[4][-1])

    f_importances = get_permutation_importance(
        iterative_method, df_train_final, y, feat_names, metric_used
    )

    f_importances = f_importances.loc[:, ~f_importances.columns.isin(ds[4])]

    return f_importances


def store_experiences(
    new_f_code: dict,
    f_importances: pd.DataFrame,
    samples,
    ds,
    embed_model,
    collection,
) -> None:
    experience = compile_experience(
        new_f_code, f_importances, 3, samples, ds[-1]
    )

    context_embedding = embed_document(embed_model, [experience["context"]])

    experience = [
        "\n\n".join([experience["context"], experience["experience"]])
    ]

    save_experience_db(
        collection, experience, context_embedding, "example", ds[0]
    )


def pre_process_fe_code(fe_code: str) -> str:
    # Remove all commments
    res = re.sub("#.*?\n", "", fe_code)
    # The resulting string can start with zero or more '\n', remove them
    res = re.sub("^\n+", "", res)
    # The resulting string can end with zero or more '\n', remove them
    res = re.sub("\n+$", "", res)
    return res


def compile_experience(
    feature_codes: dict,
    f_importances: pd.DataFrame,
    n_top: int,
    samples: str,
    data_description_unparsed: str,
) -> dict:

    fe_experiences = {"context": [], "experience": []}

    fe_experiences["context"].append("Context: ###")

    fe_experiences["context"].append(
        f"""Description of the dataset in 'df' (column dtypes might be inaccurate):\n{data_description_unparsed}\n"""
    )

    fe_experiences["context"].append(
        f"""Columns in 'df' (true feature dtypes listed here, categoricals encoded as int):\n{samples}"""
    )

    # Close context demarcation
    fe_experiences["context"].append("###")

    f_importances.index = ["f_importance"]
    f_importances = f_importances.T
    f_importances = f_importances.sort_values("f_importance", ascending=False)

    f_importances = f_importances.loc[f_importances["f_importance"] > 0]

    f_importances = f_importances.iloc[:n_top]

    fe_experiences["experience"].append("Examples: ###")

    fe_experiences["experience"].append(
        f"The following feature engineering operations resulted in the best performance increase:"
    )

    for index, row in f_importances.iterrows():
        # Remove all comments
        # feature_codes[index] = [iteration nr, feature code]
        fe_code = pre_process_fe_code(feature_codes[index][1])
        # fe_experiences["experience"].append(fe_code)

        if "remove----" in index:
            index = index.split("----")[1]
            fe_experiences["experience"].append(
                f"Removed feature '{index}' with code:\n{fe_code}"
            )
        else:
            fe_experiences["experience"].append(
                f"Added feature '{index}' with code:\n{fe_code}"
            )

    # Close examples demarcation
    fe_experiences["experience"].append("###")

    fe_experiences["context"] = "\n\n".join(fe_experiences["context"])
    fe_experiences["experience"] = "\n\n".join(fe_experiences["experience"])

    return fe_experiences


def get_permutation_importance(
    model, df_x: pd.DataFrame, df_y: pd.DataFrame, feat_names, metric_used
) -> pd.DataFrame:
    """
    Arguments:
        df (pd.DataFrame):
            Dataframe containing target in the last column and features in the
            previous columns.

        The following metrics are implemented (caafe/metrics.py):
            - roc_auc. In case of multi-class, one-vs-one (ovo) is applied.
            - accuracy.

    Returns:
        pd.DataFrame with a single row containing the averaged feature
        importances.

    """
    multi_class = True if len(set(df_y)) > 2 else False

    ss = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    all_scores = []

    for train_idx, valid_idx in ss.split(df_x, df_y):
        x_train, X_val = df_x.iloc[train_idx], df_x.iloc[valid_idx]
        y_train, y_val = df_y.iloc[train_idx], df_y.iloc[valid_idx]

        x_train, y_train = x_train.values, y_train.values.astype(int)
        X_val, y_val = X_val.values, y_val.values.astype(int)

        model.fit(x_train, y_train)

        # Retroactively determine the performance metric based on function
        # metric_used
        if metric_used.__name__ == "auc_metric":
            scoring = "roc_auc_ovo" if multi_class else "roc_auc"
            r = permutation_importance(
                model,
                X_val,
                y_val,
                scoring=scoring,
                n_repeats=1,
                random_state=0,
            )
        elif metric_used.__name__ == "accuracy_metric":
            r = permutation_importance(
                model,
                X_val,
                y_val,
                scoring="accuracy",
                n_repeats=1,
                random_state=0,
            )

        all_scores.append(r["importances_mean"])

    all_scores = np.array(all_scores)
    mean_scores = np.mean(all_scores, axis=0)
    df_res = pd.DataFrame([mean_scores], columns=feat_names)

    return df_res

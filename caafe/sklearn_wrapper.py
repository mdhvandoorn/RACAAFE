from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from .run_llm_code import run_llm_code
from .preprocessing import (
    make_datasets_numeric,
    split_target_column,
    make_dataset_numeric,
)
from .data import get_X_y
from . import caafe as caafe
from . import racaafe as racaafe
from .metrics import auc_metric, accuracy_metric
import pandas as pd
import numpy as np
from typing import Optional
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from caafe import data
from typing import Union, Tuple

IMPLEMENTED_DISTANCE_FUNCS = ["cosine", "ip", "l2"]


class CAAFEClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that uses the CAAFE algorithm to generate features and a base classifier to make predictions.

    Parameters:
    base_classifier (object, optional): The base classifier to use. If None, a default TabPFNClassifier will be used. Defaults to None.
    optimization_metric (str, optional): The metric to optimize during feature generation. Can be 'accuracy' or 'auc'. Defaults to 'accuracy'.
    iterations (int, optional): The number of iterations to run the CAAFE algorithm. Defaults to 10.
    llm_model (str, optional): The LLM model to use for generating features. Defaults to 'gpt-3.5-turbo'.
    n_splits (int, optional): The number of cross-validation splits to use during feature generation. Defaults to 10.
    n_repeats (int, optional): The number of times to repeat the cross-validation during feature generation. Defaults to 2.
    """

    def __init__(
        self,
        base_classifier: Optional[object] = None,
        optimization_metric: str = "accuracy",
        iterations: int = 10,
        llm_model: str = "gpt-3.5-turbo",
        n_splits: int = 10,
        n_repeats: int = 2,
    ) -> None:
        self.base_classifier = base_classifier
        if self.base_classifier is None:
            from tabpfn.scripts.transformer_prediction_interface import (
                TabPFNClassifier,
            )
            import torch
            from functools import partial

            self.base_classifier = TabPFNClassifier(
                N_ensemble_configurations=16,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            self.base_classifier.fit = partial(
                self.base_classifier.fit, overwrite_warning=True
            )
        self.llm_model = llm_model
        self.iterations = iterations
        self.optimization_metric = optimization_metric
        self.n_splits = n_splits
        self.n_repeats = n_repeats

    def fit_pandas(self, df, dataset_description, target_column_name, **kwargs):
        """
        Fit the classifier to a pandas DataFrame.

        Parameters:
        df (pandas.DataFrame): The DataFrame to fit the classifier to.
        dataset_description (str): A description of the dataset.
        target_column_name (str): The name of the target column in the DataFrame.
        **kwargs: Additional keyword arguments to pass to the base classifier's fit method.
        """
        feature_columns = list(df.drop(columns=[target_column_name]).columns)

        X, y = (
            df.drop(columns=[target_column_name]).values,
            df[target_column_name].values,
        )
        return self.fit(
            X, y, dataset_description, feature_columns, target_column_name, **kwargs
        )

    def fit(
        self, X, y, dataset_description, feature_names, target_name, disable_caafe=False
    ):
        """
        Fit the model to the training data.

        Parameters:
        -----------
        X : np.ndarray
            The training data features.
        y : np.ndarray
            The training data target values.
        dataset_description : str
            A description of the dataset.
        feature_names : List[str]
            The names of the features in the dataset.
        target_name : str
            The name of the target variable in the dataset.
        disable_caafe : bool, optional
            Whether to disable the CAAFE algorithm, by default False.
        """
        self.dataset_description = dataset_description
        self.feature_names = list(feature_names)
        self.target_name = target_name

        self.X_ = X
        self.y_ = y

        if X.shape[0] > 3000 and self.base_classifier.__class__.__name__ == "TabPFNClassifier":
            print(
                "WARNING: TabPFN may take a long time to run on large datasets. Consider using alternatives (e.g. RandomForestClassifier)"
            )
        elif X.shape[0] > 10000 and self.base_classifier.__class__.__name__ == "TabPFNClassifier":
            print("WARNING: CAAFE may take a long time to run on large datasets.")

        ds = [
            "dataset",
            X,
            y,
            [],
            self.feature_names + [target_name],
            {},
            dataset_description,
        ]
        # Add X and y as one dataframe
        df_train = pd.DataFrame(
            X,
            columns=self.feature_names,
        )
        df_train[target_name] = y
        if disable_caafe:
            self.code = ""
        else:
            self.code, prompt, messages = caafe.generate_features(
                ds,
                df_train,
                model=self.llm_model,
                iterative=self.iterations,
                metric_used=auc_metric,
                iterative_method=self.base_classifier,
                display_method="print",
                n_splits=self.n_splits,
                n_repeats=self.n_repeats,
            )

        df_train = run_llm_code(
            self.code,
            df_train,
        )

        df_train, _, self.mappings = make_datasets_numeric(
            df_train, df_test=None, target_column=target_name, return_mappings=True
        )

        df_train, y = split_target_column(df_train, target_name)

        X, y = df_train.values, y.values.astype(int)
        # Check that X and y have correct shape
        # X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.base_classifier.fit(X, y)

        # Return the classifier
        return self

    def predict_preprocess(self, X):
        """
        Helper functions for preprocessing the data before making predictions.

        Parameters:
        X (pandas.DataFrame): The DataFrame to make predictions on.

        Returns:
        numpy.ndarray: The preprocessed input data.
        """
        # check_is_fitted(self)

        if type(X) != pd.DataFrame:
            X = pd.DataFrame(X, columns=self.X_.columns)
        X, _ = split_target_column(X, self.target_name)

        X = run_llm_code(
            self.code,
            X,
        )

        X = make_dataset_numeric(X, mappings=self.mappings)

        X = X.values

        # Input validation
        # X = check_array(X)

        return X

    def predict_proba(self, X):
        X = self.predict_preprocess(X)
        return self.base_classifier.predict_proba(X)

    def predict(self, X):
        X = self.predict_preprocess(X)
        return self.base_classifier.predict(X)


class RACAAFEClassifier(CAAFEClassifier):
    def __init__(
        self,
        embed_model: str,
        collection_name: str,
        distance_func: str,
        ephimeral_collection: bool = True,
        overwrite_collection: bool = False,
        collection_path: str = "fe_experiences/",
        base_classifier: Optional[object] = None,
        optimization_metric: str = "accuracy",
        iterations: int = 10,
        llm_model: str = "gpt-3.5-turbo",
        n_splits: int = 10,
        n_repeats: int = 2,
    ) -> None:
        super().__init__(
            base_classifier,
            optimization_metric,
            iterations,
            llm_model,
            n_splits,
            n_repeats,
        )
        # RACAAFE uses ChromaDB for RAG. Thus, the set of usable distance
        # functions is dependent on ChromaDB's implementation.
        if distance_func not in IMPLEMENTED_DISTANCE_FUNCS:
            raise ValueError(
                "distance_func can be 'cosine', 'l2', or 'ip', see ChromaDB docs."
            )

        # For manual embeddings
        self.embed_model = SentenceTransformer(embed_model, device="cpu")

        if ephimeral_collection:
            self.client = chromadb.EphemeralClient()
        else:
            self.client = chromadb.PersistentClient(path=collection_path)

            if overwrite_collection:
                try:
                    self.client.delete_collection(collection_name)
                # Collection does not exist
                except ValueError:
                    pass

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embed_model
            ),
            metadata={"hnsw:space": distance_func},
        )

    def fit_pandas(
        self,
        df,
        dataset_description,
        target_column_name,
        exp_type,
        use_experience,
        store_experience=True,
        keep_logs=False,
        **kwargs,
    ):
        """
        Fit the classifier to a pandas DataFrame.

        Parameters:
        df (pandas.DataFrame): The DataFrame to fit the classifier to.
        dataset_description (str): A description of the dataset.
        target_column_name (str): The name of the target column in the DataFrame.
        **kwargs: Additional keyword arguments to pass to the base classifier's fit method.
        """
        feature_columns = list(df.drop(columns=[target_column_name]).columns)

        X, y = (
            df.drop(columns=[target_column_name]).values,
            df[target_column_name].values,
        )
        return self.fit(
            X,
            y,
            dataset_description,
            feature_columns,
            target_column_name,
            exp_type,
            use_experience,
            store_experience,
            **kwargs,
        )

    def fit(
        self,
        X,
        y,
        dataset_description,
        feature_names,
        target_name,
        exp_type,
        use_experience,
        store_experience=True,
        disable_caafe=False,
        keep_logs=False,
    ):
        """
        Fit the model to the training data.

        Parameters:
        -----------
        X : np.ndarray
            The training data features.
        y : np.ndarray
            The training data target values.
        dataset_description : str
            A description of the dataset.
        feature_names : List[str]
            The names of the features in the dataset.
        target_name : str
            The name of the target variable in the dataset.
        disable_caafe : bool, optional
            Whether to disable the CAAFE algorithm, by default False.
        """
        self.dataset_description = dataset_description
        self.feature_names = list(feature_names)
        self.target_name = target_name

        self.X_ = X
        self.y_ = y

        if (
            X.shape[0] > 3000
            and self.base_classifier.__class__.__name__ == "TabPFNClassifier"
        ):
            print(
                "WARNING: TabPFN may take a long time to run on large datasets. Consider using alternatives (e.g. RandomForestClassifier)"
            )
        elif (
            X.shape[0] > 10000
            and self.base_classifier.__class__.__name__ == "TabPFNClassifier"
        ):
            print(
                "WARNING: CAAFE may take a long time to run on large datasets."
            )

        ds = [
            "dataset",
            X,
            y,
            [],
            self.feature_names + [target_name],
            {},
            dataset_description,
        ]
        # Add X and y as one dataframe
        df_train = pd.DataFrame(
            X,
            columns=self.feature_names,
        )
        df_train[target_name] = y

        if disable_caafe:
            self.code = ""
        else:
            self.code, prompt, messages, new_f_code, log_feat_eng, log_errors = racaafe.generate_features(
                ds,
                df_train,
                self.collection,
                exp_type,
                use_experience,
                model=self.llm_model,
                iterative=self.iterations,
                metric_used=auc_metric,
                iterative_method=self.base_classifier,
                display_method="print",
                n_splits=self.n_splits,
                n_repeats=self.n_repeats,
                keep_logs=keep_logs,
            )

            # len(new_f_code.keys()) > 0 = Check if any engineered features were 
            # kept
            if store_experience and len(new_f_code.keys()) > 0:
                samples = racaafe.get_data_samples(df_train)    

                f_importances = racaafe.get_feat_importances(
                    self.code,
                    new_f_code,
                    df_train,
                    ds,
                    self.base_classifier,
                    auc_metric,
                    samples,
                )

                # Only proceed if at least one added features has a positive feature
                # importance.
                if f_importances.apply(lambda x: x > 0).any(axis=1).values[0]:
                    racaafe.store_experiences(
                        new_f_code, f_importances, samples, ds, self.embed_model, self.collection
                    )

        df_train = run_llm_code(
            self.code,
            df_train,
        )

        df_train, _, self.mappings = make_datasets_numeric(
            df_train,
            df_test=None,
            target_column=target_name,
            return_mappings=True,
        )

        df_train, y = split_target_column(df_train, target_name)

        X, y = df_train.values, y.values.astype(int)
        # Check that X and y have correct shape
        # X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.base_classifier.fit(X, y)

        # Return the classifier
        return self

    def gain_experience(self, datasets, exp_type: str, use_experience: bool = True, keep_logs: bool = False):
        for dataset in datasets:

            ds, df_train, df_test, _, _ = data.get_data_split(dataset, seed=0)

            print("==========================================================")
            print(f"Dataset: {ds[0]}")

            try: 
                self.code, prompt, messages, new_f_code, log_feat_eng, log_errors = racaafe.generate_features(
                    ds,
                    df_train,
                    self.collection,
                    exp_type,
                    use_experience,
                    model=self.llm_model,
                    iterative=self.iterations,
                    metric_used=auc_metric,
                    iterative_method=self.base_classifier,
                    display_method="print",
                    n_splits=self.n_splits,
                    n_repeats=self.n_repeats,
                    keep_logs=keep_logs,
                )
            # TODO: does this work as expected? No persistent states that need
            # to be resetted manually?
            except Exception as e:
                print(f'In dataset {ds[0]}, the following exception occured:\n{e}')
                print('\nRetrying once')

                self.code, prompt, messages, new_f_code, log_feat_eng, log_errors = racaafe.generate_features(
                    ds,
                    df_train,
                    self.collection,
                    exp_type,
                    use_experience,
                    model=self.llm_model,
                    iterative=self.iterations,
                    metric_used=auc_metric,
                    iterative_method=self.base_classifier,
                    display_method="print",
                    n_splits=self.n_splits,
                    n_repeats=self.n_repeats,
                    keep_logs=keep_logs,
                )
            
            # len(new_f_code.keys()) > 0 = Check if any engineered features were 
            # kept
            if len(new_f_code.keys()) > 0:
                samples = racaafe.get_data_samples(df_train)    

                f_importances = racaafe.get_feat_importances(
                    self.code,
                    new_f_code,
                    df_train,
                    ds,
                    self.base_classifier,
                    auc_metric,
                    samples,
                )

                # Only proceed if at least one added features has a positive feature
                # importance.
                if f_importances.apply(lambda x: x > 0).any(axis=1).values[0]:
                    racaafe.store_experiences(
                        new_f_code, f_importances, samples, ds, self.embed_model, self.collection
                    )

        if keep_logs:
            return log_feat_eng, log_errors
        
        return None, None

    def gain_insights(
        self, return_ids: bool = False
    ) -> Tuple[str, Union[list[str], None]]:
        ex_doc_ids = racaafe.get_example_ids(self.collection)
        if len(ex_doc_ids) == 0:
            return "No insights to add", None

        col = self.collection.get(
            ids=ex_doc_ids, include=["documents", "metadatas"]
        )
        new_insights = []
        new_ids = []
        new_metadatas = []
        for i in range(len(col["ids"])):
            doc = col["documents"][i]
            doc_id = col["ids"][i]
            metadata = col["metadatas"][i]

            insights = racaafe.extract_insight(doc)

            start_context = doc.index("Context: ###")
            context = doc[start_context:]
            end_context = context.index("###", len("Context: ###"))
            context = context[: end_context + len("###")]

            new_insight = context + f"\n\nInsights: ###\n\n{insights}\n\n###"

            doc_id_split = doc_id.split("---")
            new_id = "---".join([doc_id_split[0], "insight", doc_id_split[2]])
            
            new_insights.append(new_insight)
            new_ids.append(new_id)
            new_metadatas.append(
                {"exp_type": "insight", "dataset": metadata["dataset"]}
            )

        try: 
            self.collection.add(
                documents=new_insights,
                metadatas=new_metadatas,
                ids=new_ids,
            )
        except Exception as e:
            raise RuntimeError(f'Could not add add insights. ChromaDB raised:\n{e}')

        if return_ids:
            return f"Added {len(new_ids)} new insights", new_ids
        else:
            return f"Added {len(new_ids)} new insights", None

    def generate_features(self, ds, df_train, exp_type, use_experience, store_experience, keep_logs: bool = False):
        try: 
            self.code, prompt, messages, new_f_code, log_feat_eng, log_errors = racaafe.generate_features(
                ds,
                df_train,
                self.collection,
                exp_type,
                # use_experience,
                model=self.llm_model,
                iterative=self.iterations,
                metric_used=auc_metric,
                iterative_method=self.base_classifier,
                display_method="print",
                n_splits=self.n_splits,
                n_repeats=self.n_repeats,
                keep_logs=keep_logs,
            )
        # TODO: does this work as expected? No persistent states that need
        # to be resetted manually?
        except Exception as e:
            print(f'In dataset {ds[0]}, the following exception occured:\n{e}')
            print('\nRetrying once')

            self.code, prompt, messages, new_f_code, log_feat_eng, log_errors = racaafe.generate_features(
                ds,
                df_train,
                self.collection,
                exp_type,
                use_experience,
                model=self.llm_model,
                iterative=self.iterations,
                metric_used=auc_metric,
                iterative_method=self.base_classifier,
                display_method="print",
                n_splits=self.n_splits,
                n_repeats=self.n_repeats,
                keep_logs=keep_logs,
            )

        # len(new_f_code.keys()) > 0 = Check if any engineered features were 
        # kept
        if store_experience and len(new_f_code.keys()) > 0:
            samples = racaafe.get_data_samples(df_train)    

            f_importances = racaafe.get_feat_importances(
                self.code,
                new_f_code,
                df_train,
                ds,
                self.base_classifier,
                auc_metric,
                samples,
            )

            # Only proceed if at least one added features has a positive feature
            # importance.
            if f_importances.apply(lambda x: x > 0).any(axis=1).values[0]:
                racaafe.store_experiences(
                    new_f_code, f_importances, samples, ds, self.embed_model, self.collection
                )

        return self.code, prompt, messages, new_f_code
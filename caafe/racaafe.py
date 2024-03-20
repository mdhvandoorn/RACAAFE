def get_examples():
    f"""
Given the same instructions, these feature engineering operations on other datasets 
resulted in the biggest predictive performance gain:
...
    """


def get_insights():
    f"""
After analysing the most succesful previous feature engineering operations, 
given the same initial instructions but applied to other datasets, the 
following insights were extracted:
...
    """

# Use ChromaDB metadata to specify retrieval of insights vs. examples.
def retrieve_docs(ds):
    query_results = collection.query(
    query_texts=queries,
    n_results=4,
    )

def get_prompt(
    df, ds, examples, insights, iterative=1, data_description_unparsed=None, samples=None, **kwargs
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

{rag}

Each codeblock generates {how_many} and can drop unused columns (Feature selection).
Each codeblock ends with ```end and starts with "```python"
Codeblock:
"""


def build_prompt_from_df():
    pass


def generate_features():
    pass

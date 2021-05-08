import h2o
import streamlit as st
from h2o.automl import H2OAutoML

# Start h2o server with 6GB ram and all available threads
h2o.init(max_mem_size='6G', nthreads=-1)


@st.cache(suppress_st_warning=True)
def train_model(df, target_var, target_is_cat, max_runtime=30):
    """
    Trains an autoML model
    :param df: (pd.DataFrame) dataframe to train models on
    :param target_var: (str) name of the target column (y)
    :param target_is_cat: (bool) target column is categorical
    :param max_runtime: (int) maximum runtime allowed in secs
    :return: aml (h2o.H2OAutoML) - trained model object
    """
    # Convert pandas dataframe to h2o frame
    x_train = h2o.H2OFrame(df)

    # Convert target column in dataframe to factor type is it is categorical
    if target_is_cat is True:
        x_train[target_var] = x_train[target_var].asfactor()

    # Initialize AutoML model
    aml = H2OAutoML(max_runtime_secs=max_runtime, seed=42, exclude_algos=["StackedEnsemble"])

    # Remove target col from list of cols to get training cols
    x_cols = x_train.columns
    x_cols.remove(target_var)

    # Train the models
    training_text = st.text("Training started. Wait for " + str(max_runtime) + " seconds.")
    aml.train(x=x_cols, y=target_var, training_frame=x_train)
    training_text = st.text("Training completed...!")

    return aml


@st.cache
def predict(df, aml):
    """
    Takes a preprocessed dataset of the same type as in aml and makes a prediction on it
    :param df: (pd.DataFrame) preprocessed dataframe to be predicted upon
    :param aml: (h2o.H2OAutoML) trained model object from h2o
    :return: data_as_df (pd.DataFrame) - prediction result as a dataframe
    """
    result = aml.leader.predict(h2o.H2OFrame(df))

    # Change datatype of result from h2oframe to pd.DataFrame
    data_as_df = h2o.as_list(result)

    return data_as_df

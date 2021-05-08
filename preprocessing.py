import pandas as pd


def custom_preprocess(df, cat_cols, dt_cols, target_col, is_test=False):
    """
    Converts columns to their corresponding datatype. Works for both
    train and test datasets. Set flag `is_test` for test dataset
    :param df: (pd.DataFrame) The dataframe to be preprocessed
    :param cat_cols: (list) categorical columns
    :param dt_cols: (list) datetime columns
    :param target_col: (str) the target column (y)
    :param is_test: (bool) true if df is test dataset
    :return: df (pd.DataFrame) - preprocessed dataset,
    target_is_col (bool) - target column is categorical
    """
    # Numerical columns is all the remaining cols
    num_cols = list(set(df.columns).difference(cat_cols).difference(dt_cols))

    # Have to remove the target_col from the list of columns before preprocessing.
    # Failure to do so will result in index error as target_col is not present in
    # test dataframe.
    if target_col in cat_cols:
        target_is_cat = True
        if is_test:
            cat_cols = set(cat_cols).difference([target_col])
    else:
        target_is_cat = False
        if is_test:
            num_cols = set(num_cols).difference([target_col])

    # Changing datatype of columns in dataframe
    for col in cat_cols:
        df[col] = df[col].astype('category')
    for col in dt_cols:
        df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
    for col in num_cols:
        df[col] = df[col].astype('float64')

    return df, target_is_cat

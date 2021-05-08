from zipfile import ZipFile

import pandas as pd
import streamlit as st

import SessionState
from preprocessing import custom_preprocess
from training import train_model, predict

pd.options.mode.use_inf_as_na = True


@st.cache
def check_filetype(filename):
    """
    Takes filename string and extracts its extension.

    :param filename: (str) string denoting the filename
    :return: (str) the extension of the file
    """

    _namesplit = filename.split('.')

    return _namesplit[-1]


def main():
    upload_display_string = "Choose a file (.xlsx/.xls/.csv/.zip). In case of zip files, please make sure that only one" \
                            "file of the type (.xlsx/.xls/.csv) is present in the zip file."

    # Upload the train file
    uploaded_file = st.file_uploader(upload_display_string, type=['xlsx', 'csv', '.zip', '.xls'], key="upload1")
    # Preserve state between sessions.
    # train_uploaded - train file has been uploaded
    # done_selection - columns have been selected
    sessionstate = SessionState.get(train_uploaded=False, done_selection=False)

    if uploaded_file is not None or sessionstate.train_uploaded:
        sessionstate.train_uploaded = True
        test_display = st.text(uploaded_file.name + " has been uploaded successfully. Processing....")
        # check filetype of uploaded file
        filetype = check_filetype(uploaded_file.name)
        if filetype == 'csv':
            df = pd.read_csv(uploaded_file)
        elif filetype == 'xlsx':
            df = pd.read_excel(uploaded_file)
        # If file is inside a zipfile, extract and find extension
        else:
            target_file = None
            with ZipFile(uploaded_file, 'r') as zipObj:
                _listoffiles = zipObj.namelist()
                for filename in _listoffiles:
                    if filename.endswith('.csv') or filename.endswith('.xlsx') or filename.endswith('.xls'):
                        zipObj.extract(filename)
                        target_file = filename
                        break
            if target_file is not None:
                if check_filetype(target_file) == 'csv':
                    df = pd.read_csv(target_file)
                elif check_filetype(target_file) == 'xlsx':
                    df = pd.read_excel(target_file)

            else:
                st.text("The zip file uploaded does not contain a .csv/.xlsx/.xls file")
                st.stop()
        test_display.empty()
        # Display train dataframe
        st.dataframe(df[:100])

        # Select categorical , datetime cols. Select target variable (y)
        cat_cols = st.multiselect('Select columns that are categories (not numerical)',
                                  list(df.columns), key="cat_cols")
        dt_cols = st.multiselect('Select columns that represent date/time',
                                 list(set(df.columns).difference(cat_cols)), key="dt_cols")
        target_var = str(st.selectbox('Select the column that you want to predict', list(df.columns), key="target"))

        # Start preprocessing if columns are selected
        if st.button("Done selecting columns") or sessionstate.done_selection:
            sessionstate.done_selection = True
            preprocessed_df, target_is_col = custom_preprocess(df, cat_cols, dt_cols, target_var)
        else:
            st.stop()

        # Upload the file to predict values using trained model
        uploaded_file_predict = st.file_uploader(upload_display_string, type=['xlsx', 'csv', '.zip', '.xls'],
                                                 key="upload2")
        if uploaded_file_predict is not None:
            test_display_predict = st.text(
                uploaded_file_predict.name + " has been uploaded successfully. Processing....")
            # Check filetype of the file uploaded
            filetype = check_filetype(uploaded_file_predict.name)
            if filetype == 'csv':
                df_predict = pd.read_csv(uploaded_file_predict)
            elif filetype == 'xlsx':
                df_predict = pd.read_excel(uploaded_file_predict)
            # Check filetype if file is inside a zip file
            else:
                target_file = None
                with ZipFile(uploaded_file_predict, 'r') as zipObj:
                    _listoffiles = zipObj.namelist()
                    for filename in _listoffiles:
                        if filename.endswith('.csv') or filename.endswith('.xlsx') or filename.endswith('.xls'):
                            zipObj.extract(filename)
                            target_file = filename
                            break
                if target_file is not None:
                    if check_filetype(target_file) == 'csv':
                        df_predict = pd.read_csv(target_file)
                    elif check_filetype(target_file) == 'xlsx':
                        df_predict = pd.read_excel(target_file)

                    test_display_predict.empty()
                    st.dataframe(df_predict.head())

                else:
                    st.text("The zip file uploaded does not contain a .csv/.xlsx/.xls file")
                    test_display_predict.empty()
                    st.stop()

            # Train model on training data
            aml = train_model(preprocessed_df, target_var, target_is_col, max_runtime=600)

            # Preprocess prediction data to bring it to the same format as model
            df_predict_processed, _ = custom_preprocess(df_predict, cat_cols, dt_cols, target_var, is_test=True)

            # Get and display result as a dataframe
            result = predict(df_predict_processed, aml)
            st.dataframe(result)
        st.stop()


if __name__ == '__main__':
    main()

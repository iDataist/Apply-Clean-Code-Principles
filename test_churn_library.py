import os
import numpy as np
import logging
import pytest
import churn_library
import constants

logging.basicConfig(
    level=logging.INFO,
    filename="logs/churn_library.log",
    format="%(asctime)-15s %(message)s",
)
logger = logging.getLogger()


def test_import_data():
    """
    test data import
    """
    try:
        df = churn_library.import_data(constants.data_path)
        pytest.df = df
        logger.info("Testing import_data: SUCCESS")
    except Exception as err:
        logger.error("Testing import_eda: {err}")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logger.error(
            "Testing import_data: \
            the file doesn't appear to have rows and columns"
        )
        raise err


def test_eda():
    """
    test perform eda function
    """
    try:
        churn_library.perform_eda(pytest.df, constants.eda_output_path)
        assert os.path.isfile(constants.eda_output_path)
        logger.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logger.error(
            "Testing perform_eda: the eda_output report can't be found"
        )
        raise err


def test_scaler():
    """
    test scaler
    """
    try:
        scaled_df = churn_library.scaler(pytest.df, constants.quant_columns)
        pytest.scaled_df = scaled_df
        logger.info("Testing scaler: SUCCESS")
    except ValueError as err:
        logger.error("Testing scaler: {err}")
        raise err

    try:
        assert scaled_df.shape[0] == pytest.df.shape[0]
        assert scaled_df.shape[1] == pytest.df.shape[1]
    except AssertionError as err:
        logger.error("Testing scaler: the shape of the dataframe changed")
        raise err


def test_encoder():
    """
    test encoder
    """
    try:
        encoded_df = churn_library.encoder(
            pytest.scaled_df, constants.cat_columns
        )
        pytest.encoded_df = encoded_df
        logger.info("Testing encoder: SUCCESS")
    except Exception as err:
        logger.error("Testing encoder: {err}")
        raise err

    try:
        assert encoded_df.select_dtypes(exclude=[np.number]).shape[1] == 0
    except AssertionError as err:
        logger.error("Testing encoder: there are still non-numerical values")
        raise err


def test_perform_train_test_split():
    """
    test encoder
    """
    try:
        split = churn_library.perform_train_test_split(
            pytest.encoded_df,
            constants.target,
            constants.test_size,
            constants.random_state,
        )
        pytest.split = split
        X_train, X_test, y_train, y_test = split
        logger.info("Testing perform_train_test_split: SUCCESS")
    except Exception as err:
        logger.error(f"Testing perform_train_test_split: {err}")
        raise err

    try:
        assert len(X_train) + len(X_test) == len(pytest.df)
        assert len(y_train) + len(y_test) == len(pytest.df)
        assert round(len(X_test) / len(pytest.df), 1) == constants.test_size
    except AssertionError as err:
        logger.error("Testing perform_train_test_split: {err}")
        raise err


def test_train_models():
    """
    test train_models
    """
    X_train, X_test, y_train, y_test = pytest.split
    try:
        churn_library.train_models(
            X_train,
            X_test,
            y_train,
            y_test,
            constants.image_output_path,
            constants.model_output_path,
        )
        logger.info("Testing train_models: SUCCESS")
    except Exception as err:
        logger.error(err)
        raise err

    try:
        assert os.path.isfile(constants.model_output_path)
    except AssertionError as err:
        logger.error("Testing train_models: {err}")
        raise err


if __name__ == "__main__":
    logger.info("############################################################")
    logger.info("start tests")
    pytest.main(args=['-s', os.path.abspath(__file__)])
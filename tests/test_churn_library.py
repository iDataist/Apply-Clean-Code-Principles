import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import churn_library
import constants

logging.basicConfig(
    level=logging.INFO,
    filename="logs/test_churn_library.log",
    format="%(asctime)-15s %(message)s",
)
logger = logging.getLogger()
data = pd.read_csv(constants.data_path).iloc[:, 2:]
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(columns=['Attrition_Flag']), data['Attrition_Flag'].ravel()
)


def test_import():
    """
    test data import
    """
    try:
        df = churn_library.import_data(constants.data_path)
        logger.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logger.error("Testing import_eda: the file wasn't found")
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
        churn_library.perform_eda(data, constants.eda_output_path)
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
        df = churn_library.scaler(data, constants.quant_columns)
        logger.info("Testing scaler: SUCCESS")
    except ValueError as err:
        logger.error(
            "Testing scaler: at lease one of the columns is not numerical"
        )
        raise err

    try:
        assert df.shape[0] == data.shape[0]
        assert df.shape[1] == data.shape[1]
    except AssertionError as err:
        logger.error("Testing scaler: the shape of the dataframe changed")
        raise err


def test_encoder():
    """
    test encoder
    """
    try:
        df = churn_library.encoder(data, constants.cat_columns)
        logger.info("Testing encoder: SUCCESS")
    except KeyError as err:
        logger.error("Testing encoder: the column names do not exist")
        raise err

    try:
        assert df.select_dtypes(exclude=[np.number]).shape[1] == 0
    except AssertionError as err:
        logger.error("Testing encoder: there are still non-numerical values")
        raise err


def test_train_models():
    """
    test train_models
    """
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
        logger.error("Testing train_models: model was not saved")
        raise err


if __name__ == "__main__":
    pass

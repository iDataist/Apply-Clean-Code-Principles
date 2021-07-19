"""
libraries
"""
# import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_profiling
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import logging
import constants

logging.basicConfig(
    level=logging.INFO,
    filename="logs/churn_library.log",
    format="%(asctime)-15s %(message)s",
)
logger = logging.getLogger()


def import_data(path):
    """
    returns dataframe for the csv found at path

    input:
            path: a path to the csv
    output:
            df: pandas dataframe
    """
    df = pd.read_csv(path)
    return df.iloc[:, 2:]


def perform_eda(df, output_path):
    """
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe
            output_path: path to store the eda report

    output:
            None
    """
    profile = pandas_profiling.ProfileReport(df)
    profile.to_file(output_path)


def scaler(df, quant_columns):
    """
    helper function to normalize each numerical column
    input:
            df: pandas dataframe
    output:
            df: normalized pandas dataframe
    """
    df[quant_columns] = StandardScaler().fit_transform(df[quant_columns])
    return df


def encoder(df, cat_columns):
    """
    helper function to one-hot-encode each categorical column
    input:
            df: pandas dataframe
    output:
            df: one-hot-encoded pandas dataframe
    """
    return pd.get_dummies(df, columns=cat_columns, drop_first=True)


def perform_train_test_split(df, target, test_size, random_state):
    """
    input:
              df: pandas dataframe

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
              target: target column
    """
    X = df.drop(columns=[target])
    y = df[target].ravel()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


def classification_report_image(
    y_train, y_test, y_train_preds, y_test_preds, output_path
):

    """
    produces classification report for training and testing results
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds: training predictions
            y_test_preds: test predictions
            output_path: path to store the figure
    output:
            None
    """

    plt.rc("figure", figsize=(7, 5))
    plt.text(
        0.01, 1.1, str("Train"), {"fontsize": 10}, fontproperties="monospace"
    )
    plt.text(
        0.01,
        0.7,
        str(classification_report(y_train, y_train_preds)),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01, 0.5, str("Test"), {"fontsize": 10}, fontproperties="monospace"
    )
    plt.text(
        0.01,
        0.1,
        str(classification_report(y_test, y_test_preds)),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.axis("off")
    plt.savefig(output_path + "classification_report.png")
    plt.close()


def feature_importance_plot(model, X, output_path):
    """
    creates and stores the feature importances in output_path
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_path: path to store the figure

    output:
            None
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [X.columns[i] for i in indices]
    plt.figure(figsize=(20, 20))
    plt.title("Feature Importance")
    plt.ylabel("Importance")
    plt.bar(range(X.shape[1]), importances[indices])
    plt.xticks(range(X.shape[1]), names, rotation=75)
    plt.savefig(output_path + "feature_importance.png")
    plt.close()


def train_models(
    X_train, X_test, y_train, y_test, image_output_path, model_output_path
):
    """
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
              image_output_path: path to store the figures
              model_output_path: path to store the models
    output:
              best_model
    """
    rfc = RandomForestClassifier(random_state=42)
    param_grid = {"n_estimators": [50, 100, 200], "max_depth": [2, 8, 16]}
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    best_model = cv_rfc.best_estimator_

    # creates and stores the feature importances
    feature_importance_plot(best_model, X_test, image_output_path)
    # produces classification report for training and testing results
    y_train_preds = best_model.predict(X_train)
    y_test_preds = best_model.predict(X_test)
    classification_report_image(
        y_train, y_test, y_train_preds, y_test_preds, image_output_path
    )
    # saves best model
    joblib.dump(best_model, model_output_path)


if __name__ == "__main__":
    logger.info("############################################################")
    logger.info("import data")
    df = import_data(constants.data_path)
    logger.info(
        f"inspect dataframe: \
        \n{df.iloc[0]}"
    )
    logger.info(f"generate EDA report: {constants.eda_output_path}")
    perform_eda(df, constants.eda_output_path)
    logger.info(f"normalize numeric features: {constants.quant_columns}")
    df = scaler(df, constants.quant_columns)
    logger.info(
        f"inspect dataframe: \
        \n{df.iloc[0]}"
    )
    logger.info(f"one-hot-encode categorical features:{constants.cat_columns}")
    df = encoder(df, constants.cat_columns)
    logger.info(
        f"inspect dataframe: \
        \n{df.iloc[0]}"
    )
    logger.info(
        f"perform train test split with the test size of {constants.test_size}"
    )
    X_train, X_test, y_train, y_test = perform_train_test_split(
        df, constants.target, constants.test_size, constants.random_state
    )
    logger.info("start training")
    train_models(
        X_train,
        X_test,
        y_train,
        y_test,
        constants.image_output_path,
        constants.model_output_path,
    )
    logger.info(
        f"save models in {constants.model_output_path}, "
        + f"store results in {constants.image_output_path}"
    )

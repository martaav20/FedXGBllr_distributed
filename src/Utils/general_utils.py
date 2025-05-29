from sklearn.metrics import mean_squared_error, accuracy_score
import xgboost as xgb
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt


def load_unsurance():
    """
    https://github.com/Athpr123/Binary-Classification-Using-Machine-learning/blob/master/Travel%20Insurance%20Claim%20Status.ipynb
    """

    def age_convert(age):
        result = ""
        if age <= 21:
            result = "Child"
        elif age <= 50:
            result = "Adult"
        else:
            result = "Senior"
        return result

    def Coutry_Categories(value):
        result = ""
        if value >= 0.3:
            result = "(1) High Risk - More than 30% Claimed"
        elif value >= 0.2:
            result = "(2) Medium Risk - More than 20% Claimed"
        elif value > 0:
            result = "(3) Low Risk - More than 1% Claimed"
        else:
            result = "(4) No Risk Countries"
        return result

    df = pd.read_csv(
        "C:/Users/ssava/Desktop/FedXGBoost/dataset/classification/insurance-dataset.csv"
    )

    df["Age Group"] = df["Age"].map(lambda x: age_convert(x))
    # Dropping Feature Gender
    df.drop("Gender", axis=1, inplace=True)
    # Since the minimum duration that any travel can have is 1 day thus we impute it by the column median.
    df["Duration"][df["Duration"] < 0] = df["Duration"].median()
    # As we observed duration of any travel cannot be more than 731 we will impute it as 731.
    df["Duration"][df["Duration"] > 731] = 731
    # replacing the values that is greater than 99 with the mean of Senior Age
    df["Age"][df["Age"] > 99] = df[df["Age Group"] == "Senior"]["Age"].mean()

    df["Destination_risk"] = df.groupby("Destination")["Claim"].transform("mean")
    df["Risk of Countries"] = df["Destination_risk"].map(lambda x: Coutry_Categories(x))

    fe = df.groupby("Destination").size() / len(df)
    df.loc[:, "Dest_fe"] = df["Destination"].map(fe)
    fe_1 = df.groupby("Agency").size() / len(df)
    df.loc[:, "Agency_fe"] = df["Agency"].map(fe_1)
    fe_2 = df.groupby("Product Name").size() / len(df)
    df.loc[:, "Product Name_fe"] = df["Product Name"].map(fe_2)
    df.drop(columns="Agency", axis=1, inplace=True)
    df.drop(columns="Destination", axis=1, inplace=True)
    df.drop(columns="Product Name", axis=1, inplace=True)
    df.drop(columns="Age Group", axis=1, inplace=True)
    df = pd.get_dummies(
        df, columns=["Agency Type", "Distribution Channel"], drop_first=True
    )

    df.drop(columns=["Risk of Countries"], axis=1, inplace=True)
    df.drop(columns=["Destination_risk"], axis=1, inplace=True)
    df.drop(columns=["ID"], axis=1, inplace=True)
    # df = pd.get_dummies(
    #    df, columns=["Agency Type", "Distribution Channel"], drop_first=True
    # )
    X = df.drop("Claim", axis=1)
    Y = df["Claim"]
    return X, Y


def accuracy(y_true, y_pred):
    """Accuracy classification score."""
    error = accuracy_score(y_true, y_pred)
    # print(f"Accuracy: {error :.5f}")
    return error


def mse(y_true, y_pred):
    """Mean squared error."""
    error = mean_squared_error(y_true, y_pred)
    # print(f"MSE: {error :.5f}")
    return error


def get_basescore(model):
    """Get base score from an XGBoost sklearn estimator."""
    base_score = float(
        json.loads(model.get_booster().save_config())["learner"]["learner_model_param"][
            "base_score"
        ]
    )
    return base_score


def get_trees_predictions_xgb(X, objective, *models, numclasses = None, reshape_enabled = None):
    """
    Get predictions for each tree in each model.
    Alternatively:
    preds = []
    for n in range(n_trees):
        preds.append(
            reg.predict(
                sample,
                iteration_range=(n, n + 1),
            )
        )
    """
    xm = xgb.DMatrix(X, base_margin=np.zeros(len(X), dtype=np.float32))
    #for model in models:
    #    for booster in model.get_booster():
    #        a = booster.predict(xm)
    trees_predictions = np.array(
        [booster.predict(xm) for model in models for booster in model.get_booster()]
    ).T
    if numclasses is not None and reshape_enabled is True:
        # reshape (filter size = number of trees per client)
        trees_predictions = trees_predictions.transpose()
        trees_predictions = np.reshape(trees_predictions, (numclasses,int(trees_predictions.shape[0]/numclasses), trees_predictions.shape[1]))
        reordered_arr = np.transpose(trees_predictions, axes=[1, 0, 2])
        trees_predictions = np.reshape(reordered_arr,(trees_predictions.shape[0]*trees_predictions.shape[1],trees_predictions.shape[2]))
        trees_predictions = trees_predictions.transpose()

    if objective == "binary":
        trees_predictions = trees_predictions >= 0.5  # hard margin inputs
    elif objective == "multiclass": # only for regression inputs
        # trees_predictions = np.rint(trees_predictions)
        trees_predictions = np.clip(trees_predictions, 0, numclasses - 1)   
    elif objective == "soft":
        ampl_f = 10
        for k in range(trees_predictions.shape[0]):
            # trees_predictions[k,:] = ampl_f*(trees_predictions[k,:]-0.5)
            trees_predictions[k,:] = np.tanh(ampl_f*(trees_predictions[k,:]-0.5))            
    return trees_predictions  


def get_trees_predictions_rf(X, objective, *models):
    """
    Get predictions for each tree in each model.
    Alternatively:

    """
    trees_predictions = np.array(
        [tree.predict(X) for model in models for tree in model.estimators_]
    ).T

    # if objective == "binary":
    #    trees_predictions = trees_predictions >= 0.5  # hard margin inputs

    return trees_predictions  # shape (n_samples, n_trees * n_models)

def display_trees(model):
    """Display trees in the model."""
    trees = model.get_booster().get_dump()
    # 'trees' is a list where each element is a string representation of an individual tree
    for i, tree in enumerate(trees):
        print(f"Tree {i + 1}:\n{tree}\n")


def trees_importance(model_global, num_clients, trees_client, view_n_trees=None):
    if view_n_trees is None:  # view all the tree weights
        view_n_trees = num_clients * trees_client

    feature_names = np.array(
        [f"T_{c}_{t+1}" for c in range(num_clients) for t in range(trees_client)]
    )
    weights = model_global.get_weights()[0]

    abs_weights = np.abs(weights)
    # Get the indices that would sort the abs_weights array
    sorted_indices = np.argsort(
        abs_weights[:, 0],
    )[-view_n_trees:]

    # Sort the abs_weights and feature indices accordingly
    sorted_weights = weights[sorted_indices, 0]
    sorted_features = feature_names[sorted_indices]

    # Create a bar plot
    plt.figure(figsize=(6, view_n_trees / 5))  # Adjust the width as needed
    plt.barh(sorted_features, sorted_weights)
    plt.title("Magnitudes of Weights")
    plt.xlabel("Weight Value")
    plt.ylabel("Tree Index")
    # plt.yticks(sorted_feature_indices)  # Display feature indices on x-axis
    plt.ylim(-0.5, view_n_trees)
    plt.show()


def importance_per_client(model_global, num_clients, trees_client, view_n_trees=None):
    if view_n_trees is None:  # view all the tree weights
        view_n_trees = trees_client

    feature_names = np.array(
        [f"T_{c}_{t+1}" for c in range(num_clients) for t in range(trees_client)]
    )
    weights = model_global.get_weights()[0]
    #  Categorical Data

    plt.figure(figsize=(num_clients * 6, view_n_trees / 5))
    # plt.suptitle("SUBPLOTS with FOR loop - example #1", fontsize=18)

    for i in range(num_clients):
        plt.subplot(1, num_clients, i + 1)

        client_ind = slice(i * trees_client, (i + 1) * trees_client)
        weights_client = weights[client_ind]
        features_client = feature_names[client_ind]

        abs_weights = np.abs(weights_client)
        # Get the indices that would sort the abs_weights array
        sorted_indices = np.argsort(
            abs_weights[:, 0],
        )[-view_n_trees:]

        # Sort the abs_weights and feature indices accordingly
        sorted_weights = weights_client[sorted_indices, 0]
        sorted_features = features_client[sorted_indices]

        plt.barh(sorted_features, sorted_weights)
        plt.title(f"Magnitudes of Weights Client {i}")
        plt.xlabel("Weight Value")
        plt.ylabel("Tree Index")
        plt.ylim(-0.5, view_n_trees)
        plt.xlim(
            left=min(np.min(weights), -np.max(weights)),
            right=max(np.max(weights), -np.min(weights)),
        )
    plt.tight_layout()
    plt.show()


import logging
import sys

def get_logging_level(level_str):
    levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }
    return levels.get(level_str, logging.DEBUG)

def setup_logger(name, level_str="debug"):
    logger = logging.getLogger(name)
    logger.setLevel(get_logging_level(level_str))
    stdout = logging.StreamHandler(sys.stdout)
    stdout.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(filename)s:%(lineno)s | %(message)s"))
    logger.addHandler(stdout)
    logger.propagate = False
    return logger

from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import os
import xgboost as xgb
import tensorflow.keras as tfk

def train_centralized_xgb(x_train, y_train, x_valid, y_valid, cfg, output_path=None):
    hyperparams = cfg["hyperparams"]

    reg = xgb.XGBClassifier(**hyperparams)
    reg.fit(x_train, y_train)
    y_pred = reg.predict(x_valid)

    acc = accuracy_score(y_valid, y_pred)
    cm = confusion_matrix(y_valid, y_pred)

    print(f"[Centralized] Accuracy: {acc:.2f}")
    # print("[Centralized] Confusion Matrix:")
    # print(cm)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(reg, output_path, compress=0)

    return reg, acc, cm

def save_server_model(model, path):
    """
    Save the server model to a specified path with a round number.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path, include_optimizer=True)

        

def get_client_model_path(client_id, round):
    """
    Get the path for a client's model based on client ID and round number.
    """
    return f"cnn_models/clients/round_{round}/CNN_client_model_{client_id}.h5"


def CNN(num_clients, trees_client, n_channels, objective, n_classes=None):
    # Define 1D-CNN
    model = tfk.models.Sequential()
    model.add(
        tfk.layers.Conv1D(
            n_channels,
            kernel_size=trees_client,
            strides=trees_client,
            activation="relu",
            input_shape=(num_clients * trees_client, 1),
        )
    )

    model.add(tfk.layers.Flatten())
    model.add(tfk.layers.Dense(n_channels * num_clients, activation="relu"))

    # Output layer
    if objective == "binary":
        model.add(tfk.layers.Dense(1, activation="sigmoid"))
        loss = "binary_crossentropy"
        metrics = ["accuracy"]
    elif objective == "regression":
        model.add(tfk.layers.Dense(1, activation="linear"))
        loss = "mse"
        metrics = [None]
    elif objective == "multiclass": # regression inputs
        model.add(tfk.layers.Dense(n_classes, activation="softmax"))
        loss = "categorical_crossentropy"
        metrics = ["accuracy"]
    # Compile the model

    opt = tfk.optimizers.Adam(learning_rate=0.01, beta_1=0.5, beta_2=0.999)
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    return model

def CNN_mc(num_clients, filter_size, trees_client, n_channels, n_classes):
    # Define 1D-CNN
    model = tfk.models.Sequential()
    if n_classes == 2:
        model.add(
        tfk.layers.Conv1D(
            n_channels,
            kernel_size=filter_size,
            strides=trees_client,
            activation="relu",
            input_shape=(num_clients * trees_client, 1),
            )
        )
    else:
        model.add(
        tfk.layers.Conv1D(
            n_channels,
            kernel_size=filter_size,
            strides=trees_client,
            activation="relu",
            input_shape=(num_clients * trees_client * n_classes, 1),
            )
        )

    model.add(tfk.layers.Flatten())
    model.add(tfk.layers.Dense(n_channels * num_clients, activation="relu"))

    # Output layer
    if n_classes == 2:
        model.add(tfk.layers.Dense(1, activation="sigmoid"))
        loss = "binary_crossentropy"
        metrics = ["accuracy"]
    else:
        model.add(tfk.layers.Dense(n_classes, activation="softmax"))
        loss = "categorical_crossentropy"
        metrics = ["accuracy"]
    # Compile the model

    opt = tfk.optimizers.Adam(learning_rate=0.01, beta_1=0.5, beta_2=0.999)
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    return model


def SimpleNN(num_clients, trees_client, objective, lbd=0, n_classes=None):
    # Define 1D-CNN
    model = tfk.models.Sequential()
    model.add(tfk.layers.Input(shape=(num_clients * trees_client,)))

    # Output layer
    if objective == "binary":
        model.add(
            tfk.layers.Dense(
                1, activation="sigmoid", kernel_regularizer=tfk.regularizers.l1(lbd)
            ),
        )  # Lasso
        loss = "binary_crossentropy"
        metrics = ["accuracy"]
    elif objective == "regression":
        model.add(tfk.layers.Dense(1, activation="linear"))
        loss = "mse"
        metrics = [None]
    elif objective == "multiclass":
        model.add(tfk.layers.Dense(n_classes, activation="softmax"))
        loss = "categorical_crossentropy"
        metrics = ["accuracy"]

    # Compile the model
    opt = tfk.optimizers.Adam(learning_rate=0.01, beta_1=0.5, beta_2=0.999)
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    return model
import os
import joblib
from src.Utils.dataset_utils import load_full_dataset
from src.Utils.model_utils import train_centralized_xgb, save_server_model, get_client_model_path, CNN_mc
from tensorflow.keras.models import load_model
import numpy as np 
import time

def server_process(cfg):
    
    round = 1
    num_clients = cfg["num_clients"]
    trees_client = cfg["trees_client"]
    num_classes = cfg["num_classes"]
    R = cfg["rounds"]
    
    # 1. Centralized XGboost model training
    x_train, x_valid, y_train, y_valid = load_full_dataset(cfg)
    _, acc_centralized, _ = train_centralized_xgb(
        x_train, y_train, x_valid, y_valid, cfg,
        output_path="src/Models/xgb_models/XGB_centralized_model.h5"
    )
    
    # 2 Create the aggregated XGBoost model 
    while True:
        if os.path.exists("src/Models/xgb_models/clients") and len(os.listdir("src/Models/xgb_models/clients")) == num_clients:
            time.sleep(1)  # wait for all clients to finish training
            break
        # Aggregate all the xgboost models from all clients 
    XGB_models = []
    for c in range(num_clients):
        checkpointpath = f'src/Models/xgb_models/clients/XGB_client_model_{c}.h5'
        xgb = joblib.load(checkpointpath)
        XGB_models.append(xgb)
        # Save the aggregated model
    checkpointpath2 = f'src/Models/xgb_models/server/XGB_aggregated_model.h5'
    if not os.path.exists(os.path.dirname(checkpointpath2)):
        os.makedirs(os.path.dirname(checkpointpath2))
    joblib.dump(XGB_models, checkpointpath2, compress=0)
    # Add a flag to indicate that the aggregated model is ready
    with open("model_ready.flag", "w") as f:
        f.write("done")
        
    filters = 32
    filter_size = trees_client
    params_cnn = (
        num_clients,
        filter_size,
        filter_size,
        filters,
        num_classes
    )
    
    model_global = CNN_mc(*params_cnn)
    num_layers = len(model_global.get_weights())

    
    # Save the model architecture so that clients can load it
    model_global_path = f"src/Models/cnn_models/server/round_1/CNN_global_model.h5"
    save_server_model(model_global, model_global_path)
   
    
    while round < R:
        # wait for all clients to finish training
        # wait for the folder to be created
        while not os.path.exists(f"src/Models/cnn_models/clients/round_{round}") and len(os.listdir(f"src/Models/cnn_models/clients/round_{round}")) < num_clients:
            pass
        
        models_clients = []
        # Load client models and aggregate them # PARTI DA QUI NON CARICA I MODELLI DEI CLIENT
        for i in range(num_clients):
            client_model_path = get_client_model_path(i, round)
            if os.path.exists(client_model_path):
                model_client = load_model(client_model_path)
                models_clients.append(model_client)
                
        global_weights = []
        for i in range(num_layers):  # aggregate the weights, no memory of prev global weights
            global_weights.append(
                np.sum([model.get_weights()[i] for model in models_clients], axis=0)
                / len(models_clients)
                )
        model_global.set_weights(global_weights)
        
        # save the aggregated global model
        model_global_path = f"src/Models/cnn_models/server/round_{round}/CNN_global_model.h5"
        save_server_model(model_global, model_global_path, round)        
                
                
                

import os
import joblib
from Utils.dataset_utils import load_full_dataset
from Utils.model_utils import train_centralized_xgb, save_server_model, get_client_model_path, CNN_mc
from tensorflow.keras.models import load_model
import numpy as np 

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
        output_path="../Models/xgb_models/server/XGB_centralized_model.h5"
    )
    
    # 2 Create the aggregated XGBoost model 
    while True:
        if len(os.listdir("../Models/xgb_models/clients")) == num_clients:
            break
        # Aggregate all the xgboost models from all clients 
    XGB_models = []
    for c in range(num_clients):
        checkpointpath = f'../Models/xgb_models/clients/XGB_client_model_{c}.h5'
        xgb = joblib.load(checkpointpath)
        XGB_models.append(xgb)
        # Save the aggregated model
    checkpointpath2 = f'../Models/xgb_models/server/XGB_aggregated_model.h5'
    joblib.dump(XGB_models, checkpointpath2, compress=0)
        
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
    model_global_path = f"../Models/cnn_models/server/round_1/CNN_global_model.keras"
    save_server_model(model_global, model_global_path, round=round)
   
    
    while round < R:
        # wait for all clients to finish training
        while len(os.listdir(f"cnn_models/clients/round_{round}")) < num_clients:
            pass
        
        models_clients = []
        # Load client models and aggregate them
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
        model_global_path = f"../Models/cnn_models/server/round_{round}/CNN_global_model.keras"
        save_server_model(model_global, model_global_path, round)        
                
                
                

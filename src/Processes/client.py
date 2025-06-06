import joblib
from src.Utils.dataset_utils import load_data_for_client
from src.Utils.model_utils import CNN_mc
from src.Utils.general_utils import get_trees_predictions_xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import xgboost as xgb
import numpy as np
from tensorflow.keras.models import load_model
import time
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='keras')

def client_process(cfg, client_id): 
    num_clients = cfg["num_clients"]
    num_classes = cfg["num_classes"]
    R = cfg["rounds"]
    E = cfg["epochs"]
    
    x_train_c, y_train_c, x_valid_c, y_valid_c = load_data_for_client(client_id)
    #size_training = len(y_train_c)
    
    # # TODO: non so se sia necessario salvare i dati in un file, ma per ora lo faccio
    # dir_train = f'src/Data/client_{client_id}/train/' # create a folder with the local data for the client
    # dir_valid = f'src/Data/client_{client_id}/valid/'
    # np.save(dir_train + f'x_train.npy', x_train_c)
    # np.save(dir_train + f'y_train.npy', y_train_c)
    # np.save(dir_valid + f'x_valid.npy', x_valid_c)
    # np.save(dir_valid + f'y_valid.npy', y_valid_c)
    print(f"Client {client_id} | Training samples: {len(y_train_c)}")
    print(f"Client {client_id} | Validation samples: {len(y_valid_c)}")
    
    
    # 1. Train local XGBoost model with the client data
    hyperparams = cfg["hyperparams"]   
    reg = xgb.XGBClassifier(**hyperparams)
    reg.fit(x_train_c, y_train_c)
        # Save the model
    checkpointpath = f"src/Models/xgb_models/clients/XGB_client_model_{client_id}.h5"
    if not os.path.exists(os.path.dirname(checkpointpath)):
        os.makedirs(os.path.dirname(checkpointpath))
    joblib.dump(reg, checkpointpath, compress=0)
        # full performance tests (accuracy and confusion matrix)
    y_pred = reg.predict(x_valid_c)
    error = accuracy_score(y_valid_c, y_pred)
    cm = confusion_matrix(y_valid_c, y_pred)
    print(f"xgboost classifier local model accuracy, (Client {client_id}): {100*error :.5f}%")
    
    # 3. Retrieve the aggregated XGBoost model and convert data for the client
    while not os.path.exists("model_ready.flag"):
        time.sleep(1)
    checkpointpath2 = f'src/Models/xgb_models/server/XGB_aggregated_model.h5'  
    xgb_aggregated = joblib.load(checkpointpath2) 
    print("Converting the data of client", client_id, 100 * "-")
    reshape_enabled = False 
    inputs_obj = "soft"
        # other options: 
        # objective = "soft" # applies a tanh activation to the xgboost tree soft outputs  
        # objective = "binary" # outputs of xgboost trees are binarized, 
    x_xgb_trees_out = get_trees_predictions_xgb(x_train_c, inputs_obj, *xgb_aggregated, numclasses=num_classes, reshape_enabled=reshape_enabled) 
    y_xgb_trees_out = y_train_c 
    xgb_valid_out = get_trees_predictions_xgb(x_valid_c, inputs_obj, *xgb_aggregated, numclasses=num_classes, reshape_enabled=reshape_enabled) 
    
   
    # 4. Initialize the CNN model parameters
    filters = 32
    filter_size = hyperparams["n_estimators"] # trees_client
    params_cnn = (
        num_clients,
        filter_size,
        filter_size,
        filters,
        num_classes
    )
    model_client = CNN_mc(*params_cnn)  # create a new mode
    
    # 5. Federated learning rounds
    for round in np.arange(1,R+1): 
        model_global_path = f"src/Models/cnn_models/server/round_{round}/CNN_global_model.h5" # TODO : change to .h5 if needed
        model_client_path = f"src/Models/cnn_models/clients/round_{round}/CNN_client_model_{client_id}.h5"
        
        # wait for the global model to be ready
        while not os.path.exists(model_global_path):
            time.sleep(1)
            
        # update phase 
        print(f"Round {round}/{R}, Client {client_id}/{num_clients}")
        
        # wait for the global model to be ready
        model_global = load_model(model_global_path) # Load the client model from the global model
        model_client.set_weights(model_global.get_weights())  # set global weights (no memory of prev local weights)
        # update phase
        model_client.fit(
            x_xgb_trees_out, y_xgb_trees_out, epochs=E, verbose=False
        )  # train the model on the client data
        # Print the accuracy of the model on x_xgb_trees_out 
        #loss_train, acc_train = model_client.evaluate(x_xgb_trees_out, y_xgb_trees_out, verbose=0)
        #print(f"Training accuracy and loss Client {client_id}, round {round}: {acc_train * 100:.2f}, {loss_train}%")
        
        # save the model
        model_client.save(model_client_path)
        print(f"Client {client_id} | Model saved at {model_client_path}")
        
        
    ## FINALE EVALUATION WITH VALIDATION DATA
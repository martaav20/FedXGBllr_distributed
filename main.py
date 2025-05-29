import sys
import os
import yaml
import multiprocessing
import argparse
import shutil
from src.Processes.server import server_process
from src.Processes.client import client_process
from src.Utils.dataset_utils import *
from src.Utils.general_utils import setup_logger

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if module_path not in sys.path:
    sys.path.append(module_path)
    
def main(): 
    parser = argparse.ArgumentParser(description='Setting up the Federated Learning simulation.')
    
    # Dataset parameters
    parser.add_argument('--num_classes', type=int, help='Number of output classes')
    parser.add_argument('--n_features', type=int, help='Number of features in the dataset')
    parser.add_argument('--n_redundant', type=int, help='Number of redundant features in the dataset')
    parser.add_argument('--test_size', type=float, help='Proportion of the dataset to include in the test split')
    parser.add_argument('--training_samples_tot', type=int, help='Number of training samples before splitting')
    # Note: n_samples is computed as round(training_samples_tot / (1 - test_size)), not a direct argument
    parser.add_argument('--random_state', type=int, help='Random state for reproducibility in dataset splitting')

    # Federated learning parameters
    parser.add_argument('--num_clients', type=int, help='Number of federated clients')
    parser.add_argument('--trees_client', type=int, help='Number of trees used by each client')
    parser.add_argument('--epochs', type=int, help='Number of federated training epochs')
    parser.add_argument('--rounds', type=int, default=0, help='Number of rounds of the federated learning process')
    # Note: samples is computed as round(training_samples_tot / num_clients), not a direct argument

    # Hyperparameters for the model (e.g., XGBoost)
    parser.add_argument('--objective', type=str, help='Objective function for the model')
    parser.add_argument('--n_estimators', type=int, help='Number of boosting rounds (trees)')
    parser.add_argument('--max_depth', type=int, help='Maximum depth of a tree')
    parser.add_argument('--learning_rate', type=float, help='Boosting learning rate (eta)')
    parser.add_argument('--base_score', type=float, help='Initial prediction score (global bias)')
    parser.add_argument('--model_random_state', type=int, help='Random state for the model')
    parser.add_argument("--verbose", type=str, help="Set verbosity level (debug, info, warning, error)")

    args = parser.parse_args()

    # set logging level
    level_str = args.verbose if args.verbose else "debug"
    logger = setup_logger("FedXGB_llr", level_str=level_str)
        
    # load also config file
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, 'config.yaml')
    with open(file_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    # Set configuration parameters
    # Dataset parameters (not useful if the dataset is imported)
    config['num_classes'] = args.num_classes if args.num_classes is not None else config['num_classes']
    config['n_features'] = args.n_features if args.n_features is not None else config['n_features']
    config['n_redundant'] = args.n_redundant if args.n_redundant is not None else config['n_redundant']
    config['test_size'] = args.test_size if args.test_size is not None else config['test_size']
    config['training_samples_tot'] = args.training_samples_tot if args.training_samples_tot is not None else config['training_samples_tot']
    config['random_state'] = args.random_state if args.random_state is not None else config['random_state']

    # Federated learning parameters
    config['num_clients'] = args.num_clients if args.num_clients is not None else config['num_clients']
    config['trees_client'] = args.trees_client if args.trees_client is not None else config['trees_client']
    config['epochs'] = args.epochs if args.epochs is not None else config['epochs']
    config['num_rounds'] = args.rounds if args.rounds is not None else config['num_rounds']

    # Hyperparameters
    config['hyperparams']['objective'] = args.objective if args.objective is not None else config['hyperparams']['objective']
    config['hyperparams']['n_estimators'] = args.n_estimators if args.n_estimators is not None else config['hyperparams']['n_estimators']
    config['hyperparams']['max_depth'] = args.max_depth if args.max_depth is not None else config['hyperparams']['max_depth']
    config['hyperparams']['learning_rate'] = args.learning_rate if args.learning_rate is not None else config['hyperparams']['learning_rate']
    config['hyperparams']['base_score'] = args.base_score if args.base_score is not None else config['hyperparams']['base_score']
    config['hyperparams']['random_state'] = args.model_random_state if args.model_random_state is not None else config['hyperparams']['random_state']

    config['n_samples_tot'] = round(config['training_samples_tot'] / config['num_clients'])
    config['training_samples_client'] = round(config['training_samples_tot'] / config['num_clients'])
    
    logger.info(f"Configuration: {config}")
    
    # Load the full dataset based on the configuration
    x_train, x_valid, y_train, y_valid = load_full_dataset(config=config)
    # Save & split dataset across clients
    save_iid_data_to_clients(x_train, y_train, x_valid, y_valid, config)
    
    folder_list = [ "src/Models/xgb_models", "src/Models/cnn_models", "model_ready.flag"]
    for folder in folder_list:
        if os.path.exists(folder) and os.path.isdir(folder):
            try:
                shutil.rmtree(folder)
                print(f"Deleted folder: {folder}")
            except OSError as e:
                print(f"Error while deleting folder {folder}: {e}")
        elif os.path.exists(folder) and not os.path.isdir(folder): # if it's a file, delete it
            try:
                os.remove(folder)
                print(f"Deleted file: {folder}")
            except OSError as e:
                print(f"Error while deleting file {folder}: {e}")
        else:
            print(f"Folder {folder} does not exist.")
    
    import time
    start_time = time.time()
    
    # Set multiprocessing start method to 'spawn'
    multiprocessing.set_start_method('spawn', force=True)

    server = multiprocessing.Process(target=server_process, args=(config,))
    server.start()

    t = []
    for ii in range(config['num_clients']):
        t.append(multiprocessing.Process(target=client_process, args=(config, ii)))
        t[ii].start()

    for process in t:
        process.join()

    print('Optimization finished in {} s'.format(time.time()-start_time))

if __name__ == '__main__':
    main()
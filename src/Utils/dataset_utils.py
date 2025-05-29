import pickle
import os
import numpy as np
import shutil


def load_full_dataset(config):
    num_classes = config["num_classes"]
    n_redundant = config["n_redundant"]
    path = f"src/Dataset/dataset_{num_classes}_redundant_{n_redundant}.pkl"

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with open(path, 'rb') as f:
        x_train, x_valid, y_train, y_valid = pickle.load(f)
    return x_train, x_valid, y_train, y_valid

def save_iid_data_to_clients(x_train, y_train, x_valid, y_valid, cfg):
    num_clients = cfg["num_clients"]
    samples = round(cfg["training_samples_tot"]/num_clients)

    print("Splitting and saving IID client datasets...")

    # Delete directory if it exists
    if os.path.exists('src/Data'):
        shutil.rmtree('src/Data')

    # Split the training dataset and create folders in data/client_#i/train
    for i in range(num_clients):
        dir = f'src/Data/client_{i}/train/'
        os.makedirs(dir, exist_ok=True)
        start, end = i * samples, (i + 1) * samples
        x_part, y_part = x_train[start:end], y_train[start:end]
        np.save(f"{dir}/x_train.npy", x_part)
        np.save(f"{dir}/y_train.npy", y_part)
        print('Client {} | Samples {}'.format(i, len(y_part)))
    print(f'Saved train data')

    # Split the validation dataset and create folders in data/client_#i/valid
    for i in range(num_clients):
        dir = f'src/Data/client_{i}/valid/'
        os.makedirs(dir, exist_ok=True)
        x_part, y_part = x_valid, y_valid  # all clients have the same validation set
        np.save(f"{dir}/x_valid.npy", x_part)
        np.save(f"{dir}/y_valid.npy", y_part)
        print(f"Client {i} | Validation samples: {len(y_part)}")
    print(f'Saved validation data')


def load_data_for_all_clients(cfg): 
    num_clients = cfg["num_clients"]
    x_train_clients, y_train_clients = [], []
    x_valid_clients, y_valid_clients = [], []

    for i in range(num_clients):
        x_train_clients.append(np.load(f'src/Data/client_{i}/train/x_train.npy'))
        y_train_clients.append(np.load(f'src/Data/client_{i}/train/y_train.npy'))
        x_valid_clients.append(np.load(f'src/Data/client_{i}/valid/x_valid.npy'))
        y_valid_clients.append(np.load(f'src/Data/client_{i}/valid/y_valid.npy'))

    datasets = tuple(zip(x_train_clients, y_train_clients))
    return datasets, (x_valid_clients[0], y_valid_clients[0])  # all same valid


def load_data_for_client(client_id):
    x_train_c = np.load(f'src/Data/client_{client_id}/train/x_train.npy')
    y_train_c = np.load(f'src/Data/client_{client_id}/train/y_train.npy')
    x_valid_c = np.load(f'src/Data/client_{client_id}/valid/x_valid.npy')
    y_valid_c = np.load(f'src/Data/client_{client_id}/valid/y_valid.npy')
    return x_train_c, y_train_c, x_valid_c, y_valid_c

# import numpy as np
# from sklearn.datasets import load_svmlight_file
# from sklearn.model_selection import train_test_split
# from resources.config import Config

# # Carica il dataset in formato LIBSVM
# def load_full_dataset(path):
#     X, y = load_svmlight_file(path)
#     return X.toarray(), y

# # Suddivide il dataset equamente tra i client
# def split_dataset_among_clients(X, y, num_clients):
#     idx = np.random.permutation(len(y))
#     X, y = X[idx], y[idx]
#     splits = np.array_split(np.arange(len(y)), num_clients)
#     return [(X[split], y[split]) for split in splits]

# # Funzione richiamata da ogni client per caricare la propria porzione
# def load_data_for_client(client_id):
#     X, y = load_full_dataset(Config.DATA_PATH)
#     client_data = split_dataset_among_clients(X, y, Config.NUM_CLIENTS)
#     return client_data[client_id]

# def create_dataset():
    
import pandas as pd
import numpy as np
import torch
import logging
import itertools
from data_util import GraphData, HeteroData, z_norm, create_hetero_obj
import matplotlib.pyplot as plt
from torch_geometric.utils import to_dense_adj
from ppr_aggregator import build_split_ppr

def visualise(tr_edge_index):
    # Convert to dense adjacency matrix (counts edges between nodes)
    adj = to_dense_adj(tr_edge_index, max_num_nodes=100)[0]  # First 100 nodes for visibility

    plt.figure(figsize=(10,8))
    plt.imshow(adj.numpy(), cmap='viridis')
    plt.colorbar(label='Number of Transactions')
    plt.title("Adjacency Matrix (Edge Counts Between Nodes)")
    plt.xlabel("Target Node")
    plt.ylabel("Source Node")
    plt.show()



def get_data(args, data_config, run_local=False):
    '''Loads the AML transaction data.
    
    1. The data is loaded from the csv and the necessary features are chosen.
    2. The data is split into training, validation and test data.
    3. PyG Data objects are created with the respective data splits.
    '''
    logging.info('Loading for Graph')

    if run_local:
        transaction_file = f"{data_config['paths']['aml_data']}/{args.data}/formatted_transactions_small.csv" #replace this with your path to the respective AML data objects
    else:
        transaction_file = f"{data_config['paths']['aml_data']}/{args.data}/formatted_transactions.csv" #replace this with your path to the respective AML data objects
    df_edges = pd.read_csv(transaction_file)

    logging.info(f'Available Edge Features: {df_edges.columns.tolist()}')

    # Step 1: connvert TimeStamp into by minimizing the easliest timestamp
    df_edges['Timestamp'] = df_edges['Timestamp'] - df_edges['Timestamp'].min()
    print(f"Time stamp first 10: {df_edges['Timestamp'][:10].tolist()}")
    print(f"Time stamp last 10: {df_edges['Timestamp'][-10:].tolist()}")
    

    max_n_id = df_edges.loc[:, ['from_id', 'to_id']].to_numpy().max() + 1
    df_nodes = pd.DataFrame({'NodeID': np.arange(max_n_id), 'Feature': np.ones(max_n_id)})
    timestamps = torch.Tensor(df_edges['Timestamp'].to_numpy())
    y = torch.LongTensor(df_edges['Is Laundering'].to_numpy())

    logging.info(f"Illicit ratio = {sum(y)} / {len(y)} = {sum(y) / len(y) * 100:.2f}%")
    logging.info(f"Number of nodes (holdings doing transcations) = {df_nodes.shape[0]}")
    logging.info(f"Number of transactions = {df_edges.shape[0]}")

    # edge_features = ['Timestamp', 'Amount Received', 'Received Currency', 'Payment Format']
    edge_features = data_config['features']['edge_features']
    node_features = ['Feature']

    logging.info(f'Edge features being used: {edge_features}')
    logging.info(f'Node features being used: {node_features} ("Feature" is a placeholder feature of all 1s)')

    x = torch.tensor(df_nodes.loc[:, node_features].to_numpy()).float()
    edge_index = torch.LongTensor(df_edges.loc[:, ['from_id', 'to_id']].to_numpy().T)
    edge_attr = torch.tensor(df_edges.loc[:, edge_features].to_numpy()).float()

    n_days = int(timestamps.max() / (3600 * 24) + 1)
    n_samples = y.shape[0]
    logging.info(f'number of days and transactions in the data: {n_days} days, {n_samples} transactions')

    #data splitting
    daily_irs, weighted_daily_irs, daily_inds, daily_trans = [], [], [], [] #irs = illicit ratios, inds = indices, trans = transactions
    for day in range(n_days):
        l = day * 24 * 3600
        r = (day + 1) * 24 * 3600
        day_inds = torch.where((timestamps >= l) & (timestamps < r))[0]
        daily_irs.append(y[day_inds].float().mean())
        weighted_daily_irs.append(y[day_inds].float().mean() * day_inds.shape[0] / n_samples)
        daily_inds.append(day_inds)
        daily_trans.append(day_inds.shape[0])
    
    split_per = [0.6, 0.2, 0.2]
    daily_totals = np.array(daily_trans)
    d_ts = daily_totals
    I = list(range(len(d_ts)))
    split_scores = dict()
    for i,j in itertools.combinations(I, 2):
        if j >= i:
            split_totals = [d_ts[:i].sum(), d_ts[i:j].sum(), d_ts[j:].sum()]
            split_totals_sum = np.sum(split_totals)
            split_props = [v/split_totals_sum for v in split_totals]
            split_error = [abs(v-t)/t for v,t in zip(split_props, split_per)]
            score = max(split_error) #- (split_totals_sum/total) + 1
            split_scores[(i,j)] = score
        else:
            continue

    i,j = min(split_scores, key=split_scores.get)
    #split contains a list for each split (train, validation and test) and each list contains the days that are part of the respective split
    split = [list(range(i)), list(range(i, j)), list(range(j, len(daily_totals)))]
    logging.info(f'Calculate split: {split}')

    #Now, we seperate the transactions based on their indices in the timestamp array
    split_inds = {k: [] for k in range(3)}
    for i in range(3):
        for day in split[i]:
            split_inds[i].append(daily_inds[day]) #split_inds contains a list for each split (tr,val,te) which contains the indices of each day seperately

    tr_inds = torch.cat(split_inds[0])
    val_inds = torch.cat(split_inds[1])
    te_inds = torch.cat(split_inds[2])

    logging.info(f"Total train samples: {tr_inds.shape[0] / y.shape[0] * 100 :.2f}% || IR: "
            f"{y[tr_inds].float().mean() * 100 :.2f}% || Train days: {split[0][:5]}")
    logging.info(f"Total val samples: {val_inds.shape[0] / y.shape[0] * 100 :.2f}% || IR: "
        f"{y[val_inds].float().mean() * 100:.2f}% || Val days: {split[1][:5]}")
    logging.info(f"Total test samples: {te_inds.shape[0] / y.shape[0] * 100 :.2f}% || IR: "
        f"{y[te_inds].float().mean() * 100:.2f}% || Test days: {split[2][:5]}")
    
    #Creating the final data objects
    tr_x, val_x, te_x = x, x, x
    e_tr = tr_inds.numpy()
    e_val = np.concatenate([tr_inds, val_inds])

    tr_edge_index,  tr_edge_attr,  tr_y,  tr_edge_times  = edge_index[:,e_tr],  edge_attr[e_tr],  y[e_tr],  timestamps[e_tr]
    val_edge_index, val_edge_attr, val_y, val_edge_times = edge_index[:,e_val], edge_attr[e_val], y[e_val], timestamps[e_val]
    te_edge_index,  te_edge_attr,  te_y,  te_edge_times  = edge_index,          edge_attr,        y,        timestamps
    
    logging.info(f"--- Train: Node Feature {tr_x.shape} Edge Attr: {tr_edge_attr.shape} Edge Feature Shape: {tr_edge_attr.shape} Y: {tr_y.shape}")
    logging.info(f"--- Eval: Node Feature {val_x.shape} Edge Attr: {val_edge_attr.shape} Edge Feature Shape: {val_edge_attr.shape} Y: {val_y.shape}")
    logging.info(f"--- Test: Node Feature {te_x.shape} Edge Attr: {te_edge_attr.shape} Edge Feature Shape: {te_edge_attr.shape} Y: {te_y.shape}")
    logging.info(tr_x)

    # visualise(tr_edge_index)
    # logging.info(f"Total train samples: {tr_inds.shape[0] / y.shape[0] * 100 :.2f}% || IR: "
    #         f"{y[tr_inds].float().mean() * 100 :.2f}% || Train days: {split[0][:5]}")
    
    tr_ppr, tr_x_remap, tr_edge_index_remap, tr_nodes = build_split_ppr(tr_edge_index, x)
    logging.info(f"Built split ppr for train, tr_ppr {len(tr_ppr)}")
    val_ppr, val_x_remap, val_edge_index_remap, val_nodes = build_split_ppr(val_edge_index, x)
    logging.info(f"Built split ppr for val, val_ppr {len(tr_ppr)}")
    te_ppr, te_x_remap, te_edge_index_remap, te_nodes = build_split_ppr(te_edge_index, x)
    logging.info(f"Built split ppr for test, te_ppr {len(te_ppr)}")
   
    tr_data = GraphData (x=tr_x_remap,  y=tr_y,  edge_index=tr_edge_index_remap,  edge_attr=tr_edge_attr,  timestamps=tr_edge_times, ppr_index=tr_ppr)
    val_data = GraphData(x=val_x_remap, y=val_y, edge_index=val_edge_index_remap, edge_attr=val_edge_attr, timestamps=val_edge_times, ppr_index=val_ppr)
    te_data = GraphData (x=te_x_remap,  y=te_y,  edge_index=te_edge_index_remap,  edge_attr=te_edge_attr,  timestamps=te_edge_times, ppr_index=te_ppr)

    #Adding ports and time-deltas if applicable
    if args.ports:
        logging.info(f"Start: adding ports")
        tr_data.add_ports()
        val_data.add_ports()
        te_data.add_ports()
        logging.info(f"Done: adding ports")
    if args.tds:
        logging.info(f"Start: adding time-deltas")
        tr_data.add_time_deltas()
        val_data.add_time_deltas()
        te_data.add_time_deltas()
        logging.info(f"Done: adding time-deltas")
    
    #Normalize data
    tr_data.x = val_data.x = te_data.x = z_norm(tr_data.x)
    if not args.model == 'rgcn':
        tr_data.edge_attr, val_data.edge_attr, te_data.edge_attr = z_norm(tr_data.edge_attr), z_norm(val_data.edge_attr), z_norm(te_data.edge_attr)
    else:
        tr_data.edge_attr[:, :-1], val_data.edge_attr[:, :-1], te_data.edge_attr[:, :-1] = z_norm(tr_data.edge_attr[:, :-1]), z_norm(val_data.edge_attr[:, :-1]), z_norm(te_data.edge_attr[:, :-1])

    #Create heterogenous if reverese MP is enabled
    #TODO: if I observe wierd behaviour, maybe add .detach.clone() to all torch tensors, but I don't think they're attached to any computation graph just yet
    if args.reverse_mp:
        tr_data = create_hetero_obj(tr_data.x,  tr_data.y,  tr_data.edge_index,  tr_data.edge_attr, tr_data.timestamps, args)
        val_data = create_hetero_obj(val_data.x,  val_data.y,  val_data.edge_index,  val_data.edge_attr, val_data.timestamps, args)
        te_data = create_hetero_obj(te_data.x,  te_data.y,  te_data.edge_index,  te_data.edge_attr, te_data.timestamps, args)
    
    logging.info(f'train data object: {tr_data}')
    logging.info(f'validation data object: {val_data}')
    logging.info(f'test data object: {te_data}')

    return tr_data, val_data, te_data, tr_inds, val_inds, te_inds


def get_data_xgboost(args, data_config):
    logging.info('Loading tabular data for XGBoost')

    transaction_file = f"{data_config['paths']['aml_data']}/{args.data}/formatted_transactions.csv"
    df_edges = pd.read_csv(transaction_file)

    logging.info(f'Available Edge Features: {df_edges.columns.tolist()}')

    # Normalize timestamp
    df_edges['Timestamp'] = df_edges['Timestamp'] - df_edges['Timestamp'].min()
    timestamps = torch.Tensor(df_edges['Timestamp'].to_numpy())
    y = torch.LongTensor(df_edges['Is Laundering'].to_numpy())

    # --- Frequency Encoding of from_id and to_id ---
    for col in ['from_id', 'to_id']:
        freq = df_edges[col].value_counts().to_dict()
        df_edges[f'{col}_freq'] = df_edges[col].map(freq)

    # Define full list of features (including ID encodings)
    edge_features = data_config['features']['edge_features']
    numerical_features = ['Timestamp', 'Amount Sent']  # example
    categorical_features = ['Sent Currency', 'Payment Format']
    id_features = ['from_id_freq', 'to_id_freq']

    full_features = numerical_features + id_features + categorical_features
    logging.info(f"Using features: {full_features}")

    # Convert categorical to int (ordinal encoding)
    for cat_col in categorical_features:
        df_edges[cat_col] = df_edges[cat_col].astype("category").cat.codes

    # Normalize numerical and ID freq features using z_norm
    to_normalize = numerical_features + id_features
    df_edges[to_normalize] = z_norm(torch.tensor(df_edges[to_normalize].to_numpy()).float()).numpy()

    # Create final feature matrix
    X = df_edges[full_features].to_numpy().astype(np.float32)

    # Data splitting by days
    n_days = int(timestamps.max() / (3600 * 24) + 1)
    daily_inds = []
    daily_trans = []

    for day in range(n_days):
        l = day * 24 * 3600
        r = (day + 1) * 24 * 3600
        day_inds = torch.where((timestamps >= l) & (timestamps < r))[0]
        daily_inds.append(day_inds)
        daily_trans.append(day_inds.shape[0])

    # Optimize for approximate [0.6, 0.2, 0.2] split
    split_per = [0.6, 0.2, 0.2]
    d_ts = np.array(daily_trans)
    I = list(range(len(d_ts)))
    split_scores = {}
    for i, j in itertools.combinations(I, 2):
        if j >= i:
            split_totals = [d_ts[:i].sum(), d_ts[i:j].sum(), d_ts[j:].sum()]
            total = sum(split_totals)
            props = [v / total for v in split_totals]
            score = max([abs(v - t) / t for v, t in zip(props, split_per)])
            split_scores[(i, j)] = score

    i, j = min(split_scores, key=split_scores.get)
    split = [list(range(i)), list(range(i, j)), list(range(j, len(d_ts)))]

    split_inds = {k: [] for k in range(3)}
    for k in range(3):
        for day in split[k]:
            split_inds[k].append(daily_inds[day])
    tr_inds = torch.cat(split_inds[0])
    val_inds = torch.cat(split_inds[1])
    te_inds = torch.cat(split_inds[2])

    X_train, y_train = X[tr_inds], y[tr_inds].numpy()
    X_val, y_val = X[val_inds], y[val_inds].numpy()
    X_test, y_test = X[te_inds], y[te_inds].numpy()

    logging.info(f"Train X: {X_train.shape}, Y: {y_train.shape}")
    logging.info(f"Val X: {X_val.shape}, Y: {y_val.shape}")
    logging.info(f"Test X: {X_test.shape}, Y: {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test



if __name__ == "__main__":
    from util import create_parser, set_seed, logger_setup
    import json
    import time
    parser = create_parser()
    args = parser.parse_args()

    with open('data_config.json', 'r') as config_file:
        data_config = json.load(config_file)

    # Setup logging
    logger_setup()

    #set seed
    set_seed(args.seed)

    #get data
    logging.info("Retrieving data")
    t1 = time.perf_counter()
    
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)

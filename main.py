import time
import logging
from util import create_parser, set_seed, logger_setup
from data_loading import get_data, get_data_xgboost
from training import train_gnn, train_xgboost
from inference import infer_gnn
import json

def main():
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
    
    if args.model == "xgboost":
        X_train, y_train, X_val, y_val, X_test, y_test = get_data_xgboost(args, data_config)
    else:
        if args.run_local:
            tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config,run_local=True)
        else:
            tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config, run_local=False)

    t2 = time.perf_counter()
    logging.info(f"Retrieved data in {t2-t1:.2f}s")
    if args.model == "xgboost":
        train_xgboost(X_train, X_val, X_test,y_train, y_val, y_test, args, data_config)
    else:
        if not args.inference:
            #Training
            logging.info(f"Running Training")
            train_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config)
        else:
            #Inference
            logging.info(f"Running Inference")
            infer_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config)

if __name__ == "__main__":
    main()

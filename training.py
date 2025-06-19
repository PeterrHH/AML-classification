import torch
import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from train_util import AddEgoIds, extract_param, add_arange_ids, get_loaders, evaluate_homo, evaluate_hetero, save_model, load_model
from models import GINe, PNA, GATe, RGCN
from xgboost import XGBClassifier
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import to_hetero, summary
from torch_geometric.utils import degree
import wandb
import numpy as np
import logging

def train_homo(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config):
    #training
    best_val_f1 = 0
    best_val_accuracy = 0
    best_te_f1 = 0
    patience = 20                     
    epochs_since_improvement = 0     
    best_model = None
    best_model_epoch = 0
    
    if args.run_local:
        logging.warning("Running in local mode: test set is re-used from validation set.")
        te_loader = val_loader
        te_inds = val_inds


    for epoch in range(config.epochs):
        total_loss = total_examples = 0
        preds = []
        ground_truths = []
        for batch in tqdm.tqdm(tr_loader, disable=not args.tqdm):
            optimizer.zero_grad()
            #select the seed edges from which the batch was created
            inds = tr_inds.detach().cpu()
            batch_edge_inds = inds[batch.input_id.detach().cpu()]
            batch_edge_ids = tr_loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
            mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)

            #remove the unique edge id from the edge features, as it's no longer needed
            batch.edge_attr = batch.edge_attr[:, 1:]

            batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            pred = out[mask]
            ground_truth = batch.y[mask]
            preds.append(pred.argmax(dim=-1))
            ground_truths.append(ground_truth)
            loss = loss_fn(pred, ground_truth)

            loss.backward()
            optimizer.step()

            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
        loss = total_loss / total_examples if total_examples > 0 else 0
        pred = torch.cat(preds, dim=0).detach().cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).detach().cpu().numpy()

        accuracy = accuracy_score(ground_truth, pred)
        f1 = f1_score(ground_truth, pred)
        logging.info(f'--- Epoch: {epoch}---')
        logging.info(f"Prediction counts: {np.bincount(pred)}")
        logging.info(f"Ground truth counts: {np.bincount(ground_truth)}")
        wandb.log({"f1/train": f1}, step=epoch)
        logging.info(f'train loss: {loss:.4f}')
        logging.info(f'Train F1: {f1:.4f}')

        #evaluate
        val_accuracy, val_f1, val_precision, val_recall = evaluate_homo(val_loader, val_inds, model, val_data, device, args)
        te_accuracy, te_f1, te_precision, te_recall = evaluate_homo(te_loader, te_inds, model, te_data, device, args)


        wandb.log({
            "total_loss": loss,
            "f1/validation": val_f1,
            "accuracy/validation": val_accuracy,
            "precision/validation": val_precision,
            "recall/validation": val_recall,
            "f1/test": te_f1,
            "precision/test":te_precision,
            "recall/test":te_recall
        }, step=epoch)

        
        # wandb.log({"f1/test": te_f1}, step=epoch)
        logging.info(f'Validation F1: {val_f1:.4f}')
        logging.info(f'Validation Pre: {val_precision:.4f} Recall:{val_recall:.4f}')
        logging.info(f'Testing F1: {te_f1:.4f}')
        logging.info(f'Testing Pre: {te_precision:.4f} Recall:{te_recall:.4f}')

        if epoch == 0:
            wandb.log({"best_val_f1": val_f1}, step=epoch)
            wandb.log({"best_val_f1": te_f1}, step=epoch)
            best_model = model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            wandb.log({"best_val_f1": val_f1}, step=epoch)
            logging.info(f'Epoch: {epoch} Best Val f1: {val_f1:.4f}!!!')
            
            epochs_since_improvement = 0                     
            if args.save_model:
                save_model(model, optimizer, epoch, args, data_config,"val")
            best_model = model
            best_model_epoch = epoch
        else:
            epochs_since_improvement += 1                   
            
        if te_f1 > best_te_f1:
            best_te_f1 = te_f1
            wandb.log({"best_te_f1": te_f1}, step=epoch)
            logging.info(f'Epoch {epoch} Best Test f1: {te_f1:.4f}!!!')
            if args.save_model:
                save_model(model, optimizer, epoch, args, data_config,"test")
                
        if epochs_since_improvement >= patience:            
            logging.info(f"Early stopping at epoch {epoch} due to no val f1 improvement in {patience} epochs.")
            break

    return model

def train_hetero(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config):
    #training
    best_val_f1 = 0
    for epoch in range(config.epochs):
        total_loss = total_examples = 0
        preds = []
        ground_truths = []
        for batch in tqdm.tqdm(tr_loader, disable=not args.tqdm):
            optimizer.zero_grad()
            #select the seed edges from which the batch was created
            inds = tr_inds.detach().cpu()
            batch_edge_inds = inds[batch['node', 'to', 'node'].input_id.detach().cpu()]
            batch_edge_ids = tr_loader.data['node', 'to', 'node'].edge_attr.detach().cpu()[batch_edge_inds, 0]
            mask = torch.isin(batch['node', 'to', 'node'].edge_attr[:, 0].detach().cpu(), batch_edge_ids)
            
            #remove the unique edge id from the edge features, as it's no longer needed
            batch['node', 'to', 'node'].edge_attr = batch['node', 'to', 'node'].edge_attr[:, 1:]
            batch['node', 'rev_to', 'node'].edge_attr = batch['node', 'rev_to', 'node'].edge_attr[:, 1:]

            batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
            out = out[('node', 'to', 'node')]
            pred = out[mask]
            ground_truth = batch['node', 'to', 'node'].y[mask]
            preds.append(pred.argmax(dim=-1))
            ground_truths.append(batch['node', 'to', 'node'].y[mask])
            loss = loss_fn(pred, ground_truth)

            loss.backward()
            optimizer.step()

            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
            
        pred = torch.cat(preds, dim=0).detach().cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).detach().cpu().numpy()
        f1 = f1_score(ground_truth, pred)
        wandb.log({"f1/train": f1}, step=epoch)
        logging.info(f'Train F1: {f1:.4f}')

        #evaluate
        val_f1 = evaluate_hetero(val_loader, val_inds, model, val_data, device, args)
        te_f1 = evaluate_hetero(te_loader, te_inds, model, te_data, device, args)

        wandb.log({"f1/validation": val_f1}, step=epoch)
        wandb.log({"f1/test": te_f1}, step=epoch)
        logging.info(f'Validation F1: {val_f1:.4f}')
        logging.info(f'Test F1: {te_f1:.4f}')

        if epoch == 0:
            wandb.log({"best_test_f1": te_f1}, step=epoch)
        elif val_f1 > best_val_f1:
            best_val_f1 = val_f1
            wandb.log({"best_test_f1": te_f1}, step=epoch)
            if args.save_model:
                save_model(model, optimizer, epoch, args, data_config)
        
    return model

def get_model(sample_batch, config, args):
    n_feats = sample_batch.x.shape[1] if not isinstance(sample_batch, HeteroData) else sample_batch['node'].x.shape[1]
    e_dim = (sample_batch.edge_attr.shape[1] - 1) if not isinstance(sample_batch, HeteroData) else (sample_batch['node', 'to', 'node'].edge_attr.shape[1] - 1)

    if args.model == "gin":
        model = GINe(
                num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2,
                n_hidden=round(config.n_hidden), residual=False, edge_updates=args.emlps, edge_dim=e_dim, 
                dropout=config.dropout, final_dropout=config.final_dropout
                )
    elif args.model == "gat":
        model = GATe(
                num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2,
                n_hidden=round(config.n_hidden), n_heads=round(config.n_heads), 
                edge_updates=args.emlps, edge_dim=e_dim,
                dropout=config.dropout, final_dropout=config.final_dropout
                )
    elif args.model == "pna":
        if not isinstance(sample_batch, HeteroData):
            d = degree(sample_batch.edge_index[1], dtype=torch.long)
        else:
            index = torch.cat((sample_batch['node', 'to', 'node'].edge_index[1], sample_batch['node', 'rev_to', 'node'].edge_index[1]), 0)
            d = degree(index, dtype=torch.long)
        deg = torch.bincount(d, minlength=1)
        model = PNA(
            num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2,
            n_hidden=round(config.n_hidden), edge_updates=args.emlps, edge_dim=e_dim,
            dropout=config.dropout, deg=deg, final_dropout=config.final_dropout
            )
    elif config.model == "rgcn":
        model = RGCN(
            num_features=n_feats, edge_dim=e_dim, num_relations=8, num_gnn_layers=round(config.n_gnn_layers),
            n_classes=2, n_hidden=round(config.n_hidden),
            edge_update=args.emlps, dropout=config.dropout, final_dropout=config.final_dropout, n_bases=None #(maybe)
        )
    
    return model

def train_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config):
    #set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wandb.login(
        key= data_config['wandb']['api_key'], #replace this with your wandb api key if you want to use wandb logging
    )
    #define a model config dictionary and wandb logging at the same time
    wandb.init(
        mode="disabled" if args.testing else "online",
        project=data_config['wandb']['project_name'], 
        name = data_config['wandb']['name'], # Name of the run
        config={
            "epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "model": args.model,
            "data": args.data,
            "num_neighbors": args.num_neighs,
            "lr": extract_param("lr", args),
            "n_hidden": extract_param("n_hidden", args),
            "n_gnn_layers": extract_param("n_gnn_layers", args),
            "loss": "ce",
            "w_ce1": extract_param("w_ce1", args),
            "w_ce2": extract_param("w_ce2", args),
            "dropout": extract_param("dropout", args),
            "final_dropout": extract_param("final_dropout", args),
            "n_heads": extract_param("n_heads", args) if args.model == 'gat' else None
        }
    )

    config = wandb.config

    #set the transform if ego ids should be used
    if args.ego:
        transform = AddEgoIds()
    else:
        transform = None

    #add the unique ids to later find the seed edges
    add_arange_ids([tr_data, val_data, te_data])

    tr_loader, val_loader, te_loader = get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args)

    #get the model
    sample_batch = next(iter(tr_loader))
    model = get_model(sample_batch, config, args)

    if args.reverse_mp:
        model = to_hetero(model, te_data.metadata(), aggr='mean')
    
    if args.finetune:
        model, optimizer = load_model(model, device, args, config, data_config)
    else:
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    sample_batch.to(device)
    sample_x = sample_batch.x if not isinstance(sample_batch, HeteroData) else sample_batch.x_dict
    sample_edge_index = sample_batch.edge_index if not isinstance(sample_batch, HeteroData) else sample_batch.edge_index_dict
    if isinstance(sample_batch, HeteroData):
        sample_batch['node', 'to', 'node'].edge_attr = sample_batch['node', 'to', 'node'].edge_attr[:, 1:]
        sample_batch['node', 'rev_to', 'node'].edge_attr = sample_batch['node', 'rev_to', 'node'].edge_attr[:, 1:]
    else:
        sample_batch.edge_attr = sample_batch.edge_attr[:, 1:]
    sample_edge_attr = sample_batch.edge_attr if not isinstance(sample_batch, HeteroData) else sample_batch.edge_attr_dict
    logging.info(summary(model, sample_x, sample_edge_index, sample_edge_attr))
    
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([config.w_ce1, config.w_ce2]).to(device))

    if args.reverse_mp:
        model = train_hetero(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config)
    else:
        model = train_homo(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config)
    
    wandb.finish()


def train_xgboost(X_train, X_val, X_test, y_train, y_val, y_test, args, data_config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wandb.login(
        key= data_config['wandb']['api_key'], #replace this with your wandb api key if you want to use wandb logging
    )
    #define a model config dictionary and wandb logging at the same time
    wandb.init(
        mode="disabled" if args.testing else "online",
        project=data_config['wandb']['project_name'], 
        name = "XGBoost", # Name of the run
        config={
            "epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "model": args.model,
            "data": args.data,
            "n_hidden": extract_param("n_hidden", args),
        }
    )

    # Build Loaders
    # Extract edge features and labels (convert to numpy)
    # X_train = tr_data.edge_attr.detach().cpu().numpy()
    # y_train = tr_data.y.detach().cpu().numpy()

    # X_val = val_data.edge_attr.detach().cpu().numpy()
    # y_val = val_data.y.detach().cpu().numpy()

    # X_test = te_data.edge_attr.detach().cpu().numpy()
    # y_test = te_data.y.detach().cpu().numpy()

    logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logging.info(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    logging.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    # Define XGBoost model
    model = XGBClassifier(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.01,
        reg_lambda=10 ** 1, 
        scale_pos_weight=10,           # ∈ (1, 10)
        colsample_bytree=0.8,         # ∈ (0.5, 1.0)
        subsample=0.8,                # ∈ (0.5, 1.0)
        objective='binary:logistic',
        eval_metric='auc'
    )

    # XGBoost
        # Train
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])


    # Predict on eval
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    acc = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)

    logging.info(f"Eval Accuracy: {acc:.4f}")
    logging.info(f"Eval Precision: {precision:.4f}")
    logging.info(f"Eval Recall: {recall:.4f}")
    logging.info(f"Eval F1: {f1:.4f}")

    
    # Predict on Test
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    logging.info(f"Test Accuracy: {acc:.4f}")
    logging.info(f"Test Precision: {precision:.4f}")
    logging.info(f"Test Recall: {recall:.4f}")
    logging.info(f"Test F1: {f1:.4f}")


    # Log metrics
    wandb.log({
        "test_accuracy": acc,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
    })

    
    pass
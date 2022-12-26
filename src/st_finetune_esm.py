'''
Fine-tunes ESM Transformer models with labelled data.
'''

import os
import pathlib

import numpy as np
import pandas as pd
import torch

from esm import BatchConverter, pretrained

from utils import read_fasta
from utils.metric_utils import spearman, ndcg
from utils.esm_utils import CSVBatchedDataset

mse_criterion = torch.nn.MSELoss(reduction='mean')
ce_criterion = torch.nn.CrossEntropyLoss(reduction='none')

def step(model, labels, toks, wt_toks, mask_idx):
    labels = torch.tensor(labels)    
    if torch.cuda.is_available():
        labels = labels.to(device="cuda", non_blocking=True)
    predictions = predict(model, toks, wt_toks, mask_idx)
    loss = mse_criterion(predictions.float(), labels.float())
    return loss, predictions


def predict(model, toks, wt_toks, mask_idx):    
    if torch.cuda.is_available():
        toks = toks.to(device="cuda", non_blocking=True)
    wt_toks_rep = wt_toks.repeat(toks.shape[0], 1)
    mask = (toks != wt_toks)
    masked_toks = torch.where(mask, mask_idx, toks)
    out = model(masked_toks, return_contacts=False)
    logits = out["logits"]
    logits_tr = logits.transpose(1, 2)  # [B, E, T]
    ce_loss_mut = ce_criterion(logits_tr, toks)   # [B, E]
    ce_loss_wt = ce_criterion(logits_tr, wt_toks_rep)
    ll_diff_sum = torch.sum(
        (ce_loss_wt - ce_loss_mut) * mask, dim=1, keepdim=True)  # [B, 1]
    return ll_diff_sum[:, 0]


def finetune_esm(
    model_location,
    wt_fasta_file,
    df_train,
    df_val,
    epochs,
    learning_rate,
    log=False,
    output_dir = "/content/experiment_artifacts/",
    toks_per_batch=512,
):
    model_data = torch.load(model_location, map_location='cpu')
    model, alphabet = pretrained.load_model_and_alphabet(model_location)

    batch_converter = BatchConverter(alphabet)

    mask_idx = torch.tensor(alphabet.mask_idx)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    wt_seq = read_fasta(wt_fasta_file)[0]
    _, _, wt_toks = batch_converter([('WT', wt_seq)])

    if torch.cuda.is_available():
        model = model.cuda()
        mask_idx = mask_idx.cuda()
        wt_toks = wt_toks.cuda()
        if log:
            print("Transferred model to GPU")

    train_dataset = CSVBatchedDataset.from_dataframe(df_train)
    val_dataset = CSVBatchedDataset.from_dataframe(df_val)

    train_batches = train_dataset.get_batch_indices(
        toks_per_batch, extra_toks_per_seq=1)
    val_batches = val_dataset.get_batch_indices(
        toks_per_batch, extra_toks_per_seq=1)

    train_data_loader = torch.utils.data.DataLoader(train_dataset,
            collate_fn=batch_converter, batch_sampler=train_batches)
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
            collate_fn=batch_converter, batch_sampler=val_batches)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate)
    train_loss = np.zeros(epochs+1)
    val_loss = np.zeros(epochs+1)
    val_spearman = np.zeros(epochs+1)
    best_val_spearman = None

    for epoch in range(epochs+1):
        # Train
        if epoch > 0:
            for batch_idx, (labels, strs, toks) in enumerate(train_data_loader):
                if batch_idx % 100 == 0 and log:
                    print(
                        f"Processing {batch_idx + 1} of {len(train_batches)} "
                        f"batches ({toks.size(0)} sequences)"
                    )
                optimizer.zero_grad()
                loss, _ = step(model, labels, toks, wt_toks, mask_idx)
                loss.backward()
                optimizer.step()
                train_loss[epoch] += loss.to('cpu').item()
            train_loss[epoch] /= float(len(train_data_loader))

        # Validation 
        model_eval = model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (labels, strs, toks) in enumerate(val_data_loader):
                loss, predictions = step(
                    model, labels, toks, wt_toks, mask_idx)
                y_pred.append(predictions.to('cpu').numpy())
                y_true.append(labels)
                val_loss[epoch] += loss.to('cpu').item()
        val_loss[epoch] /= float(len(val_data_loader))
        if log:
            print('epoch %d, train loss: %.3f, val loss: %.3f' % (
                epoch + 1, train_loss[epoch], val_loss[epoch]))
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        val_spearman[epoch] = spearman(y_pred, y_true)
        if log:
            print(f'Val Spearman correlation {val_spearman[epoch]}')

        if best_val_spearman is None or val_spearman[epoch] > best_val_spearman:
            best_val_spearman = val_spearman[epoch]
            model_data["model"] = model.state_dict() 
            #model_data["toplinear"] = toplinear.state_dict()
            torch.save(model_data, os.path.join(output_dir, 'model_data.pt'))

    np.savetxt(os.path.join(output_dir, 'loss_trajectory_train.npy'), train_loss)
    np.savetxt(os.path.join(output_dir, 'loss_trajectory_val.npy'), val_loss)
    np.savetxt(os.path.join(output_dir, 'spearman_trajectory_val.npy'), val_spearman)

    # Load best saved model
    model, alphabet = pretrained.load_model_and_alphabet(
        os.path.join(output_dir, 'model_data.pt'))
    if torch.cuda.is_available():
        model = model.cuda()
    model_eval = model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(val_data_loader):
            predictions = predict(model, toks, wt_toks, mask_idx)
            y_pred.append(predictions.to('cpu').numpy())
            y_true.append(labels)
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    if log:
        print(f'Final Spearman correlation {spearman(y_pred, y_true)}')
    df_final_val = df_val.copy()
    df_final_val['pred'] = y_pred
    metric_fns = {
        'spearman': spearman,
        'ndcg': ndcg,
    }
    results_dict = {k: mf(df_final_val.pred.values, df_final_val.log_fitness.values)
            for k, mf in metric_fns.items()}
    results_dict.update({
        'predictor': model_location.split(os.sep)[-1],
        'epochs': epochs,
        'learning_rate': learning_rate,
    })

    results = pd.DataFrame(columns=sorted(results_dict.keys()))
    results = results.append(results_dict, ignore_index=True)
    results.to_csv(os.path.join(output_dir, 'metrics.csv'),
        mode='w', index=False, columns=sorted(results.columns.values))
    
    return results


def get_outputs(model_location, df, wt_fasta_file, toks_per_batch=512):
    
    model, alphabet = pretrained.load_model_and_alphabet(model_location)
    batch_converter = BatchConverter(alphabet)
    mask_idx = torch.tensor(alphabet.mask_idx)
    model_eval = model.eval()

    wt_seq = read_fasta(wt_fasta_file)[0]
    _, _, wt_toks = batch_converter([('WT', wt_seq)])
    
    dataset = CSVBatchedDataset.from_dataframe(df)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=batch_converter, batch_sampler=batches)
    
    y_pred = []
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            predictions = predict(model, toks, wt_toks, mask_idx)
            y_pred.append(predictions.to('cpu').numpy())
            
    return np.concatenate(y_pred)
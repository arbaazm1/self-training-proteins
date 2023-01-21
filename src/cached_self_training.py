from Bio import SeqIO
from collections import defaultdict
import json
import os
import pandas as pd
import pathlib
from sklearn.metrics import mean_squared_error as mse
from sklearn.utils import shuffle 
from st_finetune_esm import finetune_esm, get_outputs
import torch
from tqdm import tqdm
from utils.metric_utils import spearman, ndcg

def train_val_test_split(dataset_name, n_train, val_split, test_split, seed):
    data_path = os.path.join("..", "processed_data", dataset_name, "data.csv")
    df_full = shuffle(pd.read_csv(data_path), random_state=seed)

    n_test = int(len(df_full) * test_split)
    df_test = df_full[-n_test:]
    df_trainval = df_full.drop(df_test.index)

    if n_train == -1:
        n_train = int(len(df_trainval))
    if n_train > len(df_trainval):
        print(f"Insufficient data")
        return
    n_val = int(n_train * val_split)
    df_train = df_trainval[:n_train-n_val]
    df_val = df_trainval[n_train-n_val:n_train]

    return df_train, df_val, df_test

def create_msa_df(dataset_name, train_df, val_df):
    dataset_prefix = '_'.join(dataset_name.split('_')[:2])
    msa_a2m_path = os.path.join("..", "alignments", f"{dataset_prefix}.a2m")
    
    names, seqs = [], []

    for record in SeqIO.parse(msa_a2m_path, "fasta"):
        names.append(record.id)
        seqs.append(str(record.seq))

    alignment_df = pd.DataFrame()
    alignment_df["mutant"] = names
    alignment_df["seq"] = pd.Series(seqs).apply(lambda x: x.upper().replace(' ', ''))
    alignment_df["log_fitness"] = [0 for seq in alignment_df["seq"]]

    #Remove repeated MSA sequences
    alignment_df.drop_duplicates(subset=["seq"],inplace=True)

    #Remove MSA sequences that have labels
    labelled_train_seqs = alignment_df["seq"].isin(train_df["seq"])
    msa_train_filtered_df = alignment_df[~labelled_train_seqs]
    labelled_val_seqs = msa_train_filtered_df["seq"].isin(val_df["seq"])
    msa_filtered_df = msa_train_filtered_df[~labelled_val_seqs]

    return msa_filtered_df


#IMPORTANT: Val data split is needed to determine when to early stop (and less importantly, for hyperparam vals)

def run_cached_experiment(
    dataset_name,
    model_path,
    n_train,
    num_self_train_iters,
    finetune_learning_rate,
    finetune_epochs,
    seed,
    val_split,
    test_split,
    finetune_log,
    output_dir,
    train_toks_per_batch,
    eval_toks_per_batch,
):

    metric_fns = {
        'spearman': spearman,
        'ndcg': ndcg,
        "mse": mse
    }

    if os.path.exists(os.path.join(output_dir, "experiment_results.json")):
        with open(os.path.join(output_dir, "experiment_results.json"), "r") as f:
            res_dict = json.load(f)
            if all([f"test_{k}" in res_dict for k in metric_fns.keys()]):
                print(f"Experiment already complete with data at {output_dir}")
                return

    # Create experiment_artifacts directory for storing (labeled +) pseudolabeled sequences, 
    # global best model, and per_iteration model for logging metrics 
    if not os.path.exists(output_dir):
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    wt_fasta_path = os.path.join("..", "processed_data", dataset_name, "wt.fasta")

    #Create train, val, test dataframes (use data infra section of Chloe's script)
    if not os.path.exists(os.path.join(output_dir, "train_data.csv")):
        train_df, val_df, test_df = train_val_test_split(dataset_name, n_train, val_split, test_split, seed)
        train_df.to_csv(os.path.join(output_dir, "train_data.csv"))
        val_df.to_csv(os.path.join(output_dir, "val_data.csv"))
        test_df.to_csv(os.path.join(output_dir, "test_data.csv"))
        # Artifact folder should now contain train_df.csv, val_df.csv, test_df.csv
    else:
        train_df = pd.read_csv(os.path.join(output_dir, "train_data.csv"))
        val_df = pd.read_csv(os.path.join(output_dir, "val_data.csv"))
        test_df = pd.read_csv(os.path.join(output_dir, "test_data.csv"))
    
    # Create or load MSA df with dummy pseudolabels
    if not os.path.exists(os.path.join(output_dir, 'msa_data.csv')):
        msa_df = create_msa_df(dataset_name, train_df, val_df)
        msa_df.to_csv(os.path.join(output_dir, 'msa_data.csv'))
    else:
        msa_df = pd.read_csv(os.path.join(output_dir, 'msa_data.csv'))
    
    # Baseline logging
    baseline_folder = os.path.join(output_dir, "baseline_data")
    if not os.path.exists(baseline_folder):
        pathlib.Path(baseline_folder).mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(os.path.join(baseline_folder, "model_data.pt")):
        finetune_esm(
            model_path, 
            wt_fasta_path,
            train_df, 
            val_df, 
            finetune_epochs,
            finetune_learning_rate,
            log=finetune_log,
            output_dir=baseline_folder,
            toks_per_batch=train_toks_per_batch
            )
    
    baseline_model_path = os.path.join(baseline_folder, "model_data.pt")

    if not os.path.exists(os.path.join(baseline_folder, 'baseline_metrics.csv')):
        baseline_dict = {}
        for i in ["train", "val", "test"]:
            preds = get_outputs(model_path, baseline_model_path, eval(f"{i}_df"), wt_fasta_path, toks_per_batch=eval_toks_per_batch)
            for k, mf in metric_fns.items():
                baseline_dict[f"best_st_{i}_{k}"] = mf(preds, eval(f"{i}_df").log_fitness.values)

        baseline_results = pd.DataFrame(columns=sorted(baseline_dict.keys()))
        baseline_results = baseline_results.append(baseline_dict, ignore_index=True)
        baseline_results.to_csv(os.path.join(baseline_folder, 'baseline_metrics.csv'),
            mode='w', index=False, columns=sorted(baseline_results.columns.values))


    ###SELF TRAINING SETUP###
    # train_mse = []
    # val_mse = []
    # train_spearman = []
    # val_spearman = []
    # train_ndcg = []
    # val_ndcg = []

    if os.path.exists(os.path.join(output_dir, "experiment_results.json")):
        with open(os.path.join(output_dir, "experiment_results.json")) as f:
            res_dict = json.load(f)

        iter_num = len(res_dict["val_spearman_trajectory"])
        teacher_model_path = os.path.join(output_dir, 'model_data.pt')
    else:
        res_dict = defaultdict(list)
        res_dict["best_val_spearman"] = None
        
        iter_num = 0
        teacher_model_path = baseline_model_path

    ###SELF TRAINING LOOP#
    with tqdm(total=num_self_train_iters, initial=iter_num) as pbar:
        while iter_num < num_self_train_iters:
            #Get teacher_model pseudolabels for MSA sequences
            msa_df = pd.read_csv(os.path.join(output_dir, 'msa_data.csv'))
            pseudolabels = get_outputs(model_path, teacher_model_path, msa_df, wt_fasta_path, toks_per_batch=eval_toks_per_batch)
            msa_df['log_fitness'] = pseudolabels
            #Concat pseudolabels with actual labels
            combined_labelled_df = pd.concat([msa_df[["seq", "log_fitness"]], train_df[["seq", "log_fitness"]]])
            
            student_model = finetune_esm(
                        model_path, 
                        wt_fasta_path,
                        combined_labelled_df, 
                        val_df, 
                        finetune_epochs,
                        finetune_learning_rate,
                        log=finetune_log,
                        output_dir=output_dir,
                        toks_per_batch=train_toks_per_batch
            )
            #Log train, val Spearmen + MSE for student
            student_model_path = os.path.join(output_dir, 'model_data.pt')
            for i in ["train", "val"]:
                preds = get_outputs(model_path, student_model_path, eval(f"{i}_df"), wt_fasta_path, toks_per_batch=eval_toks_per_batch)
                for k, mf in metric_fns.items():
                    res_dict[f"{i}_{k}_trajectory"].append(mf(preds, eval(f"{i}_df").log_fitness.values))

            #Early stopping variables update
            if res_dict["best_val_spearman"] is None or res_dict["val_spearman_trajectory"][-1] > res_dict["best_val_spearman"]:
                res_dict["best_val_spearman"] = res_dict["val_spearman_trajectory"][-1]
                torch.save(student_model.state_dict(), os.path.join(output_dir, 'early_stopped_st_model_data.pt'))
            
            teacher_model_path = student_model_path
            
            with open(os.path.join(output_dir, "experiment_results.json"), "w") as f:
                json.dump(res_dict, f, indent=4)
            
            iter_num += 1
            pbar.update(1)
    ###FIN SELF TRAINING LOOP###

    #Log test Spearman from early-stopped model in self-training iterations
    best_st_path = os.path.join(output_dir, 'early_stopped_st_model_data.pt')
    
    for i in ["train", "val", "test"]:
        best_st_preds = get_outputs(model_path, best_st_path, eval(f"{i}_df"), wt_fasta_path, toks_per_batch=eval_toks_per_batch)
        for k, mf in metric_fns.items():
            res_dict[f"best_st_{i}_{k}"] = mf(best_st_preds, eval(f"{i}_df").log_fitness.values)
    
    with open(os.path.join(output_dir, "experiment_results.json"), "w") as f:
        json.dump(res_dict, f, indent=4)

    #Clean up large .pt files
    os.remove(os.path.join(output_dir, 'early_stopped_st_model_data.pt'))
    os.remove(os.path.join(output_dir, 'model_data.pt'))
    os.remove(os.path.join(baseline_folder, "model_data.pt"))

    return res_dict
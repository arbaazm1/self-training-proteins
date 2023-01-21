import argparse
import pathlib

from cached_self_training import run_cached_experiment

def create_parser():
    
    parser = argparse.ArgumentParser(
        description="Self-training for protein sequences"  
    )
    parser.add_argument(
        "dataset_name",
        type=str,
        help="paper/dataset of interest",
    )
    parser.add_argument(
        "model_path",
        type=pathlib.Path,
        help="location of baseline and initial teacher model",
    )
    parser.add_argument(
        "n_train", type=int, help="training data size"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="output directory",
    )
    parser.add_argument(
        "--num_self_train_iters", type=int, default=25, help="number of self-training iterations to run"
    )
    parser.add_argument(
        "--finetune_learning_rate", type=float, default=3e-7, help="learning rate to use while finetuning"
    )
    parser.add_argument(
        "--finetune_epochs", type=int, default=5, help="number of epochs while finetuning"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed"
    )
    parser.add_argument(
        "--val_split", type=float, default=0.2, help="proportion of train set to use for validation"
    )
    parser.add_argument(
        "--test_split", type=float, default=0.2, help="proportion of data to use as test set"
    )
    parser.add_argument(
        "--train_toks_per_batch", type=int, default=256, help="batch size for training"
    )
    parser.add_argument(
        "--eval_toks_per_batch", type=int, default=512, help="batch size for inference"
    )
    parser.add_argument(
        "--finetune_log", dest="finetune_log", action="store_true"
    )

    parser.set_defaults(finetune_log=False)

    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    run_cached_experiment(
                    args.dataset_name,
                    args.model_path,
                    args.n_train,
                    args.num_self_train_iters,
                    args.finetune_learning_rate,
                    args.finetune_epochs,
                    args.seed,
                    args.val_split,
                    args.test_split,
                    args.finetune_log,
                    args.output_dir,
                    args.train_toks_per_batch,
                    args.eval_toks_per_batch
                    )
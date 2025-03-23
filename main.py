import argparse
import torch
import numpy as np
from copy import deepcopy
import shutil
from pathlib import Path

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_supervised_finetuning import Exp_Supervised_Finetuning
from utils.tools import set_seed, print_formatted_dict
from data_provider.data_loader import update_args_from_dataset


def get_args_from_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM4TS")

    # * basic config
    parser.add_argument(
        "--task_name",
        type=str,
        default="long_term_forecast",
        help="time-series task",
        choices=["long_term_forecast"],
    )
    parser.add_argument("--model_id", type=str, default="test", help="model id")
    parser.add_argument(
        "--model",
        type=str,
        default="LLM4TS",
        help="model name",
    )
    parser.add_argument(
        "--overwrite_args",
        action="store_true",
        help="overwrite args with fixed_params and tunable_params",
        default=False,
    )
    parser.add_argument(
        "--delete_checkpoints",
        action="store_true",
        help="delete checkpoints after training",
        # default=False,
        default=True,
    )

    # * data loader
    parser.add_argument(
        "--data_name",
        type=str,
        default="ETTh1",
        help="dataset name",
        choices=[
            "Weather",
            "ETTh1",
            "ETTh2",
            "ETTm1",
            "ETTm2",
            "ECL",
            "Traffic",
        ],
    )
    parser.add_argument("--data", type=str, default="ETTh1", help="dataset type")
    parser.add_argument(
        "--root_path",
        type=str,
        default="./data/ETT/",
        help="root path of the data file",
    )
    parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="data file")
    parser.add_argument(
        "--features",
        type=str,
        default="M",
        help="forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate (only for downstream tasks)",
    )
    parser.add_argument(
        "--target", type=str, default="OT", help="target feature in S or MS task"
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="h",
        help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="./checkpoints/",
        help="location of model checkpoints",
    )
    parser.add_argument(
        "--pred_len_list",
        type=int,
        nargs="+",
        default=[96, 192, 336, 720],
        help="the prediction length list",
    )
    parser.add_argument(
        "--percent",
        type=int,
        default=100,
        help="the percentage of the training set",
    )

    # * forecasting task
    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument("--label_len", type=int, default=0, help="start token length")
    parser.add_argument(
        "--pred_len", type=int, default=96, help="prediction sequence length"
    )

    # * model architecture
    parser.add_argument(
        "--LLM",
        type=str,
        default="gpt2",
        help="the pretrained LLM model",
        # choices=["gpt2", "llama", "falcon"],
        choices=["gpt2"],
    )
    parser.add_argument(
        "--no_freeze",
        action="store_true",
        help="if False, we will freeze the parameters of the pretrained LLM model",
        default=False,
    )
    parser.add_argument(
        "--no_pretrain",
        action="store_true",
        help="if False, we will use the pretrained weights of the LLM model",
        default=False,
    )
    parser.add_argument(
        "--first_k_layers",
        type=int,
        default=6,
        help="the number of initial layers to be used in LLM",
    )
    parser.add_argument("--enc_in", type=int, default=7, help="encoder input size (C)")
    parser.add_argument("--d_model", type=int, default=512, help="dimension of model")
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument(
        "--embed",
        type=str,
        default="timeF",
        help="time features encoding, options:[timeF, fixed, learned]",
    )
    parser.add_argument(
        "--token_embed_type",
        type=str,
        default="conv",
        choices=["linear", "conv"],
        help="token embedding type",
    )
    parser.add_argument(
        "--token_embed_kernel_size",
        type=int,
        default=3,
        help="token embedding kernel size (for conv)",
    )
    parser.add_argument(
        "--temporal_embed_type",
        type=str,
        default="learned",
        choices=["none", "fixed", "learned", "timeF"],
        help="temporal embedding type",
    )
    parser.add_argument("--activation", type=str, default="gelu", help="activation")
    parser.add_argument(
        "--patch_len", type=int, default=16, help="the length of the patch"
    )
    parser.add_argument(
        "--stride", type=int, default=16, help="the stride of the patch"
    )

    # * peft (LoRA-related)
    parser.add_argument(
        "--peft_method",
        type=str,
        default="lora",
        choices=["none", "lora", "adalora"],
        help="PEFT method",
    )
    parser.add_argument(
        "--peft_params_r",
        type=int,
        default=8,
        help="the dimension of the low-rank matrices",
    )
    parser.add_argument(
        "--peft_params_lora_alpha",
        type=int,
        default=32,
        help="the scaling factor for the low-rank matrices",
    )
    parser.add_argument(
        "--peft_params_lora_dropout",
        type=float,
        default=0.1,
        help="the dropout probability of the LoRA layers",
    )

    # * training_stage_params (sft)
    parser.add_argument(
        "--enable_supervised_finetuning",
        type=bool,
        default=True,
        help="enable supervised finetuning (sft)",
    )
    parser.add_argument(
        "--sft_optim",
        type=str,
        default="Adam",
        help="optimizer",
        choices=["Adam", "AdamW", "RMSprop"],
    )
    parser.add_argument(
        "--sft_learning_rate", type=float, default=0.001, help="optimizer learning rate"
    )
    parser.add_argument(
        "--sft_lradj", type=str, default="type1", help="adjust learning rate"
    )
    parser.add_argument("--sft_weight_decay", type=float, default=0.001)
    parser.add_argument("--sft_train_epochs", type=int, default=10, help="train epochs")

    # * training_stage_params (dft)
    parser.add_argument(
        "--dft_optim",
        type=str,
        default="Adam",
        help="optimizer",
        choices=["Adam", "AdamW", "RMSprop"],
    )
    parser.add_argument(
        "--dft_learning_rate", type=float, default=0.001, help="optimizer learning rate"
    )
    parser.add_argument(
        "--dft_lradj", type=str, default="type1", help="adjust learning rate"
    )
    parser.add_argument("--dft_weight_decay", type=float, default=0.001)
    parser.add_argument("--dft_train_epochs", type=int, default=10, help="train epochs")

    # * training_stage_params (shared)
    parser.add_argument(
        "--num_workers", type=int, default=8, help="data loader num workers"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="batch size of train input data"
    )
    parser.add_argument(
        "--patience", type=int, default=3, help="early stopping patience"
    )
    parser.add_argument(
        "--delta", type=float, default=0.0001, help="early stopping delta"
    )
    parser.add_argument(
        "--ft_mode",
        type=str,
        default="lp_ft",
        choices=["lp_ft", "lp", "ft"],
        help="fine-tuning mode (it should be `ft` for sft and `lp_ft` for dft)",
    )

    # * Hardware
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="use automatic mixed precision training",
        # default=False,
        default=True,
    )

    args, _ = parser.parse_known_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.root_path = Path.cwd()  # Set this outside of the trainable function

    args.return_single_feature = (
        True  # make batch_size invariant to the number of features
    )

    return args


def update_args_from_fixed_params(
    args: argparse.Namespace, fixed_params: dict
) -> argparse.Namespace:
    # Update args
    for key, value in fixed_params.items():
        print("### [Fixed] Set {} to {}".format(key, value))
        setattr(args, key, value)

    return args


def update_args_from_tunable_params(
    args: argparse.Namespace, tunable_params: dict
) -> argparse.Namespace:
    for key, value in tunable_params.items():
        print("### [Tunable] Set {} to {}".format(key, value))
        setattr(args, key, value)

    return args


def update_args(args, fixed_params, tunable_params):
    # Check if there are duplicated keys
    duplicated_keys = set(fixed_params.keys()) & set(tunable_params.keys())
    assert not duplicated_keys, f"Duplicated keys found: {duplicated_keys}"

    # Update args from fixed_params, tunable_params, and dataset
    if args.overwrite_args:
        args = update_args_from_fixed_params(args, fixed_params)
        args = update_args_from_tunable_params(args, tunable_params)
    args = update_args_from_dataset(args)

    args.setting = "{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_eb{}".format(
        args.task_name,
        args.model_id,
        args.model,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.embed,
    )
    print(f"Args in experiment: {args}")

    # Create sft_args
    sft_args = deepcopy(args)
    sft_args.task_name = "supervised_finetuning"
    sft_args.ft_mode = "ft"  # no need to probe
    sft_args.features = "M" # there's no univariate task in sft
    sft_args.pred_len = sft_args.stride
    sft_args.label_len = sft_args.seq_len - sft_args.pred_len
    sft_args.setting = "sft_" + sft_args.setting

    # Create dft_args (just like args)
    dft_args = deepcopy(args)

    return sft_args, dft_args


def get_exp(args):
    # Downstream task
    if args.task_name == "long_term_forecast":
        exp = Exp_Long_Term_Forecast(args)
    elif args.task_name == "supervised_finetuning":
        exp = Exp_Supervised_Finetuning(args)
    else:
        raise NotImplementedError

    return exp


def trainable(
    tunable_params: dict,
    fixed_params: dict,
    args: argparse.Namespace,
) -> dict:
    # Update args
    sft_args, dft_args = update_args(args, fixed_params, tunable_params)

    # sft
    if dft_args.enable_supervised_finetuning:
        sft_exp = get_exp(sft_args)
        sft_metrics = sft_exp.train(use_tqdm=True)

    # dft
    dft_metrics_dict = {}
    for pred_len in dft_args.pred_len_list:
        # Update pred_len
        dft_args.pred_len = pred_len

        dft_exp = get_exp(dft_args)
        dft_metrics_dict[pred_len] = dft_exp.train(use_tqdm=True)

    # Return metrics
    return_metrics = {}
    return_metrics["avg_mse"] = np.mean(
        [v["best_test_loss"] for v in dft_metrics_dict.values()]
    )
    return_metrics["avg_mae"] = np.mean(
        [v["best_test_mae"] for v in dft_metrics_dict.values()]
    )
    for pred_len in dft_args.pred_len_list:
        return_metrics[f"{pred_len}_mse"] = dft_metrics_dict[pred_len]["best_test_loss"]
        return_metrics[f"{pred_len}_mae"] = dft_metrics_dict[pred_len]["best_test_mae"]

    if args.delete_checkpoints:
        # Delete both sft and dft checkpoints
        shutil.rmtree(args.checkpoints)

    return return_metrics  # we only care about downstream task's best_test_loss


if __name__ == "__main__":
    """------------------------------------"""
    # data_name = "Weather"  # 21
    data_name = "ETTh1"  # 7
    # data_name = "ETTh2"  # 7
    # data_name = "ETTm1"  # 7
    # data_name = "ETTm2"  # 7
    # data_name = "ECL"  # 321
    # data_name = "Traffic"  # 862

    # pred_len_list = [96, 192, 336, 720]
    # pred_len_list = [96, 192, 336]  # 5% (ETTh1, ETTh2, Traffic)
    # pred_len_list = [24, 48, 168, 336, 720] # linear probe
    pred_len_list = [96]
    # pred_len_list = [192]
    # pred_len_list = [336]
    # pred_len_list = [720]

    percent = 100
    # percent = 10
    # percent = 5

    # num_workers = 4
    # num_workers = 6
    num_workers = 8

    batch_size = 128  # 8G
    # batch_size = 512 # 24G
    """------------------------------------"""
    set_seed(seed=2023)

    # Setup args
    args = get_args_from_parser()

    # Setup fixed params
    fixed_params = {
        "data_name": data_name,
        "pred_len_list": pred_len_list,
        "percent": percent,
        "num_workers": num_workers,
        "batch_size": batch_size,
    }

    # Setup tunable params
    # TODO: copy `config` from `exp_settings_and_results` (be careful with the boolean values)
    tunable_params = {
        "enable_supervised_finetuning": True,
        "first_k_layers": 6,
        "patch_len": 16,
        "stride": 8,
        "seq_len": 336,
        "ft_mode": "lp_ft",
        "dropout": 0.05,
        "token_embed_type": "conv",
        "token_embed_kernel_size": 3,
        "temporal_embed_type": "learned",
        "freq": "h",
        # "peft_method": "lora",
        "peft_method": "adalora",
        "sft_optim": "AdamW",
        "sft_learning_rate": 7.912045141879411e-05,
        "sft_lradj": "constant",
        "sft_weight_decay": 0.0005542494992024964,
        "sft_train_epochs": 5,
        "dft_optim": "AdamW",
        "dft_learning_rate": 1.8257759510439175e-05,
        "dft_lradj": "constant",
        "dft_weight_decay": 0.0014555863788252605,
        "dft_train_epochs": 15,
        "peft_params_r": 8,
        "peft_params_lora_alpha": 64,
        "peft_params_lora_dropout": 0,
    }

    # Run
    return_metrics = trainable(tunable_params, fixed_params, args)
    print_formatted_dict(return_metrics)

from data_provider.data_factory import data_provider
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from tqdm import tqdm
import shutil
from einops import rearrange
import time
from utils.tools import print_params

warnings.filterwarnings("ignore")


def patching_ci(x, patch_len, stride):
    # Prepare padding
    padding_patch_layer = nn.ReplicationPad1d((0, stride))

    # do patching and embedding on tokens
    x = rearrange(x, "B T C -> B C T")  # (B, C, T)
    x = padding_patch_layer(x)  # (B, C, T+S)
    x = x.unfold(dimension=-1, size=patch_len, step=stride)  # (B, C, T_p, P)
    x = rearrange(x, "B C T_p P -> (B C) T_p P")  # (B * C, T_p, P)

    return x


def update_batch_x_y(batch_x, batch_y, args, enable_y_patching=True):
    # Apply instance normalization to batch_x and batch_y
    batch_all = torch.cat((batch_x, batch_y[:, -args.stride :, :]), dim=1)
    dim2reduce = tuple(range(1, batch_all.ndim - 1))
    mean = torch.mean(batch_all, dim=dim2reduce, keepdim=True).detach()
    stdev = torch.sqrt(
        torch.var(batch_all, dim=dim2reduce, keepdim=True, unbiased=False) + 1e-5
    ).detach()
    batch_all = (batch_all - mean) / stdev
    batch_x = batch_all[:, : args.seq_len, :]
    batch_y = batch_all[:, -(args.label_len + args.pred_len) :, :]

    # Apply patching_ci to batch_y
    if enable_y_patching:
        batch_y = patching_ci(batch_y, args.patch_len, args.stride)

    return batch_x, batch_y


class Exp_Supervised_Finetuning(Exp_Basic):
    def __init__(self, args):
        super(Exp_Supervised_Finetuning, self).__init__(args)
        # 1. set args, model_dict, device into self
        # 2. build model

    def _build_model(self):
        if getattr(self, "model", None) is not None:
            raise ValueError("Model already exists!")

        # Try to save C_t into args
        train_data, train_loader = self._get_data(flag="train")
        batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(train_loader))
        self.args.C_t = batch_x_mark.shape[2]

        # Build model
        model = self.model_dict[self.args.model].Model(self.args).float()  # Feed `args`
        print_params(model)

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = getattr(optim, self.args.sft_optim)(
            self.model.parameters(),
            lr=self.args.sft_learning_rate,
            weight_decay=self.args.sft_weight_decay,
        )
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def train(self, use_tqdm=False):
        print(
            ">>>>> start training (supervised finetuning) : {}>>>>>".format(
                self.args.setting
            )
        )

        # Get data
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")
        assert self.args.seq_len == self.args.label_len + self.args.pred_len, (
            "seq_len should be equal to label_len + pred_len, but got "
            f"seq_len:{self.args.seq_len}, label_len:{self.args.label_len}, "
            f"pred_len:{self.args.pred_len}"
        )
        assert self.args.pred_len == self.args.stride
        assert len(train_loader) > 0, "The train_loader is empty!"
        assert len(vali_loader) > 0, "The vali_loader is empty!"
        assert len(test_loader) > 0, "The test_loader is empty!"

        path = os.path.join(
            self.args.checkpoints, self.args.setting
        )  # `setting` is just a path storing config
        if not os.path.exists(path):
            os.makedirs(path)

        start_time = time.time()
        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # Automatic Mixed Precision (some op. are fp32, some are fp16)
        scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)  # type: ignore

        for epoch in range(self.args.sft_train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in (
                tqdm(enumerate(train_loader), total=len(train_loader))
                if use_tqdm
                else enumerate(train_loader)
            ):
                batch_x, batch_y = update_batch_x_y(batch_x, batch_y, self.args)
                batch_y_shape = batch_y.shape
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # encoder - decoder
                with torch.cuda.amp.autocast(enabled=self.args.use_amp):  # type: ignore
                    outputs = self.model(
                        batch_x, batch_x_mark, None, batch_y_mark
                    )  # embedding + encoder + decoder

                    assert (
                        batch_y_shape == batch_y.shape
                    ), f"batch_y_shape: {batch_y_shape}, batch_y.shape: {batch_y.shape}"
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                # Show loss
                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.sft_train_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                # Backward
                scaler.scale(loss).backward()  # type: ignore
                scaler.step(model_optim)
                scaler.update()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            # At the end of each epoch, we evaluate the validation set and test set
            print(">>>>> start validation >>>>>")
            vali_loss, vali_mae = self.get_metrics(vali_loader)
            # print(">>>>> start testing >>>>>")
            # test_loss, test_mae = self.get_metrics(test_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss
                )
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(
                model_optim, epoch + 1, self.args.sft_learning_rate, self.args.sft_lradj
            )
            print("------------------------------------------------------------------")

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path), strict=False)
        # shutil.rmtree(path, ignore_errors=True)  # delete the checkpoint folder

        return None

        metrics = {}  # loss = mse
        print("### Calculating metrics for train ###")
        metrics["train_loss"], metrics["train_mae"] = self.get_metrics(train_loader)
        print("### Calculating metrics for vali ###")
        metrics["val_loss"], metrics["val_mae"] = self.get_metrics(vali_loader)
        print("### Calculating metrics for test ###")
        metrics["test_loss"], metrics["test_mae"] = self.get_metrics(test_loader)
        print("===============================")
        print(metrics)
        print("===============================")

        end_time = time.time()
        self.spent_time = end_time - start_time

        return metrics

    def get_metrics(self, data_loader, use_tqdm=False):
        total_mse = 0
        total_mae = 0
        total_samples = 0

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in (
                tqdm(enumerate(data_loader), total=len(data_loader))
                if use_tqdm
                else enumerate(data_loader)
            ):
                batch_x, batch_y = update_batch_x_y(batch_x, batch_y, self.args)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )
                # encoder - decoder
                with torch.cuda.amp.autocast(enabled=self.args.use_amp):  # type: ignore
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                pred = outputs.detach()
                true = batch_y.detach()

                batch_mse = torch.mean((pred - true) ** 2).item()
                batch_mae = torch.mean(torch.abs(pred - true)).item()

                total_mse += batch_mse * len(batch_x)
                total_mae += batch_mae * len(batch_x)
                total_samples += len(batch_x)

        mse = total_mse / total_samples
        mae = total_mae / total_samples

        return mse, mae

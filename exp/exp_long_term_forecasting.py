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
from utils.tools import print_params

warnings.filterwarnings("ignore")


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
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
        model = (
            self.model_dict[self.args.model].Model(self.args).float()
        )  # Feed `args`
        print_params(model)

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = getattr(optim, self.args.dft_optim)(
            self.model.parameters(),
            lr=self.args.dft_learning_rate,
            weight_decay=self.args.dft_weight_decay,
        )
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def train(self, use_tqdm=False):
        print(
            f">>>>> start training (long-term forecasting: {self.args.pred_len}) : {self.args.setting}>>>>>"
        )

        # Load the model (if we have already trained it with sft)
        if self.args.enable_supervised_finetuning:
            checkpoint = torch.load(
                os.path.join("./checkpoints/sft_" + self.args.setting, "checkpoint.pth")
            )
            # Get a list of keys related to the output layer to delete
            keys_related_to_output_layer = [
                k for k in checkpoint.keys() if "output_layer" in k
            ]
            for key in keys_related_to_output_layer:
                del checkpoint[key]

            # Load the modified state dict
            self.model.load_state_dict(checkpoint, strict=False)
            print("### Successfully loaded the model trained with sft ###")

        # Get data
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")
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

        best_test_loss, best_test_mae, best_epoch = (
            np.inf,
            np.inf,
            0,
        )  # for capturing the best test loss during training
        for epoch in range(self.args.dft_train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            # Change from linear probing to fine-tuning in the middle of training
            # (Only happens in the downstream task, not in the supervised fine-tuning)
            if (
                self.args.ft_mode == "lp_ft"
                and epoch == self.args.dft_train_epochs // 2
            ):
                self.model.linear_probe_to_fine_tuning()
                print_params(self.model)

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in (
                tqdm(enumerate(train_loader), total=len(train_loader))
                if use_tqdm
                else enumerate(train_loader)
            ):
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

                    # M: multivariate predict multivariate, S: univariate predict univariate, MS: multivariate predict univariate
                    f_dim = -1 if self.args.features == "MS" else 0
                    outputs = outputs[:, -self.args.pred_len :, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len :, f_dim:]
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
                        (self.args.dft_train_epochs - epoch) * train_steps - i
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
            print(">>>>> start testing >>>>>")
            test_loss, test_mae = self.get_metrics(test_loader)
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_test_mae = test_mae
                best_epoch = epoch + 1

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(
                model_optim, epoch + 1, self.args.dft_learning_rate, self.args.dft_lradj
            )
            print("------------------------------------------------------------------")

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))
        # shutil.rmtree(path, ignore_errors=True)  # delete the checkpoint folder

        metrics = {}  # loss = mse
        # print("### Calculating metrics for train ###")
        # metrics["train_loss"], metrics["train_mae"] = self.get_metrics(train_loader)
        # print("### Calculating metrics for vali ###")
        # metrics["val_loss"], metrics["val_mae"] = self.get_metrics(vali_loader)
        # print("### Calculating metrics for test ###")
        # metrics["test_loss"], metrics["test_mae"] = self.get_metrics(test_loader)
        metrics["best_test_loss"], metrics["best_test_mae"], metrics["best_epoch"] = (
            best_test_loss,
            best_test_mae,
            best_epoch,
        )
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

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:]
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

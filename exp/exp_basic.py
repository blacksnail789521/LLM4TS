import os
import torch
from models import LLM4TS


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "LLM4TS": LLM4TS,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )  # Assume we only use 1 gpu at most
            print("Use GPU")
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _get_data(self):
        pass

    def train(self):
        pass

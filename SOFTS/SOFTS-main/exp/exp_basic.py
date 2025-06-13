import os

import torch

from models import SOFTS


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'SOFTS': SOFTS,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError

    # def _acquire_device(self):
    #     if self.args.use_gpu:
    #         os.environ["CUDA_VISIBLE_DEVICES"] = str(
    #             self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
    #         device = torch.device('cuda:{}'.format(self.args.gpu))
    #         print('Use GPU: cuda:{}'.format(self.args.gpu))
    #     else:
    #         device = torch.device('cpu')
    #         print('Use CPU')
    #     return device
    def _acquire_device(self):
        if 'use_gpu' in self.args and self.args['use_gpu']:
            # 使用 GPU
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            # 使用 CPU
            device = torch.device('cpu')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

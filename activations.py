

import torch
import torch.nn as nn

class ActivationsTracker():

    def __init__(self, param_threshold=0.99):
        '''
        :param param_threshold: float; threshold above which the PReLU layer is deactivated
        '''

        self.inactive_layers = []
        self.active_layers = []
        self.num_active = 0
        self.init_num_active = 0
        self.param_threshold = param_threshold
        self.regularise = False


    def add_layer(self, layer):

        self.active_layers.append(layer)
        self.num_active += 1
        self.init_num_active += 1


    def start_regularising(self):

        self.regularise = True


    def _sparsity_loss(self, params):
        '''calculate sparsity regularisation loss Σ|1 − αi|^0.5'''
        return torch.sum(torch.abs(1 - params)**0.5)


    def calc_regularisation_term(self):

        # if we don't want to regularise yet, return 0
        if not self.regularise:
            return torch.tensor(0, dtype=torch.float)

        reg_loss = 0

        for al in self.active_layers:
            for p in al.parameters():
                reg_loss += self._sparsity_loss(p)

        return reg_loss

    def print_active_params(self):

        params = []
        for idx, al in enumerate(self.active_layers):
            for p in al.parameters():
                params += p.data.tolist()

        print(f"{self.num_active} / {self.init_num_active} activation layers remain active")
        print(['%.4f' % p for p in params])

    def _deactivate_layer(self, layer_idx):

        # remove from active layers list
        layer = self.active_layers.pop(layer_idx)
        print(f"PReLU param before removal = {layer.weight.data}")

        # fix parameter to 1 and freeze layer
        layer.weight.data = torch.ones_like(layer.weight.data)
        layer.weight.requires_grad = False

        # add to inactive layers
        self.inactive_layers.append(layer)
        self.num_active -= 1


    def _validate_deactivated_layers(self):
        '''check that all inactive layers output identity function'''
        for layer in self.inactive_layers:
            assert layer(torch.tensor(-100, dtype=torch.float).to(layer.weight.device)) == -100


    def update_active_layers(self):

        # if we're not regularising the layers yet, don't want to deactive either
        if not self.regularise:
            return

        for idx, al in enumerate(self.active_layers):
            for p in al.parameters():
                if (torch.abs(1 - p) < (1 - self.param_threshold)).all():
                    self._deactivate_layer(idx)

        self._validate_deactivated_layers()

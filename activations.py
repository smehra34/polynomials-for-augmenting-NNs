

import torch
import torch.nn as nn

class ActivationsTracker():

    def __init__(self, param_threshold=0.99, init_regularisation_w=0.003, **kwargs):
        '''
        :param param_threshold: float; threshold above which the PReLU layer is deactivated
        '''

        self.inactive_layers = []
        self.active_layers = []
        self.num_active = 0
        self.init_num_active = 0
        self.param_threshold = param_threshold
        self.regularise = False
        self.regularisation_w = init_regularisation_w


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

        return reg_loss * self.regularisation_w

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


class RegularisationWeightScheduler():

    def __init__(self, activations_tracker, increase_factor=2, patience=10,
                 verbose=True, mode='min'):
        '''
        :param activations_tracker: ActivationsTracker;
        :param increase_factor: float; factor by which to increase regularisation weight
        :param patience: int; number of epochs without improvement before increasing w
        :param verbose: bool; whether to print a message at each weight update
        '''

        self.activations_tracker = activations_tracker
        self.increase_factor = increase_factor
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.metric_values = []


    def step(self):

        self.metric_values.append(self.activations_tracker.num_active)

        # only track the number of values we need based on the patience
        if len(self.metric_values) >= self.patience:
            self.metric_values[-self.patience:]

            if not self._is_improving():

                self._update_w()

                if self.verbose:
                    print('\n')
                    print(f"Regularisation losses have not improved in {self.patience} epochs.")
                    print(self.metric_values)
                    print(f"New regularisation weight is {self.activations_tracker.regularisation_w:0.4f}")
                    print('\n')

                self.metric_values = []

    def _is_improving(self):

        # if oldest tracked metric (assuming loss buffer is full) is the
        # min/max (depending on mode), means there has been no improvement
        # since <patience> epochs
        fct = min if self.mode == 'min' else max
        return self.metric_values[0] != fct(self.metric_values)

    def _update_w(self):

        self.activations_tracker.regularisation_w *= self.increase_factor

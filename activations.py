

import torch
import torch.nn as nn
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

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
        return self.num_active

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
            self.metric_values = self.metric_values[-self.patience:]

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


class ActivationsVisualiser():


    def __init__(self, net):
        '''
        :param net: torch.nn.Module;
        '''

        self.activ_layers = [l for l in net.state_dict().keys() if 'activ' in l]
        self.values = {l:[] for l in self.activ_layers}
        self.step(net)

    def step(self, net):

        net_state = net.state_dict()
        for al in self.activ_layers:
            self.values[al].append(net_state[al].item())

    def save_values(self, outdir):

        with open(os.path.join(outdir, 'activ_param_values.pkl'), 'wb') as fp:
            pickle.dump(self.values, fp)
        self.plot_values(outdir)


    def plot_values(self, outdir):

        max_dist = 0
        values = {}
        for al in self.activ_layers:
            values[al] = [abs(1-v) for v in self.values[al]]
            max_dist = max(max_dist, max(values[al]))

        plt.imshow(values.values(), cmap='viridis', vmin=0, vmax=max_dist, aspect='auto')

        plt.yticks(np.arange(len(values)), list(values.keys()))
        plt.ylabel('Activation layer', fontsize=12)

        x_ticks_step = max(1, len(values[al]) // 20)
        plt.xticks(np.arange(0, len(values[al]), step=x_ticks_step),
                   [str(i) for i in range(0, len(values[al]), x_ticks_step)])
        plt.xlabel('Epoch', fontsize=12)

        plt.colorbar(label='| 1 - α |')
        plt.tight_layout()

        plt.savefig(os.path.join(outdir, 'activations_heatmap.png'))
        plt.clf()


# define as global var to make it like a singleton since we want a shared layer
# everywhere
SHARED_LEAKYRELU_LAYER = nn.LeakyReLU(negative_slope=0, inplace=True)

class ActivationIncrementer():

    def __init__(self, start_increment_epoch, end_increment_epoch,
                 increment_patience=1, **kwargs):
        '''
        :param start_increment_epoch: int; when to start incrementing leakyrelu slope
        :param end_increment_epoch: int; when to stop incrementing leakyrelu slope
        :param increment_patience: int; number of epochs to wait before incrementing again
        '''

        self.start_epoch = start_increment_epoch
        self.end_epoch = end_increment_epoch
        self.increment_patience = increment_patience
        self.current_epoch = 0
        self.epochs_since_increment = 0
        self.increment = increment_patience / (self.end_epoch - self.start_epoch)

    def step(self):

        self.current_epoch += 1
        if self.current_epoch <= self.start_epoch:
            SHARED_LEAKYRELU_LAYER.negative_slope = 0

        elif self.current_epoch >= self.end_epoch:
            SHARED_LEAKYRELU_LAYER.negative_slope = 1

        else:
            if self.epochs_since_increment >= self.increment_patience:
                SHARED_LEAKYRELU_LAYER.negative_slope += self.increment
                self.epochs_since_increment = 0

            self.epochs_since_increment += 1

        assert(SHARED_LEAKYRELU_LAYER.negative_slope <= 1)
        print(f"LeakyReLU slope (epoch {self.current_epoch}): {SHARED_LEAKYRELU_LAYER.negative_slope}")

        return SHARED_LEAKYRELU_LAYER.negative_slope

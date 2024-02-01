import os
import matplotlib.pyplot as plt
import numpy as np
import pickle

class MetricsOverEpochsViz():

    def __init__(self, train_val_metrics, other_metrics, **kwargs):
        '''
        :param train_val_metrics: list; list of metrics which will be tracked
            both train and val sets (plotted on same axis)
        :param other_metrics: list; list of additional metrics which will
            be tracked individually (plotted on individual axis)
        '''

        self.epochs = 0
        train_metrics = {m:[] for m in train_val_metrics}
        val_metrics = {m:[] for m in train_val_metrics}
        other_metrics = {m:[] for m in other_metrics}
        self.metrics = {
            'train': train_metrics,
            'val': val_metrics,
            'other': other_metrics
        }

    def add_value(self, metric, value, type):

        '''
        :param metric: str; name of metric
        :param value: float/int; value of metric
        :param type: ['train', 'val', 'other']: whether it's a train, val or
            other value
        '''
        assert type in self.metrics, "Invalid type of metric. Must be one of ['train', 'val', 'other']"
        assert metric in self.metrics[type], f"{metric} is not being tracked for {type}"

        self.metrics[type][metric].append(value)

    def step(self):

        '''
        Call every epoch to validate that each metric was registered for the
        epoch exactly once
        '''
        self.epochs += 1
        invalid_metrics = []
        for type in self.metrics:
            for metric in self.metrics[type]:
                if len(self.metrics[type][metric]) != self.epochs:
                    invalid_metrics.append((type, metric,
                                            len(self.metrics[type][metric])))

        assert not invalid_metrics, f"{self.epochs} epochs complete, but the following metrics had an invalid number of values: {invalid_metrics}"


    def save_values(self, outdir):

        with open(os.path.join(outdir, 'metrics_over_epochs_values.pkl'), 'wb') as fp:
            pickle.dump(self.metrics, fp)

    def plot_values(self, outdir):

        for metric in self.metrics['train']:
            train_vals = self.metrics['train'][metric]
            val_vals = self.metrics['val'][metric]

            plt.plot(train_vals)
            plt.plot(val_vals)
            plt.title(f"{metric} curve")
            plt.ylabel(metric)
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"train_val_{metric}.png"))
            plt.clf()

        for metric in self.metrics['other']:

            vals = self.metrics['other'][metric]
            plt.plot(vals)
            plt.title(f"{metric} curve")
            plt.ylabel(metric)
            plt.xlabel('epoch')
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"{metric}.png"))
            plt.clf()



class PReLUVisualiser():


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

        max_val = 0
        min_val = 0
        for al in self.activ_layers:
            max_val = max(max_val, max(self.values[al]))
            min_val = min(min_val, min(self.values[al]))

        max_mag_val = max(max_val, -min_val)

        plt.imshow(self.values.values(), cmap='coolwarm', vmin=-max_mag_val, vmax=max_mag_val, aspect='auto')

        plt.yticks(np.arange(len(self.values)), list(self.values.keys()))
        plt.ylabel('Activation layer', fontsize=12)

        x_ticks_step = max(1, len(self.values[al]) // 5)
        plt.xticks(np.arange(0, len(self.values[al]), step=x_ticks_step),
                   [str(i) for i in range(0, len(self.values[al]), x_ticks_step)])
        plt.xlabel('Epoch', fontsize=12)

        plt.colorbar(label='α')
        plt.tight_layout()

        plt.savefig(os.path.join(outdir, 'prelu_params_heatmap.png'))
        plt.clf()

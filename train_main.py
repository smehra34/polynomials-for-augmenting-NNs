# Train CIFAR10 with pytorch
from __future__ import print_function
import yaml
import sys
from time import time
import random
from os.path import abspath, dirname, join, isdir
from os import curdir, makedirs
import logging
import shutil

import torch
import torch.optim as optim

from utils import (save_checkpoints, load_model, return_loaders)
from activations import (ActivationsTracker, RegularisationWeightScheduler,
                         ActivationsVisualiser, ActivationIncrementer)
from visualisations import MetricsOverEpochsViz

torch.backends.cudnn.benchmark = True
base = dirname(abspath(__file__))
sys.path.append(base)


def train(train_loader, net, optimizer, criterion, train_info, epoch, device,
          activations_tracker=None, reg_w_scheduler=None, metric_logger=None):
    """ Perform single epoch of the training."""
    net.train()
    # # initialize variables that are augmented in every batch.
    train_loss, reg_loss, correct, total = 0, 0, 0, 0
    start_time = time()
    for idx, data_dict in enumerate(train_loader):
        img, label = data_dict[0], data_dict[1]
        inputs, label = img.to(device), label.to(device)
        optimizer.zero_grad()
        pred = net(inputs)
        loss = criterion(pred, label)
        if activations_tracker is not None:
            reg = activations_tracker.calc_regularisation_term()
            if torch.isnan(reg):
                print('Regularisation loss is nan')
                activations_tracker.print_active_params()
                sys.stdout.flush()
            loss +=  reg
        assert not torch.isnan(loss), 'NaN loss.'
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if activations_tracker is not None:
            reg_loss += reg.item()
            train_loss += reg.item()

        _, predicted = torch.max(pred.data, 1)
        total += label.size(0)
        correct += predicted.eq(label).cpu().sum()
        if idx % train_info['display_interval'] == 0:
            reg_loss_str = '' if activations_tracker is None else f", Reg loss: {reg_loss:.04f}"
            m2 = ('Time: {:.04f}, Epoch: {}, Epoch iters: {} / {}\t'
                  'Loss: {:.04f}{}, Acc: {:.06f}')
            print(m2.format(time() - start_time, epoch, idx, len(train_loader),
                            float(train_loss), reg_loss_str,
                            float(correct) / total))
            start_time = time()

    if metric_logger is not None:
        metric_logger.add_value('acc', float(correct) / total, 'train')
        metric_logger.add_value('train_loss', float(train_loss), 'other')
        if activations_tracker is not None:
            metric_logger.add_value('reg_loss', float(reg_loss), 'other')

    return net


def test(net, test_loader, device='cuda', verbose=False):
    """ Perform testing, i.e. run net on test_loader data
        and return the accuracy. """
    net.eval()
    correct, total = 0, 0
    if hasattr(net, 'is_training'):
        net.is_training = False
    for (idx, data) in enumerate(test_loader):
        if verbose:
            sys.stdout.write('\r [%d/%d]' % (idx + 1, len(test_loader)))
        sys.stdout.flush()
        img, label = data[0].to(device), data[1].to(device)
        with torch.no_grad():
             pred = net(img)
        _, predicted = pred.max(1)
        total += label.size(0)
        correct += predicted.eq(label).sum().item()
    if hasattr(net, 'is_training'):
        net.is_training = True
    return correct / total


def main(seed=None, use_cuda=True):
    # # set the seed for all.
    if seed is None:
        seed = random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # # set the cuda availability.
    cuda = torch.cuda.is_available() and use_cuda
    device = torch.device('cuda' if cuda else 'cpu')
    print(device)
    yml = yaml.safe_load(open('pdc_so_nosharing.yml'))  # # file that includes the configuration.
    cur_path = abspath(curdir)
    # # define the output path
    outdir = 'results_poly'
    if 'exp_name' in yml:
        outdir = join(outdir, yml['exp_name'])
    out = join(cur_path, outdir, '')
    if not isdir(out):
        makedirs(out)

    shutil.copyfile('pdc_so_nosharing.yml', join(out, 'config.yml'))

    # # set the dataset options.
    train_loader, test_loader = return_loaders(**yml['dataset'])
    m1 = 'Current path: {}. Length of iters per epoch: {}. Length of testing batches: {}.'
    print(m1.format(cur_path, len(train_loader), len(test_loader)))

    activation_metrics_to_track = []

    modc = yml['model']
    # add an activations tracker object to the model args if an activation
    # layer parameter threshold was provided
    activations_tracker = None
    if modc['args']['train_time_activ'] == 'regularised':
        activations_tracker = ActivationsTracker(**yml['training_info']['train_time_activations'])
        modc['args']['activations_tracker'] = activations_tracker
        activation_metrics_to_track += ['num_active', 'reg_loss']

    elif modc['args']['train_time_activ'] == 'fixed_increment':
        activation_incrementer = ActivationIncrementer(**yml['training_info']['train_time_activations'])
        activation_metrics_to_track += ['leakyrelu_slope']

    # load the model.
    net = load_model(modc['fn'], modc['name'], modc['args']).to(device)

    # if using pretrained model parameters, print initial accuracy as a sanity
    # check
    if 'pretrained_params_path' in modc['args']:
        acc = test(net, test_loader, device=device)
        print(f"\nInitial accuracy with pretrained weights: {acc}\n")

    # report number of train time activation layers being tracked
    if activations_tracker is not None:
        print(f"Registered {activations_tracker.num_active} train time activation layers")
        activations_visualiser = ActivationsVisualiser(net)

    # # define the criterion and the optimizer.
    criterion = torch.nn.CrossEntropyLoss().to(device)
    sub_params = [p for p in list(net.parameters()) if p.requires_grad]
    decay = yml['training_info']['weight_dec'] if 'weight_dec' in yml['training_info'].keys() else 5e-4
    optimizer = optim.SGD(sub_params, lr=yml['learning_rate'],
                          momentum=0.9, weight_decay=decay)

    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('total params: {}'.format(total_params))

    metric_logger = MetricsOverEpochsViz(train_val_metrics = ['acc'],
                                         other_metrics = ['train_loss'] + activation_metrics_to_track)

    # # get the milestones/gamma for the optimizer.
    tinfo = yml['training_info']
    mil = tinfo['lr_milestones'] if 'lr_milestones' in tinfo.keys() else [40, 60, 80, 100]
    gamma = tinfo['lr_gamma'] if 'lr_gamma' in tinfo.keys() else 0.1
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=mil, gamma=gamma)
    reg_w_scheduler = None
    best_acc, best_epoch, accuracies = 0, 0, []

    for epoch in range(1, tinfo['total_epochs'] + 1):

        # remove activation layers from model if using regularised train time
        # activ and epoch threshold is reached
        if modc['args']['train_time_activ'] == 'regularised' and epoch == tinfo['train_time_activations']['epochs_before_regularisation'] + 1:
            activations_tracker.start_regularising()
            msg = f"\n\n----- Activations now being penalised at epoch {epoch} with weight {activations_tracker.regularisation_w} -----\n\n"
            print(msg)
            logging.info(msg)

            # if reg weight scheduler specified in config, create this
            if 'scheduler' in tinfo['train_time_activations']:
                reg_w_scheduler = RegularisationWeightScheduler(activations_tracker,
                                                                **tinfo['train_time_activations']['scheduler'])

        elif modc['args']['train_time_activ'] == 'fixed_increment':
            slope = activation_incrementer.step()
            metric_logger.add_value('leakyrelu_slope', slope, 'other')

        scheduler.step()
        net = train(train_loader, net, optimizer, criterion, yml['training_info'],
                    epoch, device, activations_tracker=activations_tracker,
                    reg_w_scheduler=reg_w_scheduler, metric_logger=metric_logger)

        if activations_tracker is not None:
            activations_tracker.update_active_layers()
            num_active = activations_tracker.print_active_params()
            metric_logger.add_value('num_active', num_active, 'other')

            activations_visualiser.step(net)
            activations_visualiser.save_values(out)

            if reg_w_scheduler is not None:
                reg_w_scheduler.step()

        save_checkpoints(net, optimizer, epoch, out)
        # # testing mode to evaluate accuracy.
        acc = test(net, test_loader, device=device)
        metric_logger.add_value('acc', acc, 'val')
        if acc > best_acc:
            out_path = join(out, 'net_best_1.pth')
            state = {'net': net.state_dict(), 'acc': acc,
                     'epoch': epoch, 'n_params': total_params}

            if activations_tracker is not None:
                state['active_activations'] = activations_tracker.num_active
                state['init_activations'] = activations_tracker.init_num_active
                state['regularisation_w'] = activations_tracker.regularisation_w

            torch.save(state, out_path)
            best_acc = acc
            best_epoch = epoch
        accuracies.append(float(acc))
        msg = '\nEpoch:{}.\tAcc: {:.03f}.\t Best_Acc:{:.03f} (epoch: {}).\n'
        print(msg.format(epoch,  acc, best_acc, best_epoch))
        logging.info(msg.format(epoch, acc, best_acc, best_epoch))

        metric_logger.step()
        metric_logger.save_values(out)

    metric_logger.plot_values(out)

if __name__ == '__main__':
    main()

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from src.load_data import DataModule
from src.models import LinearRegression
from src.evaluate import Evaluator
from src.utils import Plotter


def main():

    torch.manual_seed(54)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load normalization stats
    y_stats = np.load('data/y_stats.npz')

    # load data
    train_loader, val_loader, test_loader = DataModule.get_dataloaders(
        train_dir='data/train',
        val_dir='data/val',
        test_dir='data/test'
    )

    # model setup
    model_name = 'FinalRidgeRegression'
    model = LinearRegression(in_dim=7).to(device)

    # load trained weights from notebook
    weights_path = 'results/models/final_ridge_regression.pth'
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # hyperparams only for plot annotation
    lr = 9e-3
    l2_lambda = 1e-3

    # helpers
    evaluator = Evaluator(model, device, y_stats)
    plotter = Plotter(save_dir='results/plots')

    # evaluate
    y_pred, y_true, mse, mae, r2, m_rel_err, rms_rel_err = evaluator.evaluate(test_loader)

    # empty losses since we skip training
    train_losses = []
    val_losses = []

    # plot kwargs
    plot_kwargs = {
        'model_name': model_name,
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'y_true': y_true,
        'y_pred': y_pred,
        'lr': lr,
        'l2_lambda': l2_lambda,
        'loss_name': 'MSE',
        'opt_name': 'AdamW'
    }

    # general plots
    plotter.generate_plot(plot_type='train', **plot_kwargs)
    plotter.generate_plot(plot_type='accuracy', **plot_kwargs)
    plotter.generate_plot(plot_type='relative_error', **plot_kwargs)
    plotter.generate_plot(plot_type='relative_error_2d', **plot_kwargs)

    # specific error plots
    plotter.plot_unclipped_relative_error_lg(y_true, y_pred, model_name)
    plotter.plot_absolute_error_hexbin_lg(y_true, y_pred, model_name)

    # plot a detector pulse example
    sample_file = os.path.join('data/train', os.listdir('data/train')[0])
    sample_data = torch.load(sample_file)

    event = sample_data['X'][0].numpy()

    plotter.plot_detector_pulse(
        hg=event[0],
        lg=event[1]
    )


if __name__ == '__main__':
    main()
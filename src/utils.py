import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch


class Plotter:

    # initialize plotter and create directory for saving figures
    def __init__(self, save_dir='results/plots'):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)


    # helper function to add a small box showing model hyperparameters
    def _add_hyperparams_box(self, ax, model_name, lr, l2, opt, loss_fn, pos='right'):

        textstr = f'Model: {model_name}'

        if lr:
            textstr += f'\nLR: {lr}'
        if l2:
            textstr += f'\nL2: {l2}'
        if opt:
            textstr += f'\nOptimizer: {opt}'
        if loss_fn:
            textstr += f'\nLoss: {loss_fn}'

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # choose box position on plot
        x, ha = (0.02, 'left') if pos == 'left' else (0.98, 'right')

        ax.text(
            x,
            0.85,
            textstr,
            transform=ax.transAxes,
            fontsize=9,
            va='top',
            ha=ha,
            bbox=props
        )


    # general plotting function used for different evaluation plots
    def generate_plot(
        self,
        plot_type,
        model_name,
        model=None,
        train_losses=None,
        val_losses=None,
        y_true=None,
        y_pred=None,
        lr=None,
        l2_lambda=None,
        opt_name=None,
        loss_name=None
    ):

        filepath = os.path.join(self.save_dir, f"{model_name}_{plot_type}.png")

        fig, ax = plt.subplots(figsize=(8, 6))


        # training loss vs epochs
        if plot_type == 'train':

            ax.plot(train_losses, label='Train Loss', color='blue', lw=2)

            if val_losses:
                ax.plot(val_losses, label='Val Loss', color='orange', lw=2)

            ax.set_title('Training Curves', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')

            ax.legend()

            self._add_hyperparams_box(
                ax,
                model_name,
                lr,
                l2_lambda,
                opt_name,
                loss_name,
                'right'
            )


        # scatter plot comparing predicted energy vs true energy
        elif plot_type == 'accuracy':

            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())

            ax.scatter(
                y_true,
                y_pred,
                alpha=0.5,
                color='royalblue',
                edgecolors='none'
            )

            # diagonal line = perfect prediction
            ax.plot(
                [min_val, max_val],
                [min_val, max_val],
                'r--',
                lw=2,
                label='Perfect Fit'
            )

            ax.set_title('Predictions vs True', fontweight='bold')
            ax.set_xlabel('True Signal Energy')
            ax.set_ylabel('Predicted Signal Energy')

            ax.legend()

            self._add_hyperparams_box(
                ax,
                model_name,
                lr,
                l2_lambda,
                opt_name,
                loss_name,
                'left'
            )


        # histogram showing distribution of relative prediction error
        elif plot_type == 'relative_error':

            mask = y_true > 1.0

            relative_error = (y_pred[mask] - y_true[mask]) / y_true[mask]

            ax.hist(
                relative_error,
                bins=100,
                density=True,
                color='green',
                alpha=0.7,
                edgecolor='black'
            )

            ax.axvline(
                np.mean(relative_error),
                color='r',
                linestyle='--',
                label=f'Mean: {np.mean(relative_error):.4f}'
            )

            ax.set_title('Relative Error Distribution', fontweight='bold')
            ax.set_xlabel('(E_pred - E_true) / E_true')
            ax.set_ylabel('Density')

            ax.legend()

            self._add_hyperparams_box(
                ax,
                model_name,
                lr,
                l2_lambda,
                opt_name,
                loss_name,
                'right'
            )


        # 2D histogram showing how relative error varies with energy
        elif plot_type == 'relative_error_2d':

            # convert tensors to numpy if needed
            if isinstance(y_true, torch.Tensor):
                y_true = y_true.cpu().numpy()

            if isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.cpu().numpy()

            mask = y_true > 1.0

            y_t = y_true[mask]
            y_p = y_pred[mask]

            rel_err = (y_p - y_t) / y_t

            rel_err_min, rel_err_max = -2.5, 2.5

            # clip extreme errors
            clip_mask = (rel_err >= rel_err_min) & (rel_err <= rel_err_max)

            yt_plot = y_t[clip_mask]
            re_plot = rel_err[clip_mask]

            x_max = np.percentile(yt_plot, 99.5)

            # density heatmap
            h = ax.hist2d(
                yt_plot,
                re_plot,
                bins=[200, 100],
                range=[[0, x_max], [rel_err_min, rel_err_max]],
                cmap='jet',
                norm=LogNorm(vmin=1, vmax=1e5),
                cmin=1
            )

            plt.colorbar(h[3], ax=ax).set_label('Counts', rotation=270, labelpad=15)

            ax.axhline(0, color='white', linestyle='-', lw=1, alpha=0.7)

            # compute mean error per energy bin
            x_edges = np.linspace(0, x_max, 31)

            b_centers, b_means, b_stds = [], [], []

            for i in range(30):

                in_bin = (yt_plot >= x_edges[i]) & (yt_plot < x_edges[i + 1])

                if in_bin.sum() > 10:

                    b_centers.append((x_edges[i] + x_edges[i + 1]) / 2)

                    b_means.append(re_plot[in_bin].mean())

                    b_stds.append(re_plot[in_bin].std())

            # overlay bin statistics
            ax.errorbar(
                b_centers,
                b_means,
                yerr=b_stds,
                fmt='+',
                color='red',
                markersize=8,
                lw=2,
                label='Bin mean ± σ',
                capsize=3,
                alpha=0.8
            )

            ax.set(
                xlim=(0, x_max),
                ylim=(rel_err_min, rel_err_max),
                xlabel='E_true [ADC Counts]',
                ylabel='(E_pred - E_true) / E_true',
                title='Relative Error vs True Energy'
            )

            ax.legend(loc='upper right', fontsize=9)

            self._add_hyperparams_box(
                ax,
                model_name,
                lr,
                l2_lambda,
                opt_name,
                loss_name,
                'right'
            )


        ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()

        plt.savefig(filepath, dpi=300, bbox_inches='tight')

        plt.close()


    # relative error heatmap for low-gain events
    def plot_unclipped_relative_error_lg(self, y_true, y_pred, model_name):

        mask = y_true > 10.0

        yt_v = y_true[mask]
        yp_v = y_pred[mask]

        rel_err = (yp_v - yt_v) / yt_v

        rmin, rmax = -0.15, 0.15

        clip_mask = (rel_err >= rmin) & (rel_err <= rmax)

        yt_plot = yt_v[clip_mask]
        re_plot = rel_err[clip_mask]

        fig, ax = plt.subplots(figsize=(10, 8))

        x_max = min(np.percentile(yt_plot, 99.5), 4000)
        x_min = 0

        h = ax.hist2d(
            yt_plot,
            re_plot,
            bins=[200, 100],
            range=[[x_min, x_max], [rmin, rmax]],
            cmap='jet',
            norm=LogNorm(vmin=1, vmax=1e4),
            cmin=1
        )

        plt.colorbar(h[3], ax=ax).set_label('Counts', rotation=270, labelpad=15)

        ax.axhline(0, color='white', linestyle='-', lw=1.5, alpha=0.8)

        ax.set(
            xlim=(x_min, x_max),
            ylim=(rmin, rmax),
            xlabel='E_true [ADC]',
            ylabel='(E_pred - E_true)/E_true',
            title='Relative Error vs True Energy (Low-Gain)'
        )

        plt.tight_layout()

        plt.savefig(f'{self.save_dir}/{model_name}_relative_error_2d_LG.png', dpi=300)

        plt.close()


    # hexbin plot showing absolute error vs true energy
    def plot_absolute_error_hexbin_lg(self, y_true, y_pred, model_name):

        mask = y_true > 10.0

        abs_err = y_pred[mask] - y_true[mask]

        amin, amax = -60, 60

        clip_mask = (abs_err >= amin) & (abs_err <= amax)

        yt_plot = y_true[mask][clip_mask]
        ae_plot = abs_err[clip_mask]

        fig, ax = plt.subplots(figsize=(10, 8))

        x_max = min(np.percentile(yt_plot, 99.5), 4000)

        h = ax.hexbin(
            yt_plot,
            ae_plot,
            gridsize=150,
            extent=[0, x_max, amin, amax],
            cmap='jet',
            norm=LogNorm(vmin=1, vmax=1e4),
            mincnt=1
        )

        plt.colorbar(h, ax=ax).set_label('Counts', rotation=270, labelpad=15)

        ax.axhline(0, color='white', linestyle='-', lw=1.5, alpha=0.8)

        ax.set(
            xlim=(0, x_max),
            ylim=(amin, amax),
            xlabel='E_true [ADC]',
            ylabel='E_pred - E_true',
            title='Absolute Error vs True Energy (Low-Gain) - Hexbin'
        )

        plt.tight_layout()

        plt.savefig(f'{self.save_dir}/{model_name}_absolute_error_hexbin_LG.png', dpi=300)

        plt.close()


    # visualize a detector waveform for high gain and low gain channels
    def plot_detector_pulse(self, hg, lg):

        time = np.arange(len(hg))

        plt.figure(figsize=(7, 4))

        plt.plot(time, hg, label="High Gain (HG)", linewidth=2)
        plt.plot(time, lg, label="Low Gain (LG)", linewidth=2)

        plt.xlabel("Time Sample")
        plt.ylabel("ADC Counts")

        plt.title("Detector Pulse (HG vs LG)")

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(f'{self.save_dir}/detector_pulse.png', dpi=300)

        plt.close()
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.utils import Progbar


class Evaluator:

    # initialize evaluator with model
    def __init__(self, model, device, y_stats):
        self.model = model
        self.device = device
        self.y_stats = y_stats


    # run inference on test dataset and compute metrics
    def evaluate(self, test_loader, hi_gain=False):

        num_steps = len(test_loader)

        print("Evaluating Model...")
        progbar = Progbar(target=num_steps)

        # switching model to evaluation mode
        self.model.eval()

        # storing predictions and ground truth
        y_pred_list, y_true_list = [], []

        # disabling gradient computation (faster inference)
        with torch.no_grad():

            for batch_idx, (feat_hi, feat_lo, y_true) in enumerate(test_loader, 1):

                # choosing which features to use
                if hi_gain:
                    feat_hi, feat_lo = feat_hi.to(self.device), feat_lo.to(self.device)
                    outputs = self.model(feat_hi, feat_lo)

                else:
                    feat_lo = feat_lo.to(self.device)
                    outputs = self.model(feat_lo)

                # storing batch predictions and labels
                y_pred_list.append(outputs.cpu())
                y_true_list.append(y_true.cpu())

                # update progress bar
                progbar.update(batch_idx)

        # concatenate all batches into one array
        y_pred = torch.cat(y_pred_list).numpy().flatten()
        y_true = torch.cat(y_true_list).numpy().flatten()

        # compute regression metrics
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # denormalize predictions and targets back to original scale
        y_true = y_true * self.y_stats['std'][:, 1] + self.y_stats['mean'][:, 1]
        y_pred = y_pred * self.y_stats['std'][:, 1] + self.y_stats['mean'][:, 1]

        # compute relative error
        eps = 1e-8
        relative_error = (y_pred - y_true) / (y_true + eps)

        # mean and rms of relative error
        mean_relative_error = np.mean(relative_error)
        rms_relative_error = np.sqrt(np.mean(relative_error**2))

        # print evaluation results
        print(f"\nMSE: {mse:.4f}\nMAE: {mae:.4f}\nR² Score: {r2:.4f}")
        print(f"Mean Relative Error: {mean_relative_error:.4f}\nRMS Relative Error: {rms_relative_error:.4f}")

        # return predictions, targets and metrics
        return y_pred, y_true, mse, mae, r2, mean_relative_error, rms_relative_error
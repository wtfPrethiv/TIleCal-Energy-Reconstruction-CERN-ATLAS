import torch
import numpy as np
from tensorflow.keras.utils import Progbar


class Trainer:

    # trainer holds the model, optimizer and loss function used for training
    def __init__(self, model, optimizer, loss_fn, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device


    # training loop
    # runs for a given number of epochs and optionally evaluates on validation data
    def train(self, train_loader, val_loader, epochs, hi_gain=False):

        num_steps = len(train_loader)

        train_losses, val_losses = [], []

        # store predictions from the final validation epoch
        final_val_trues, final_val_preds = [], []

        for epoch in range(epochs):

            print(f"\nEpoch {epoch+1}/{epochs}")

            progbar = Progbar(target=num_steps)

            # switch model to training mode
            self.model.train()

            running_loss = 0.0

            # iterate over batches
            for batch_idx, (feat_hi, feat_lo, y_true) in enumerate(train_loader, 1):

                # move targets to device and match output shape
                y_true = y_true.to(self.device).unsqueeze(1)

                # choose which features to use
                if hi_gain:
                    feat_hi, feat_lo = feat_hi.to(self.device), feat_lo.to(self.device)
                    y_pred = self.model(feat_hi, feat_lo)
                else:
                    feat_lo = feat_lo.to(self.device)
                    y_pred = self.model(feat_lo)

                # compute training loss
                loss = self.loss_fn(y_pred, y_true)

                # standard gradient descent step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # update running loss for progress tracking
                running_loss += loss.item()
                avg_loss = running_loss / batch_idx

                if batch_idx < num_steps:
                    progbar.update(batch_idx, values=[("loss", avg_loss)])

            # store epoch training loss
            train_losses.append(avg_loss)


            # validation phase
            if val_loader is not None:

                self.model.eval()

                val_running_loss = 0.0

                # check if this is the final epoch
                is_final_epoch = (epoch == epochs - 1)

                with torch.no_grad():

                    for val_idx, (v_hi, v_lo, v_y) in enumerate(val_loader, 1):

                        v_y = v_y.to(self.device).unsqueeze(1)

                        # forward pass on validation data
                        if hi_gain:
                            v_hi, v_lo = v_hi.to(self.device), v_lo.to(self.device)
                            v_pred = self.model(v_hi, v_lo)
                        else:
                            v_lo = v_lo.to(self.device)
                            v_pred = self.model(v_lo)

                        # compute validation loss
                        v_loss = self.loss_fn(v_pred, v_y)

                        val_running_loss += v_loss.item()

                        # store predictions from final epoch for later analysis
                        if is_final_epoch:
                            final_val_trues.append(v_y.cpu().numpy())
                            final_val_preds.append(v_pred.cpu().numpy())

                # average validation loss
                val_avg_loss = val_running_loss / len(val_loader)

                val_losses.append(val_avg_loss)

                progbar.update(num_steps, values=[("loss", avg_loss), ("val_loss", val_avg_loss)])

            else:
                progbar.update(num_steps, values=[("loss", avg_loss)])


        # combine stored validation predictions
        if val_loader is not None:
            final_val_trues = np.concatenate(final_val_trues, axis=0)
            final_val_preds = np.concatenate(final_val_preds, axis=0).squeeze()

        return train_losses, val_losses, final_val_trues, final_val_preds
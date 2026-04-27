import torch
import glob
from torch.utils.data import Dataset, DataLoader


class TileCalDataset(Dataset):

    # dataset reads all .pt files from the given directory
    # each file contains detector samples with waveform features (X) and target energy (y)
    def __init__(self, dataDir, norm=False):

        # collect all dataset files
        self.files = sorted(glob.glob(f'{dataDir}/*.pt'))

        # temporary lists to accumulate data from multiple files
        li_X, li_y = [], []

        # load each .pt file
        for file in self.files:

            # each file contains a dictionary with keys 'X' and 'y'
            data = torch.load(file)

            # store feature tensor and target tensor
            li_X.append(data['X'].float())
            li_y.append(data['y'].float())

        # combine all loaded files into one tensor
        # this creates a single dataset stored in memory
        X = torch.cat(li_X, dim=0)
        y = torch.cat(li_y, dim=0)

        # X structure: [events, gain_channel, features]
        # channel 0 = high gain waveform
        # channel 1 = low gain waveform
        self.X_hi = X[:, 0, :]
        self.X_lo = X[:, 1, :]

        # target tensor contains energy values for both channels
        # we use the low gain energy as the regression target
        self.y = y[:, 1]

    # total number of events in the dataset
    def __len__(self):
        return len(self.y)

    # return one event sample
    # dataloader will call this to construct batches
    def __getitem__(self, idx):
        return self.X_hi[idx], self.X_lo[idx], self.y[idx]


class DataModule:

    # helper function to create train / validation / test dataloaders
    @staticmethod
    def get_dataloaders(train_dir, val_dir, test_dir, batch_train=1024, batch_eval=2048):

        # initialize datasets from directories
        train_ds = TileCalDataset(train_dir)
        val_ds = TileCalDataset(val_dir)
        test_ds = TileCalDataset(test_dir)

        # training loader
        # shuffle=True ensures batches are randomly sampled
        # drop_last avoids incomplete batches
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_train,
            shuffle=True,
            drop_last=True,
            pin_memory=True
        )

        # validation loader (no shuffle for deterministic evaluation)
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_eval,
            shuffle=False,
            pin_memory=True
        )

        # test loader used for final model evaluation
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_eval,
            shuffle=False,
            pin_memory=True
        )

        return train_loader, val_loader, test_loader
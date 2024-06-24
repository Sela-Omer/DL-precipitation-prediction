import pytorch_lightning as pl
from torch.utils.data import DataLoader


class MeteorologicalDataModule(pl.LightningDataModule):
    def __init__(self, dataset_class, data_dir, parameters, years, batch_size=1, cache_dir='cache'):
        super().__init__()
        self.dataset_class = dataset_class
        self.data_dir = data_dir
        self.parameters = parameters
        self.years = years
        self.batch_size = batch_size
        self.cache_dir = cache_dir

    def prepare_data(self):
        # This will initialize and cache the index files
        self.train_dataset = self.dataset_class(self.data_dir, self.parameters, self.years, self.cache_dir)
        self.val_dataset = self.dataset_class(self.data_dir, self.parameters, self.years, self.cache_dir)

    def setup(self, stage=None):
        # Called on every GPU separately - set state which is made inside prepare_data
        if stage == 'fit' or stage is None:
            self.train_dataset = self.dataset_class(self.data_dir, self.parameters, self.years, self.cache_dir)
            self.val_dataset = self.dataset_class(self.data_dir, self.parameters, self.years, self.cache_dir)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

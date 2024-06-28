import lightning as pl
from torch.utils.data import DataLoader

from src.dataloader.all_times_collate_fn import all_times_collate_fn
from src.helper.dataset_helper import split_dataset


class MeteorologicalDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for meteorological datasets.
    :param service: A service object that contains the configuration parameters.
    :param dataset_cls: The dataset class to use for loading the data.
    :param data_dir: The directory containing the data files.

    """

    def __init__(self, service, dataset_cls, data_dir):
        super().__init__()
        self.service = service
        self.dataset_cls = dataset_cls
        self.data_dir = data_dir
        self.parameters = service.data_parameters
        self.years = service.data_years
        self.batch_size = service.batch_size
        self.cache_dir = service.data_cache
        self.val_ratio = service.val_ratio
        self.train_index_files = None
        self.val_index_files = None
        self.collate_fn = all_times_collate_fn(service.lookback_range, service.forecast_range)

    def prepare_data(self):
        """
        Prepare the data for training and validation.
        :return:
        """
        # This will initialize and cache the index files if needed by the dataset
        temp_dataset = self.dataset_cls(self.data_dir, self.parameters, self.years, self.cache_dir)
        self.train_index_files, self.val_index_files = split_dataset(temp_dataset.index_files, self.val_ratio)

    def setup(self, stage=None):
        """
        Set up the data for training and validation.
        :param stage: The stage for which to set up the data.
        :return:
        """
        # Called on every GPU separately - set state which is made inside prepare_data
        train_dataset = self.dataset_cls(self.data_dir, self.parameters, self.years, self.cache_dir)
        val_dataset = self.dataset_cls(self.data_dir, self.parameters, self.years, self.cache_dir)

        train_dataset.set_index_files(self.train_index_files)
        val_dataset.set_index_files(self.val_index_files)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train_dataloader(self):
        """
        Return the training dataloader.
        :return:
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=self.service.cpu_workers)

    def val_dataloader(self):
        """
        Return the validation dataloader.
        :return:
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=self.service.cpu_workers)

import os.path
from abc import ABC

import lightning as pl
import torch
import torchaudio
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar, DeviceStatsMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from matplotlib import pyplot as plt

from src.script.script import Script
from src.transform.noise_tfm import NoiseTfm


class EvalScript(Script, ABC):

    def create_trainer(self, callbacks: list):
        """
        Create a trainer with specified configurations.

        Args:
            callbacks (list): A list of callbacks to be used during training.

        Returns:
            pl.Trainer: The created trainer object.
        """
        # Create the trainer with specified configurations
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator=self.service.config['APP']['ACCELERATOR'],
            log_every_n_steps=1,
            callbacks=callbacks,
            logger=None,
            devices=int(self.service.config['APP']['DEVICES']),
            num_nodes=int(self.service.config['APP']['NUM_NODES']),
            strategy=self.service.config['APP']['STRATEGY'],
        )
        return trainer

    def _get_model_checkpoint(self):
        """
        This method returns the path to the model checkpoint.
        :return: The path to the model checkpoint.

        """
        model_dir = f'model/{self.service.model_name}'
        assert os.path.isdir(model_dir), f"Model directory {model_dir} does not exist."
        version_lst = os.listdir(model_dir)

        version_dict = {}
        for version in version_lst:
            version_dir_lst = version.split('_')
            if len(version_dir_lst) != 2 or not version_dir_lst[1].isdigit() or version_dir_lst[0] != 'version':
                continue
            version_ind = int(version_dir_lst[1])
            version_dict[version_ind] = version
        assert len(version_dict) > 0, f"No model versions found in {model_dir}."

        version = None
        if self.service.config['EVAL']['CHECKPOINT_VERSION'] == 'highest':
            version = max(version_dict.keys())
        if self.service.config['EVAL']['CHECKPOINT_VERSION'] == 'loweset':
            version = min(version_dict.keys())
        if self.service.config['EVAL']['CHECKPOINT_VERSION'].isdigit():
            version = int(self.service.config['EVAL']['CHECKPOINT_VERSION'])
        assert version in version_dict, f"Version {version} not found in {model_dir}."
        checkpoint_dir = f'{model_dir}/{version_dict[version]}/checkpoints'
        assert os.path.isdir(checkpoint_dir), f"Checkpoint directory {checkpoint_dir} does not exist."
        checkpoint_lst = os.listdir(checkpoint_dir)
        checkpoint_lst = [checkpoint for checkpoint in checkpoint_lst if checkpoint.endswith('.ckpt')]
        assert len(checkpoint_lst) > 0, f"No checkpoints found in {checkpoint_dir}."
        if len(checkpoint_lst) > 1:
            print(f"Multiple checkpoints found in {checkpoint_dir}. Using the first one: {checkpoint_lst[0]}")
        return f'{checkpoint_dir}/{checkpoint_lst[0]}'

    def _denormalize_metric(self, metric_value):
        """
        Denormalize the metric value if needed.
        :param metric_name:
        :param metric_value:
        :return:
        """

        if self.service.config['EVAL']['APPLY_DENORMALIZATION'] == 'True':
            std_tensor = torch.load(f'stats/{self.service.model_name}-std.pt')
            std_val_at_target = std_tensor[self.service.get_parameter_index(self.service.target_parameters[0])]
            denormalized_metric = metric_value * std_val_at_target
            return denormalized_metric
        return None

    def __call__(self):
        """
        This method orchestrates the training process.
        It creates the data module, architecture, callbacks, and trainer,
        and then fits the model using the trainer.
        """
        # Create the data module
        datamodule = self.create_datamodule()

        # Create the architecture
        arch = self.create_architecture(datamodule)

        # Create the trainer
        trainer = self.create_trainer([])

        metric_dict_lst = trainer.test(model=arch, datamodule=datamodule,
                                       ckpt_path=self._get_model_checkpoint(),
                                       verbose=True)
        metric_dict = metric_dict_lst[0]
        metric_name = self.service.config['EVAL']['METRIC']
        metric_value = metric_dict[metric_name]

        # attempt to denormalize the metric
        denormalized_metric = self._denormalize_metric(metric_value)
        if denormalized_metric is not None:
            print(f'denormalized {metric_name}: {denormalized_metric}')

        # run noise analysis
        noise_metric_dict = {}
        noise_tfm = NoiseTfm(self.service, 0, noise_mean=0, noise_std=0.5)
        self.service.add_tfm(noise_tfm)
        for param in self.service.input_parameters:
            noise_tfm.set_noise_param_index(self.service.get_parameter_index(param))
            metric_dict_lst = trainer.test(model=arch, datamodule=datamodule,
                                           ckpt_path=self._get_model_checkpoint(),
                                           verbose=True)
            metric_dict = metric_dict_lst[0]
            noise_metric_value = metric_dict[metric_name]
            noise_metric_dict[param] = noise_metric_value

        noise_metric_differences = {key: value - metric_value for key, value in noise_metric_dict.items()}
        print(f'Noise analysis results: {noise_metric_differences}')

        # Extracting keys and values for plotting
        parameters = list(noise_metric_differences.keys())
        diff_values = list(noise_metric_differences.values())

        # Sort the parameters and differences based on the difference values in descending order
        parameters, diff_values = zip(
            *sorted(zip(parameters, diff_values), key=lambda x: x[1], reverse=True))

        # Define colors based on the sign of the differences
        colors = ['blue' if diff > 0 else 'red' for diff in diff_values]

        # Creating the horizontal bar chart
        plt.figure(figsize=(10, 6))
        bars = plt.barh(parameters, diff_values, color=colors)
        plt.xlabel('Difference from Baseline')
        plt.ylabel('Parameters')
        plt.title(
            f'Differences between Baseline and Noisy Values \n For: {metric_name} Metric with {f"denorm: {denormalized_metric}, norm: {metric_value}" if denormalized_metric is not None else metric_value} Value')
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        # Adjust the bar labels for better visibility
        for bar, diff in zip(bars, diff_values):
            width = bar.get_width()
            plt.text(width if width > 0 else width - 3,
                     bar.get_y() + bar.get_height() / 2,
                     f'{diff:.5f}',
                     ha='left' if width > 0 else 'right',
                     va='center',
                     color='black')

        f = self._get_model_checkpoint().replace('.ckpt', f'-eval_noise_diff-{metric_name}.png')
        plt.savefig(f)
        plt.show()

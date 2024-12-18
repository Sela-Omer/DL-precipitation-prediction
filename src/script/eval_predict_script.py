from abc import ABC

import torch
from tqdm import tqdm

from src.script.eval_script import EvalScript


class EvalPredictScript(EvalScript, ABC):
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

        # load the model from the checkpoint
        arch.load_state_dict(torch.load(self._get_model_checkpoint())['state_dict'])

        # Get the dataloader
        test_dataloader = datamodule.test_dataloader()

        # Get the dataset
        test_dataset = test_dataloader.dataset
        test_index_files = test_dataset.index_files

        preds_dict = {}
        for idx, item in tqdm(zip(test_index_files, test_dataset)):
            X, y = test_dataloader.collate_fn([item])
            y_hat, y = arch.forward_passthrough(X, y)
            preds_dict[idx] = (y_hat, y)

        # Save the predictions to a file with pickle
        f = self._get_model_checkpoint().replace('.ckpt', f'-predictions.pt')
        torch.save(preds_dict, f)

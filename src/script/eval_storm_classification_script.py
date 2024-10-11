import json
from abc import ABC

import pandas as pd

from src.script.eval_script import EvalScript


class EvalStormClassificationScript(EvalScript, ABC):

    def __call__(self):
        """
        This method orchestrates the training process.
        It creates the data module, architecture, callbacks, and trainer,
        and then fits the model using the trainer.
        """
        # Create the data module
        datamodule = self.create_datamodule()

        datamodule.prepare_data = lambda *_, **__: None
        datamodule.setup = lambda *_, **__: None

        # Create the architecture
        arch = self.create_architecture(datamodule)

        # Read Storm Classification Json
        with open('stats/STORM_CLASSIFICATION.json', 'r') as f:
            storm_classification_dict = json.load(f)

        # Get all the storm classification values
        storm_clf_types = set()
        for storm_clf_lst in storm_classification_dict.values():
            storm_clf_types.update(storm_clf_lst)
        storm_clf_types = list(storm_clf_types)

        assert hasattr(datamodule, 'train_index_files'), 'The datamodule must have a train_index_files attribute'
        assert hasattr(datamodule, 'val_index_files'), 'The datamodule must have a val_index_files attribute'
        all_train_indices = set(datamodule.train_index_files)
        all_val_indices = set(datamodule.val_index_files)

        metric_name = self.service.config['EVAL']['METRIC']
        eval_dict = {'storm_classification_type': [], metric_name: [], f'denormalized_{metric_name}': [],
                     'num_storms': []}

        for clf_type in storm_clf_types:
            # Get all the storm indices for the current classification type
            storm_indices = set([key for key, value in storm_classification_dict.items() if clf_type in value])

            # Set storm indices for the datamodule
            assert hasattr(datamodule, 'train_dataset'), 'The datamodule must have a train_dataset attribute'
            assert hasattr(datamodule, 'val_dataset'), 'The datamodule must have a val_dataset attribute'
            assert hasattr(datamodule.train_dataset,
                           'set_index_files'), 'The train dataset must have a set_index_files method'
            assert hasattr(datamodule.val_dataset,
                           'set_index_files'), 'The val dataset must have a set_index_files method'

            # intersect the storm indices with the train and val indices
            train_storm_indices = list(all_train_indices.intersection(storm_indices))
            val_storm_indices = list(all_val_indices.intersection(storm_indices))

            if len(val_storm_indices) == 0:
                continue

            datamodule.train_dataset.set_index_files(train_storm_indices)
            datamodule.val_dataset.set_index_files(val_storm_indices)
            datamodule.train_index_files = train_storm_indices
            datamodule.val_index_files = val_storm_indices

            # Create the trainer
            trainer = self.create_trainer([])

            # Run the evaluation
            metric_dict_lst = trainer.test(model=arch, datamodule=datamodule,
                                           ckpt_path=self._get_model_checkpoint(),
                                           verbose=True)

            metric_dict = metric_dict_lst[0]

            metric_value = metric_dict[metric_name]

            # attempt to denormalize the metric
            denormalized_metric = self._denormalize_metric(metric_value)

            eval_dict['storm_classification_type'].append(clf_type)
            eval_dict[metric_name].append(metric_value)
            eval_dict[f'denormalized_{metric_name}'].append(denormalized_metric.item() if denormalized_metric is not None else -999)
            eval_dict['num_storms'].append(len(val_storm_indices))

            # print partial results
            print(pd.DataFrame(eval_dict))

        # Create dataframe from evaluation results
        eval_df = pd.DataFrame(eval_dict)

        # Print the evaluation results
        print(eval_df)

        # Save the evaluation results
        f = self._get_model_checkpoint().replace('.ckpt', f'-eval_storm_classification-{metric_name}.csv')
        eval_df.to_csv(f)

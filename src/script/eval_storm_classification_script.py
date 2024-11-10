import json
from abc import ABC

import pandas as pd

from src.helper.param_helper import convert_param_to_list
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

        metric_names = convert_param_to_list(self.service.config['EVAL']['STORM_CLASSIFICATION_METRICS'])
        eval_dict = {'storm_classification_type': [], 'num_storms': [], 'num_samples': []}
        for metric_name in metric_names:
            eval_dict[metric_name] = []
            eval_dict[f'denormalized_{metric_name}'] = []

        for clf_type in storm_clf_types:
            # Get all the storm indices for the current classification type
            storm_indices = [key for key, value in storm_classification_dict.items() if clf_type in value]
            storm_indices_set = set(storm_indices)
            # Get number of appearances of the current classification type in each storm index
            num_samples = []
            for storm_index in storm_indices:
                num_samples.append(storm_classification_dict[storm_index].count(clf_type))

            # Set storm indices for the datamodule
            assert hasattr(datamodule, 'train_dataset'), 'The datamodule must have a train_dataset attribute'
            assert hasattr(datamodule, 'val_dataset'), 'The datamodule must have a val_dataset attribute'
            assert hasattr(datamodule.train_dataset,
                           'set_index_files'), 'The train dataset must have a set_index_files method'
            assert hasattr(datamodule.val_dataset,
                           'set_index_files'), 'The val dataset must have a set_index_files method'

            # intersect the storm indices with the train and val indices
            train_storm_indices = list(all_train_indices.intersection(storm_indices_set))
            val_storm_indices = list(all_val_indices.intersection(storm_indices_set))

            # replicate the storm indices according to the number of samples
            train_storm_indices = [index for index in train_storm_indices for _ in
                                   range(num_samples[storm_indices.index(index)])]
            val_storm_indices = [index for index in val_storm_indices for _ in
                                 range(num_samples[storm_indices.index(index)])]

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

            for metric_name in metric_names:
                metric_value = metric_dict[metric_name]

                # attempt to denormalize the metric
                denormalized_metric = self._denormalize_metric(metric_value)

                eval_dict[metric_name].append(metric_value)
                eval_dict[f'denormalized_{metric_name}'].append(
                    denormalized_metric.item() if denormalized_metric is not None else -999)

            eval_dict['storm_classification_type'].append(clf_type)
            eval_dict['num_storms'].append(len(all_val_indices.intersection(storm_indices_set)))
            eval_dict['num_samples'].append(sum(num_samples))

            # print partial results
            print(pd.DataFrame(eval_dict))

        # Create dataframe from evaluation results
        eval_df = pd.DataFrame(eval_dict)

        # Print the evaluation results
        print(eval_df)

        # Save the evaluation results
        f = self._get_model_checkpoint().replace('.ckpt', f'-eval_storm_classification-{metric_names}.csv')
        eval_df.to_csv(f)

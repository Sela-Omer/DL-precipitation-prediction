from typing import Callable, Dict

import torch

from src.helper.param_helper import convert_param_to_type
from src.script.fit_naive_module_script import FitNaiveModuleScript
from src.script.fit_residual_nn_script import FitResidualNNScript
from src.script.fit_resnet_rnn_predictor_script import FitResNetRNN_PredictorScript
from src.script.fit_resnet_script import FitResNetScript
from src.script.fit_simple_nn_polynomial_predictor_script import FitSimpleNNPolynomialPredictorScript
from src.script.fit_simple_nn_rnn_predictor_script import FitSimpleNN_RNNPredictorScript
from src.script.fit_simple_nn_script import FitSimpleNNScript
from src.script.fit_simple_nn_skip_connection_script import FitSimpleNNSkipConnectionScript
from src.script.fit_skip_connection_cnn_script import FitSkipConnectionCNNScript
from src.service.service import Service


class ServiceFit(Service):
    """
    This class is responsible for initializing a fit service.

    """

    def __init__(self, config):
        """
        Initializes a new instance of the ServiceFit class.

        Args:
            config (dict): A dictionary containing the configuration settings for the service.
                It should include 'FIT' key with 'TORCH_PRECISION' and 'MODEL_HYPERPARAMS' subkeys.

        Returns:
            None
        """
        # Call the parent class's __init__ method
        super(ServiceFit, self).__init__(config)

        # Set the float32 matmul precision based on the config
        torch.set_float32_matmul_precision(config['FIT']['TORCH_PRECISION'])

        # Initialize the model hyperparameters dictionary
        self.model_hyperparams = {}

        # Split the hyperparameters string in the config and add them to the dictionary
        hyperparam_lst = config['FIT']['MODEL_HYPERPARAMS'].split(',')
        for hyperparam in hyperparam_lst:
            # Skip the hyperparameter if it doesn't have an '=' sign
            if '=' not in hyperparam:
                continue
            # Split the key-value pair and add it to the dictionary
            key, value = hyperparam.split('=')
            self.model_hyperparams[key] = convert_param_to_type(value)

    @property
    def scripts(self) -> Dict[str, Callable]:
        """
        Returns a dictionary of scripts.

        The dictionary contains a single key-value pair.
        The key is the name of the ARCH , and the value is an instance of the Fit Script class.
        """
        # Create a dictionary with the ARCH name and its corresponding instance
        script_dict = {
            'SIMPLE_NN': FitSimpleNNScript(self),
            'RESIDUAL_NN': FitResidualNNScript(self),
            'NAIVE_MODULE': FitNaiveModuleScript(self),
            'RESNET': FitResNetScript(self),
            'RESNET_RNN_PREDICTOR': FitResNetRNN_PredictorScript(self),
            'SIMPLE_NN_SKIP_CONNECTION': FitSimpleNNSkipConnectionScript(self),
            'SIMPLE_NN_POLYNOMIAL_PREDICTOR': FitSimpleNNPolynomialPredictorScript(self),
            'SIMPLE_NN_RNN_PREDICTOR': FitSimpleNN_RNNPredictorScript(self),
            'CNN_SKIP_CONNECTION': FitSkipConnectionCNNScript(self),
        }

        return script_dict

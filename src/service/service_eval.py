from typing import Callable, Dict

from src.script.eval_naive_module_script import EvalNaiveModuleScript
from src.script.eval_residual_nn_script import EvalResidualNNScript
from src.script.eval_resnet_script import EvalResNetScript
from src.script.eval_simple_nn_polynomial_predictor_script import EvalSimpleNNPolynomialPredictorScript
from src.script.eval_simple_nn_rnn_predictor_script import EvalSimpleNN_RNNPredictorScript
from src.script.eval_simple_nn_script import EvalSimpleNNScript
from src.script.eval_simple_nn_skip_connection_script import EvalSimpleNNSkipConnectionScript
from src.service.service import Service


class ServiceEval(Service):

    def __init__(self, config):
        """
        Initializes a new instance of the ServiceEval class.

        Args:
            config (dict): A dictionary containing the configuration settings for the service.

        Returns:
            None
        """
        super(ServiceEval, self).__init__(config)

    @property
    def scripts(self) -> Dict[str, Callable]:
        """
        Returns a dictionary of scripts.

        The dictionary contains a single key-value pair.
        The key is the name of the ARCH , and the value is an instance of the Fit Script class.
        """
        # Create a dictionary with the ARCH name and its corresponding instance
        script_dict = {
            'SIMPLE_NN': EvalSimpleNNScript(self),
            'RESIDUAL_NN': EvalResidualNNScript(self),
            'NAIVE_MODULE': EvalNaiveModuleScript(self),
            'RESNET': EvalResNetScript(self),
            'SIMPLE_NN_SKIP_CONNECTION': EvalSimpleNNSkipConnectionScript(self),
            'SIMPLE_NN_POLYNOMIAL_PREDICTOR': EvalSimpleNNPolynomialPredictorScript(self),
            'SIMPLE_NN_RNN_PREDICTOR': EvalSimpleNN_RNNPredictorScript(self),
        }

        return script_dict

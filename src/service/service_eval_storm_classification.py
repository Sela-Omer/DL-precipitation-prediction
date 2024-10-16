from typing import Callable, Dict

from src.script.eval_naive_module_script import EvalNaiveModuleScript
from src.script.eval_residual_nn_script import EvalResidualNNScript
from src.script.eval_resnet_rnn_predictor_script import EvalResNetRNN_PredictorScript
from src.script.eval_resnet_script import EvalResNetScript
from src.script.eval_simple_nn_polynomial_predictor_script import EvalSimpleNNPolynomialPredictorScript
from src.script.eval_simple_nn_rnn_predictor_script import EvalSimpleNN_RNNPredictorScript
from src.script.eval_simple_nn_script import EvalSimpleNNScript
from src.script.eval_simple_nn_skip_connection_script import EvalSimpleNNSkipConnectionScript
from src.script.eval_skip_connection_cnn_script import EvalSkipConnectionCNNScript
from src.script.eval_storm_classification_skip_connection_cnn_script import \
    EvalStormClassificationSkipConnectionCNNScript
from src.service.service import Service


class ServiceEvalStormClassification(Service):
    @property
    def scripts(self) -> Dict[str, Callable]:
        """
        Returns a dictionary of scripts.

        The dictionary contains a single key-value pair.
        The key is the name of the ARCH , and the value is an instance of the Fit Script class.
        """
        # Create a dictionary with the ARCH name and its corresponding instance
        script_dict = {
            'CNN_SKIP_CONNECTION': EvalStormClassificationSkipConnectionCNNScript(self),
        }

        return script_dict

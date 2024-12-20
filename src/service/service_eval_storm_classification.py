from typing import Callable, Dict

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

from typing import Callable, Dict

from src.service.service import Service


class ServiceDataAnalysis(Service):

    @property
    def scripts(self) -> Dict[str, Callable]:
        """
        Returns a dictionary of scripts.

        The dictionary contains a single key-value pair.
        The key is the name of the ARCH , and the value is an instance of the Script class.
        """
        # Create a dictionary with the ARCH name and its corresponding instance
        script_dict = {
            'REALTIME_PARAMETER_TO_INTENSITY_NN': None,
        }

        return script_dict

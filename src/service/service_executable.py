from typing import Callable, Dict

from src.script.storm_classification_script import StormClassificationScript
from src.service.service import Service


class ServiceExecutable(Service):
    def __init__(self, *arg, **kwargs):
        """
        Initializes a new instance of the ServiceDataAnalysis class.

        Args:
            config (dict): A dictionary containing the configuration settings for the service.

        Returns:
            None
        """
        super().__init__(*arg, **kwargs)
        self.tfms = [tfm for tfm in self.tfms if tfm.tfm_name != 'normalize']

    @property
    def scripts(self) -> Dict[str, Callable]:
        """
        Returns a dictionary of scripts.

        The dictionary contains a single key-value pair.
        The key is the name of the ARCH , and the value is an instance of the Script class.
        """
        # Create a dictionary with the ARCH name and its corresponding instance
        script_dict = {
            'STORM_CLASSIFICATION': StormClassificationScript(self),
        }

        return script_dict

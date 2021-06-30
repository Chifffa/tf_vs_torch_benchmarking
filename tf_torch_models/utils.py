from abc import abstractmethod
from typing import Tuple, Optional

from nvsmi import get_gpu_processes

AVAILABLE_MODELS = [
    'efficientnetb0', 'efficientnetb1', 'efficientnetb2', 'efficientnetb3', 'efficientnetb4', 'efficientnetb5',
    'efficientnetb6', 'efficientnetb7', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'vgg16', 'vgg19',
    'densenet121', 'densenet169', 'densenet201', 'inception_v3', 'mobilenetv2'
]


class BaseModel:
    def __init__(self, random_seed: int = 0, steps_number: int = 100, gpu_number: Optional[int] = 0) -> None:
        """
        Base class for TensorFlow and PyTorch models for benchmark testing.

        :param random_seed: random generators seed.
        :param steps_number: number of steps for testing.
        :param gpu_number: number of GPU to use or None to use CPU.
        """
        self.random_seed = random_seed
        self.steps_number = steps_number + 5
        self.gpu_number = gpu_number

    @abstractmethod
    def train(self, model_name: str, batch_size: int, input_shape: Tuple[int, int, int]) -> Tuple[str, str]:
        pass

    @abstractmethod
    def inference(self, model_name: str, batch_size: int, input_shape: Tuple[int, int, int]) -> Tuple[str, str]:
        pass

    @abstractmethod
    def get_model(self, model_name: str, input_shape: Tuple[int, int, int]):
        pass

    @staticmethod
    def check_model_name(model_name: str) -> None:
        """
        Check if given model is available for benchmark testing.

        :param model_name: model name.
        """
        if model_name not in AVAILABLE_MODELS:
            msg = f'Wrong model name "{model_name}". Available models are: {AVAILABLE_MODELS}.'
            raise RuntimeError(msg)

    def get_gpu_memory(self) -> str:
        """
        Using nvidia-smi python wrapper to get used GPU memory value.

        :return: used GPU memory or "-" if using CPU.
        """
        if self.gpu_number is None:
            return '-'
        processes = [str(p.used_memory) for p in list(get_gpu_processes()) if p.gpu_id == str(self.gpu_number)]
        return ','.join(processes)

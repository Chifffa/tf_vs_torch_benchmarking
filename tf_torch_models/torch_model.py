import time
from typing import Tuple, Optional

import torch
import numpy as np
from tqdm import trange
from torchvision.models import (
    resnet18, resnet34, resnet50, resnet101, resnet152, vgg16, vgg19, densenet121, densenet169, densenet201,
    inception_v3, mobilenet_v2
)
from efficientnet_pytorch import EfficientNet

from .utils import BaseModel


def init_random_generators(seed: int) -> None:
    """
    Initialize all random generators with given seed.

    :param seed: int >= 0 to initialize generators.
    """
    np.random.seed(seed)
    torch.random.manual_seed(seed)


class PyTorchModel(BaseModel):
    def __init__(self, random_seed: int = 0, steps_number: int = 100, gpu_number: Optional[int] = 0) -> None:
        """
        PyTorch model class for benchmark testing.

        :param random_seed: random generators seed.
        :param steps_number: number of steps for testing.
        :param gpu_number: number of GPU to use or None to use CPU.
        """
        super().__init__(random_seed, steps_number, gpu_number)
        if self.gpu_number is None:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(f'cuda:{gpu_number}')

    def train(self, model_name: str, batch_size: int, input_shape: Tuple[int, int, int]) -> Tuple[str, str]:
        """
        Measure training step time and GPU memory using provided model with given parameters.

        :param model_name: one of the available architectures (see tf_torch_models.utils.AVAILABLE_MODELS).
        :param batch_size: batch size.
        :param input_shape: input image shape (height, width, channels).
        :return: measured mean train step time and GPU memory.
        """
        torch.cuda.empty_cache()

        init_random_generators(self.random_seed)
        image_data = np.random.randint(0, 256,
                                       (self.steps_number, batch_size, input_shape[2], input_shape[0], input_shape[1]))
        labels_data = np.random.randint(0, 1000, (self.steps_number, batch_size))

        model = self.get_model(model_name, input_shape)
        model.train()

        loss_func = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        image_data = torch.tensor(image_data, dtype=torch.float32)
        labels_data = torch.tensor(labels_data, dtype=torch.long)

        desc_str = f'Train: Architecture: "{model_name}". Image size: "{input_shape}". Batch size: "{batch_size}".'
        all_times = []
        for step_idx in trange(self.steps_number, leave=False, desc=desc_str):

            start_time = time.time()
            inputs = image_data[step_idx, :, :, :, :].to(self.device)
            labels = labels_data[step_idx, :].to(self.device)
            optimizer.zero_grad()
            # Forward.
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            # Backward.
            loss.backward()
            optimizer.step()
            finish_time = time.time()

            all_times.append(finish_time - start_time)

        mean_time = float(np.mean(all_times[5:]))
        memory = self.get_gpu_memory()
        return str(mean_time), memory

    def inference(self, model_name: str, batch_size: int, input_shape: Tuple[int, int, int]) -> Tuple[str, str]:
        """
        Measure inference step time and GPU memory using provided model with given parameters.

        :param model_name: one of the available architectures (see tf_torch_models.utils.AVAILABLE_MODELS).
        :param batch_size: batch size.
        :param input_shape: input image shape (height, width, channels).
        :return: measured mean inference step time and GPU memory.
        """
        torch.cuda.empty_cache()

        init_random_generators(self.random_seed)
        image_data = np.random.randint(0, 256,
                                       (self.steps_number, batch_size, input_shape[2], input_shape[0], input_shape[1]))

        model = self.get_model(model_name, input_shape)
        model.eval()

        image_data = torch.tensor(image_data, dtype=torch.float32)

        desc_str = f'Inference: Architecture: "{model_name}". Image size: "{input_shape}". Batch size: "{batch_size}".'
        all_times = []
        with torch.no_grad():
            for step_idx in trange(self.steps_number, leave=False, desc=desc_str):
                start_time = time.time()
                inputs = image_data[step_idx, :, :, :, :].to(self.device)
                model(inputs)
                finish_time = time.time()
                all_times.append(finish_time - start_time)

        mean_time = float(np.mean(all_times[5:]))
        memory = self.get_gpu_memory()
        return str(mean_time), memory

    def get_model(self, model_name: str, input_shape: Tuple[int, int, int]) -> torch.nn.Module:
        """
        Creating model object with given architecture name.

        :param model_name: one of the available architectures (see tf_torch_models.utils.AVAILABLE_MODELS).
        :param input_shape: input image shape (height, width, channels).
        :return: created model.
        """
        self.check_model_name(model_name)

        models_dict = {
            'resnet18': resnet18,
            'resnet34': resnet34,
            'resnet50': resnet50,
            'resnet101': resnet101,
            'resnet152': resnet152,
            'vgg16': vgg16,
            'vgg19': vgg19,
            'densenet121': densenet121,
            'densenet169': densenet169,
            'densenet201': densenet201,
            'inception_v3': inception_v3,
            'mobilenetv2': mobilenet_v2
        }

        if model_name.startswith('efficientnet'):
            model = EfficientNet.from_name('efficientnet-' + model_name.replace('efficientnet', ''))
        else:
            model = models_dict[model_name](pretrained=False)

        model = model.to(self.device)
        return model

import time
from typing import Tuple, Optional

import keras
import numpy as np
import tensorflow as tf
from tqdm import trange
from classification_models.keras import Classifiers
from efficientnet.keras import (
    EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6,
    EfficientNetB7
)

from .utils import BaseModel

# Disable TensorFlow logging.
tf.compat.v1.logging.set_verbosity('CRITICAL')


def session_config(gpu_number: Optional[int] = None) -> None:
    """
    Configure keras session with provided GPU or CPU device.

    :param gpu_number: GPU index or None to use CPU.
    """
    if gpu_number is not None:
        tf_sess_config = tf.ConfigProto(allow_soft_placement=True)
        tf_sess_config.gpu_options.allow_growth = True
        tf_sess_config.gpu_options.visible_device_list = str(gpu_number)
    else:
        tf_sess_config = tf.ConfigProto(allow_soft_placement=True, device_count={'GPU': 0})
    keras.backend.set_session(tf.Session(config=tf_sess_config))


def init_random_generators(seed: int) -> None:
    """
    Initialize all random generators with given seed.

    :param seed: int >= 0 to initialize generators.
    """
    np.random.seed(seed)
    tf.set_random_seed(seed)


class TensorFlow1Model(BaseModel):
    def __init__(self, random_seed: int = 0, steps_number: int = 100, gpu_number: Optional[int] = 0) -> None:
        """
        TensorFlow 1.x model class for benchmark testing.

        :param random_seed: random generators seed.
        :param steps_number: number of steps for testing.
        :param gpu_number: number of GPU to use or None to use CPU.
        """
        super().__init__(random_seed, steps_number, gpu_number)
        session_config(gpu_number)

    def train(self, model_name: str, batch_size: int, input_shape: Tuple[int, int, int]) -> Tuple[str, str]:
        """
        Measure training step time and GPU memory using provided model with given parameters.

        :param model_name: one of the available architectures (see tf_torch_models.utils.AVAILABLE_MODELS).
        :param batch_size: batch size.
        :param input_shape: input image shape (height, width, channels).
        :return: measured mean train step time and GPU memory.
        """
        init_random_generators(self.random_seed)
        image_data = np.random.randint(0, 256,
                                       (self.steps_number, batch_size, input_shape[0], input_shape[1], input_shape[2]))
        labels_data = np.random.randint(0, 2, (self.steps_number, batch_size, 1000))
        image_data = image_data.astype(np.float32)
        labels_data = labels_data.astype(np.float32)

        model = self.get_model(model_name, input_shape)

        desc_str = f'Train: Architecture: "{model_name}". Image shape: "{input_shape}". Batch size: "{batch_size}".'
        all_times = []
        for step_idx in trange(self.steps_number, leave=False, desc=desc_str):
            start_time = time.time()
            model.train_on_batch(image_data[step_idx, :, :, :, :], labels_data[step_idx, :, :])
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
        # Deactivate training phase.
        keras.backend.set_learning_phase(0)

        init_random_generators(self.random_seed)
        image_data = np.random.randint(0, 256,
                                       (self.steps_number, batch_size, input_shape[0], input_shape[1], input_shape[2]))
        image_data = image_data.astype(np.float32)

        model = self.get_model(model_name, input_shape)

        desc_str = f'Inference: Architecture: "{model_name}". Image shape: "{input_shape}". Batch size: "{batch_size}".'
        all_times = []
        for step_idx in trange(self.steps_number, leave=False, desc=desc_str):
            start_time = time.time()
            model.predict_on_batch(image_data[step_idx, :, :, :, :])
            finish_time = time.time()
            all_times.append(finish_time - start_time)
        mean_time = float(np.mean(all_times[5:]))
        memory = self.get_gpu_memory()
        return str(mean_time), memory

    def get_model(self, model_name: str, input_shape: Tuple[int, int, int]) -> keras.models.Model:
        """
        Creating keras.models.Model object with given architecture name.

        :param model_name: one of the available architectures (see tf_torch_models.utils.AVAILABLE_MODELS).
        :param input_shape: input image shape (height, width, channels).
        :return: created and compiled keras.models.Model object.
        """
        self.check_model_name(model_name)

        if model_name.lower() == 'efficientnetb0':
            model = EfficientNetB0(weights=None, input_shape=input_shape)
        elif model_name.lower() == 'efficientnetb1':
            model = EfficientNetB1(weights=None, input_shape=input_shape)
        elif model_name.lower() == 'efficientnetb2':
            model = EfficientNetB2(weights=None, input_shape=input_shape)
        elif model_name.lower() == 'efficientnetb3':
            model = EfficientNetB3(weights=None, input_shape=input_shape)
        elif model_name.lower() == 'efficientnetb4':
            model = EfficientNetB4(weights=None, input_shape=input_shape)
        elif model_name.lower() == 'efficientnetb5':
            model = EfficientNetB5(weights=None, input_shape=input_shape)
        elif model_name.lower() == 'efficientnetb6':
            model = EfficientNetB6(weights=None, input_shape=input_shape)
        elif model_name.lower() == 'efficientnetb7':
            model = EfficientNetB7(weights=None, input_shape=input_shape)
        else:
            model_class, _ = Classifiers.get(model_name)
            model = model_class(weights=None, input_shape=input_shape)

        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(0.001))
        return model

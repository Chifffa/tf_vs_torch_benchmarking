import argparse
import os
import sys
import warnings
from typing import Optional, Tuple

import pandas as pd


def single_test(
        file_path: str, framework: str, architecture: str, input_size: int, batch_size: int, mode: str,
        gpu_number: Optional[int], random_seed: int, steps_number: int
) -> None:
    """
    Run single time and GPU memory measuring with provided parameters.

    :param file_path: path to log csv file with measured time and memory.
    :param framework: "tf1", "tf2" or "pytorch".
    :param architecture: one of the available architectures (see tf_torch_models.utils.AVAILABLE_MODELS).
    :param input_size: input image size.
    :param batch_size: batch size.
    :param mode: "train", "inference_gpu" or "inference_cpu".
    :param gpu_number: GPU index or None for using CPU.
    :param random_seed: random generators seed.
    :param steps_number: number of steps for testing.
    """
    # Disable TensorFlow logging.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if framework == 'tf1':
        # Disable "Using TensorFlow backend." printing after keras importing.
        stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        import keras
        sys.stderr.close()
        sys.stderr = stderr

        from tf_torch_models.tf_1x_model import TensorFlow1Model
        model_object = TensorFlow1Model(random_seed, steps_number, gpu_number)
    elif framework == 'tf2':
        from tf_torch_models.tf_2x_model import TensorFlow2Model
        model_object = TensorFlow2Model(random_seed, steps_number, gpu_number)
    elif framework == 'pytorch':
        # Disable UserWarning in PyTorch 1.9.0.
        warnings.filterwarnings('ignore', category=UserWarning)

        from tf_torch_models.torch_model import PyTorchModel
        model_object = PyTorchModel(random_seed, steps_number, gpu_number)
    else:
        raise ValueError(f'Wrong framework "{framework}". Must be "tf1", "tf2" or "pytorch".')

    memory, res_time, exc_info = '-', '-', ''
    input_shape = (input_size, input_size, 3)
    try:
        if mode == 'train':
            res_time, memory = model_object.train(architecture, batch_size, input_shape)
        elif mode == 'inference_gpu' or mode == 'inference_cpu':
            res_time, memory = model_object.inference(architecture, batch_size, input_shape)
        else:
            raise ValueError(f'Wrong mode "{mode}". Must be "train", "inference_gpu" or "inference_cpu".')
    except Exception as e:
        exc_info = f'Exception "{e.__class__.__name__}". Text: "{e}".'
        exit(1)
    finally:
        update_file(
            file_path, framework, architecture, input_shape, batch_size, mode, gpu_number, res_time, memory, exc_info
        )


def update_file(file_path: str, framework: str, architecture: str, input_shape: Tuple[int, int, int], batch_size: int,
                mode: str, gpu_number: Optional[int], time: str, memory: str, exc_info: str = '') -> None:
    """
    Updating (or creating new if not exists) log csv file with measured time and memory and other parameters.
    Creates directory if not exists. Printing logs to console.

    :param file_path: path to log csv file with measured time and memory.
    :param framework: "tf1", "tf2" or "pytorch".
    :param architecture: one of the available architectures (see tf_torch_models.utils.AVAILABLE_MODELS).
    :param input_shape: input image shape (height, width, channels).
    :param batch_size: batch size.
    :param mode: "train", "inference_gpu" or "inference_cpu".
    :param gpu_number: GPU index or None for using CPU.
    :param time: measured time.
    :param memory: measured GPU memory ("-" if CPU was used).
    :param exc_info: any exception info for printing.
    """
    if os.path.dirname(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.exists(file_path):
        df = pd.DataFrame(
            columns=['Framework', 'Architecture', 'Input size', 'Batch size', 'Mode', 'GPU index', 'Time', 'Memory']
        )
    else:
        df = pd.read_csv(file_path, sep=';', index_col=0)
    update_list = [framework, architecture, str(input_shape), batch_size, mode, gpu_number, time, memory]
    df.loc[len(df)] = update_list
    df.to_csv(file_path, sep=';')

    cell_len = 20
    mode = mode.capitalize().replace('_', ' ')
    input_shape = str(input_shape)
    batch_size = str(batch_size)
    if time != '-':
        time = str(round(float(time), 6))

    msg = f'{mode + " "*(cell_len - len(mode))}{architecture + " "*(cell_len - len(architecture))}'
    msg += f'{input_shape + " "*(cell_len - len(input_shape))}{batch_size + " "*(cell_len - len(batch_size))}'
    msg += f'{time + " "*(cell_len - len(time))}{memory + " "*(cell_len - len(memory))}' + exc_info
    print(msg)


def parse_args() -> argparse.Namespace:
    """
    Parsing command line arguments with argparse.
    """
    parser = argparse.ArgumentParser(
        'Script for running single measurement of time and GPU memory using provided parameters.'
    )
    parser.add_argument('--file', type=str, help='Path to log csv file with measured time and memory.')
    parser.add_argument('--framework',  type=str, help='Framework: "tf1", "tf2" or "pytorch".')
    parser.add_argument('-a', '--architecture', type=str, help='One of the available architectures.')
    parser.add_argument('-i', '--input_size', type=int, help='Input image size.')
    parser.add_argument('-b', '--batch_size', type=int, help='Batch size.')
    parser.add_argument('-m', '--mode', type=str, help='Work mode: "train", "inference_gpu" or "inference_cpu".')
    parser.add_argument('--gpu', type=int, help='GPU index. Pass "-1" to use CPU.')
    parser.add_argument('--seed', type=int, help='Random generators seed.')
    parser.add_argument('--steps', type=int, help='Number of steps for testing.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.gpu == -1:
        args.gpu = None

    single_test(
        file_path=args.file,
        framework=args.framework,
        architecture=args.architecture,
        input_size=args.input_size,
        batch_size=args.batch_size,
        mode=args.mode,
        gpu_number=args.gpu,
        random_seed=args.seed,
        steps_number=args.steps
    )

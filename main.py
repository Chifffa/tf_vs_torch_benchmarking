import argparse
import os

from config import (
    TRAIN_LOGS_FILE_PATH, INFERENCE_GPU_LOGS_FILE_PATH, INFERENCE_CPU_LOGS_FILE_PATH, GPU_NUMBER, MODES, ARCHITECTURES,
    INPUT_SIZES, BATCH_SIZES, RANDOM_SEED, STEPS_NUMBER
)
from run_single_test import update_file


def run_process(
        file_path: str, framework: str, architecture: str, input_size: int, batch_size: int, mode: str,
        gpu_number: int, random_seed: int, steps_number: int
) -> int:
    """
    Use os.system() to run single time and GPU memory measuring with provided parameters as separate subprocess.

    :param file_path: path to log csv file with measured time and memory.
    :param framework: "tf1", "tf2" or "pytorch".
    :param architecture: one of the available architectures (see tf_torch_models.utils.AVAILABLE_MODELS).
    :param input_size: input image size.
    :param batch_size: batch size.
    :param mode: "train", "inference_gpu" or "inference_cpu".
    :param gpu_number: GPU index or -1 for using CPU.
    :param random_seed: random generators seed.
    :param steps_number: number of steps for testing.
    :return exit code value.
    """
    command = f'python run_single_test.py --file {file_path} --framework {framework} --architecture {architecture} '
    command += f'--input_size {input_size} --batch_size {batch_size} --mode {mode} --gpu {gpu_number} '
    command += f'--seed {random_seed} --steps {steps_number}'
    exit_code = os.system(command)
    if exit_code == 2:
        raise KeyboardInterrupt
    return exit_code


def parse_args() -> argparse.Namespace:
    """
    Parsing command line arguments with argparse.
    """
    parser = argparse.ArgumentParser(
        'Script for running measurements of time and GPU memory using provided parameters on single framework.'
    )
    parser.add_argument(
        'framework', type=str, help='Framework: "tf1", "tf2" or "pytorch".', choices=['tf1', 'tf2', 'pytorch']
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    mode_parameters = {
        'train': {'file': TRAIN_LOGS_FILE_PATH, 'gpu': GPU_NUMBER},
        'inference_gpu': {'file': INFERENCE_GPU_LOGS_FILE_PATH, 'gpu': GPU_NUMBER},
        'inference_cpu': {'file': INFERENCE_CPU_LOGS_FILE_PATH, 'gpu': -1}
    }

    cell_len = 20
    msg = 'Mode' + ' '*(cell_len - len('Mode')) + 'Architecture' + ' '*(cell_len - len('Architecture'))
    msg += 'Input shape' + ' '*(cell_len - len('Input shape')) + 'Batch size' + ' '*(cell_len - len('Batch size'))
    msg += 'Time' + ' '*(cell_len - len('Time')) + 'Memory' + ' '*(cell_len - len('Memory'))
    print(msg)

    for mode_name in MODES:
        for arc in ARCHITECTURES:
            for inp_s in INPUT_SIZES:
                break_cycle = False
                for bs in BATCH_SIZES:
                    if not break_cycle:
                        code = run_process(
                            file_path=mode_parameters[mode_name]['file'],
                            framework=args.framework,
                            architecture=arc,
                            input_size=inp_s,
                            batch_size=bs,
                            mode=mode_name,
                            gpu_number=mode_parameters[mode_name]['gpu'],
                            random_seed=RANDOM_SEED,
                            steps_number=STEPS_NUMBER
                        )
                        # Don't try greater batches if any exception occurs.
                        if code != 0:
                            break_cycle = True
                    else:
                        update_file(
                            file_path=mode_parameters[mode_name]['file'],
                            framework=args.framework,
                            architecture=arc,
                            input_shape=(inp_s, inp_s, 3),
                            batch_size=bs,
                            mode=mode_name,
                            gpu_number=mode_parameters[mode_name]['gpu'],
                            time='-',
                            memory='-',
                            exc_info='Not checked.'
                        )

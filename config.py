import os

RANDOM_SEED = 0
STEPS_NUMBER = 100

BENCHMARK_LOGS_PATH = 'benchmark_logs'
PLOTS_SAVE_PATH = 'figures'
TRAIN_LOGS_FILE_PATH = os.path.join(BENCHMARK_LOGS_PATH, 'train.csv')
INFERENCE_GPU_LOGS_FILE_PATH = os.path.join(BENCHMARK_LOGS_PATH, 'inference_gpu.csv')
INFERENCE_CPU_LOGS_FILE_PATH = os.path.join(BENCHMARK_LOGS_PATH, 'inference_cpu.csv')
GPU_NUMBER = 0

ARCHITECTURES = ['resnet18', 'resnet50', 'resnet152', 'vgg16', 'mobilenetv2', 'efficientnetb4', 'efficientnetb7']
INPUT_SIZES = [128, 224, 320]
BATCH_SIZES = [1, 8, 16, 32]

MODES = ['train', 'inference_gpu', 'inference_cpu']

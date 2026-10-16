import os
import argparse

import torch

import tool
from model import RNN, CVNN, CNP
from long_term_test import Long_Term_Forecast

# ============================================================
# GPU acceleration
# ============================================================

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def parse_bool(value):
    if value.lower() in ('true', '1', 'yes'):
        return True
    if value.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError('Expected true or false')


parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--max_epoch', type=int, default=30, help='epoch to run')
parser.add_argument('--i_x', type=int, default=1024, help='i_x')
parser.add_argument('--window_size', type=int, default=10, help='window_size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--min_lr', type=float, default=0.0001, help='min learning rate')
parser.add_argument('--decay_epoch', type=int, default=10, help='decay epoch')
parser.add_argument('--test_rounds', type=int, default=100, help='test rounds')
parser.add_argument('--use_gpu', type=bool, default=True, help='use GPU')
parser.add_argument('--gpu', type=int, default=0, help='GPU')
parser.add_argument('--use_multi_gpu', type=int, default=0, help='multi GPU')
parser.add_argument('--filename', default='Data\\ML_for_opticsEOcomb_space_25GHz_Vratio_8.9_PMHNLF5_20m_Time_field.mat', help='data')
parser.add_argument('--data_file', default='Figure\\Data\\Real_world\\CNP', help='CNP')
parser.add_argument('--tem_file', default='outside/Temporary Files_40m_my', help='Temporary Files')
parser.add_argument('--train_evo', type=float, default=2688, help='train samples')
parser.add_argument('--test_evo', type=float, default=512, help='test samples')
parser.add_argument('--steps', type=float, default=101, help='steps')
parser.add_argument('--model_file', default='model_saved.pt', help='model filename')
parser.add_argument('--use_train', type=parse_bool, default=True, help='use train')
parser.add_argument('--single_test', type=parse_bool, default=False, help='run selected test samples through the complete multi-step inference rollout without saving prediction files')
parser.add_argument('--profile', type=parse_bool, nargs='?', const=True, default=False, help='measure inference time and resource usage (default: False)')
parser.add_argument('--single_test_samples', type=int, default=10000, choices=[1, 10, 100, 1000, 10000], help='number of test samples used when single_test=True')

args = parser.parse_args()
if args.single_test:
    args.batch_size = args.single_test_samples
    args.use_train = False
args.use_gpu = bool(torch.cuda.is_available() and args.use_gpu)

directory = args.data_file
os.makedirs(directory, exist_ok=True)

log_file_path = os.path.join(directory, 'log(LSTM)')
log = open(log_file_path, 'w')

tool.log_string(log, str(args)[10:-1])


# ============================================================
# Environment
# ============================================================

tool.log_string(log, '================ Environment ================')
tool.log_string(log, 'PyTorch: %s' % torch.__version__)
tool.log_string(log, 'CUDA: %s' % torch.version.cuda)
tool.log_string(log, 'cuDNN: %s' % torch.backends.cudnn.version())
tool.log_string(log, 'cuDNN enabled: %s' % torch.backends.cudnn.enabled)
tool.log_string(log, 'cuDNN benchmark: %s' % torch.backends.cudnn.benchmark)
tool.log_string(log, 'cuDNN deterministic: %s' % torch.backends.cudnn.deterministic)

if torch.cuda.is_available():
    tool.log_string(log, 'GPU: %s' % torch.cuda.get_device_name(args.gpu))
    tool.log_string(log, 'GPU capability: %s' % str(torch.cuda.get_device_capability(args.gpu)))
    tool.log_string(log, 'TF32 matmul: %s' % torch.backends.cuda.matmul.allow_tf32)
    tool.log_string(log, 'TF32 cuDNN: %s' % torch.backends.cudnn.allow_tf32)
    tool.log_string(log, 'Float32 matmul precision: %s' % torch.get_float32_matmul_precision())

tool.log_string(log, 'PyTorch CPU threads: %d' % torch.get_num_threads())
tool.log_string(log, 'PyTorch interop threads: %d' % torch.get_num_interop_threads())
tool.log_string(log, '================================================')


# ============================================================
# Model
# ============================================================

tool.log_string(log, 'compiling model...')

model = CNP(args.window_size, args.i_x)

parameters = tool.count_parameters(model)

tool.log_string(log, 'trainable parameters: {:,}'.format(parameters))


if __name__ == '__main__':
    try:
        forecast = Long_Term_Forecast(model, 1, args, log)
        forecast.load_model()
        tool.run_forecast_test(forecast, args, log)
    finally:
        log.close()

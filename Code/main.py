import argparse
import os
import random
import numpy as np
import torch
import tool
from model import RNN, CVNN, CNP
from long_term import Long_Term_Forecast

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

# ========================== Args ========================== #
def parse_bool(value):
    if value.lower() in ('true', '1', 'yes'):
        return True
    if value.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError('Expected true or false')


parser = argparse.ArgumentParser()

# model define
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--max_epoch', type=int, default=30, help='epoch to run')
parser.add_argument('--i_x', type=int, default=1024, help='i_x')
parser.add_argument('--window_size', type=int, default=10, help='window_size')
parser.add_argument('--learning_rate', type=float, default=0.001,help='initial learning rate 0.0005')
parser.add_argument('--min_lr', type=float, default=0.00005, help='min learning rate 0.00001')
parser.add_argument('--decay_epoch', type=int, default=8, help='decay epoch')
parser.add_argument('--patience', type=int, default=5, help='patience')
parser.add_argument('--monitor_interval', type=float, default=1.0,help='hardware monitoring interval in seconds')
parser.add_argument('--profile', type=parse_bool, nargs='?', const=True, default=False, help='measure training/testing time and resource usage (default: False)')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', type=int, default=1, help='use multi gpu')

# basic config
parser.add_argument('--filename', default='Data\\Real_world\\Forth_EOcomb_space_25GHz_Vratio_8.9_PM1605_40m_Time_field.mat',help='data')
parser.add_argument('--data_file',default='outside/Temporary Files_40m_my',help='Temporary Files')
parser.add_argument('--train_evo', type=float, default=2688, help='(3200, 101, 1024)')
parser.add_argument('--test_evo', type=float, default=512, help='(3200, 101, 1024)')
parser.add_argument('--steps', type=float, default=101, help='(3200, 101, 1024)')
parser.add_argument('--model_file', default='model_saved.pt', help='model filename; saved in the experiment result directory')

parser.add_argument('--resume', default='', help='trusted local checkpoint for fine tuning')
parser.add_argument('--train_samples', type=int, default=0, help='0 uses all training windows')
parser.add_argument('--input_dropout', type=float, default=0.1)
parser.add_argument('--loss_mode', choices=['l2', 'physical_mse'], default='l2')
parser.add_argument('--weight_alpha', type=float, default=3.0)
parser.add_argument('--output_dir', default='')
parser.add_argument('--skip_test', action='store_true', help='run rollout separately with main_test.py')
parser.add_argument('--boundary_fraction', type=float, default=0.1, help='fraction of trajectories at each training-power boundary')
parser.add_argument('--boundary_boost', type=float, default=1.0, help='training window sampling weight for boundary trajectories')

args = parser.parse_args()
args.use_gpu = bool(torch.cuda.is_available() and args.use_gpu)

# ========================== Experiment Setting ========================== #
setting = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
    args.filename,
    args.batch_size,
    args.max_epoch,
    args.learning_rate,
    args.decay_epoch,
    args.use_gpu,
    args.gpu,
    args.use_multi_gpu,
    args.train_evo,
    args.test_evo
)

directory = args.output_dir or 'Figure/{}'.format(setting)
os.makedirs(directory, exist_ok=True)

args.model_file = os.path.join(directory, os.path.basename(args.model_file))

log_file_path = os.path.join(directory, 'log(LSTM)')
log = open(log_file_path, 'w')
tool.log_string(log, str(args)[10:-1])


# ========================== Load Model ========================== #
tool.log_string(log, 'compiling model...')

random.seed(2014)
torch.manual_seed(2014)
np.random.seed(2014)
model = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else CNP(args.window_size, args.i_x)
model.input_real_dropout.p = args.input_dropout
model.input_imag_dropout.p = args.input_dropout
model.output_clamp = not args.no_output_clamp

parameters = tool.count_parameters(model)
tool.log_string(log, 'trainable parameters: {:,}'.format(parameters))


# ========================== Main ========================== #
if __name__ == '__main__':
    try:
        fix_seed = 2014
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(fix_seed)
        forecast = Long_Term_Forecast(model, args, log)
        tool.run_forecast_training(forecast, args, log, directory)
    finally:
        log.close()

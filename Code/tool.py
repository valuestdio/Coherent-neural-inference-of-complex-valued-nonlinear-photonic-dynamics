import h5py
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import time
import threading
import psutil
import GPUtil

# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

def load_variable_from_hdf5(file_path, dataset_name):
    with h5py.File(file_path, 'r') as f:
        print(list(f.keys()))
        data = f[dataset_name][()]
        return data.astype(np.complex64) if isinstance(data, np.ndarray) else np.array(data, dtype=np.complex64)
    

def load_alldata(folder_path):
    # 定义变量名列表
    variable_names = ['i_x', 'X_train', 'X_test', 'Y_train', 'Y_test_0']

    # 初始化空列表以保存加载的变量
    loaded_variables = []

    # 逐个加载变量
    for name in variable_names:
        # 构造 hdf5 文件路径
        file_path = os.path.join(folder_path, f'{name}.h5')
        # 加载变量
        loaded_variables.append(load_variable_from_hdf5(file_path, name))

    return tuple(loaded_variables)

def load_test_data(folder_path, tem_folder_path, i):
    # 定义变量名列表
    # i_x 从 tem_file 读取，X_test_new 从 data_file 读取，Y_test 从 tem_file 读取
    paths_and_names = [
        (tem_folder_path, 'i_x'),
        (folder_path, f'X_test_new{i-1}'),
        (tem_folder_path, f'Y_test_{i-1}'),
    ]

    loaded_variables = []
    for base_path, name in paths_and_names:
        file_path = os.path.join(base_path, f'{name}.h5')
        loaded_variables.append(load_variable_from_hdf5(file_path, name))

    return tuple(loaded_variables)

def load_test_first(folder_path, tem_folder_path, i):
    # 定义变量名列表
    # i_x、X_test、Y_test_0 均从 tem_file 读取
    paths_and_names = [
        (tem_folder_path, 'i_x'),
        (tem_folder_path, 'X_test'),
        (tem_folder_path, 'Y_test_0'),
    ]

    loaded_variables = []
    for base_path, name in paths_and_names:
        file_path = os.path.join(base_path, f'{name}.h5')
        loaded_variables.append(load_variable_from_hdf5(file_path, name))

    return tuple(loaded_variables)


def load_train_data(folder_path, tem_folder_path, i):
    # 定义变量名列表
    # i_x 和 Y_train 从 tem_file 读取，X_train_new 从 data_file 读取
    paths_and_names = [
        (tem_folder_path, 'i_x'),
        (folder_path, f'X_train_new{i-1}'),
        (tem_folder_path, f'Y_train_{i-1}'),
    ]

    loaded_variables = []
    for base_path, name in paths_and_names:
        file_path = os.path.join(base_path, f'{name}.h5')
        loaded_variables.append(load_variable_from_hdf5(file_path, name))

    return tuple(loaded_variables)

def load_train_first(folder_path, tem_folder_path, i):
    # 定义变量名列表
    # i_x 和 Y_train_0 从 tem_file 读取，X_train_first 从 data_file 读取
    paths_and_names = [
        (tem_folder_path, 'i_x'),
        (tem_folder_path, 'X_train_first'),
        (tem_folder_path, 'Y_train_0'),
    ]

    loaded_variables = []
    for base_path, name in paths_and_names:
        file_path = os.path.join(base_path, f'{name}.h5')
        loaded_variables.append(load_variable_from_hdf5(file_path, name))

    return tuple(loaded_variables)

def metric(pred_real, pred_imag, target_complex):
    """
    Compute average L2 loss and average phase difference for complex prediction.

    Parameters:
    - pred_real: torch.Tensor, 实部预测
    - pred_imag: torch.Tensor, 虚部预测
    - target_complex: torch.Tensor, 复数目标 (dtype=torch.cfloat)

    Returns:
    - l2_loss_scaled: float, L2损失 (乘以10000后的值)
    - mean_phase_diff_deg: float, 平均相角差 (单位：度)
    """
    if isinstance(target_complex, np.ndarray):
        target_complex = torch.from_numpy(target_complex)
    
    target_real = target_complex.real
    target_imag = target_complex.imag
    diff_squared = (pred_real.float() - target_real.float()) ** 2 + (pred_imag.float() - target_imag.float()) ** 2
    l2_diff = torch.sqrt(diff_squared)
    l2_loss = torch.mean(l2_diff)

    # 相角差
    pred_complex = torch.complex(pred_real.float(), pred_imag.float())
    phase_diff = torch.angle(pred_complex) - torch.angle(target_complex)
    phase_diff = torch.remainder(phase_diff + torch.pi, 2 * torch.pi) - torch.pi  # wrap to [-π, π]
    mean_phase_diff_deg = torch.mean(torch.abs(phase_diff)) * 180 / torch.pi
    phase = mean_phase_diff_deg/180

    return l2_loss + phase, l2_loss*10000, mean_phase_diff_deg*10000

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_train_loss(train_total_loss, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_total_loss) + 1), train_total_loss, c='b', marker='s', label='Train')
    plt.legend(loc='best')
    plt.title('Train loss vs Validation loss')
    plt.savefig(file_path)

def log_resource_growth_multi_gpu(log, start_time, stats_before):
    """
    记录相对计算开始前的资源增长
    stats_before: dict, 存储计算开始前的状态
        {
            'cpu': cpu_percent_before,
            'cpu_peak': cpu_peak_before,
            'gpu_mem': [gpu_mem_used_before...],
            'gpu_util': [gpu_util_before...]
        }
    """
    # CPU 利用率
    cpu_now = psutil.cpu_percent(interval=0.1)
    cpu_peak_now = max(psutil.cpu_percent(interval=0.1) for _ in range(10))
    cpu_growth = cpu_now - stats_before['cpu']
    cpu_peak_growth = cpu_peak_now - stats_before['cpu_peak']
    
    # 系统内存
    mem_now = psutil.virtual_memory().used / (1024**2)  # MB
    mem_growth = mem_now - stats_before['mem']  # 保留原来的增长计算
    
    # GPU
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu_mem_now = [gpu.memoryUsed for gpu in gpus]
        gpu_util_now = [gpu.load*100 for gpu in gpus]

        # 计算增长
        gpu_mem_growth = [now - before for now, before in zip(gpu_mem_now, stats_before['gpu_mem'])]
        gpu_util_growth = [now - before for now, before in zip(gpu_util_now, stats_before['gpu_util'])]

        gpu_mem_growth_avg = np.mean(gpu_mem_growth)
        gpu_util_growth_avg = np.mean(gpu_util_growth)
        gpu_util_growth_peak = max(gpu_util_growth)
    else:
        gpu_mem_growth = []
        gpu_util_growth = []
        gpu_mem_growth_avg = gpu_util_growth_avg = gpu_util_growth_peak = 0

    # 总时间
    total_time_s = time.time() - start_time

    log_str = (
        f"Memory growth (MB): {mem_growth:.1f}, "
        f"CPU growth avg: {cpu_growth:.1f}%, CPU growth peak: {cpu_peak_growth:.1f}%, "
        f"GPU memory growth avg: {gpu_mem_growth_avg:.1f} MB, "
        f"GPU util growth avg: {gpu_util_growth_avg:.1f}%, GPU util growth peak: {gpu_util_growth_peak:.1f}%, "
        f"Total time: {total_time_s:.1f}s, "
        f"Individual GPU mem growth: {gpu_mem_growth}, GPU util growth: {gpu_util_growth}"
    )
    print(log_str)
    log.write(log_str + "\n")

def visual(preds, gts, name='./pic/test.pdf'):
    """
    Visualization of magnitude of complex predictions and ground truths
    After inverse normalization.
    """
    # 定义归一化时使用的 min 和 max
    min_real = -11.410256765891187
    max_real = 11.410256765891187
    min_imag = -10.608256128626813
    max_imag = 10.608256128626813
    
    # 反归一化函数
    def inverse_normalize(data_real_norm, data_imag_norm):
        real = data_real_norm * (max_real - min_real) + min_real
        imag = data_imag_norm * (max_real - min_real) + min_real
        return real + 1j * imag

    # 反归一化 preds 和 gts
    preds = inverse_normalize(preds.real, preds.imag)
    gts = inverse_normalize(gts.real, gts.imag)

    plt.figure()
    plt.plot(np.abs(gts), label='GT Magnitude', linewidth=2)
    if preds is not None:
        plt.plot(np.abs(preds), label='Pred Magnitude', linewidth=2)
    plt.title("Magnitude")
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    plt.close()

def torch_unwrap(p, dim=-1, discont=torch.pi):
    """
    Unwrap a tensor of phase angles (in radians) along the specified dimension.
    Similar to numpy.unwrap.
    """
    # 如果是复数，先取角度
    if torch.is_complex(p):
        p = torch.angle(p)

    dd = torch.diff(p, dim=dim)
    ddmod = torch.remainder(dd + discont, 2 * discont) - discont
    ddmod = torch.where((ddmod == -discont) & (dd > 0), discont, ddmod)

    ph_correct = ddmod - dd
    ph_correct = torch.cumsum(ph_correct, dim=dim)

    shape = list(p.shape)
    shape[dim] = 1
    ph_correct = torch.cat([torch.zeros(shape, dtype=ph_correct.dtype, device=ph_correct.device), ph_correct], dim=dim)

    return p + ph_correct

class ProfileClock:
    """Disable timer reads and profiling-only CUDA synchronization when not requested."""
    def __init__(self, enabled=False):
        self.enabled = enabled

    def now(self):
        return time.perf_counter() if self.enabled else 0.0

    def synchronize(self, device):
        if self.enabled and device is not None and device.type == 'cuda':
            torch.cuda.synchronize(device)


class ResourceMonitor:
    def __init__(self, interval=1.0):
        import psutil
        try:
            import GPUtil
            self.get_gpus = GPUtil.getGPUs
        except ImportError:
            self.get_gpus = lambda: []
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.running = False
        self.thread = None
        self.stop_event = threading.Event()
        self.error = None
        self.ram_samples = []
        self.cpu_samples = []
        self.gpu_util_samples = []
        self.gpu_mem_samples = []
        self.gpu_each_mem_samples = []
        self.gpu_each_util_samples = []

    def start(self):
        self.stop_event.clear()
        self.running = True
        self.process.cpu_percent(interval=None)
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def _monitor(self):
        try:
            self._sample_loop()
        except Exception as error:
            self.error = str(error)

    def _sample_loop(self):
        while self.running:
            self.ram_samples.append(self.process.memory_info().rss / 1024 ** 2)
            self.cpu_samples.append(self.process.cpu_percent(interval=None))

            gpus = self.get_gpus()

            if gpus:
                gpu_utils = [gpu.load * 100 for gpu in gpus]
                gpu_mems = [gpu.memoryUsed for gpu in gpus]

                self.gpu_each_util_samples.append(gpu_utils)
                self.gpu_each_mem_samples.append(gpu_mems)
                self.gpu_util_samples.append(np.mean(gpu_utils))
                self.gpu_mem_samples.append(np.sum(gpu_mems))

            self.stop_event.wait(self.interval)

    def stop(self):
        self.running = False
        self.stop_event.set()

        if self.thread is not None:
            self.thread.join()

    def summary(self):
        result = {}

        result['ram_start'] = self.ram_samples[0] if self.ram_samples else 0
        result['ram_end'] = self.ram_samples[-1] if self.ram_samples else 0
        result['ram_avg'] = np.mean(self.ram_samples) if self.ram_samples else 0
        result['ram_peak'] = np.max(self.ram_samples) if self.ram_samples else 0

        result['cpu_avg'] = np.mean(self.cpu_samples) if self.cpu_samples else 0
        result['cpu_peak'] = np.max(self.cpu_samples) if self.cpu_samples else 0

        result['gpu_util_avg'] = np.mean(self.gpu_util_samples) if self.gpu_util_samples else 0
        result['gpu_util_peak'] = np.max(self.gpu_util_samples) if self.gpu_util_samples else 0

        result['gpu_mem_avg'] = np.mean(self.gpu_mem_samples) if self.gpu_mem_samples else 0
        result['gpu_mem_peak'] = np.max(self.gpu_mem_samples) if self.gpu_mem_samples else 0

        if self.gpu_each_mem_samples:
            gpu_each_mem_arr = np.array(self.gpu_each_mem_samples)
            gpu_each_util_arr = np.array(self.gpu_each_util_samples)

            result['gpu_each_mem_peak'] = np.max(gpu_each_mem_arr, axis=0).tolist()
            result['gpu_each_mem_avg'] = np.mean(gpu_each_mem_arr, axis=0).tolist()
            result['gpu_each_util_peak'] = np.max(gpu_each_util_arr, axis=0).tolist()
            result['gpu_each_util_avg'] = np.mean(gpu_each_util_arr, axis=0).tolist()

        else:
            result['gpu_each_mem_peak'] = []
            result['gpu_each_mem_avg'] = []
            result['gpu_each_util_peak'] = []
            result['gpu_each_util_avg'] = []

        return result

    def log_summary(self, log, total_seconds):
        result = self.summary()
        if self.error:
            log_string(log, 'Resource monitoring stopped early: %s' % self.error)

        log_string(log, '================ Resource Usage ================')
        log_string(log, 'Pipeline runtime: %.2fs, %.4fmin' % (total_seconds, total_seconds / 60))
        log_string(log, 'Program RAM start: %.1f MB, end: %.1f MB, avg: %.2f GB, peak: %.1f MB' % (result['ram_start'], result['ram_end'], result['ram_avg']/1024, result['ram_peak']))
        log_string(log, 'Program CPU utilization avg: %.1f%%, peak: %.1f%%' % (result['cpu_avg'], result['cpu_peak']))
        log_string(log, 'GPU utilization avg: %.1f%%, peak: %.1f%%' % (result['gpu_util_avg'], result['gpu_util_peak']))
        log_string(log, 'GPU memory total avg: %.2f GB, peak: %.1f MB' % (result['gpu_mem_avg']/1024, result['gpu_mem_peak']))
        log_string(log, 'Individual GPU memory avg: %s' % result['gpu_each_mem_avg'])
        log_string(log, 'Individual GPU memory peak: %s' % result['gpu_each_mem_peak'])
        log_string(log, 'Individual GPU util avg: %s' % result['gpu_each_util_avg'])
        log_string(log, 'Individual GPU util peak: %s' % result['gpu_each_util_peak'])
        log_string(log, '================================================')

def log_forecast_timing(log, test_time, num_samples, timing, data_load_time, preprocess_time, metric_time):
    """Write per-call inference timing and throughput."""
    time_per_sample_ms = test_time / max(num_samples, 1) * 1000
    throughput = num_samples / max(test_time, 1e-12)
    log_string(log, 'inference time: %.6fs' % test_time)
    log_string(log, 'time per sample: %.6f ms/sample' % time_per_sample_ms)
    log_string(log, 'throughput: %.3f samples/s' % throughput)

    log_string(log, '---------------- Inference Breakdown ----------------')
    log_string(log, 'H2D transfer:      %.6fs (%.2f%%)' % (timing['h2d'], timing['h2d'] / max(test_time, 1e-12) * 100))
    log_string(log, 'Padding:           %.6fs (%.2f%%)' % (timing['padding'], timing['padding'] / max(test_time, 1e-12) * 100))
    log_string(log, 'Model forward:     %.6fs (%.2f%%)' % (timing['forward'], timing['forward'] / max(test_time, 1e-12) * 100))
    log_string(log, 'Output handling:   %.6fs (%.2f%%)' % (timing['output_handling'], timing['output_handling'] / max(test_time, 1e-12) * 100))
    log_string(log, 'D2H transfer:      %.6fs (%.2f%%)' % (timing['d2h'], timing['d2h'] / max(test_time, 1e-12) * 100))
    log_string(log, 'Other inference:   %.6fs (%.2f%%)' % (timing['other_inference'], timing['other_inference'] / max(test_time, 1e-12) * 100))
    log_string(log, '-----------------------------------------------------')

    log_string(log, 'Data loading:      %.6fs' % data_load_time)
    log_string(log, 'Preprocessing:     %.6fs' % preprocess_time)
    log_string(log, 'Metric:            %.6fs' % metric_time)



def run_forecast_test(forecast, args, log):
    """Run inference with optional timing and process/device resource reporting."""
    monitor = ResourceMonitor(interval=1.0) if args.profile else None
    if monitor is not None:
        monitor.start()
    clock = ProfileClock(args.profile)
    try:
        total_start = clock.now()

        if args.single_test:
            # ============================================================
            # Single sample, complete multi-step autoregressive rollout
            # ============================================================
            X_final, prediction, inference_time_all, rollout_total, data_load_time, timing_all = forecast.test_single_sample_rollout()

            total_seconds = clock.now() - total_start

            steps = args.test_rounds
            sample_count = X_final.shape[0]
            if args.profile:
                avg_inference_per_step_ms = inference_time_all / steps * 1000
                avg_rollout_per_step_ms = rollout_total / steps * 1000
                avg_inference_per_sample_ms = inference_time_all / sample_count * 1000
                avg_rollout_per_sample_ms = rollout_total / sample_count * 1000
                avg_inference_per_sample_step_ms = inference_time_all / (sample_count * steps) * 1000

                log_string(log, '================ Selected-Sample Full Rollout ===============')
                log_string(log, 'Rollout steps:                    %d' % steps)
                log_string(log, 'Batch size:                       %d' % args.batch_size)
                log_string(log, 'Test sample count:                %d' % sample_count)
                log_string(log, 'Model inference total:            %.6fs' % inference_time_all)
                log_string(log, 'Full rollout total:               %.6fs' % rollout_total)
                log_string(log, 'Average inference / step:         %.6f ms' % avg_inference_per_step_ms)
                log_string(log, 'Average rollout / step:           %.6f ms' % avg_rollout_per_step_ms)
                log_string(log, 'Average inference / sample:       %.6f ms' % avg_inference_per_sample_ms)
                log_string(log, 'Average rollout / sample:         %.6f ms' % avg_rollout_per_sample_ms)
                log_string(log, 'Average inference / sample-step:  %.6f ms' % avg_inference_per_sample_step_ms)
                log_string(log, 'Data loading:                     %.6fs' % data_load_time)
                log_string(log, '============================================================')

                log_string(log, '================ Inference Breakdown Total ================')
                log_string(log, 'H2D total:             %.6fs (%.2f%%)' % (timing_all['h2d'], timing_all['h2d'] / max(inference_time_all, 1e-12) * 100))
                log_string(log, 'Padding total:         %.6fs (%.2f%%)' % (timing_all['padding'], timing_all['padding'] / max(inference_time_all, 1e-12) * 100))
                log_string(log, 'Model forward total:   %.6fs (%.2f%%)' % (timing_all['forward'], timing_all['forward'] / max(inference_time_all, 1e-12) * 100))
                log_string(log, 'Output handling total: %.6fs (%.2f%%)' % (timing_all['output_handling'], timing_all['output_handling'] / max(inference_time_all, 1e-12) * 100))
                log_string(log, 'D2H total:             %.6fs (%.2f%%)' % (timing_all['d2h'], timing_all['d2h'] / max(inference_time_all, 1e-12) * 100))
                log_string(log, 'Other inference:       %.6fs (%.2f%%)' % (timing_all['other_inference'], timing_all['other_inference'] / max(inference_time_all, 1e-12) * 100))
                log_string(log, '===========================================================')

                log_string(log, '================ Rollout Breakdown ========================')
                log_string(log, 'Preprocessing total:      %.6fs' % timing_all['preprocess'])
                log_string(log, 'Complex conversion total: %.6fs' % timing_all['complex_conversion'])
                log_string(log, 'Concatenate total:        %.6fs' % timing_all['concatenate'])
                log_string(log, 'Full rollout total:       %.6fs' % rollout_total)
                log_string(log, '===========================================================')

                log_string(log, 'pipeline total time: %.2fs' % total_seconds)
                log_string(log, 'pipeline total time: %.4fmin' % (total_seconds / 60))
                monitor.stop()
                monitor.log_summary(log, total_seconds)

            print('%d test samples completed through all %d inference steps. No prediction files were generated.' % (sample_count, steps))

        else:
            # ============================================================
            # Original full-dataset inference
            # ============================================================
            inference_time_all = 0.0
            inference_samples_all = 0

            timing_all = {
                'h2d': 0.0,
                'padding': 0.0,
                'forward': 0.0,
                'output_handling': 0.0,
                'd2h': 0.0,
                'profile_total': 0.0,
                'other_inference': 0.0,
                'data_load': 0.0,
                'preprocess': 0.0,
                'metric': 0.0,
                'complex_conversion': 0.0,
                'concatenate': 0.0,
                'hdf5_save': 0.0
            }

            for i in range(1, args.test_rounds + 1):
                forecast.test_rounds = i
                round_start = clock.now()

                if i == 1:
                    X_real, X_imag, pred_real, pred_imag, Y_real, Y_imag, inference_time, timing = forecast.test_model_0()
                else:
                    X_real, X_imag, pred_real, pred_imag, Y_real, Y_imag, inference_time, timing = forecast.test_model()

                t0 = clock.now()
                X = X_real + 1j * X_imag
                prediction = pred_real.numpy() + 1j * pred_imag.numpy()
                complex_time = clock.now() - t0

                t0 = clock.now()
                n = min(X.shape[0], prediction.shape[0])
                X_new = np.concatenate([X[:n, 1:, :], prediction[:n]], axis=1)
                concat_time = clock.now() - t0

                t0 = clock.now()
                prefix_name = 'train' if args.use_train else 'test'
                arrays = [X_new, prediction]
                names = [f'X_{prefix_name}_new{i}', f'{prefix_name}_Pred_{i-1}']

                for name, data in zip(names, arrays):
                    with h5py.File(os.path.join(args.data_file, name + '.h5'), 'w') as hf:
                        hf.create_dataset(name, data=data)

                save_time = clock.now() - t0
                round_seconds = clock.now() - round_start

                if args.profile:
                    log_string(log, '================ Round %d Timing ================' % i)
                    log_string(log, 'Total round:        %.6fs' % round_seconds)
                    log_string(log, 'Inference:          %.6fs' % inference_time)
                    log_string(log, 'Data loading:       %.6fs' % timing['data_load'])
                    log_string(log, 'Preprocessing:      %.6fs' % timing['preprocess'])
                    log_string(log, 'Metric:             %.6fs' % timing['metric'])
                    log_string(log, 'Complex conversion: %.6fs' % complex_time)
                    log_string(log, 'Concatenate:        %.6fs' % concat_time)
                    log_string(log, 'HDF5 save:          %.6fs' % save_time)
                    log_string(log, '=================================================')

                inference_time_all += inference_time
                inference_samples_all += n

                for key in ['h2d', 'padding', 'forward', 'output_handling', 'd2h', 'profile_total', 'other_inference', 'data_load', 'preprocess', 'metric']:
                    timing_all[key] += timing[key]

                timing_all['complex_conversion'] += complex_time
                timing_all['concatenate'] += concat_time
                timing_all['hdf5_save'] += save_time

            total_seconds = clock.now() - total_start

            if args.profile:
                inference_time_per_sample_ms = inference_time_all / max(inference_samples_all, 1) * 1000
                inference_throughput = inference_samples_all / max(inference_time_all, 1e-12)

                log_string(log, '================ Inference Timing ================')
                log_string(log, 'model inference total: %.6fs' % inference_time_all)
                log_string(log, 'inference rounds: %d' % args.test_rounds)
                log_string(log, 'inference samples: %d' % inference_samples_all)
                log_string(log, 'average time per sample: %.6f ms/sample' % inference_time_per_sample_ms)
                log_string(log, 'average throughput: %.3f samples/s' % inference_throughput)
                log_string(log, '==================================================')

                log_string(log, '================ Inference Breakdown Total ================')
                log_string(log, 'H2D total:             %.6fs (%.2f%%)' % (timing_all['h2d'], timing_all['h2d'] / max(inference_time_all, 1e-12) * 100))
                log_string(log, 'Padding total:         %.6fs (%.2f%%)' % (timing_all['padding'], timing_all['padding'] / max(inference_time_all, 1e-12) * 100))
                log_string(log, 'Model forward total:   %.6fs (%.2f%%)' % (timing_all['forward'], timing_all['forward'] / max(inference_time_all, 1e-12) * 100))
                log_string(log, 'Output handling total: %.6fs (%.2f%%)' % (timing_all['output_handling'], timing_all['output_handling'] / max(inference_time_all, 1e-12) * 100))
                log_string(log, 'D2H total:             %.6fs (%.2f%%)' % (timing_all['d2h'], timing_all['d2h'] / max(inference_time_all, 1e-12) * 100))
                log_string(log, 'Other inference:       %.6fs (%.2f%%)' % (timing_all['other_inference'], timing_all['other_inference'] / max(inference_time_all, 1e-12) * 100))
                log_string(log, '===========================================================')

                known_pipeline = (
                    inference_time_all
                    + timing_all['data_load']
                    + timing_all['preprocess']
                    + timing_all['metric']
                    + timing_all['complex_conversion']
                    + timing_all['concatenate']
                    + timing_all['hdf5_save']
                )
                pipeline_other = total_seconds - known_pipeline

                log_string(log, '================ Pipeline Breakdown Total =================')
                log_string(log, 'Data loading total:       %.6fs' % timing_all['data_load'])
                log_string(log, 'Preprocessing total:      %.6fs' % timing_all['preprocess'])
                log_string(log, 'Metric total:             %.6fs' % timing_all['metric'])
                log_string(log, 'Complex conversion total: %.6fs' % timing_all['complex_conversion'])
                log_string(log, 'Concatenate total:        %.6fs' % timing_all['concatenate'])
                log_string(log, 'HDF5 save total:          %.6fs' % timing_all['hdf5_save'])
                log_string(log, 'Other pipeline overhead:  %.6fs' % pipeline_other)
                log_string(log, 'Full pipeline total:      %.6fs' % total_seconds)
                log_string(log, '===========================================================')

                log_string(log, 'pipeline total time: %.2fs' % total_seconds)
                log_string(log, 'pipeline total time: %.4fmin' % (total_seconds / 60))
                monitor.stop()
                monitor.log_summary(log, total_seconds)

            print('All predictions have been saved.')
    finally:
        if monitor is not None:
            monitor.stop()


def run_forecast_training(forecast, args, log, directory):
    """Run training and optional evaluation using the shared profiling utilities."""
    clock = ProfileClock(args.profile)
    monitor = ResourceMonitor(interval=args.monitor_interval) if args.profile else None
    total_start = clock.now()
    try:
        if monitor is not None:
            monitor.start()
        clock.synchronize(forecast.device)
        train_start = clock.now()
        loss_train = forecast.train_model()
        clock.synchronize(forecast.device)
        train_seconds = clock.now() - train_start
        if args.profile:
            log_string(log, 'training time: %.2fs, %.4fmin' % (train_seconds, train_seconds / 60))

        plot_start = clock.now()
        plot_train_loss(loss_train, os.path.join(directory, 'train_val_loss.png'))
        plot_seconds = clock.now() - plot_start
        if args.skip_test:
            print('Training and loss plotting completed. Results saved in: %s' % directory)
            return

        clock.synchronize(forecast.device)
        test_start = clock.now()
        trainPred_real, trainPred_imag, testPred_real, testPred_imag, Y_train_real, Y_train_imag, Y_test_real, Y_test_imag = forecast.test_model()
        clock.synchronize(forecast.device)
        test_seconds = clock.now() - test_start
        if args.profile:
            log_string(log, 'testing time: %.2fs, %.4fmin' % (test_seconds, test_seconds / 60))

        post_start = clock.now()
        outputs = [trainPred_real + 1j * trainPred_imag, Y_train_real + 1j * Y_train_imag, testPred_real + 1j * testPred_imag, Y_test_real + 1j * Y_test_imag]
        names = ['train_Pred', 'train_Y', 'test_Pred_0', 'test_Y']
        for name, data in zip(names, outputs):
            with h5py.File(os.path.join(directory, name + '.h5'), 'w') as file:
                file.create_dataset(name, data=data)
        post_seconds = clock.now() - post_start
        if args.profile:
            log_string(log, 'plotting time: %.2fs, %.4fmin' % (plot_seconds, plot_seconds / 60))
            log_string(log, 'postprocess and H5 saving time: %.2fs, %.4fmin' % (post_seconds, post_seconds / 60))
        print('All plots and prediction results have been saved in: %s' % directory)
    finally:
        if monitor is not None:
            try:
                clock.synchronize(forecast.device)
                total_seconds = clock.now() - total_start
            finally:
                monitor.stop()
            monitor.log_summary(log, total_seconds)

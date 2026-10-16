import datetime
import os
import time
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, TensorDataset
import tool


# Let cuDNN select a fast convolution algorithm for fixed input shapes.
torch.backends.cudnn.benchmark = True

# Enable TF32 acceleration for FP32 matrix multiplication and convolution on supported GPUs.
# Floating-point results may differ slightly when TF32 is enabled.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class ExponentialLRSchedulerWithMinLr(_LRScheduler):
    def __init__(self, optimizer, gamma, min_lr, last_epoch=-1):
        self.gamma = gamma
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            max(base_lr * self.gamma ** self.last_epoch, self.min_lr)
            for base_lr in self.base_lrs
        ]


class WeightedComplexL2Loss(nn.Module):
    def __init__(self, weight_path, alpha=None):
        super().__init__()
        weight_file = os.path.join(weight_path, "feature_weights.npz")
        try:
            weights = np.load(weight_file)["weights"].astype(np.float32)
            if alpha is not None:
                with np.load(os.path.join(weight_path, 'norm_params.npz')) as params:
                    power = params['max_power'].reshape(-1)
                weights = ((1 + power / max(float(power.max()), 1e-12)) ** alpha).astype(np.float32)
            self.register_buffer("weights", torch.from_numpy(weights))
        except Exception:
            self.register_buffer("weights", None)
        self._printed = False

    def forward(self, pred_real, pred_imag, target_complex):
        # Compute the loss in FP32 for stable square roots and small errors.
        pred_real = pred_real.float()
        pred_imag = pred_imag.float()
        target_real = target_complex.real.float()
        target_imag = target_complex.imag.float()

        diff_squared = (
            (pred_real - target_real).square()
            + (pred_imag - target_imag).square()
        )
        l2_diff = torch.sqrt(diff_squared + 1e-8)

        if self.weights is not None:
            # criterion.to(device) moves the registered weight buffer to the target device.
            # Reuse the buffer without transferring it again for each batch.
            l2_diff = l2_diff * self.weights.view(1, 1, -1)
            if not self._printed:
                print("Using weighted complex L2 loss.")
                self._printed = True
        elif not self._printed:
            print("Using unweighted complex L2 loss.")
            self._printed = True

        return l2_diff.mean()


class PhysicalMSELoss(nn.Module):
    """Squared complex error in physical units after per-feature normalization."""
    def __init__(self, folder, alpha=0.0):
        super().__init__()
        with np.load(os.path.join(folder, 'norm_params.npz')) as p:
            real_scale = (p['max_real'] - p['min_real']).reshape(-1)
            imag_scale = (p['max_imag'] - p['min_imag']).reshape(-1)
            power = p['max_power'].reshape(-1)
        emphasis = (1 + power / max(float(power.max()), 1e-12)) ** alpha
        normalizer = np.mean(real_scale ** 2 + imag_scale ** 2)
        self.register_buffer('real_weight', torch.tensor(real_scale ** 2 * emphasis / normalizer, dtype=torch.float32))
        self.register_buffer('imag_weight', torch.tensor(imag_scale ** 2 * emphasis / normalizer, dtype=torch.float32))

    def forward(self, pred_real, pred_imag, target_complex):
        return ((pred_real.float() - target_complex.real.float()).square() * self.real_weight +
                (pred_imag.float() - target_complex.imag.float()).square() * self.imag_weight).mean()


class Long_Term_Forecast(object):
    def __init__(self, model, args, log):
        super().__init__()
        self.model = model
        self.args = args
        self.device_ids = list(range(torch.cuda.device_count()))
        self.log = log
        self.data_file = args.data_file
        self.device = None

    def _acquire_device(self):
        if self.args.use_gpu and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.device_ids[0]}")
            if self.args.use_multi_gpu and len(self.device_ids) > 1:
                print("main GPU used", device)
            return device

        if self.args.use_gpu and not torch.cuda.is_available():
            print("Warning: GPU specified but no GPU available. Using CPU instead.")
        return torch.device("cpu")

    def _build_model(self, device):
        available_gpus = torch.cuda.device_count()
        print(f"Available GPUs: {available_gpus}")

        model = self.model.to(device)
        if (
            self.args.use_gpu
            and torch.cuda.is_available()
            and self.args.use_multi_gpu
            and len(self.device_ids) > 1
        ):
            model = nn.DataParallel(model, device_ids=self.device_ids)
        return model

    def _get_data(self):
        sample_count = getattr(self.args, 'train_samples', 0)
        if sample_count:
            rng = np.random.default_rng(2014)
            with h5py.File(os.path.join(self.data_file, 'X_train.h5')) as xf, h5py.File(os.path.join(self.data_file, 'Y_train.h5')) as yf:
                probability = None
                boost = getattr(self.args, 'boundary_boost', 1.0)
                if boost != 1.0:
                    with h5py.File(os.path.join(self.data_file, 'X_train_first.h5')) as first, np.load(os.path.join(self.data_file, 'norm_params.npz')) as params:
                        initial = first['X_train_first'][:, 0]
                        real = initial.real * (params['max_real'] - params['min_real']).reshape(-1) + params['min_real'].reshape(-1)
                        imag = initial.imag * (params['max_imag'] - params['min_imag']).reshape(-1) + params['min_imag'].reshape(-1)
                    power = (real ** 2 + imag ** 2).max(axis=-1)
                    fraction = self.args.boundary_fraction
                    if not 0 < fraction < 0.5 or boost <= 0:
                        raise ValueError('boundary_fraction must be in (0, .5); boost must be positive')
                    low, high = np.quantile(power, [fraction, 1 - fraction])
                    weights = np.where((power <= low) | (power >= high), boost, 1.0)
                    windows_per_trajectory = xf['X_train'].shape[0] // len(power)
                    probability = np.repeat(weights, windows_per_trajectory)
                    probability /= probability.sum()
                indices = np.sort(rng.choice(xf['X_train'].shape[0], sample_count, replace=False, p=probability))
                x = xf['X_train'][indices].astype(np.complex64)
                y = yf['Y_train'][indices].astype(np.complex64)
            return x.shape[-1], x, x[:1], y, y[:1]
        return tool.load_alldata(self.args.data_file)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_scheduler(self, num_train, optimizer):
        steps_per_epoch = max(1, num_train // self.args.batch_size)
        decay_steps = max(1, self.args.decay_epoch * steps_per_epoch)
        gamma = 0.5 ** (1 / decay_steps)
        return ExponentialLRSchedulerWithMinLr(
            optimizer,
            gamma=gamma,
            min_lr=self.args.min_lr,
        )

    def _select_criterion(self):
        if getattr(self.args, 'loss_mode', 'l2') == 'physical_mse':
            return PhysicalMSELoss(self.data_file, self.args.weight_alpha)
        return WeightedComplexL2Loss(self.data_file, getattr(self.args, 'weight_alpha', None))

    def _build_train_loader(self, X_train, Y_train):
        # Keep the dataset on CPU and transfer one batch at a time to the GPU.
        X_train_real = torch.from_numpy(
            np.ascontiguousarray(X_train.real, dtype=np.float32)
        )
        X_train_imag = torch.from_numpy(
            np.ascontiguousarray(X_train.imag, dtype=np.float32)
        )
        Y_train_tensor = torch.from_numpy(
            np.ascontiguousarray(Y_train, dtype=np.complex64)
        )

        dataset = TensorDataset(X_train_real, X_train_imag, Y_train_tensor)

        # Default to zero workers to avoid copying an in-memory dataset into Windows subprocesses.
        # Use args.num_workers when explicitly provided.
        num_workers = int(getattr(self.args, "num_workers", 0))
        loader_kwargs = {
            "dataset": dataset,
            "batch_size": self.args.batch_size,
            "shuffle": True,
            "drop_last": True,
            "num_workers": num_workers,
            "pin_memory": self.device.type == "cuda",
        }

        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = int(
                getattr(self.args, "prefetch_factor", 2)
            )

        return DataLoader(**loader_kwargs)

    def train_model(self):
        self.device = self._acquire_device()
        self.model = self._build_model(self.device)

        tool.log_string(self.log, "loading data...")
        i_x, X_train, X_test, Y_train, Y_test = self._get_data()

        tool.log_string(self.log, "i_x: %s" % i_x)
        tool.log_string(
            self.log,
            "X_train:   %s\t\tX_test   %s" % (X_train.shape, X_test.shape),
        )
        tool.log_string(
            self.log,
            "Y_train:  %s\t\tY_test:  %s" % (Y_train.shape, Y_test.shape),
        )
        tool.log_string(self.log, "data loaded!")

        del X_test, Y_test

        train_loader = self._build_train_loader(X_train, Y_train)
        train_sample = len(train_loader) * self.args.batch_size
        if len(train_loader) == 0:
            raise ValueError(
                f"Not enough training samples for one full batch: "
                f"num_train={len(X_train)}, batch_size={self.args.batch_size}"
            )

        optimizer = self._select_optimizer()
        criterion = self._select_criterion().to(self.device)
        amp_enabled = self.device.type == "cuda" and not getattr(self.args, 'full_precision', False)
        scaler = GradScaler(enabled=amp_enabled)
        scheduler = self._select_scheduler(len(X_train), optimizer)

        train_total_loss = []
        best_loss = float("inf")
        best_epoch = 0
        no_improve_count = 0
        log_interval = max(1, int(getattr(self.args, "log_interval", 20)))

        tool.log_string(self.log, "**** training model ****")
        print(self.args.max_epoch)

        for epoch in range(self.args.max_epoch):
            print("------------------")
            print(
                f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} | "
                f"epoch: {epoch}"
            )

            self.model.train()
            start_train = time.time()

            # Accumulate the epoch loss on the GPU and read the total at the end of the epoch.
            # This avoids synchronizing the running total after every batch.
            train_loss_sum = torch.zeros(
                (), device=self.device, dtype=torch.float32
            )

            for batch_idx, (batchX_real, batchX_imag, batchY) in enumerate(
                train_loader
            ):
                batchX_real = batchX_real.to(
                    self.device, non_blocking=amp_enabled
                )
                batchX_imag = batchX_imag.to(
                    self.device, non_blocking=amp_enabled
                )
                batchY = batchY.to(
                    self.device, non_blocking=amp_enabled
                )

                optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=amp_enabled):
                    output_real, output_imag = self.model(
                        batchX_real, batchX_imag
                    )
                    batch_loss = criterion(
                        output_real, output_imag, batchY
                    )

                scaler.scale(batch_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss_sum += (
                    batch_loss.detach().float() * batchX_real.size(0)
                )

                # Read and print batch loss only at the logging interval to reduce GPU synchronization.
                if (
                    (batch_idx + 1) % log_interval == 0
                    or batch_idx + 1 == len(train_loader)
                ):
                    current_batch_loss = batch_loss.detach().float().item()
                    print(
                        f"Epoch {epoch + 1}, "
                        f"Batch {batch_idx + 1}/{len(train_loader)},"
                        f"batch_loss={current_batch_loss:.6f}"
                    )

            train_loss = (train_loss_sum / train_sample).item()
            train_total_loss.append(train_loss)
            scheduler.step()

            epoch_time = time.time() - start_train
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch + 1}, "
                f"loss={train_loss:.6f}, "
                f"time={epoch_time:.1f}s, "
                f"learning_rate={current_lr:.10f}"
            )

            if train_loss < best_loss:
                best_loss = train_loss
                best_epoch = epoch + 1
                no_improve_count = 0

                # Save the complete model for compatibility with test_model loading.
                # Unwrap DataParallel so evaluation does not require the training wrapper.
                model_to_save = (
                    self.model.module
                    if isinstance(self.model, nn.DataParallel)
                    else self.model
                )
                torch.save(model_to_save, self.args.model_file)
                print(
                    f"New best model found at epoch {best_epoch} "
                    f"with loss {best_loss:.6f}"
                )
            else:
                no_improve_count += 1
                print(
                    f"No improvement for {no_improve_count} epoch(s)"
                )

                if no_improve_count >= self.args.patience:
                    print("Early stopping triggered!")
                    break

        tool.log_string(
            self.log,
            "Training and validation are completed, and model has been stored "
            f"as {self.args.model_file}",
        )
        tool.log_string(
            self.log,
            f"The best loss {best_loss} at epoch {best_epoch}",
        )

        del X_train, Y_train, train_loader
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return train_total_loss

    def test_model(self):
        """Run batched inference while limiting GPU memory usage.

        Keep input arrays and preallocated outputs on CPU; move one batch at a time
        to the selected device and evaluate with inference_mode and optional AMP.
        Only complete batches are evaluated by this implementation.
        """
        self.device = self._acquire_device()
        i_x, X_train, X_test, Y_train, Y_test = self._get_data()

        X_train_real = np.ascontiguousarray(X_train.real, dtype=np.float32)
        X_train_imag = np.ascontiguousarray(X_train.imag, dtype=np.float32)
        X_test_real = np.ascontiguousarray(X_test.real, dtype=np.float32)
        X_test_imag = np.ascontiguousarray(X_test.imag, dtype=np.float32)

        Y_train_real = Y_train.real
        Y_train_imag = Y_train.imag
        Y_test_real = Y_test.real
        Y_test_imag = Y_test.imag
        del X_train, X_test

        tool.log_string(self.log, "**** testing model ****")
        tool.log_string(
            self.log, "loading model from %s" % self.args.model_file
        )

        model = torch.load(self.args.model_file, map_location=self.device)
        model = model.to(self.device)
        model.eval()

        tool.log_string(self.log, "model restored!")
        tool.log_string(self.log, "evaluating...")

        batch_size = self.args.batch_size
        amp_enabled = self.device.type == "cuda"

        def evaluate_model(real_np, imag_np):
            num_data = real_np.shape[0]
            num_batch = num_data // batch_size
            used_data = num_batch * batch_size

            if used_data == 0:
                raise ValueError(
                    f"Not enough data for one full batch: "
                    f"num_data={num_data}, batch_size={batch_size}"
                )

            pred_real_cpu = None
            pred_imag_cpu = None

            with torch.inference_mode():
                for batch_idx in range(num_batch):
                    start_idx = batch_idx * batch_size
                    end_idx = start_idx + batch_size

                    real_cpu = torch.from_numpy(
                        np.ascontiguousarray(
                            real_np[start_idx:end_idx], dtype=np.float32
                        )
                    )
                    imag_cpu = torch.from_numpy(
                        np.ascontiguousarray(
                            imag_np[start_idx:end_idx], dtype=np.float32
                        )
                    )

                    if amp_enabled:
                        real_cpu = real_cpu.pin_memory()
                        imag_cpu = imag_cpu.pin_memory()

                    batch_real = real_cpu.to(
                        self.device, non_blocking=amp_enabled
                    )
                    batch_imag = imag_cpu.to(
                        self.device, non_blocking=amp_enabled
                    )

                    with autocast(enabled=amp_enabled):
                        output_real, output_imag = model(
                            batch_real, batch_imag
                        )

                    if pred_real_cpu is None:
                        pred_real_cpu = torch.empty(
                            (used_data,) + tuple(output_real.shape[1:]),
                            dtype=torch.float32,
                            device="cpu",
                        )
                        pred_imag_cpu = torch.empty(
                            (used_data,) + tuple(output_imag.shape[1:]),
                            dtype=torch.float32,
                            device="cpu",
                        )

                    pred_real_cpu[start_idx:end_idx].copy_(
                        output_real.float().cpu()
                    )
                    pred_imag_cpu[start_idx:end_idx].copy_(
                        output_imag.float().cpu()
                    )

            return pred_real_cpu, pred_imag_cpu

        if amp_enabled:
            torch.cuda.synchronize(self.device)
        start_test = time.time()

        trainPred_real, trainPred_imag = evaluate_model(
            X_train_real, X_train_imag
        )
        testPred_real, testPred_imag = evaluate_model(
            X_test_real, X_test_imag
        )

        if amp_enabled:
            torch.cuda.synchronize(self.device)
        end_test = time.time()

        Y_train = Y_train[: trainPred_real.shape[0], ...]
        Y_test = Y_test[: testPred_real.shape[0], ...]
        Y_train_real = Y_train_real[: trainPred_real.shape[0], ...]
        Y_train_imag = Y_train_imag[: trainPred_imag.shape[0], ...]
        Y_test_real = Y_test_real[: testPred_real.shape[0], ...]
        Y_test_imag = Y_test_imag[: testPred_imag.shape[0], ...]

        train_loss, train_L2, train_phase = tool.metric(trainPred_real, trainPred_imag, Y_train)
        test_loss, test_L2, test_phase = tool.metric(testPred_real, testPred_imag, Y_test)

        tool.log_string(self.log, "testing time: %.1fs" % (end_test - start_test))
        tool.log_string(self.log, "Total Loss/   train_L2 (1e4) / Phase Diff (1e4°)")
        tool.log_string(self.log,"train %.4f/ %.2f / %.2f"% (train_loss, train_L2, train_phase))
        tool.log_string(self.log, "test %.4f/ %.2f / %.2f"% (test_loss, test_L2, test_phase))
        tool.log_string(self.log, "test time: %.1fmin" % ((time.time() - start_test) / 60))
        
        if amp_enabled:
            torch.cuda.empty_cache()

        return (trainPred_real, trainPred_imag, testPred_real, testPred_imag, Y_train_real, Y_train_imag, Y_test_real, Y_test_imag)

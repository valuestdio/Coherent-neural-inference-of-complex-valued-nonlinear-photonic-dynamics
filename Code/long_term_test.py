import os
import torch
import numpy as np
import tool


class Long_Term_Forecast:
    def __init__(self, model, test_rounds, args, log):
        self.model = model
        self.args = args
        self.log = log
        self.test_rounds = test_rounds
        self.profile = tool.ProfileClock(getattr(args, 'profile', False))
        self.device_ids = list(range(torch.cuda.device_count()))
        self.device = self._acquire_device()
        self.model_loaded = False
        self.gpu_warmed_up = False

    def _acquire_device(self):
        if self.args.use_gpu and torch.cuda.is_available():
            device_id = self.args.gpu if self.args.gpu < torch.cuda.device_count() else 0
            device = torch.device(f'cuda:{device_id}')
            print('GPU used:', device)
            return device

        if self.args.use_gpu:
            print('Warning: GPU specified but no GPU available. Using CPU instead.')

        return torch.device('cpu')

    def _synchronize(self):
        self.profile.synchronize(self.device)

    def load_model(self):
        """Load model once. Model loading is excluded from inference timing."""
        if self.model_loaded:
            return

        tool.log_string(self.log, '**** loading model ****')
        model_path = os.path.join(self.args.data_file, self.args.model_file)
        tool.log_string(self.log, 'loading model from %s' % model_path)

        self.model = torch.load(model_path, map_location=self.device)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.model_loaded = True
        tool.log_string(self.log, 'model restored!')

    def _get_data(self):
        if self.args.use_train:
            return tool.load_train_data(self.args.data_file, self.args.tem_file, self.test_rounds)

        return tool.load_test_data(self.args.data_file, self.args.tem_file, self.test_rounds)

    def _get_data_0(self):
        if self.args.use_train:
            return tool.load_train_first(self.args.data_file, self.args.tem_file, self.test_rounds)

        return tool.load_test_first(self.args.data_file, self.args.tem_file, self.test_rounds)


    def _warm_up_gpu(self, real_np, imag_np):
        """Run one untimed forward pass to exclude CUDA/kernel initialization."""
        if not self.profile.enabled or self.device.type != 'cuda' or self.gpu_warmed_up:
            return

        warm_size = min(self.args.batch_size, real_np.shape[0])
        warm_real = torch.from_numpy(real_np[:warm_size]).to(self.device)
        warm_imag = torch.from_numpy(imag_np[:warm_size]).to(self.device)

        with torch.inference_mode():
            self.model(warm_real, warm_imag)

        self._synchronize()
        del warm_real, warm_imag
        self.gpu_warmed_up = True

    def _evaluate_model(self, real_np, imag_np):
        """
        Inference breakdown:
        H2D              : CPU -> GPU
        Padding          : padding incomplete batch
        Forward          : pure model forward
        Output handling  : GPU output allocation + copy
        D2H              : GPU -> CPU
        """

        num_data = real_np.shape[0]
        batch_size = self.args.batch_size

        timing = {
            'h2d': 0.0,
            'padding': 0.0,
            'forward': 0.0,
            'output_handling': 0.0,
            'd2h': 0.0,
            'profile_total': 0.0
        }

        # ==================== H2D ====================
        self._synchronize()
        t0 = self.profile.now()

        real_tensor = torch.from_numpy(real_np).to(self.device)
        imag_tensor = torch.from_numpy(imag_np).to(self.device)

        self._synchronize()
        timing['h2d'] = self.profile.now() - t0

        pred_real = None
        pred_imag = None

        with torch.inference_mode():
            for start_idx in range(0, num_data, batch_size):
                end_idx = min(start_idx + batch_size, num_data)
                batch_real = real_tensor[start_idx:end_idx]
                batch_imag = imag_tensor[start_idx:end_idx]
                actual_len = batch_real.shape[0]

                # ==================== Padding ====================
                if actual_len < batch_size:
                    self._synchronize()
                    t0 = self.profile.now()

                    pad_shape = (batch_size - actual_len, *batch_real.shape[1:])
                    batch_real = torch.cat([batch_real, torch.zeros(pad_shape, dtype=batch_real.dtype, device=self.device)], dim=0)
                    batch_imag = torch.cat([batch_imag, torch.zeros(pad_shape, dtype=batch_imag.dtype, device=self.device)], dim=0)

                    self._synchronize()
                    timing['padding'] += self.profile.now() - t0

                # ==================== Forward ====================
                self._synchronize()
                t0 = self.profile.now()

                output_real, output_imag = self.model(batch_real, batch_imag)

                self._synchronize()
                timing['forward'] += self.profile.now() - t0

                if actual_len < batch_size:
                    output_real = output_real[:actual_len]
                    output_imag = output_imag[:actual_len]

                # ==================== Output ====================
                self._synchronize()
                t0 = self.profile.now()

                if pred_real is None:
                    pred_real = torch.empty((num_data,) + output_real.shape[1:], dtype=output_real.dtype, device=self.device)
                    pred_imag = torch.empty((num_data,) + output_imag.shape[1:], dtype=output_imag.dtype, device=self.device)

                pred_real[start_idx:end_idx].copy_(output_real)
                pred_imag[start_idx:end_idx].copy_(output_imag)

                self._synchronize()
                timing['output_handling'] += self.profile.now() - t0

        # ==================== D2H ====================
        self._synchronize()
        t0 = self.profile.now()

        predictions_real = pred_real.cpu()
        predictions_imag = pred_imag.cpu()

        self._synchronize()
        timing['d2h'] = self.profile.now() - t0

        timing['profile_total'] = timing['h2d'] + timing['padding'] + timing['forward'] + timing['output_handling'] + timing['d2h']

        del real_tensor, imag_tensor, pred_real, pred_imag

        return predictions_real, predictions_imag, timing

    def _test(self, first_round):
        self.load_model()

        # ==================== Data loading ====================
        t0 = self.profile.now()
        _, X_test, Y_test = self._get_data_0() if first_round else self._get_data()
        if getattr(self.args, 'single_test', False):
            X_test = X_test[:1]
            Y_test = Y_test[:1]
        data_load_time = self.profile.now() - t0

        # ==================== Preprocessing ====================
        t0 = self.profile.now()

        X_test_real = np.ascontiguousarray(X_test.real, dtype=np.float32)
        X_test_imag = np.ascontiguousarray(X_test.imag, dtype=np.float32)
        Y_test_real = Y_test.real
        Y_test_imag = Y_test.imag

        preprocess_time = self.profile.now() - t0

        tool.log_string(self.log, '**** testing model ****')
        tool.log_string(self.log, 'evaluating...')

        self._warm_up_gpu(X_test_real, X_test_imag)

        # ==================== Inference ====================
        self._synchronize()
        start_test = self.profile.now()

        testPred_real, testPred_imag, timing = self._evaluate_model(X_test_real, X_test_imag)

        self._synchronize()
        test_time = self.profile.now() - start_test

        num_samples = testPred_real.shape[0]

        Y_test = Y_test[:num_samples, ...]
        Y_test_real = Y_test_real[:num_samples, ...]
        Y_test_imag = Y_test_imag[:num_samples, ...]

        # ==================== Metric ====================
        t0 = self.profile.now()
        test_loss, test_L2, test_phase = tool.metric(testPred_real, testPred_imag, Y_test)
        metric_time = self.profile.now() - t0

        timing['data_load'] = data_load_time
        timing['preprocess'] = preprocess_time
        timing['metric'] = metric_time
        timing['other_inference'] = test_time - timing['profile_total']

        # ==================== Log ====================
        if self.profile.enabled:
            tool.log_forecast_timing(self.log, test_time, num_samples, timing, data_load_time, preprocess_time, metric_time)

        tool.log_string(self.log, '          Total Loss      train_L2 (1e4) / Phase Diff (1e4°)')
        tool.log_string(self.log, 'test         %.4f /     %.2f / %.2f' % (test_loss, test_L2, test_phase))

        del X_test

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return X_test_real, X_test_imag, testPred_real, testPred_imag, Y_test_real, Y_test_imag, test_time, timing


    def test_single_sample_rollout(self):
        """Run selected training samples through the complete autoregressive inference rollout in memory.

        No intermediate prediction/input files are written. For 1/10/100/1000,
        the first N training samples are used. For 3200 or more, the complete training
        and test sets form the base sample pool. If the requested sample count exceeds
        the available pool, existing samples are repeated in order and truncated to the
        exact requested size. Each prediction is then fed back into the corresponding
        input window for the next inference round.
        """
        self.load_model()

        # ==================== Load samples ====================
        t0 = self.profile.now()
        sample_count = self.args.single_test_samples

        if sample_count >= 3200:
            # Use the complete training set + complete test set as the base sample pool
            _, X_train, _ = tool.load_train_first(self.args.data_file, self.args.tem_file, 1)
            _, X_test, _ = tool.load_test_first(self.args.data_file, self.args.tem_file, 1)
            X_all = np.concatenate([X_train, X_test], axis=0)

            if X_all.shape[0] == 0:
                raise ValueError('The combined training and test sample pool is empty.')

            # Repeat existing samples in order until reaching the requested count
            repeat_times = int(np.ceil(sample_count / X_all.shape[0]))
            X_current = np.concatenate([X_all] * repeat_times, axis=0)[:sample_count]

            tool.log_string(
                self.log,
                'Sample pool: %d train + %d test = %d original samples; requested=%d, repeat_times=%d.'
                % (X_train.shape[0], X_test.shape[0], X_all.shape[0], sample_count, repeat_times)
            )
        else:
            # For 1 / 10 / 100 / 1000, use the first N training samples
            _, X_train, _ = tool.load_train_first(self.args.data_file, self.args.tem_file, 1)

            if sample_count > X_train.shape[0]:
                repeat_times = int(np.ceil(sample_count / X_train.shape[0]))
                X_current = np.concatenate([X_train] * repeat_times, axis=0)[:sample_count]
            else:
                X_current = X_train[:sample_count]

        data_load_time = self.profile.now() - t0

        # ==================== GPU warm-up ====================
        X_real = np.ascontiguousarray(X_current.real, dtype=np.float32)
        X_imag = np.ascontiguousarray(X_current.imag, dtype=np.float32)
        self._warm_up_gpu(X_real, X_imag)

        # ==================== Timing ====================
        timing_all = {
            'h2d': 0.0,
            'padding': 0.0,
            'forward': 0.0,
            'output_handling': 0.0,
            'd2h': 0.0,
            'profile_total': 0.0,
            'other_inference': 0.0,
            'preprocess': 0.0,
            'complex_conversion': 0.0,
            'concatenate': 0.0
        }

        model_inference_total = 0.0
        rollout_start = self.profile.now()

        # ==================== Full autoregressive rollout ====================
        for i in range(1, self.args.test_rounds + 1):
            self.test_rounds = i

            # Preprocess current input
            t0 = self.profile.now()
            X_real = np.ascontiguousarray(X_current.real, dtype=np.float32)
            X_imag = np.ascontiguousarray(X_current.imag, dtype=np.float32)
            timing_all['preprocess'] += self.profile.now() - t0

            # Model inference
            self._synchronize()
            start_test = self.profile.now()

            pred_real, pred_imag, timing = self._evaluate_model(X_real, X_imag)

            self._synchronize()
            inference_time = self.profile.now() - start_test
            model_inference_total += inference_time

            for key in ['h2d', 'padding', 'forward', 'output_handling', 'd2h', 'profile_total']:
                timing_all[key] += timing[key]

            timing_all['other_inference'] += inference_time - timing['profile_total']

            # Complex conversion
            t0 = self.profile.now()
            prediction = pred_real.numpy() + 1j * pred_imag.numpy()
            timing_all['complex_conversion'] += self.profile.now() - t0

            # Autoregressive update
            t0 = self.profile.now()
            X_current = np.concatenate([X_current[:, 1:, :], prediction], axis=1)
            timing_all['concatenate'] += self.profile.now() - t0

            if self.profile.enabled:
                tool.log_string(self.log, 'Round %d/%d, samples=%d, inference: %.6fs, forward: %.6fs' % (i, self.args.test_rounds, sample_count, inference_time, timing['forward']))
            else:
                tool.log_string(self.log, 'Round %d/%d, samples=%d' % (i, self.args.test_rounds, sample_count))

        rollout_total = self.profile.now() - rollout_start

        return X_current, prediction, model_inference_total, rollout_total, data_load_time, timing_all

    def test_model(self):
        return self._test(first_round=False)

    def test_model_0(self):
        return self._test(first_round=True)

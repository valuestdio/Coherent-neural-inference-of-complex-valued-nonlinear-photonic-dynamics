import torch
import numpy as np
import h5py
import os

# Save one array as an HDF5 dataset whose name matches the file stem.
def save_h5(var, name, save_dir):
    with h5py.File(os.path.join(save_dir, f'{name}.h5'), 'w') as hf:
        hf.create_dataset(name, data=var)
    print(f"Saved {name}.h5")

def load_data_sort_new(filename, train_evo, test_evo, steps, window_size, pred_size, stride,
                        normalization='none',
                        sort_mode='input_power',
                        save_dir='./saved_h5'):
    """Load complex trajectories, split samples, build prediction windows and save HDF5 files.

    The MATLAB v7.3 file must contain a compound `data_t` dataset with `real` and
    `imag` fields. After permutation, trajectories have shape [samples, features, steps].
    `window_size` is the input history length; `pred_size` is the target length;
    `stride` controls window spacing. Training targets currently span `stride` steps,
    so use pred_size == stride for matching training and test target lengths.

    Split modes: input_power/output_power reserve both ends of a centered sorted
    subset for testing; random uses a shuffled split; eq selects test samples at
    evenly spaced ranks in the input-peak ordering, not evenly spaced power values.
    `train_evo` and `test_evo` specify the trajectory counts used to build windows.

    Normalization modes: none, min-max, location min-max, observe, dBm, manual.
    min-max uses training statistics; location min-max extends per-feature training
    bounds with test initial-frame extrema; observe uses all loaded trajectories.
    The latter two modes save norm_params.npz. The output directory must exist.

    Returns (i_x, X_train, X_test, Y_train, Y_test), where i_x is the feature count:
    X_train: [train_evo * K, window_size, i_x]; X_test: [test_evo, window_size, i_x];
    Y_train: [train_evo * K, pred_size, i_x]; Y_test: [test_evo, K, pred_size, i_x].
    K = (steps - 1) // stride. Saved Y_train_k/Y_test_k files contain targets per round.
    """

    # Read the MATLAB v7.3 compound dataset.
    with h5py.File(filename, 'r') as f:
        data_struct = f['data_t'][:]
    print("Loaded data shape:", data_struct.shape)

    # Combine field components and reorder to [trajectories, features, steps].
    data = data_struct['real'] + 1j * data_struct['imag']
    data = torch.from_numpy(data).to(torch.cfloat)
    data = data.permute(2, 0, 1)
    print("Complex data shape:", data.shape)

    i_x = data.shape[1]
    num_total = data.shape[0]

    data_real = data.real.numpy()
    data_imag = data.imag.numpy()

    # Peak amplitude gives the same trajectory ordering as peak power.
    input_modulus = torch.abs(data[:, :, 0])
    output_modulus = torch.abs(data[:, :, -1])
    max_input_modulus = torch.max(input_modulus, dim=1).values
    max_output_modulus = torch.max(output_modulus, dim=1).values

    num_test = test_evo
    num_test_half = num_test // 2
    num_train = train_evo

    # Choose the trajectory ordering used for the train/test split.
    if sort_mode == 'input_power':
        print(">>> Sorting by INPUT power")
        sorted_indices = torch.argsort(max_input_modulus)

    elif sort_mode == 'output_power':
        print(">>> Sorting by OUTPUT power")
        sorted_indices = torch.argsort(max_output_modulus)

    elif sort_mode == 'random':
        print(">>> Randomly selecting samples")
        sorted_indices = torch.randperm(num_total)

    elif sort_mode == 'eq':
        print(">>> Equally sampling by INPUT peak power")

        # Select test trajectories uniformly by sorted rank, not by power intervals.
        sorted_by_power = torch.argsort(max_input_modulus)

        num_select = num_test
        num_total = len(sorted_by_power)
        indices = torch.linspace(0, num_total - 1, num_select).long()
        test_indices = sorted_by_power[indices]

        # Keep the remaining trajectories as training candidates.
        all_indices = torch.arange(num_total)
        train_mask = torch.ones(num_total, dtype=torch.bool)
        train_mask[test_indices] = False
        train_indices = all_indices[train_mask]

    else:
        raise ValueError("sort_mode must be one of: 'input_power', 'output_power', 'random'")

    # Use a centered subset with num_train middle samples and test samples at both ends.
    if sort_mode in ['input_power', 'output_power']:
        num_middle = num_train + 2 * num_test_half
        assert num_middle <= num_total, "Not enough samples to satisfy train/test split"

        # Discard any unused trajectories outside the centered subset.
        mid_start = (num_total - num_middle) // 2
        mid_end   = mid_start + num_middle
        mid_indices = sorted_indices[mid_start:mid_end]

        # Combine the low- and high-power ends into the test set.
        low_indices  = mid_indices[:num_test_half]
        high_indices = mid_indices[-num_test_half:]
        test_indices = torch.cat([low_indices, high_indices])

        # Reserve the interior trajectories for training.
        train_indices = mid_indices[num_test_half:-num_test_half]

    elif sort_mode == 'random':
        test_indices = sorted_indices[:num_test]
        train_indices = sorted_indices[num_test:]

    print("train_indices:", train_indices, "test_indices:", test_indices)

    # Extract raw split components for normalization statistics.
    train_real = data_real[train_indices.numpy()]
    train_imag = data_imag[train_indices.numpy()]
    test_real  = data_real[test_indices.numpy()]
    test_imag  = data_imag[test_indices.numpy()]

    # Report field ranges and peak amplitudes before normalization.
    print(f"Training real-component range: min={train_real.min():.6f}, max={train_real.max():.6f}")
    print(f"Training imaginary-component range: min={train_imag.min():.6f}, max={train_imag.max():.6f}")
    print(f"Test real-component range: min={test_real.min():.6f}, max={test_real.max():.6f}")
    print(f"Test imaginary-component range: min={test_imag.min():.6f}, max={test_imag.max():.6f}")

    print(f"Training input peak-amplitude range: min={max_input_modulus[train_indices].min():.6f}, max={max_input_modulus[train_indices].max():.6f}")
    print(f"Test input peak-amplitude range: min={max_input_modulus[test_indices].min():.6f}, max={max_input_modulus[test_indices].max():.6f}")
    print(f"Training output peak-amplitude range: min={max_output_modulus[train_indices].min():.6f}, max={max_output_modulus[train_indices].max():.6f}")
    print(f"Test output peak-amplitude range: min={max_output_modulus[test_indices].min():.6f}, max={max_output_modulus[test_indices].max():.6f}")

    # Apply the selected normalization to the full array after computing its statistics.
    if normalization == 'none':
        pass

    # Use global training bounds; the upper bound is the maximum absolute component.
    elif normalization == 'min-max':
        min_real = np.min(train_real)
        max_real = np.max(np.abs(train_real))
        min_imag = np.min(train_imag)
        max_imag = np.max(np.abs(train_imag))
        print('min(data_real)',min_real)
        print('max(data_real)',max_real)
        print('max(data_real)',np.max(train_real))
        print('min(data_imag)',min_imag)
        print('max(data_imag)',max_imag)
        print('max(data_imag)',np.max(train_imag))

        data_real = (data_real - min_real) / (max_real - min_real)
        data_imag = (data_imag - min_imag) / (max_imag -min_imag)

    # Use per-feature training bounds extended by test initial-frame extrema.
    elif normalization == 'location min-max':

        # Record maximum training power per feature for downstream loss weighting.
        train_power = train_real**2 + train_imag**2
        max_power_train = np.max(train_power, axis=(0, 2), keepdims=True).reshape(1, i_x, 1)

        # Use only the initial frame of each test trajectory to extend the bounds.
        min_real_test = np.min(test_real[:, :, 0], axis=0, keepdims=True).reshape(1, i_x, 1)
        max_real_test = np.max(test_real[:, :, 0], axis=0, keepdims=True).reshape(1, i_x, 1)
        min_imag_test = np.min(test_imag[:, :, 0], axis=0, keepdims=True).reshape(1, i_x, 1)
        max_imag_test = np.max(test_imag[:, :, 0], axis=0, keepdims=True).reshape(1, i_x, 1)

        min_real = np.min(train_real, axis=(0, 2), keepdims=True)
        max_real = np.max(np.abs(train_real), axis=(0, 2), keepdims=True)
        min_imag = np.min(train_imag, axis=(0, 2), keepdims=True)
        max_imag = np.max(np.abs(train_imag), axis=(0, 2), keepdims=True)

        # Expand the training bounds wherever test initial values lie outside them.
        min_real = np.minimum(min_real, min_real_test)
        min_imag = np.minimum(min_imag, min_imag_test)

        max_real = np.maximum(max_real, max_real_test)
        max_imag = np.maximum(max_imag, max_imag_test)
        print('min_real.shape:', min_real.shape)
        print('max_real.shape:', max_real.shape)
        print('min_imag.shape:', min_imag.shape)
        print('max_imag.shape:', max_imag.shape)

        data_real = (data_real - min_real) / (max_real - min_real)
        data_imag = (data_imag - min_imag) / (max_imag - min_imag)

        norm_params = {
            "min_real": min_real,
            "max_real": max_real,
            "min_imag": min_imag,
            "max_imag": max_imag,
            "max_power": max_power_train
        }

        # Save component bounds and per-feature maximum power for later use.
        norm_params_path = os.path.join(save_dir, "norm_params.npz")
        np.savez(norm_params_path,
                min_real=min_real, max_real=max_real,
                min_imag=min_imag, max_imag=max_imag,
                max_power = max_power_train)

        print("Normalization parameters saved to norm_params.npz")

    # Compute per-feature bounds and power statistics from all loaded trajectories.
    elif normalization == 'observe':

        power = data_real**2 + data_imag**2
        max_power = np.max(power, axis=(0, 2), keepdims=True).reshape(1, i_x, 1)

        min_real = np.min(data_real, axis=(0, 2), keepdims=True)
        max_real = np.max(np.abs(data_real), axis=(0, 2), keepdims=True)
        min_imag = np.min(data_imag, axis=(0, 2), keepdims=True)
        max_imag = np.max(np.abs(data_imag), axis=(0, 2), keepdims=True)

        print('min_real.shape:', min_real.shape)
        print('max_real.shape:', max_real.shape)
        print('min_imag.shape:', min_imag.shape)
        print('max_imag.shape:', max_imag.shape)

        data_real = (data_real - min_real) / (max_real - min_real)
        data_imag = (data_imag - min_imag) / (max_imag - min_imag)

        norm_params = {
            "min_real": min_real,
            "max_real": max_real,
            "min_imag": min_imag,
            "max_imag": max_imag,
            "max_power": max_power
        }

        # Save component bounds and per-feature maximum power for later use.
        norm_params_path = os.path.join(save_dir, "norm_params.npz")
        np.savez(norm_params_path,
                min_real=min_real, max_real=max_real,
                min_imag=min_imag, max_imag=max_imag,
                max_power = max_power)

        print("Normalization parameters saved to norm_params.npz")

    # Apply the existing logarithmic transform, floor at -55 and rescale.
    elif normalization == 'dBm':
        m_max = np.max(np.abs(data))
        data = data / m_max
        data = 10 * np.log10(data)
        data[data < -55] = -55
        data = data / 55 + 1
        data_real = np.real(data)
        data_imag = np.imag(data)

    # Scale both field components by a fixed reference amplitude.
    elif normalization == 'manual':
        m_max = 10369993.175721595
        data_real = data_real / m_max
        data_imag = data_imag / m_max

    # Recombine normalized components: [trajectories, features, steps].
    normal_data = data_real + 1j * data_imag
    normal_data = torch.from_numpy(normal_data).to(torch.cfloat)

    # Apply the chosen trajectory split to the normalized data.
    normal_data_train = normal_data[train_indices]
    normal_data_test = normal_data[test_indices]

    # Allocate one training window per trajectory and rollout position.
    evo_size = (steps - 1) // stride
    num_samples = train_evo * evo_size

    X_train = torch.zeros((num_samples, window_size, i_x), dtype=torch.cfloat)
    Y_train = torch.zeros((num_samples, pred_size, i_x), dtype=torch.cfloat)
    X_train_first = torch.zeros((train_evo, window_size, i_x), dtype=torch.cfloat)
    Y_train_list = torch.zeros((train_evo, evo_size, pred_size, i_x), dtype=torch.cfloat)

    for evo in range(train_evo):
        evo_data = normal_data_train[evo]

        # Repeat the initial frame to fill the input history before the first prediction.
        start_frame = evo_data[:, 0].unsqueeze(1)
        padding = start_frame.repeat(1, window_size - 1)
        padded_data = torch.cat((padding, evo_data), dim=1)
        X_train_first[evo] = start_frame.repeat(1, window_size).T

        for i in range(evo_size):
            # Advance the history window and collect the following stride target frames.
            step = i * stride
            idx = evo * evo_size + i

            input_window = padded_data[:, step:step + window_size]
            output_window = padded_data[:, step + window_size:step + window_size + stride]

            X_train[idx] = input_window.T
            Y_train[idx] = output_window.T
            Y_train_list[evo, i] = output_window.T

    # Test rollouts start from repeated initial frames with separate per-round targets.
    test_X = torch.zeros((test_evo, window_size, i_x), dtype=torch.cfloat)
    test_Y_list = torch.zeros((test_evo, evo_size, pred_size, i_x), dtype=torch.cfloat)

    for evo in range(test_evo):
        evo_data = normal_data_test[evo]

        # Repeat the initial frame to fill the input history before the first prediction.
        start_frame = evo_data[:, 0].unsqueeze(1).repeat(1, window_size)
        test_X[evo] = start_frame.T

        for i in range(evo_size):
            # Collect pred_size target frames for this rollout position.
            y_start = 1 + i * stride
            y_end = y_start + pred_size
            y_window = evo_data[:, y_start:y_end]
            test_Y_list[evo, i] = y_window.T

    # Save training windows, initial histories and targets for autoregressive evaluation.
    save_h5(np.array(i_x), 'i_x', save_dir)
    save_h5(X_train, 'X_train', save_dir)
    save_h5(Y_train, 'Y_train', save_dir)
    save_h5(X_train_first, 'X_train_first', save_dir)
    save_h5(test_X, 'X_test', save_dir)

    for i in range(evo_size):
        # Store each rollout round separately so inference can load one target at a time.
        Y_train_i = Y_train_list[:, i]
        save_h5(Y_train_i, f'Y_train_{i}', save_dir)

    for i in range(evo_size):
        Y_test_i = test_Y_list[:, i]
        save_h5(Y_test_i, f'Y_test_{i}', save_dir)

    return i_x, X_train, test_X, Y_train, test_Y_list

def main():

    # Fix the split seed for reproducibility.
    torch.manual_seed(47)
    # Set the input file and preprocessing options.
    filename = 'Data\Real_world\EOcomb_space_25GHz_Vratio_8.9_PM1605_40m_Time_field.mat'
    train_evo = 2688
    test_evo = 512
    steps = 101
    window_size = 10
    pred_size = 1
    stride = 1
    normalization = 'location min-max'
    sort_mode =    'random'
    save_dir = 'Temporary Files_40m'

    # Build the splits and windows, then write the preprocessing results.
    i_x, X_train, X_test, Y_train, Y_test= load_data_sort_new(
        filename, train_evo, test_evo, steps, window_size, pred_size, stride, normalization, sort_mode, save_dir
    )

    print(f"\ni_x = {i_x}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Y_train shape: {Y_train.shape}")
    print(f"Y_test shape: {Y_test.shape}")

if __name__ == '__main__':
    main()

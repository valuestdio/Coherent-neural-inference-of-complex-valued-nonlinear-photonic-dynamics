import h5py
import numpy as np
import os

def read_hdf5(
    input_path='Temporary Files/X_train_relation.h5',
    dataset_name='X_train_relation',
):
    '''
    读取hdf5的数据
    '''
    # ----------------------
    # 加载数据
    # ----------------------
    i = 29440
    with h5py.File(input_path, 'r') as f:
        data = f[dataset_name]
        print(data.shape)

def load_h5(path):
    with h5py.File(path, "r") as f:
        keys = list(f.keys())

        if len(keys) == 0:
            raise ValueError(f"No dataset found in: {path}")

        key = keys[0]
        obj = f[key]

        # 顶层直接是 Dataset
        if isinstance(obj, h5py.Dataset):
            return obj[()]

        # 顶层是 Group，则继续找其中的 Dataset
        if isinstance(obj, h5py.Group):
            datasets = []

            def collect_dataset(name, item):
                if isinstance(item, h5py.Dataset):
                    datasets.append(name)

            obj.visititems(collect_dataset)

            if len(datasets) == 0:
                raise ValueError(f"No dataset found inside group '{key}' in: {path}")

            return obj[datasets[0]][()]

        raise ValueError(f"Unsupported HDF5 object in: {path}")

def merge_menhir_mat_files(
    folder="Data/Real_world/2.5G_14m",
    prefix="Menhir_space_2.502GHz_PMHN5_1.14m_Time_field",
    num_files=16,
    input_key="dataSet",
    output_name="Menhir_space_2.502GHz_PMHN5_1.14m_Time_field.mat",
    output_key="data_t"
):
    """
    使用 h5py 读取多个 MATLAB v7.3 文件：
    每个数据形状为 (16384, 101, 200)，
    取第一个维度中 7680:8704 的 1024 个样本，
    沿最后一维拼接得到输出形状 (1024, 101, 3200)。
    """
    merged_data = None
    start_idx, end_idx = 7680, 8704  # 取 1024 个点

    for i in range(1, num_files + 1):
        file_path = os.path.join(folder, f"{i}-{prefix}.mat")
        print(f"Reading slice [{start_idx}:{end_idx}] from: {file_path}")

        # 用 h5py 读取数据
        with h5py.File(file_path, 'r') as f:
            data_struct = f[input_key][()]  # 结构化数组

        data = data_struct['real'] + 1j * data_struct['imag']
        print("  original shape:", data.shape)

        # 取中间的 1024 个点
        data_slice = data[start_idx:end_idx, :, :]
        print("  sliced shape:", data_slice.shape)

        # 拼接
        if merged_data is None:
            merged_data = data_slice
        else:
            merged_data = np.concatenate((merged_data, data_slice), axis=-1)

    print(f"\nFinal merged shape: {merged_data.shape}")

    # 保存为结构化数组，方便之后直接用 data_struct['real'] + 1j*data_struct['imag'] 读取
    dt = np.dtype([('real', np.float64), ('imag', np.float64)])
    merged_struct = np.empty(merged_data.shape, dtype=dt)
    merged_struct['real'] = merged_data.real
    merged_struct['imag'] = merged_data.imag

    save_path = os.path.join(folder, output_name)
    with h5py.File(save_path, 'w') as f:
        f.create_dataset(output_key, data=merged_struct)

    print(f"Saved merged data to: {save_path}")

def merge_hdf5(folder='Figure/Data/Real_world/RNN', prefix='test_Pred', output_name='test_Pred'):
    all_data = []

    for i in range(10):
        file_name = f'{prefix}_{i}.h5'
        file_path = os.path.join(folder, file_name)
        key_name = f'{prefix}_{i}'

        with h5py.File(file_path, 'r') as f:
            data = f[key_name][()]  # 读取数据
            all_data.append(data)

    # 拼接成 (200, 100, 512)
    merged_data = np.concatenate(all_data, axis=1)

    # 保存合并后的数据
    output_path = os.path.join(folder, f'{output_name}.h5')
    with h5py.File(output_path, 'w') as f:
        f.create_dataset(output_name, data=merged_data)

def split_train():
    # 原始数据路径
    x_train_path = 'Temporary Files/X_train.h5'
    y_train_path = 'Temporary Files/Y_train.h5'
    # 读取数据
    with h5py.File(x_train_path, 'r') as f_x:
        key_x = list(f_x.keys())[0]
        X_train = f_x[key_x][:]  # shape: (58880, 10, 512)

    with h5py.File(y_train_path, 'r') as f_y:
        key_y = list(f_y.keys())[0]
        Y_train = f_y[key_y][:]  # shape: (58880, 10, 512)

    # 检查形状一致性
    print('X_train',X_train.shape)
    print('Y_train',Y_train.shape)
    

    # 构造 X_train_0 到 X_train_9 和 Y_train_0 到 Y_train_9
    for i in range(10):
        X_group = X_train[i::10]
        Y_group = Y_train[i::10]

        assert X_group.shape == (2688, 10, 1024)
        assert Y_group.shape == (2688, 10, 1024)

        # 保存路径
        x_save_path = f'Temporary Files/X_train_{i}.h5'
        y_save_path = f'Temporary Files/Y_train_{i}.h5'
        relation_save_path = f'Temporary Files/X_train_relation_{i}.h5'

        # 保存数据
        with h5py.File(x_save_path, 'w') as f_x_out:
            f_x_out.create_dataset(f'X_train_{i}', data=X_group)

        with h5py.File(y_save_path, 'w') as f_y_out:
            f_y_out.create_dataset(f'Y_train_{i}', data=Y_group)

def split_train_10to1():
    # 原始数据路径
    x_train_path = 'Temporary Files/X_train.h5'
    y_train_path = 'Temporary Files/Y_train.h5'
    # 读取数据
    with h5py.File(x_train_path, 'r') as f_x:
        key_x = list(f_x.keys())[0]
        X_train = f_x[key_x][:]  # shape: (58880, 10, 512)

    with h5py.File(y_train_path, 'r') as f_y:
        key_y = list(f_y.keys())[0]
        Y_train = f_y[key_y][:]  # shape: (58880, 10, 512)

    # 检查形状一致性
    print('X_train',X_train.shape)
    print('Y_train',Y_train.shape)
    

    # 构造 X_train_0 到 X_train_9 和 Y_train_0 到 Y_train_9
    for i in range(100):
        X_group = X_train[i::100]
        Y_group = Y_train[i::100]

        assert X_group.shape == (2688, 10, 1024)
        assert Y_group.shape == (2688, 1, 1024)

        # 保存路径
        x_save_path = f'Temporary Files/X_train_{i}.h5'
        y_save_path = f'Temporary Files/Y_train_{i}.h5'

        # 保存数据
        with h5py.File(x_save_path, 'w') as f_x_out:
            f_x_out.create_dataset(f'X_train_{i}', data=X_group)

        with h5py.File(y_save_path, 'w') as f_y_out:
            f_y_out.create_dataset(f'Y_train_{i}', data=Y_group)

def split_train_pred_single():
    rnn_path = 'Figure/Data/Real_world/10to1/CNNLSTM/train_Pred.h5'
    #rnn_path = 'Figure/Data/Real_world/RNN/train_Pred.h5'

    # 读取原始数据
    with h5py.File(rnn_path, 'r') as f:
        rnn_pred = f['train_Pred'][:]  # [58880, 10, 512]
    print(rnn_pred.shape)
    # 每隔10个取一个（即第0,10,20,...）
    rnn_pred_0 = rnn_pred[::100]        # [5888, 10, 512]
    print(rnn_pred_0.shape)
    # 保存为新的h5文件
    with h5py.File('Figure/Data/Real_world/10to1/CNNLSTM/train_Pred_0.h5', 'w') as f_out:
        f_out.create_dataset('train_Pred_0', data=rnn_pred_0)

def split_train_pred():
    rnn_path = 'Figure/Data/Real_world/RNN/train_Pred.h5'
    rnn_emb_path = 'Figure/Data/Real_world/RNN embedding/train_Pred.h5'

    # 读取原始数据
    with h5py.File(rnn_path, 'r') as f:
        rnn_pred = f['train_Pred'][:]  # [58880, 10, 512]

    with h5py.File(rnn_emb_path, 'r') as f_e:
        rnn_emb_pred = f_e['train_Pred'][:]  # [58880, 10, 512]

    # 每隔10个取一个（即第0,10,20,...）
    rnn_pred_0 = rnn_pred[::10]        # [5888, 10, 512]
    rnn_emb_pred_0 = rnn_emb_pred[::10]  # [5888, 10, 512]

    # 保存为新的h5文件
    with h5py.File('Figure/Data/Real_world/RNN/train_Pred_0.h5', 'w') as f_out:
        f_out.create_dataset('train_Pred_0', data=rnn_pred_0)

    with h5py.File('Figure/Data/Real_world/RNN embedding/train_Pred_0.h5', 'w') as f_e_out:
        f_e_out.create_dataset('train_Pred_0', data=rnn_emb_pred_0)


def compute_feature_weights(param_path="Temporary Files_sort/Temporary Files_9m_my/norm_params.npz", 
                            weight_path="Temporary Files_sort/Temporary Files_9m_my/feature_weights.npz",
                            alpha=4.0):
    # 从 npz 文件加载归一化参数
    params = np.load(param_path)

    max_power = params["max_power"]  # (1,1024,1)

    # reshape 成 (1024,)
    max_power = max_power.reshape(-1)

    # 全局最大幅度
    global_max_amp = np.max(max_power)

    # 原始权重
    base_weights = 1.0 + max_power / (global_max_amp)  # (1024,)

    # 权重的 α 次方
    weights = base_weights ** alpha

    # 保存权重
    np.savez(weight_path, weights=weights)

    return weights

def main():
    compute_feature_weights()

if __name__ == '__main__':
    main()

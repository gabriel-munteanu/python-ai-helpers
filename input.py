import numpy as np


def normalize_0_1(data):
    # Normalize data between 0 and 1
    # Warning! New input for networks must have the same min and max as
    # the traininging data, for correct results
    offset_and_scale = []
    data = data.astype(float)
    _, cols = data.shape
    for col in range(cols):
        maxv = max(data[:, col])
        offset_and_scale.append([0, maxv])
        data[:, col] = data[:, col] / maxv
    return data, offset_and_scale


def normalize_m1_1(data):
    # Normalize data between -1 and 1
    # Warning! New input for networks must have the same min and max as
    # the traininging data, for correct results
    offset_and_scale = []
    data = data.astype(float)
    _, cols = data.shape
    for col in range(cols):
        minv = min(data[:, col])
        maxv = max(data[:, col])
        middle = (minv + maxv) / 2
        width = (maxv - minv) / 2
        offset_and_scale.append([middle, width])
        data[:, col] = (data[:, col] - middle) / width
    return data, offset_and_scale


def denormalize(data, offset_and_scale):
    # Denormalize data
    _, cols = data.shape
    for col in range(cols):
        data[:, col] = data[:, col] * \
            offset_and_scale[col][1] + offset_and_scale[col][0]
    return data


def split_data(data, training_pct=0.6, dev_pct=0.2):
    # split training, dev, test
    test_pct = 1 - training_pct - dev_pct
    if test_pct < 0:
        raise Exception('training percent and dev percent exced 100%')
    length = len(data)
    training_length = int(length * training_pct)
    training = data[0:training_length]
    dev_length = int(length * dev_pct)
    dev = data[training_length:training_length + dev_length]
    test = data[training_length + dev_length:]
    return training, dev, test


def split_x_y(data, y_col):
    y = data[:, y_col]
    return np.delete(data, y_col, axis=1), y

# test_data = np.array([[1, 2], [2, 3], [3, 4]])
# print(test_data)
# print(test_data.shape)
# res, mw = normalize_m1_1(test_data)
# print(denormalize(res, mw))

# test_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# training, dev, test = split_data(test_data)
# print(training)
# print(dev)
# print(test)

# data = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
# print(data)
# x, y = split_x_y(data, 0)
# print(x)
# print(y)

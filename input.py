import numpy as np

# doar numpy array


"""
Normalize data between 0 and 1

Warning! New input for networks must have the same min and max as 
the training data, for correct results
"""
def normalize_0_1(data):
    offset_and_scale = []
    data = data.astype(float)
    _, cols = data.shape
    for col in range(cols):
        maxv = max(data[:, col])
        offset_and_scale.append([0, maxv])
        data[:, col] = data[:, col] / maxv
    return data, offset_and_scale


"""
Normalize data between -1 and 1

Warning! New input for networks must have the same min and max as 
the training data, for correct results
"""
def normalize_m1_1(data):
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


"""
Denormalize data
"""
def denormalize(data, offset_and_scale):
    _, cols = data.shape
    for col in range(cols):
        data[:, col] = data[:, col] * \
            offset_and_scale[col][1] + offset_and_scale[col][0]
    return data

# split traint, dev, test


def split_data(data):
    # todo
    return data


test_data = np.array([[1, 2], [2, 3], [3, 4]])
print(test_data)
print(test_data.shape)
res, mw = normalize_m1_1(test_data)
print(denormalize(res, mw))

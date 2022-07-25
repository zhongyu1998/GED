import h5py
import numpy as np
import scipy.io as sio
import torch

from sklearn.model_selection import KFold


class Subject(object):
    def __init__(self, labels=None, features=None, area_avgs=None):
        """
            labels: a torch tensor of emotion scores, with the shape of 2196 x 34
            features: a torch tensor of node features (34 emotion categories + 370 brain areas),
                      with the shape of 2196 x (34 + 370) x 64
            area_avgs: a torch tensor of average activities, with the shape of 2196 x 370, where each row
                       represents a stimulus, each column represents a brain area, and the (i, j) element
                       represents the average activity of voxels in the j-th brain area for the i-th stimulus
        """
        self.labels = labels
        self.features = features
        self.area_avgs = area_avgs


def area_pooling(n, area_cors, area_voxels):
    """
        n: number of equal parts along each (x or y or z) axis in the cuboid
        area_cors: coordinates of all voxels in this brain area
        area_voxels: activities of all voxels in this brain area
    """
    x_min, y_min, z_min = np.min(area_cors, axis=1)
    x_max, y_max, z_max = np.max(area_cors, axis=1)

    x_bound = np.linspace(x_min, x_max, n+1)
    y_bound = np.linspace(y_min, y_max, n+1)
    z_bound = np.linspace(z_min, z_max, n+1)

    count = 0
    area_features = np.zeros((area_voxels.shape[0], n*n*n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                lower_bound = np.array([x_bound[i], y_bound[j], z_bound[k]]).reshape(-1, 1)
                upper_bound = np.array([x_bound[i+1], y_bound[j+1], z_bound[k+1]]).reshape(-1, 1)
                if i == n-1 or j == n-1 or k == n-1:  # consider the voxels at the upper boundary
                    indicator_bool = np.all((lower_bound <= area_cors) * (area_cors <= upper_bound), axis=0)
                else:
                    indicator_bool = np.all((lower_bound <= area_cors) * (area_cors < upper_bound), axis=0)
                indicator = np.nonzero(indicator_bool)[0]
                block_features = area_voxels[:, indicator]
                if block_features.shape[1]:
                    area_features[:, count] = np.mean(block_features, axis=1)
                count = count + 1

    return area_features


def load_data(category_file, subject_id, num_sessions, num_parts):
    """
        category_file: type of emotion category scores: binary (category) or continuous (categcontinuous)
        subject_id: identifier of subject
        num_sessions: number of sessions
        num_parts: number of equal parts along each (x or y or z) axis
    """
    label_file = 'data/feature/%s.mat' % category_file
    data_label = sio.loadmat(label_file)

    category = data_label['L'][0][0]
    if category_file == 'categcontinuous':
        category_cont = category[0]
        category_feat = np.zeros_like(category_cont)
        # normalization
        for j in range(category_cont.shape[1]):
            cat_min = min(category_cont[:, j])
            cat_max = max(category_cont[:, j])
            for i in range(category_cont.shape[0]):
                category_feat[i][j] = (category_cont[i][j] - cat_min) / (cat_max - cat_min)
    else:
        category_feat = category[0]
    num_emotions = category_feat.shape[1]

    category_name = []
    for c in category[1]:
        category_name.append(c[0][0])
    print('Category name:', category_name)

    data_list = []

    for i in range(num_sessions):
        data_file = 'data/fmri/Subject%d/preprocessed/fmri_Subject%d_Session%d.h5' % (subject_id, subject_id, i+1)
        data = h5py.File(data_file)
        data_list.append(data['dataset'][:])

        if i == 0:
            key = data['metadata']['key'][:]
            value = data['metadata']['value'][:]

    dataset = np.concatenate(data_list, axis=0)
    print('# shape of dataset:', dataset.shape)

    voxel_key = np.nonzero(np.char.count(key, b'VoxelData'))[0]
    voxel_index = np.where(value[voxel_key] == 1)[1]
    voxels = dataset[:, voxel_index]
    print('# shape of voxels:', voxels.shape)

    voxcor_key_i = np.nonzero(np.char.count(key, b'voxel_i'))[0]
    voxcor_key_j = np.nonzero(np.char.count(key, b'voxel_j'))[0]
    voxcor_key_k = np.nonzero(np.char.count(key, b'voxel_k'))[0]
    voxcor_key = np.concatenate((voxcor_key_i, voxcor_key_j, voxcor_key_k))
    voxcors = value[voxcor_key, :][:, voxel_index]

    hcp_key = np.nonzero(np.char.count(key, b'hcp180'))[0]
    hcp_index = [np.where(value[k] == 1)[0] for k in hcp_key]
    hcp_voxels = [dataset[:, i] for i in hcp_index]

    subcortical = ['V4_Thalamus', 'V4_Hippocampus', 'Hypothalamus', 'V4_Pallidum', 'Brainstem',
                   'V4_Caudate', 'V4_Putamen', 'brodmann_area_34', 'V4_Amygdala', 't_Cerebellum']
    subcor_key = [np.nonzero(np.char.count(key, bytes(s, encoding='utf-8')))[0] for s in subcortical]
    subcor_index = [np.where(value[k] == 1)[1] for k in subcor_key]
    subcor_voxels = [dataset[:, i] for i in subcor_index]

    print(len(hcp_voxels), 'cortical areas,', len(subcor_voxels), 'subcortical areas')

    features = np.zeros((voxels.shape[0], num_emotions+len(hcp_voxels)+len(subcor_voxels), num_parts*num_parts*num_parts))
    for i in range(len(hcp_voxels)):
        area_cors = voxcors[:, hcp_index[i]]
        area_voxels = hcp_voxels[i]
        features[:, num_emotions+i, :] = area_pooling(num_parts, area_cors, area_voxels)
    for i in range(len(subcor_voxels)):
        area_cors = voxcors[:, subcor_index[i]]
        area_voxels = subcor_voxels[i]
        features[:, num_emotions+len(hcp_voxels)+i, :] = area_pooling(num_parts, area_cors, area_voxels)
    print('# shape of features:', features.shape)

    hcp_avgs = [np.mean(i, axis=1) for i in hcp_voxels]
    subcor_avgs = [np.mean(i, axis=1) for i in subcor_voxels]
    area_avgs = np.concatenate((hcp_avgs, subcor_avgs)).T

    sj = Subject()

    sj.features = torch.from_numpy(features).float()
    sj.area_avgs = torch.from_numpy(area_avgs).float()

    stim_key = np.nonzero(np.char.count(key, b'stim_index'))[0]
    stim_index = np.where(value[stim_key] == 1)[1]
    stims = dataset[:, stim_index].reshape(-1).astype(int)
    sj.labels = torch.from_numpy(category_feat[stims-1, :]).float()
    print('# shape of %s labels:' % category_file, category_feat[stims-1, :].shape)

    return sj


def load_adj(category_feat, area_avgs, num_activations, num_interactions):
    """
        category_feat: a torch tensor of emotion scores (on the training set)
        area_avgs: a torch tensor of average activities (on the training set)
        num_activations: number of active brain areas (with high average activity) for each stimulus
        num_interactions: number of connected/interactive brain areas for each emotion
    """
    num_emotions = category_feat.shape[1]
    category_feat = category_feat.numpy()
    area_avgs = area_avgs.numpy()

    index_list = []  # the i-th sublist contains indices of stimuli whose i-th emotion category has the highest score

    category_temp = category_feat.copy()  # deep copy
    category_comp = np.tile(np.max(category_feat, axis=1), (num_emotions, 1)).T

    for i in range(num_emotions):
        # stimuli with the highest emotion score in the i-th emotion category
        index = np.nonzero(np.argmax(category_temp, axis=1) == i)[0]

        # stimuli with the highest emotion score in multiple emotion categories
        multi_index = np.nonzero(np.sum(category_temp == category_comp, axis=1) > 1)[0]
        zero_index = np.intersect1d(index, multi_index)
        if zero_index.shape[0]:
            # prevent np.argmax from always returning i (the first occurrence of the highest emotion score) in subsequent loops
            category_temp[zero_index, i] = 0

        if index.shape[0] < 5:
            index = np.argsort(-category_feat[:, i])[0:10]

        index_list.append(index)

    interaction_list = []  # the i-th sublist contains potentially active brain areas for the i-th emotion category

    for i in range(num_emotions):
        rep_feat = area_avgs[index_list[i]]
        area_index = np.argsort(-rep_feat, axis=1)[:, 0:num_activations]
        interaction = np.argsort(-np.bincount(area_index.reshape(-1)))[0:num_interactions]
        interaction_list.append(interaction)

    source = []
    for i in range(num_emotions):
        source.extend([i] * num_interactions)
    destination = np.concatenate(interaction_list) + num_emotions

    edges = [[s, d] for s, d in zip(source, destination)]
    edges.extend([[i, j] for j, i in edges])

    return edges


def split_data(num_samples, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    kf = KFold(n_splits=10, shuffle=True, random_state=seed)

    idx_list = []
    for idx in kf.split(np.zeros(num_samples)):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    return train_idx, test_idx

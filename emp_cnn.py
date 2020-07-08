import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

DataSetName = 'KSC'  # Indianpines  Salinas  PaviaU  KSC
train_num = 50.0

# 根据参数载入数据
if DataSetName == "Indianpines":
    mat_data = sio.loadmat('datasets/Indian_pines_corrected.mat')
    pixels = mat_data['indian_pines_corrected']
    mat_gt = sio.loadmat('datasets/Indian_pines_gt.mat')
    gt = mat_gt['indian_pines_gt']
elif DataSetName == "Salinas":
    mat_data = sio.loadmat('datasets/Salinas_corrected.mat')
    pixels = mat_data['salinas_corrected']
    mat_gt = sio.loadmat('datasets/Salinas_gt.mat')
    gt = mat_gt['salinas_gt']
elif DataSetName == "PaviaU":
    mat_data = sio.loadmat('datasets/PaviaU.mat')
    pixels = mat_data['paviaU']
    mat_gt = sio.loadmat('datasets/PaviaU_gt.mat')
    gt = mat_gt['paviaU_gt']
elif DataSetName == "KSC":
    mat_data = sio.loadmat('datasets/KSC.mat')
    pixels = mat_data['KSC']
    mat_gt = sio.loadmat('datasets/KSC_gt.mat')
    gt = mat_gt['KSC_gt']
else:
    print('非法输入')
    exit()

number_of_rows = pixels.shape[0]
number_of_columns = pixels.shape[1]
pixels = pixels.reshape(np.prod(pixels.shape[:2]), np.prod(pixels.shape[2:]))
gt_flatten = gt.flatten()

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
pixels = sc.fit_transform(pixels)

# Preprocessing
# Applying Principal Components Analysis (PCA)
from sklearn.decomposition import PCA

number_of_pc = 4
pca = PCA(n_components=number_of_pc)
pc = pca.fit_transform(pixels)

# Visualizing PCs
# print(f"The accumulated explained variance for the {number_of_pc} principal components is {np.sum(pca.explained_variance_ratio_)}")
# print(f"Individual Explained Variance: {pca.explained_variance_ratio_}")

columns = number_of_pc
rows = 1
pc_images = np.zeros(shape=(number_of_rows, number_of_columns, number_of_pc))
for i in range(number_of_pc):
    pc_images[:, :, i] = np.reshape(pc[:, i], (number_of_rows, number_of_columns))

# Building the Extended Morphological Profiles (EMP)
from skimage.morphology import reconstruction
from skimage.morphology import erosion
from skimage.morphology import disk
from skimage import util


def opening_by_reconstruction(image, se):
    """
        Performs an Opening by Reconstruction.

        Parameters:
            image: 2D matrix.
            se: structuring element
        Returns:
            2D matrix of the reconstructed image.
    """
    eroded = erosion(image, se)
    reconstructed = reconstruction(eroded, image)
    return reconstructed


def closing_by_reconstruction(image, se):
    """
        Performs a Closing by Reconstruction.

        Parameters:
            image: 2D matrix.
            se: structuring element
        Returns:
            2D matrix of the reconstructed image.
    """
    obr = opening_by_reconstruction(image, se)

    obr_inverted = util.invert(obr)
    obr_inverted_eroded = erosion(obr_inverted, se)
    obr_inverted_eroded_rec = reconstruction(
        obr_inverted_eroded, obr_inverted)
    obr_inverted_eroded_rec_inverted = util.invert(obr_inverted_eroded_rec)
    return obr_inverted_eroded_rec_inverted


def build_morphological_profiles(image, se_size=4, se_size_increment=2, num_openings_closings=4):
    """
        Build the morphological profiles for a given image.

        Parameters:
            base_image: 2d matrix, it is the spectral information part of the MP.
            se_size: int, initial size of the structuring element (or kernel). Structuring Element used: disk
            se_size_increment: int, structuring element increment step
            num_openings_closings: int, number of openings and closings by reconstruction to perform.
        Returns: 
            emp: 3d matrix with both spectral (from the base_image) and spatial information         
    """
    x, y = image.shape

    cbr = np.zeros(shape=(x, y, num_openings_closings))
    obr = np.zeros(shape=(x, y, num_openings_closings))

    it = 0
    tam = se_size
    while it < num_openings_closings:
        se = disk(tam)
        temp = closing_by_reconstruction(image, se)
        cbr[:, :, it] = temp[:, :]
        temp = opening_by_reconstruction(image, se)
        obr[:, :, it] = temp[:, :]
        tam += se_size_increment
        it += 1

    mp = np.zeros(shape=(x, y, (num_openings_closings * 2) + 1))
    cont = num_openings_closings - 1
    for i in range(num_openings_closings):
        mp[:, :, i] = cbr[:, :, cont]
        cont = cont - 1

    mp[:, :, num_openings_closings] = image[:, :]

    cont = 0
    for i in range(num_openings_closings + 1, num_openings_closings * 2 + 1):
        mp[:, :, i] = obr[:, :, cont]
        cont += 1

    return mp


def build_emp(base_image, se_size=4, se_size_increment=2, num_openings_closings=4):
    """
        Build the extended morphological profiles for a given set of images.

        Parameters:
            base_image: 3d matrix, each 'channel' is considered for applying the morphological profile. It is the spectral information part of the EMP.
            se_size: int, initial size of the structuring element (or kernel). Structuring Element used: disk
            se_size_increment: int, structuring element increment step
            num_openings_closings: int, number of openings and closings by reconstruction to perform.
        Returns:
            emp: 3d matrix with both spectral (from the base_image) and spatial information
    """
    base_image_rows, base_image_columns, base_image_channels = base_image.shape
    se_size = se_size
    se_size_increment = se_size_increment
    num_openings_closings = num_openings_closings
    morphological_profile_size = (num_openings_closings * 2) + 1
    emp_size = morphological_profile_size * base_image_channels
    emp = np.zeros(
        shape=(base_image_rows, base_image_columns, emp_size))

    cont = 0
    for i in range(base_image_channels):
        # build MPs
        mp_temp = build_morphological_profiles(
            base_image[:, :, i], se_size, se_size_increment, num_openings_closings)

        aux = morphological_profile_size * (i + 1)

        # build the EMP
        cont_aux = 0
        for k in range(cont, aux):
            emp[:, :, k] = mp_temp[:, :, cont_aux]
            cont_aux += 1

        cont = morphological_profile_size * (i + 1)

    return emp


pc_images.shape
num_openings_closings = 4
morphological_profile_size = (num_openings_closings * 2) + 1
emp_image = build_emp(base_image=pc_images, num_openings_closings=num_openings_closings)

# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D, Conv3D, MaxPooling3D, ZeroPadding3D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, Input
from keras.utils.np_utils import to_categorical
from sklearn.decomposition import PCA
from keras.optimizers import Adam, SGD, Adadelta, RMSprop, Nadam
import keras.callbacks as kcallbacks
from keras.regularizers import l2
from copy import deepcopy
import time
import scipy
import math
import collections
from sklearn import metrics, preprocessing
from Utils import averageAccuracy
from Utils import zeroPadding, normalization, doPCA, modelStatsRecord, ssrn_SS_IN
import sys

np.set_printoptions(threshold=np.inf)

now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
# DataSetName = 'PaviaU'      #  Indianpines  Salinas  PaviaU
save_name = DataSetName + now


def indexToAssignment(index_, Row, Col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index_):
        assign_0 = value // Col + pad_length
        assign_1 = value % Col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign


def assignmentToIndex(assign_0, assign_1, Row, Col):
    new_index = assign_0 * Col + assign_1
    return new_index


def selectNeighboringPatch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row - ex_len, pos_row + ex_len + 1), :]
    selected_patch = selected_rows[:, range(pos_col - ex_len, pos_col + ex_len + 1)]
    return selected_patch


def sampling(proptionVal, groundTruth):  # divide dataset into train and test datasets
    labels_loc = {}
    train = {}
    test = {}
    m = max(groundTruth)
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        labels_loc[i] = indices
        nb_val = int(proptionVal * len(indices))
        train[i] = indices[:-nb_val]
        test[i] = indices[-nb_val:]
    #    whole_indices = []
    train_indices = []
    test_indices = []
    for i in range(m):
        #        whole_indices += labels_loc[i]
        train_indices += train[i]
        test_indices += test[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    return train_indices, test_indices


def step_decay(epoch):
    initial_lrate = 0.03
    drop = 0.5
    epochs_drop = 30.0
    if epoch > 120:
        lrate = initial_lrate * math.pow(drop, 3)
    else:
        lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    # lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


def res4_model_ss():
    # model_res4 = ssrn_SS_IN.ResnetBuilder.build_resnet_8((1, PATCH_LENGTH*2+1, PATCH_LENGTH*2+1, img_channels), CATEGORY)
    model_res4 = ssrn_SS_IN.ResnetBuilder.gaborcnn((1, PATCH_LENGTH * 2 + 1, PATCH_LENGTH * 2 + 1, img_channels),
                                                   CATEGORY)

    RMS = RMSprop(lr=0.03)
    # Let's train the model using RMSprop
    model_res4.compile(loss='categorical_crossentropy', optimizer=RMS, metrics=['accuracy'])

    lrate = kcallbacks.LearningRateScheduler(step_decay)
    callbacks_list = lrate

    return model_res4, callbacks_list


data_IN = emp_image
gt_IN = gt

# 配置信息
PATCH_LENGTH = 12  # Patch_size (12*2+1)*(12*2+1)
batch_size = 32
CATEGORY = len(set(gt_IN.flatten())) - 1  # 类别数
nb_epoch = 120  # 400
patience = 200
ITER = 10  # 训练-测试 轮数
img_channels = data_IN.shape[-1]  # PCA降维后的维度数
ValMode = "none"  # 验证集模式

print('图像尺寸：')
print(data_IN.shape)
print('图像有效点数量：')
print(len(np.nonzero(gt_IN)[0]))

# 裁剪图像（切边）
cut_gt_IN = gt_IN[PATCH_LENGTH:gt_IN.shape[0] - PATCH_LENGTH, PATCH_LENGTH:gt_IN.shape[1] - PATCH_LENGTH]
print('图像裁剪后尺寸：')
print(cut_gt_IN.shape)
print('图像裁决后有效点数量：')
TOTAL_SIZE = len(np.nonzero(cut_gt_IN)[0])
VALIDATION_SPLIT = 1 - (train_num / TOTAL_SIZE)

# 划分训练数据和测试数据集
gt = cut_gt_IN.reshape(np.prod(cut_gt_IN.shape[:2]), )
train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)
TRAIN_SIZE = len(train_indices)
TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
VAL_RATIO = 0.1

# 将图像拉成向量进行标准化后再还原
data = data_IN.reshape(np.prod(data_IN.shape[:2]), np.prod(data_IN.shape[2:]))
data = preprocessing.scale(data)
data = data.reshape(data_IN.shape[0], data_IN.shape[1], data_IN.shape[2])
# data = myPCA(data, img_channels) (纯CNN不进行PCA降维)

# whole_data是裁剪后的图片, padded_data是原始的整张图片
whole_data = data[PATCH_LENGTH:data_IN.shape[0] - PATCH_LENGTH, PATCH_LENGTH:data.shape[1] - PATCH_LENGTH, :]
padded_data = data

# 创建样本方框
train_data = np.zeros((TRAIN_SIZE, 2 * PATCH_LENGTH + 1, 2 * PATCH_LENGTH + 1, img_channels))
test_data = np.zeros((TEST_SIZE, 2 * PATCH_LENGTH + 1, 2 * PATCH_LENGTH + 1, img_channels))

KAPPA_RES_SS4 = []
OA_RES_SS4 = []
AA_RES_SS4 = []
TRAINING_TIME_RES_SS4 = []
TESTING_TIME_RES_SS4 = []
ELEMENT_ACC_RES_SS4 = np.zeros((ITER, CATEGORY))

seeds = [1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229]

for index_iter in np.arange(ITER):
    print("# %d Iteration" % (index_iter + 1))

    # save the best validated model
    best_weights_RES_path_ss4 = 'models/' + str(save_name) + '_best_RES_3D_SS4_10_' + str(
        index_iter + 1) + '.hdf5'

    np.random.seed(seeds[index_iter])
    train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)
    # train_indices = point_data['index_training'][:,index_iter] - 1
    # test_indices = point_data['index_testing'][:,index_iter] - 1
    y_train = gt[train_indices] - 1  # 将1-16调整到0-15
    y_train = to_categorical(np.asarray(y_train))  # 转成one-hot形式

    y_test = gt[test_indices] - 1  # 将1-16调整到0-15
    y_test = to_categorical(np.asarray(y_test))  # 转成one-hot形式

    # 将裁剪数据坐标恢复为完整坐标
    # 将train_indices在whole_data中的裁剪坐标值更新为padded_data中的完整坐标
    train_assign = indexToAssignment(train_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
    for i in range(len(train_assign)):
        # 将padded_data中的单点样本扩充窗口样本块
        train_data[i] = selectNeighboringPatch(padded_data, train_assign[i][0], train_assign[i][1], PATCH_LENGTH)

    test_assign = indexToAssignment(test_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
    for i in range(len(test_assign)):
        test_data[i] = selectNeighboringPatch(padded_data, test_assign[i][0], test_assign[i][1], PATCH_LENGTH)

    x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], img_channels)
    x_test = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], img_channels)

    # 根据模式参数选择验证集
    if ValMode == "train":  # 从训练集取验证集
        VAL_SIZE = int(TRAIN_SIZE * VAL_RATIO)
        x_val = x_train[-VAL_SIZE:]
        y_val = y_train[-VAL_SIZE:]
        x_train = x_train[:-VAL_SIZE]
        y_train = y_train[:-VAL_SIZE]
    elif ValMode == "test":  # 从测试集取验证集
        VAL_SIZE = int(TEST_SIZE * VAL_RATIO)
        x_val = x_test[-VAL_SIZE:]
        y_val = y_test[-VAL_SIZE:]
        x_test = x_test[:-VAL_SIZE]
        y_test = y_test[:-VAL_SIZE]
    elif ValMode == "none":  # 不使用验证集
        x_val = x_train
        y_val = y_train
    else:
        print('非法模式')
        exit()

    # SS Residual Network 4 with BN
    model_res4_SS_BN, callbacks_list = res4_model_ss()

    earlyStopping6 = kcallbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='auto')
    saveBestModel6 = kcallbacks.ModelCheckpoint(best_weights_RES_path_ss4, monitor='val_loss', verbose=1,
                                                save_best_only=True,
                                                mode='auto')

    tic6 = time.clock()
    print(x_train.shape, x_test.shape)
    print(y_train.shape, y_val.shape)
    history_res4_SS_BN = model_res4_SS_BN.fit(
        x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3]), y_train,
        validation_data=(x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], x_val.shape[3]), y_val),
        batch_size=batch_size,
        nb_epoch=nb_epoch, shuffle=True, callbacks=[earlyStopping6, saveBestModel6, callbacks_list])
    toc6 = time.clock()

    tic7 = time.clock()
    loss_and_metrics_res4_SS_BN = model_res4_SS_BN.evaluate(
        x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3]), y_test,
        batch_size=batch_size)
    toc7 = time.clock()

    print('3D RES_SS4 without BN Training Time: ', toc6 - tic6)
    print('3D RES_SS4 without BN Test time:', toc7 - tic7)

    print('3D RES_SS4 without BN Test score:', loss_and_metrics_res4_SS_BN[0])
    print('3D RES_SS4 without BN Test accuracy:', loss_and_metrics_res4_SS_BN[1])

    print(history_res4_SS_BN.history.keys())

    pred_test_res4 = model_res4_SS_BN.predict(
        x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3])).argmax(axis=1)
    collections.Counter(pred_test_res4)
    gt_test = gt[test_indices] - 1
    overall_acc_res4 = metrics.accuracy_score(pred_test_res4, gt_test)
    confusion_matrix_res4 = metrics.confusion_matrix(pred_test_res4, gt_test)
    each_acc_res4, average_acc_res4 = averageAccuracy.AA_andEachClassAccuracy(confusion_matrix_res4)
    kappa = metrics.cohen_kappa_score(pred_test_res4, gt_test)
    KAPPA_RES_SS4.append(kappa)
    OA_RES_SS4.append(overall_acc_res4)
    AA_RES_SS4.append(average_acc_res4)
    TRAINING_TIME_RES_SS4.append(toc6 - tic6)
    TESTING_TIME_RES_SS4.append(toc7 - tic7)
    print(len(each_acc_res4))
    ELEMENT_ACC_RES_SS4[index_iter, :] = each_acc_res4

    # sio.savemat('./pred_gt' + str(DataSetName) + '_' + str((index_iter + 1)) + '.mat', {'train_indices':train_indices,'test_indices':test_indices,'pred':pred_test_res4,'gt':gt_test})

    print("3D RESNET_SS4 without BN training finished.")

# sio.savemat('./pred_gt' + str(DataSetName) + '_' + str((index_iter + 1)) + '.mat', {'train_indices':train_indices,'test_indices':test_indices,'pred':pred_test_res4,'gt':gt_test})
modelStatsRecord.outputStats(KAPPA_RES_SS4, OA_RES_SS4, AA_RES_SS4, ELEMENT_ACC_RES_SS4,
                             TRAINING_TIME_RES_SS4, TESTING_TIME_RES_SS4,
                             history_res4_SS_BN, loss_and_metrics_res4_SS_BN, CATEGORY,
                             'records/' + str(save_name) + '_emp_cnn.txt',
                             'records/' + str(save_name) + '_train_SS_element_10.txt')
print(np.mean(OA_RES_SS4) * 100, np.std(OA_RES_SS4) * 100)
print(np.mean(AA_RES_SS4) * 100, np.std(AA_RES_SS4) * 100)
print(np.mean(KAPPA_RES_SS4) * 100, np.std(KAPPA_RES_SS4) * 100)
# sendMsg("cnn finish")


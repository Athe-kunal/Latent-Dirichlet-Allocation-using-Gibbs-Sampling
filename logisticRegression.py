import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.linalg import inv as inv_
import random
import time


class LogisticAlgorithms:
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.bayesian_data = []
        num_features = len(data[0])
        #Adding the extra column for bayesian data
        for d in data:
            d_ = [1] + d
            self.bayesian_data.append(d_)
            assert len(d_) == num_features + 1

    def train_test_split(self, data, label, test_data_size, shuffle=True):
        data_arr = np.array(data)
        label_arr = np.array(label)
        label_arr = np.expand_dims(label_arr, axis=1)
        indices = np.arange(data_arr.shape[0])
        if shuffle:
            np.random.shuffle(indices)

        data_arr = data_arr[indices]
        label_arr = label_arr[indices]

        train_data_arr = data_arr[:-test_data_size]
        test_data_arr = data_arr[-test_data_size:]
        train_label_arr = label_arr[:-test_data_size]
        test_label_arr = label_arr[-test_data_size:]

        return train_data_arr, test_data_arr, train_label_arr, test_label_arr


    def sigmoid(self, data_arr, w_n, w_0=0):
        try:
            x = (data_arr @ w_n) + w_0
        except:
            print(data_arr.shape, w_n.shape)
        return 1 / (1 + np.exp(-x))

    #Predictive distribution using equation 4.155
    def prediction(self, data_arr, w_n: np.array):
        # data_arr:Nxd and w_n:dx1
        alpha = 0.1
        y = self.sigmoid(data_arr, w_n)
        r_i = np.multiply(y, 1 - y)
        R = np.eye(data_arr.shape[0])
        np.fill_diagonal(R, r_i)

        M = data_arr.shape[1]
        S_N = inv_(alpha * np.eye(M) + (data_arr.T) @ R @ data_arr)

        sigs = []
        for data in data_arr:
            # 1xd -- dxd -- dx1
            sigs.append(((data.T) @ S_N) @ data)
        sig_2 = np.array(sigs)
        sig_2 = np.expand_dims(sig_2, axis=1)
        mean = data_arr @ w_n
        kappa_sigma = np.power((1 + (math.pi * sig_2) / 8), -0.5)
        pred = 1 / (1 + np.exp(-(kappa_sigma * mean)))
        return pred

    def BayesianLogisticRegression(
        self, data_fraction=1.0, logger: bool = False, shuffle=True
    ):
        N = len(self.bayesian_data)
        test_data_size = N // 3
        #Randomly shuffling the data for Task 1 and not shuffling in Task 2
        (
            train_data_list,
            test_data_list,
            train_label_list,
            test_label_list,
        ) = self.train_test_split(
            self.bayesian_data, self.label, test_data_size, shuffle=shuffle
        )
        num_features = train_data_list.shape[1]
        train_length = len(train_label_list)
        train_data_list = train_data_list[: int(data_fraction * train_length)]
        train_label_list = train_label_list[: int(data_fraction * train_length)]
        w_n = np.zeros((num_features, 1))
        alpha = 0.01
        data_arr = train_data_list
        label_arr = train_label_list
        converge_bool = False
        iterations = 0

        start_time = time.time()
        weight_matrix = []
        time_counter = []
        while not converge_bool:
            y = self.sigmoid(data_arr, w_n)
            r_i = np.multiply(y, 1 - y)
            R = np.eye(data_arr.shape[0])
            np.fill_diagonal(R, r_i)
            w_update = w_n - inv_(
                (alpha * np.eye(num_features) + ((data_arr.T) @ R @ data_arr))
            ) @ (((data_arr.T) @ (y - label_arr)) + alpha * w_n)
            assert (
                w_update.shape == w_n.shape
            ), f"Shape of w_n {w_n.shape} and w_u is {w_update.shape}"
            if iterations == 0:
                pass
            else:
                if (
                    np.linalg.norm(w_update - w_n) / np.linalg.norm(w_n) < 0.001
                    or iterations == 100
                ):
                    converge_bool = True
            w_n = w_update
            iterations += 1
            if logger:
                weight_matrix.append(w_n)
                time_counter.append(time.time() - start_time)

        test_data_arr = test_data_list
        test_label_arr = test_label_list
        # test_label_arr = np.expand_dims(test_label_arr,axis=1)
        sig_preds = self.prediction(test_data_arr, w_n)

        preds = np.where(sig_preds >= 0.5, 1, 0)
        incorrect = np.where(preds != test_label_arr, 1, 0)
        correct = np.where(preds == test_label_arr, 1, 0)

        error_rate = sum(incorrect)[0] / len(test_label_arr)

        errors_weight = []
        for weights in weight_matrix:
            assert weights.shape == w_n.shape
            sig_preds = self.prediction(test_data_arr, weights)
            preds = np.where(sig_preds >= 0.5, 1.0, 0.0)
            incorrect_ = np.where(preds != test_label_arr, 1, 0)
            correct_ = np.where(preds == test_label_arr, 1, 0)

            error_rate_weight = sum(incorrect_)[0] / len(test_label_list)
            errors_weight.append(error_rate_weight)

        return error_rate, weight_matrix, time_counter, errors_weight

    def GradientDescentMethod(
        self, data_fraction=1.0, logger: bool = True, shuffle=False
    ):
        N = len(self.bayesian_data)
        test_data_size = N // 3
        (
            train_data_list,
            test_data_list,
            train_label_list,
            test_label_list,
        ) = self.train_test_split(
            self.bayesian_data, self.label, test_data_size, shuffle=shuffle
        )
        num_features = train_data_list.shape[1]
        train_length = len(train_label_list)
        train_data_list = train_data_list[: int(data_fraction * train_length)]
        train_label_list = train_label_list[: int(data_fraction * train_length)]
        w_n = np.zeros((num_features, 1))
        alpha = 0.1
        data_arr = train_data_list
        label_arr = train_label_list
        converge_bool = False
        iterations = 0

        start_time = time.time()
        weight_matrix = []
        time_counter = []
        eta = 0.001
        while not converge_bool:
            y = self.sigmoid(data_arr, w_n)
            w_update = w_n - eta * ((data_arr.T @ (y - label_arr)) + alpha * w_n)
            assert (
                w_update.shape == w_n.shape
            ), f"Shape of w_n {w_n.shape} and w_u is {w_update.shape}"
            if iterations == 0:
                pass
            else:
                if (
                    np.linalg.norm(w_update - w_n) / np.linalg.norm(w_n) < 0.001
                    or iterations == 6000
                ):
                    converge_bool = True
            w_n = w_update
            iterations += 1
            if logger:
                if iterations % 10 == 0:
                    weight_matrix.append(w_n)
                    time_counter.append(time.time() - start_time)

        test_data_arr = test_data_list
        test_label_arr = test_label_list
        errors_weight = []
        for weights in weight_matrix:
            assert weights.shape == w_n.shape
            sig_preds = self.prediction(test_data_arr, weights)
            preds = np.where(sig_preds >= 0.5, 1.0, 0.0)
            incorrect_ = np.where(preds != test_label_arr, 1, 0)
            correct_ = np.where(preds == test_label_arr, 1, 0)

            error_rate = sum(incorrect_)[0] / len(test_label_list)
            errors_weight.append(error_rate)

        return errors_weight, weight_matrix, time_counter

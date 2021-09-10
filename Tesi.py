#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17

@author: Maurizio Ferrari Dacrema
"""

from Recommender_import_list import *
from Conferences.WWW.NeuMF_our_interface.NeuMF_RecommenderWrapper import NeuMF_RecommenderWrapper


from ParameterTuning.run_parameter_search import runParameterSearch_Collaborative
from ParameterTuning.SearchSingleCase import SearchSingleCase
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs

import os, traceback, argparse
from functools import partial
import pandas as pd
import numpy as np

from Utils.assertions_on_data_for_experiments import assert_implicit_data, assert_disjoint_matrices
from Utils.plot_popularity import plot_popularity_bias, save_popularity_statistics


from Utils.ResultFolderLoader import ResultFolderLoader, generate_latex_hyperparameters

def user_to_ignore_array(file_path, gender):
    data = pd.read_csv(file_path)
    sub_data = data.loc[data['gender'] == (gender)]
    user_id = sub_data['user_id']
    array_id = user_id.to_numpy()
    return array_id

def trainSlimBpr ():
    from Conferences.WWW.NeuMF_our_interface.Movielens1M.Movielens1MReader import Movielens1MReader

    result_folder_path = "result_experiments/{}/{}_{}/".format(CONFERENCE_NAME, ALGORITHM_NAME, "movielens1m")
    dataset = Movielens1MReader(result_folder_path)

    URM_train = dataset.URM_DICT["URM_train"].copy()
    URM_validation = dataset.URM_DICT["URM_validation"].copy()
    URM_test = dataset.URM_DICT["URM_test"].copy()
    URM_test_negative = dataset.URM_DICT["URM_test_negative"].copy()

    # Ensure IMPLICIT data and DISJOINT sets
    assert_implicit_data([URM_train, URM_validation, URM_test, URM_test_negative])

    assert_disjoint_matrices([URM_train, URM_validation, URM_test])
    assert_disjoint_matrices([URM_train, URM_validation, URM_test_negative])

    # If directory does not exist, create
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)

    algorithm_dataset_string = "{}_{}_".format(ALGORITHM_NAME, "movielens1m")

    plot_popularity_bias([URM_train + URM_validation, URM_test],
                         ["Training data", "Test data"],
                         result_folder_path + algorithm_dataset_string + "popularity_plot")

    save_popularity_statistics([URM_train + URM_validation + URM_test, URM_train + URM_validation, URM_test],
                               ["Full data", "Training data", "Test data"],
                               result_folder_path + algorithm_dataset_string + "popularity_statistics")

    from Base.Evaluation.Evaluator import EvaluatorNegativeItemSample

    evaluator_validation = EvaluatorNegativeItemSample(URM_validation, URM_test_negative, cutoff_list=[10])
    evaluator_test_ALL = EvaluatorNegativeItemSample(URM_test, URM_test_negative,
                                                     cutoff_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    evaluator_test_F = EvaluatorNegativeItemSample(URM_test, URM_test_negative,
                                                   cutoff_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                                   ignore_users=user_to_ignore_array("./User_movielens1M/users.csv",
                                                                                     "M"))
    evaluator_test_M = EvaluatorNegativeItemSample(URM_test, URM_test_negative,
                                                   cutoff_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                                   ignore_users=user_to_ignore_array("./User_movielens1M/users.csv",
                                                                                     "F"))

    slimBpr = SLIM_BPR_Cython(URM_train,evaluator_test_all = evaluator_test_ALL, evaluator_test_M = evaluator_test_M, evaluator_test_F = evaluator_test_F)

    slimBpr.fit(epochs=1,#325,
            symmetric = True,
            batch_size = 1000,
            lambda_i = 0.01,
            lambda_j = 1e-05,
            learning_rate = 0.021697364494861038,
            topK = 859,
            sgd_mode='adagrad'
            )

def trainNeuMF ():

    from Conferences.WWW.NeuMF_our_interface.Movielens1M.Movielens1MReader import Movielens1MReader

    result_folder_path = "result_experiments/{}/{}_{}/".format(CONFERENCE_NAME, ALGORITHM_NAME, "movielens1m")
    dataset = Movielens1MReader(result_folder_path)

    URM_train = dataset.URM_DICT["URM_train"].copy()
    URM_validation = dataset.URM_DICT["URM_validation"].copy()
    URM_test = dataset.URM_DICT["URM_test"].copy()
    URM_test_negative = dataset.URM_DICT["URM_test_negative"].copy()

    # Ensure IMPLICIT data and DISJOINT sets
    assert_implicit_data([URM_train, URM_validation, URM_test, URM_test_negative])

    assert_disjoint_matrices([URM_train, URM_validation, URM_test])
    assert_disjoint_matrices([URM_train, URM_validation, URM_test_negative])

    # If directory does not exist, create
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)

    algorithm_dataset_string = "{}_{}_".format(ALGORITHM_NAME, "movielens1m")

    plot_popularity_bias([URM_train + URM_validation, URM_test],
                         ["Training data", "Test data"],
                         result_folder_path + algorithm_dataset_string + "popularity_plot")

    save_popularity_statistics([URM_train + URM_validation + URM_test, URM_train + URM_validation, URM_test],
                               ["Full data", "Training data", "Test data"],
                               result_folder_path + algorithm_dataset_string + "popularity_statistics")

    from Base.Evaluation.Evaluator import EvaluatorNegativeItemSample

    evaluator_validation = EvaluatorNegativeItemSample(URM_validation, URM_test_negative, cutoff_list=[10])
    evaluator_test_ALL = EvaluatorNegativeItemSample(URM_test, URM_test_negative,
                                                 cutoff_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    evaluator_test_F = EvaluatorNegativeItemSample(URM_test, URM_test_negative,
                                                     cutoff_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                                     ignore_users= user_to_ignore_array("./User_movielens1M/users.csv","M"))
    evaluator_test_M = EvaluatorNegativeItemSample(URM_test, URM_test_negative,
                                                     cutoff_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                                     ignore_users= user_to_ignore_array("./User_movielens1M/users.csv","F"))

    neuMF_parameters = {
        "epochs": 5,
        "epochs_gmf": 20,
        "epochs_mlp": 10,
        "batch_size": 256,
        "num_factors": 64,
        "layers": [256, 128, 64],
        "reg_mf": 0.0,
        "reg_layers": [0, 0, 0],
        "num_negatives": 4,
        "learning_rate": 1e-3,
        "learning_rate_pretrain": 1e-3,
        "learner": "sgd",
        "learner_pretrain": "adam",
        "pretrain": True
    }


    NeuMF =  NeuMF_RecommenderWrapper(URM_train, evaluator_test_ALL,evaluator_test_M,evaluator_test_F)

    NeuMF.fit(
        epochs=10,
        epochs_gmf=20,
        epochs_mlp=10,
        batch_size=256,
        num_factors=64,
        layers=[256, 128, 64],
        reg_mf=0.0,
        reg_layers=[0, 0, 0],
        num_negatives=4,
        learning_rate=1e-3,
        learning_rate_pretrain=1e-3,
        learner='sgd',
        learner_pretrain='adam',
        pretrain=True
    )


if __name__ == '__main__':

    ALGORITHM_NAME = "Tesi"
    CONFERENCE_NAME = "2021"

    trainNeuMF()
    #trainSlimBpr()
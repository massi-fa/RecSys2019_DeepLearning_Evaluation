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
import re

from Utils.assertions_on_data_for_experiments import assert_implicit_data, assert_disjoint_matrices
from Utils.plot_popularity import plot_popularity_bias, save_popularity_statistics


from Utils.ResultFolderLoader import ResultFolderLoader, generate_latex_hyperparameters

def cleaned_create(path):
    path_file = path + "Result.txt"
    f = open (path_file, "r")
    lines = f.readlines()
    print(lines)
    f.close()
    path_final = path + "Cleaned.txt"
    f = open(path_final, "w")
    f.close()
    f = open(path_final, "a")
    for line in lines:
        if re.search("^CUTOFF", line):
            line = line.replace("CUTOFF: ", "")
            line = line.replace(" - ", ", ")
            line = line.replace("ROC_AUC: ", "")
            line = line.replace("RECALL: ", "")
            line = line.replace("PRECISION: ", "")
            line = line.replace("PRECISION_RECALL_MIN_DEN: ", "")
            line = line.replace("MAP: ", "")
            line = line.replace("MRR: ", "")
            line = line.replace("NDCG: ", "")
            line = line.replace("F1: ", "")
            line = line.replace("HIT_RATE: ", "")
            line = line.replace("ARHR: ", "")
            line = line.replace("NOVELTY: ", "")
            line = line.replace("AVERAGE_POPULARITY: ", "")
            line = line.replace("DIVERSITY_MEAN_INTER_LIST: ", "")
            line = line.replace("DIVERSITY_HERFINDAHL: ", "")
            line = line.replace("COVERAGE_USER: ", "")
            line = line.replace("DIVERSITY_GINI: ", "")
            line = line.replace("COVERAGE_ITEM: ", "")
            line = line.replace("SHANNON_ENTROPY: ", "")
            line = re.sub('\, $', '', line)
            print(line)
            f.write(line)
    f.close()

def definisci_path(path, count):
    if count == 1:
        path = path + "CSV_RESULT/ALL/"
    if count == 2:
        path = path + "CSV_RESULT/M/"
    if count == 3:
        path = path + "CSV_RESULT/F/"

    return path

def crea_list_string (list_string):
    list_final = []
    list_final.append("CUTOFF, ROC_AUC , PRECISION , PRECISION_RECALL_MIN_DEN , RECALL , MAP , MRR , NDCG , F1 , HIT_RATE , ARHR , NOVELTY , AVERAGE_POPULARITY , DIVERSITY_MEAN_INTER_LIST , DIVERSITY_HERFINDAHL , COVERAGE_ITEM , COVERAGE_USER , DIVERSITY_GINI , SHANNON_ENTROPY\n")
    list_final = list_final + list_string
    return list_final


def crea_file(path, path_dir, epochs, list_string):
    if path == path_dir + "CSV_RESULT/ALL/":
        path = path + "epoca" + str(epochs) + ".txt"
        path_csv = path + str(epochs) + ".csv"
        f = open(path, "w+")
        f.close()
        f = open(path, "a")
        for string in list_string:
            f.write(string)
        f.close()
    if path == path_dir + "CSV_RESULT/M/":
        path = path + "epoca" + str(epochs) + ".txt"
        path_csv = path + str(epochs) + ".csv"
        f = open(path, "w+")
        f.close()
        f = open(path, "a")
        for string in list_string:
            f.write(string)
        f.close()
    if path == path_dir + "CSV_RESULT/F/":
        path = path + "epoca" + str(epochs) + ".txt"
        path_csv = path + str(epochs) + ".csv"
        f = open(path, "w+")
        f.close()
        f = open(path, "a")
        for string in list_string:
            f.write(string)
        f.close()
    data = pd.read_csv(path)
    path = path.replace(".txt", "")
    data.to_csv(path + ".csv", index=None)

def split_file_cleaned (path_dir):
    count = 1
    epochs = 1
    list_string = []
    path_file = path_dir + "Cleaned.txt"
    f = open (path_file, "r")
    lines = f.readlines()
    for line in lines:
        if re.search("^10", line):
            print(count)
            print(line)
            list_string.append(line)
            path = definisci_path(path_dir, count)
            list_string = crea_list_string (list_string)
            crea_file(path, path_dir, epochs,list_string)
            if count < 3:
                count = count + 1
            else:
                count = 1
                epochs = epochs + 1
            list_string = []
        else:
            list_string.append(line)

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

    slimBpr.fit(epochs=325,
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

    path_slim = "./ReusltSlim_Bpr_Tesi/"
    path_NeuMF = "./ResultNeuMF_Tesi/"

    result_path_slim = path_slim + "Result.txt"
    #file = open(result_path_slim,"w")
    #file.write("\n")
    result_path_NeuMF = path_slim + "Result.txt"
    #file = open(result_path_slim, "w")
    #file.write("\n")

    """"
    trainSlimBpr()
    trainNeuMF()
    """

    cleanded_path_SlimBpr = path_slim + "Cleaned.txt"
    cleaned_create(path_slim)
    cleanded_path_NeuMF = path_slim + "Cleaned.txt"
    cleaned_create(path_NeuMF)



    split_file_cleaned(path_slim)
    split_file_cleaned(path_NeuMF)
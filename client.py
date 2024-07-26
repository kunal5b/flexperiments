from collections import OrderedDict
from typing import Dict
from flwr.common import NDArrays, Scalar, GetParametersIns, GetParametersRes, Status, Code, Parameters, FitIns, FitRes, \
    EvaluateRes, EvaluateIns

import torch
from torch.utils.data import Dataset, DataLoader
import flwr as fl

from model import SimpleCNN, MLP, train, test, RNNModel, testRNN, trainRNN, LSTM
from typing import List
from attacks import label_flipping_attack, targeted_label_flipping_attack, gan_attack, partial_dataset_for_GAN_attack
from attacks import mpaf_attack_nn, mpaf_attack_sklearn
from dataset import get_data_numpy

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import log_loss, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import numpy as np
import warnings
from logging import INFO
from flwr.common.logger import log
import xgboost as xgb
from omegaconf import DictConfig

import time
import logging

logging.basicConfig(level=logging.INFO)

def generate_client_fn(traindataset_list: List[Dataset], valdataset_list: List[Dataset], num_classes: int, model: str,
                       cfg: DictConfig):
    """Return a function that can be used by the VirtualClientEngine.

    to spawn a FlowerClient with client id `cid`.
    """
    def client_fn_xgboost(cid: str):
        # This function will be called internally by the VirtualClientEngine
        # Each time the cid-th client is told to participate in the FL
        # simulation (whether it is for doing fit() or evaluate())

        # Returns a normal FLowerClient that will use the cid-th train/val
        # dataloaders as it's local data.
        return FlowerClientXGB(
            traindataset=traindataset_list[int(cid)],
            valdataset=valdataset_list[int(cid)],
            num_classes=num_classes,
            label_ratio=cfg["label_attack_ratio"],
            num_features=28 * 28,  # TODO configurable?
            train_method=cfg.train_method,
            model_name=model
        ).to_client()
    return client_fn_xgboost
    
class FlowerClientXGB(fl.client.Client):
    '''Define a Flower Client'''

    def __init__(self, traindataset: Dataset, valdataset: Dataset, num_classes: int, num_features,
                 train_method, label_ratio: float, model_name: str = "XGB") -> None:
        super().__init__()
        self.model_name = model_name
        # the dataloaders that point to the data associated to this client
        self.traindataset = traindataset
        self.valdataset = valdataset

        self.num_classes = num_classes
        self.num_features = num_features
        self.train_method = train_method
        self.label_ratio = label_ratio
    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        _ = (self, ins)
        return GetParametersRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[]),
        )

    def _local_boost(self, bst_input, num_local_round: int, train_dmatrix):
        # Update trees based on local training data.
        for i in range(num_local_round):
            bst_input.update(train_dmatrix, bst_input.num_boosted_rounds())

        # Bagging: extract the last N=num_local_round trees for sever aggregation
        # Cyclic: return the entire model
        bst = (
            bst_input[
            bst_input.num_boosted_rounds()
            - num_local_round: bst_input.num_boosted_rounds()
            ]
            if self.train_method == "bagging"
            else bst_input
        )

        return bst

    def fit(self, ins: FitIns) -> FitRes:
        # Poison the dataset if the client is malicious
        self.traindataset = applyAttacks(trainset=self.traindataset, config=ins.config, label_ratio=self.label_ratio, model=self.model_name)
        params = {
            "num_class": self.num_classes,
            "eta": float(ins.config["eta"]),
            "max_depth": int(ins.config["max_depth"]),
            "subsample": float(ins.config["subsample"]),
            "colsample_bytree": float(ins.config["colsample_bytree"]),
            "objective": ins.config["objective"],
            "eval_metric": ins.config["eval_metric"],
            "alpha": int(ins.config["alpha"]),
            "lambda": int(ins.config["lambda"]),
            "tree_method": ins.config["tree_method"],
            "device": ins.config["device"]
        }
        # early_stopping = ins.config["early_stopping"]
        # Random Forest Params
        if self.model_name == "RF":
            params["num_parallel_tree"] = ins.config["num_parallel_tree"]

        # self.params = params

        X_train, y_train = get_data_numpy(DataLoader(self.traindataset))
        train_dmatrix = xgb.DMatrix(X_train, label=y_train)
        X_val, y_val = get_data_numpy(DataLoader(self.valdataset))
        val_dmatrix = xgb.DMatrix(X_val, label=y_val)

        global_round = int(ins.config["server_round"])
        num_local_round = int(ins.config["local_epochs"])

        if global_round == 1:
            # First round local training
            bst = xgb.train(
                params,
                train_dmatrix,
                num_boost_round=num_local_round,  # num_boost_round = 1 for Random Forest (configured)
                evals=[(val_dmatrix, "validate"), (train_dmatrix, "train")],
            )
        else:
            global_model = None
            bst = xgb.Booster(params=params)
            for item in ins.parameters.tensors:
                global_model = bytearray(item)

            # Load global model into booster
            bst.load_model(global_model)

            # Local training
            bst = self._local_boost(bst, num_local_round=num_local_round, train_dmatrix=train_dmatrix)

        # Save model
        local_model = bst.save_raw("json")
        local_model_bytes = bytes(local_model)

        if ins.config["defence"]:
            ins.parameters = Parameters(tensor_type="", tensors=[local_model_bytes])
            eval = self.evaluate(self, ins)
            metrics = eval.metrics
            metrics["loss"] = eval.loss
            return FitRes(
                status=Status(
                    code=Code.OK,
                    message="OK",
                ),
                parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
                num_examples=len(X_train),
                metrics=metrics,
            )

        return FitRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=len(X_train),
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # Load global model
        params = {
            "num_class": self.num_classes,
            "eta": float(ins.config["eta"]),
            "max_depth": int(ins.config["max_depth"]),
            "subsample": float(ins.config["subsample"]),
            "colsample_bytree": float(ins.config["colsample_bytree"]),
            "objective": ins.config["objective"],
            "eval_metric": ins.config["eval_metric"],
            "alpha": int(ins.config["alpha"]),
            "lambda": int(ins.config["lambda"]),
            "tree_method": ins.config["tree_method"],
            "device": ins.config["device"]
        }
        bst = xgb.Booster(params=params)
        para_b = None
        for para in ins.parameters.tensors:
            para_b = bytearray(para)
        bst.load_model(para_b)

        # Run evaluation
        X_val, y_val = get_data_numpy(DataLoader(self.valdataset))
        val_dmatrix = xgb.DMatrix(X_val, label=y_val)

        eval_results = bst.eval_set(
            evals=[(val_dmatrix, "valid")],
            iteration=bst.num_boosted_rounds() - 1,
        )
        # auc = round(float(eval_results.split("\t")[1].split(":")[1]), 4)
        mlogloss = float(eval_results.split("\t")[1].split(":")[1])  # merror is 1 - accuracy in XGBoost

        global_round = ins.config["server_round"]
        log(INFO, f"loss = {mlogloss} at round {global_round}")

        # Making predictions
        y_pred = bst.predict(val_dmatrix)
        y_pred_classes = [round(value) for value in y_pred]

        # Calculating additional metrics
        accuracy = accuracy_score(y_val, y_pred_classes)
        precision = precision_score(y_val, y_pred_classes, average='weighted')
        recall = recall_score(y_val, y_pred_classes, average='weighted')
        f1 = f1_score(y_val, y_pred_classes, average='weighted')
        conf_matrix = confusion_matrix(y_val, y_pred_classes)

        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=mlogloss,
            num_examples=len(X_val),
            metrics={"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
                     "confusion_matrix": conf_matrix},
        )


def applyAttacks(trainset: Dataset, label_ratio: float, config, model: str = None) -> Dataset:
    # NOTE: this attack ratio is different, This is for number of samples to attack.
    ## The one in the config file is to select number of malicious clients

    if config["attack_type"] == "TLF":
        if config["is_malicious"]:
            print("----------------------------------Dataset Attacked------------------------------")
            return targeted_label_flipping_attack(trainset=trainset, attack_ratio=1.0)
    elif config["attack_type"] == "GAN":
        if config["is_malicious"]:
            print("----------------------------------Dataset Attacked------------------------------")
            return gan_attack(trainset=trainset)  # Change this if the program crashes
        # LGR model needs samples for all labels
        if model != "LGR":
            return partial_dataset_for_GAN_attack(trainset=trainset)
    elif config["attack_type"] == "MPAF":
        if config["is_malicious"]:
            print("----------------------------------Model Attacked------------------------------")
            return trainset
    else:
        if config["is_malicious"]:
            print("----------------------------------Dataset Attacked------------------------------")
            return label_flipping_attack(dataset=trainset, num_classes=10, attack_ratio=label_ratio)

    return trainset
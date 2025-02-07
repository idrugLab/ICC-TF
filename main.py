import copy
import os
import shutil
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
import logging

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from metric import accuracy, precision, recall, f1_score, bacc_score, roc_auc, mcc_score, kappa, ap_score

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy(
            args.config,
            os.path.join(model_checkpoints_folder, "config_finetune_LINCS.yaml"),
        )

def load_my_state_dict(model, state_dict):
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.parameter.Parameter):
                # backward compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)


class Train(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()

        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        dir_name = (
            current_time + "_" + config["task_name"]
        )
        log_dir = os.path.join("finetune", dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        logging.info("Logging to: {}".format(log_dir))
        print("Logging to:", log_dir)
        self.dataset = dataset
        self.criterion = nn.BCEWithLogitsLoss()

    def _get_device(self):
        if torch.cuda.is_available() and self.config["gpu"] != "cpu":
            device = self.config["gpu"]
            torch.cuda.set_device(device)
        else:
            device = "cpu"
        logging.info("Running on: {}".format(device))
        print("Running on:", device)

        return device

    def _step(self, model, data, n_iter):
        # get the prediction
        _, pred = model(data)   # [N, C]

        y = data.y.view(-1, 1)

        loss = self.criterion(pred, y)

        return loss
    
    def train(self):
        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()

        from ICC_TF_model import CellFormer
        model = CellFormer(**self.config["model"]).to(self.device)
        model = self._load_pre_trained_weights(model)

        layer_list = []
        for name, param in model.named_parameters():
            if "pred_" in name:
                logging.info('{},{}'.format(name, param.requires_grad))
                print(name, param.requires_grad)
                layer_list.append(name)
        
        params = list(
            map(
                lambda x: x[1],
                list(filter(lambda kv: kv[0] in layer_list, model.named_parameters())),
            )
        )
        base_params = list(
            map(
                lambda x: x[1],
                list(
                    filter(lambda kv: kv[0] not in layer_list, model.named_parameters())
                ),
            )
        )

        optimizer = torch.optim.Adam(
            [
                {"params": base_params, "lr": self.config["init_base_lr"]},
                {"params": params}
            ],
            self.config["init_lr"],
            weight_decay=eval(self.config["weight_decay"]),
        )

        model_checkpoints_folder = os.path.join(self.writer.log_dir, "checkpoints")

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        best_valid_auc = 0
        best_valid_corr = -1
        not_improved_count = 0

        for epoch_counter in range(self.config["epochs"]):
            for bn, data in enumerate(train_loader):
                optimizer.zero_grad()

                data = data.to(self.device)
                loss = self._step(model, data, bn)

                if n_iter % self.config["log_every_n_steps"] == 0:
                    self.writer.add_scalar("train_loss", loss, global_step=n_iter)
                    print("Epoch:", epoch_counter, "Iteration:", bn, "Train loss:", loss.item())
                
                loss.backward()
                optimizer.step()
                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config["eval_every_n_epochs"] == 0:
                # if self.config["dataset"]["task"] == "classification":
                    # valid_loss, acc, prec, rec, f1, bacc, auc_roc, mcc, kap, ap
                valid_loss, valid_acc, valid_prec, valid_rec, valid_f1, valid_bacc, valid_auc, valid_mcc, valid_kap, valid_ap = self._validate(
                    model, valid_loader
                )
                if best_valid_auc < valid_auc:
                    # save the model weights
                    torch.save(
                        model.state_dict(),
                        os.path.join(model_checkpoints_folder, "model.pth"),
                    )
                    best_model = copy.deepcopy(model)
                    best_epoch = epoch_counter
                    best_valid_loss = valid_loss
                    best_valid_auc = valid_auc
                    not_improved_count = 0
                else:
                    not_improved_count += 1
                
                if not_improved_count > 30:
                    break

                self.writer.add_scalar(
                    "validation_loss", valid_loss, global_step=valid_n_iter
                )
                self.writer.add_scalar(
                    "validation_auc", valid_auc, global_step=valid_n_iter
                )
                valid_n_iter += 1
        
        logging.info(f"Testing model epoch {best_epoch}, "
            f"best validation AUC: {best_valid_auc:.3f}"
        )
        print(
            f"Testing model epoch {best_epoch}, "
            f"best validation AUC: {best_valid_auc:.3f}"
        )
        self._test(best_model, test_loader)
    
    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join(
                "./", self.config["fine_tune_from"], "checkpoints"
            )
            print(checkpoints_folder)
            state_dict = torch.load(
                os.path.join(checkpoints_folder, "model.pth"), map_location=self.device
            )
            # model.load_state_dict(state_dict)
            load_my_state_dict(model, state_dict)
            logging.info("Loaded pre-trained model with success.")
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            logging.info("Pre-trained weights not found. Training from scratch.")
            print("Pre-trained weights not found. Training from scratch.")
        
        return model
    
    def _validate(self, model, valid_loader):
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            num_data = 0
            for bn, data in enumerate(valid_loader):
                data = data.to(self.device)

                _, pred = model(data)
                loss = self._step(model, data, bn)

                valid_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                pred = torch.sigmoid(pred)

                if self.device == "cpu":
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())
                
            valid_loss /= num_data

        model.train()

        # if self.config["dataset"]["task"] == "classification":
        predictions = np.array(predictions).flatten()
        labels = np.array(labels)
        acc = accuracy(predictions, labels)
        prec = precision(predictions, labels)
        rec = recall(predictions, labels)
        f1 = f1_score(predictions, labels)
        bacc = bacc_score(predictions, labels)
        auc_roc = roc_auc(predictions, labels)
        mcc = mcc_score(predictions, labels)
        kap = kappa(predictions, labels)
        ap = ap_score(predictions, labels)
        
        logging.info("Validation loss: {}, ACC: {}, Prec: {}, Rec: {}, F1: {}, BACC: {}, roc_auc: {}, mcc: {}, kappa: {}, ap: {}".format(valid_loss,acc,prec, rec, f1, bacc, auc_roc, mcc, kap, ap))
        print("Validation loss:", valid_loss, "ACC", acc, "Prec", prec, "Rec", rec, "F1", f1, "BACC", bacc, "roc_auc", auc_roc, "mcc", mcc, "kappa",kap, "ap", ap)
        return valid_loss, acc, prec, rec, f1, bacc, auc_roc, mcc, kap, ap
        
    def _test(self, model, test_loader):
        model_path = os.path.join(self.writer.log_dir, "checkpoints", "model.pth")
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)

        # test steps
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            test_loss = 0.0
            num_data = 0
            for bn, data in enumerate(test_loader):
                data = data.to(self.device)

                _, pred = model(data)
                loss = self._step(model, data, bn)

                test_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                pred = torch.sigmoid(pred)

                if self.device == "cpu":
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())
                
            test_loss /= num_data

        model.train()

        # if self.config["dataset"]["task"] == "classification":
        predictions = np.array(predictions).flatten()
        labels = np.array(labels)
        predictions = np.array(predictions).flatten()
        labels = np.array(labels)
        self.acc = accuracy(predictions, labels)
        self.prec = precision(predictions, labels)
        self.rec = recall(predictions, labels)
        self.f1 = f1_score(predictions, labels)
        self.bacc = bacc_score(predictions, labels)
        self.auc_roc = roc_auc(predictions, labels)
        self.mcc = mcc_score(predictions, labels)
        self.kap = kappa(predictions, labels)
        self.ap = ap_score(predictions, labels)
        
        logging.info(f"Test loss: {test_loss:.4f},"
            f"Test ACC: {self.acc:.3f}, Test Prec: {self.prec:.3f}, Test Rec: {self.rec:.3f}, Test F1: {self.f1:.3f}, Test BACC: {self.bacc:.3f}, Test roc_auc: {self.auc_roc:.3f}, Test mcc: {self.mcc:.3f}, Test kappa: {self.kap:.3f}, Test ap: {self.ap:.3f}")
        print(
            f"Test loss: {test_loss:.4f}, "
            f"Test ACC: {self.acc:.3f}, Test Prec: {self.prec:.3f}, Test Rec: {self.rec:.3f}, Test F1: {self.f1:.3f}, Test BACC: {self.bacc:.3f}, Test roc_auc: {self.auc_roc:.3f}, Test mcc: {self.mcc:.3f}, Test kappa: {self.kap:.3f}, Test ap: {self.ap:.3f}"
        )
        


def main(config):
    setup_seed(42)
    from  dataset.dataset_test import ExprTestDatasetWrapper
    dataset = ExprTestDatasetWrapper(config["batch_size"], **config["dataset"])

    icc_TF = Train(dataset, config)
    icc_TF.train()



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the config file.")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

    log_file = 'my_log_train.log'
    logging.basicConfig(filename=log_file,
                        format='%(asctime)s %(message)s',
                        datefmt='[%Y-%m-%d %H:%M:%S]',
                        level=logging.INFO)
    
    logging.info('cofig: {}'.format(config))
    print(config)
    main(config)

    


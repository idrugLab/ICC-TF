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


class Inference(object):
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

    def _get_device(self):
        if torch.cuda.is_available() and self.config["gpu"] != "cpu":
            device = self.config["gpu"]
            torch.cuda.set_device(device)
        else:
            device = "cpu"
        logging.info("Running on: {}".format(device))
        print("Running on:", device)

        return device

    
    def predict(self):
        test_loader = self.dataset.get_fulldata_loader()

        from ICC_TF_model import CellFormer
        model = CellFormer(**self.config["model"]).to(self.device)
        model = self._load_pre_trained_weights(model)
                
        self._test(model, test_loader)
    
    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join(
                "./", self.config["fine_tune_from"], "checkpoints"
            )
            print(checkpoints_folder)
            state_dict = torch.load(
                os.path.join(checkpoints_folder, "model.pth"), map_location=self.device
            )
            model.load_state_dict(state_dict)
            # load_my_state_dict(model, state_dict)
            logging.info("Loaded pre-trained model with success.")
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            logging.info("Pre-trained weights not found. Training from scratch.")
            print("Pre-trained weights not found. Training from scratch.")
        
        return model
    

    def _test(self, model, test_loader):
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()
            for bn, data in enumerate(test_loader):
                data = data.to(self.device)


                _, pred = model(data)

                pred = torch.sigmoid(pred)

                if self.device == "cpu":
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())
        # model.train()

        # if self.config["dataset"]["task"] == "classification":
        predictions = np.array(predictions).flatten()
        labels = np.array(labels).flatten()

        # np.savetxt( "AGS_predict_comb.csv", predictions, delimiter=",", fmt='%.3f', header='probability')
        # np.savetxt( "A673_index.csv", labels, delimiter=",", fmt='%d', header='index')



def main(config):
    setup_seed(42)
    from  dataset.dataset_test import ExprTestDatasetWrapper
    dataset = ExprTestDatasetWrapper(config["batch_size"], **config["dataset"])

    drugcellTF = Inference(dataset, config)
    drugcellTF.predict()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the config file.")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

    log_file = 'my_log_pretrain_AGS.log'
    logging.basicConfig(filename=log_file,
                        format='%(asctime)s %(message)s',
                        datefmt='[%Y-%m-%d %H:%M:%S]',
                        level=logging.INFO)
    
    logging.info('cofig: {}'.format(config))
    print(config)
    main(config)

    


from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
import csv
import numpy as np
from sklearn import preprocessing
import pickle
from datasets import load_from_disk
from rdkit import Chem
from rdkit.Chem import AllChem


def from_gene_to_node(dictionary, PPI_df):
    humannet_lm_a = PPI_df.copy()
    PPI_df.columns = ['','node2', 'node1', 'score']
    humannet_lm_b = PPI_df[['node1', 'node2', 'score']]
    humannet_lm_all = pd.concat([humannet_lm_a, humannet_lm_b])  # bi-directional
    humannet_lm_all2 = humannet_lm_all.drop_duplicates().reset_index(drop=True, inplace=False)
    geneid1 = [dictionary[gene] for gene in humannet_lm_all2['node1']]
    geneid2 = [dictionary[gene] for gene in humannet_lm_all2['node2']]
    gene_edge = torch.tensor([geneid1, geneid2])
    return gene_edge

def read_expr(expr_file, dictionary):
    expr_df = pd.read_csv(expr_file)
    data_df = expr_df.loc[:, list(dictionary.keys())]
    label_list = expr_df['label'].tolist()
    num_label_list = expr_df['num_label'].to_list()
    return data_df.values, label_list, num_label_list

def read_pathway(expr_file):
    expr_df = pd.read_csv(expr_file, index_col=0)
    pathway_df = expr_df.loc[:, ['C1 Antigen Processing and Presentation', 'C2 Natural Killer Cell Cytotoxicity', 'C3 TCR Signaling Pathway', 'C4 Cytotoxicity of ImmuneCellAI', 'M1 Antimicrobials', 'M2 BCR Singnaling Pathway']]
    return pathway_df.values

class exprTestDataset(Dataset):
    def __init__(self, expr_file, net_file, f_path):
        super(exprTestDataset, self).__init__()

        print('processing...')
        with open('data/token_gene_dictionary_greater_800_CO.pickle', 'rb') as file:
            direction_token_gene = pickle.load(file)
    
        self.bert_dict = pickle.load(open('data/FG_BERT_embedding_256_fine_tune.pkl', 'rb'))
        self.expr,label_list,num_label_list = read_expr(expr_file, direction_token_gene)
        self.drug_list = []
        self.drug_list = [label.split('_')[1] if '_' in label else label for label in label_list]
        self.label = num_label_list
        # self.expr = preprocess(self.expr)
        net_df = pd.read_csv(net_file,dtype='str')
        self.gene_edge = from_gene_to_node(direction_token_gene, net_df)
        self.num_expr_feature = 1
        self.dataset = load_from_disk(f_path)
        self.tokens = self.dataset['input_ids']


    def get(self, index):
        if isinstance(index, int):
            x = torch.tensor(self.expr[index, :].reshape(-1, self.num_expr_feature)).float()    # [#gene, 1]
            
            bert = self.bert_dict[self.drug_list[index]]
            bert = np.asarray(bert)
            bert = torch.FloatTensor(bert)

            label = self.label[index]
            label = torch.FloatTensor([label])
            
            token = torch.tensor(self.tokens[index])
            data = Data(x=x, edge_index=self.gene_edge, token=token, bert=bert, y=label)
            return data
        raise IndexError(
            'Only integers are valid'
            'indices (got ()).'.format(type(index).__name__)
        )
    def len(self):
        return len(self.label)
    

class ExprTestDatasetWithAdditionalFeatures(exprTestDataset):
    def __init__(self,expr_file, net_file, f_path):
        super().__init__(expr_file, net_file, f_path)
        self.feat_values = read_pathway(expr_file)

    def get(self, index):
        data: Data = super().get(index)
        feature = torch.tensor(self.feat_values[index], dtype=torch.float)
        data.feat = feature
        return data


class ExprTestDatasetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_size, test_size,
                Expr_file, net_file, f_path, path_feat, i):
        super(ExprTestDatasetWrapper, self).__init__()
        self.expr_file = Expr_file
        self.net_file = net_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.f_path = f_path
        self.path_feat = path_feat
        self.i = i

    @property
    def dataset(self):
        if not hasattr(self, "_dataset"):
            if self.path_feat:
                self._dataset = ExprTestDatasetWithAdditionalFeatures(
                    self.expr_file, self.net_file, self.f_path
                )
            else:
                self._dataset = exprTestDataset(
                    self.expr_file, self.net_file, self.f_path
                )
        return self._dataset
    
    def get_data_loaders(self):
        (
            train_loader,
            valid_loader,
            test_loader
        ) = self.get_train_validation_data_loaders(self.dataset, self.i)
        return train_loader, valid_loader, test_loader
    
    def get_fulldata_loader(self):
        loader = DataLoader(
            self.dataset,
             batch_size=self.batch_size,
             shuffle=False,
             num_workers=self.num_workers,
             drop_last=False,
        )
        return loader
    
    def get_train_validation_data_loaders(self, train_dataset, i):
        # obtain training indices that will be used for validation
        
        num_train = len(train_dataset)
        print("train number", num_train)

        num_train = len(train_dataset)
        indices = list(range(num_train))
        # np.random.shuffle(indices)

        split = int(np.floor(self.test_size * num_train * i))
        split2 = split + int(np.floor(self.valid_size * num_train))
        split3 = int(np.floor(self.test_size * num_train * (i + 1)))

        valid_idx, test_idx, train_idx = (
            indices[split: split2],
            indices[split2 : split3],
            indices[:split] + indices[split3 :]
        )
        
        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=self.num_workers,
            drop_last=False
        )

        valid_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=valid_sampler,
            num_workers=self.num_workers,
            drop_last=False
        )

        test_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=test_sampler,
            num_workers=self.num_workers,
            drop_last=False,
        )

        return train_loader, valid_loader, test_loader



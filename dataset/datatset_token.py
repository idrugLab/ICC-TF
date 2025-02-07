import csv
import os
import pickle

from datasets import Dataset


with open('data/token_gene_dictionary_greater_800_CO.pickle', 'rb') as file:
    dictionary_genename_token_pair = pickle.load(file)

csv.field_size_limit(500000)


def rebuilder(csv_file_path):
    # files_list = os.listdir(directory_path)
    labels_cell = []
    samples = []
    dt = {}

    headr = True
    with open(csv_file_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        pattern = []
        # sequence level
        for row in csv_reader:
            if headr:
                row_data = row[:]
                for gene in row_data:
                    if gene in dictionary_genename_token_pair:
                        pattern.append(dictionary_genename_token_pair[gene])
                    else:
                        pattern.append(-99999)
                headr = False
                # print(pattern)
            else:
                seq_pattern_order_id_EXPscore = []
                # token level
                for i in range(len(row)):
                    if i == 0:
                        pass
                    elif i == 1:
                        labels_cell.append(row[1])
                    else:
                        if pattern[i] == -99999:
                            pass
                        else:
                            seq_pattern_order_id_EXPscore.append((pattern[i], row[i]))
                
                sorted_seq_pattern = sorted(seq_pattern_order_id_EXPscore, key=lambda x: x[1], reverse=True)
                sample = [item[0] for item in sorted_seq_pattern]

                sample = sample[:2048]
                samples.append(sample)

    dt['input_ids'] = samples
    dt['cell_label'] = labels_cell

    my_dataset = Dataset.from_dict(dt)
    my_dataset.save_to_disk('data/pos_neg_LINCS_cell_dt_cancer')


# test: expr_path_neg_pos_ppi_01.csv

csv_file_path = 'data/pos_neg_LINCS_cancer.csv'
rebuilder(csv_file_path)

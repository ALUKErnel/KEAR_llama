import os
import numpy as np

import pickle
import torch
from torch.utils.data import Dataset
import time
import jsonlines
import jsonlines

# test for tokenizer 
import torch
import csv
from sklearn.model_selection import train_test_split


class CICDataset(Dataset):
    def __init__(self,
                 lang,
                 split,
                 train_file_path,
                 valid_file_path,
                 test_file_path,
                 tokenizer,
                 num_train_lines,
                 max_seq_len): # TODO: max_seq_len的处理方式待定
        self._max_seq_len = max_seq_len
        self.raw_X_question = []
        self.raw_X_comment = [] 
        self.question_id = []
        # self.X_question = [] # (ids, lengths)
        # self.X_comment = [] # (ids, lengths)
        self.X = [] # (ids, lengths)
        # self.X_embedding = [] 
        self.Y = []
        self.Y_t = []
        self.Y_l = []
        self.num_labels = 3  # stance label 个数
        
        self.label_dict = {"FAVOR": 1, "AGAINST": 0, "NEUTRAL": 2}
        self._lang = lang

        if split == "train":
            cnt = 0 
            with open(train_file_path, 'r', encoding='utf-8') as f:
                print("fine")
                reader = csv.reader(f, delimiter="\t")
                header = next(reader)
                for i, line in enumerate(reader):
                    label = line[2]
                    label_index = self.label_dict[label]

                    if self._lang == "Spanish":
                        question = "Independencia de Cataluña"
                    else:
                        question = "Independència de Catalunya"
                        
                    self.raw_X_question.append(question)
                    

                    comment = line[1]
                    self.raw_X_comment.append(comment)        

                    self.X.append((question, comment[:self._max_seq_len])) 

                    self.Y.append(label_index)

                    cnt += 1
                    if num_train_lines > 0 and cnt >= num_train_lines:
                        break
        if split == "valid":
            cnt = 0 
            with open(valid_file_path, 'r', encoding='utf-8') as f:
                print("fine")
                reader = csv.reader(f, delimiter="\t")
                header = next(reader)
                for i, line in enumerate(reader):
                    label = line[2]
                    label_index = self.label_dict[label]

                    if self._lang == "Spanish":
                        question = "Independencia de Cataluña"
                    else:
                        question = "Independència de Catalunya"

                    self.raw_X_question.append(question)
                   

                    comment = line[1]
                    self.raw_X_comment.append(comment)
   

                    self.X.append((question, comment[:self._max_seq_len])) 

                    self.Y.append(label_index)

                    cnt += 1
                    if num_train_lines > 0 and cnt >= num_train_lines:
                        break
        
        if split == "test":
            with open(test_file_path, 'r', encoding='utf-8') as f:
                print("fine")
                reader = csv.reader(f, delimiter="\t")
                header = next(reader)
                cnt = 0
                for i, line in enumerate(reader):
                    label = line[2]
                    label_index = self.label_dict[label]
                
                    if self._lang == "Spanish":
                        question = "Independencia de Cataluña"
                    else:
                        question = "Independència de Catalunya"
                        
                    self.raw_X_question.append(question)
                

                    comment = line[1]
                    self.raw_X_comment.append(comment)

                    self.X.append((question, comment[:self._max_seq_len])) 

                    self.Y.append(label_index)


                    cnt += 1
                    if num_train_lines > 0 and cnt >= num_train_lines:
                        break


    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])   

def get_datasets_main_CIC(lang,
                      train_file_path,
                      valid_file_path,
                      test_file_path,
                      tokenizer,
                      num_train_lines,
                      max_seq_len):
    # 按照每个target分别获取样本
    
    train_dataset = CICDataset(lang, "train", train_file_path, valid_file_path, test_file_path, tokenizer, num_train_lines, max_seq_len)
    valid_dataset = CICDataset(lang, "valid", train_file_path, valid_file_path, test_file_path, tokenizer, num_train_lines, max_seq_len)
    test_dataset = CICDataset(lang, "test", train_file_path, valid_file_path, test_file_path, tokenizer, num_train_lines, max_seq_len)

    
    return train_dataset, valid_dataset, test_dataset


if __name__ == "__main__":
    data_dir = "./dataset_CIC/" # mention the work dir

    es_train_file_path = os.path.join(data_dir, "spanish_train.csv")
    es_valid_file_path = os.path.join(data_dir, "spanish_val.csv")
    es_test_file_path = os.path.join(data_dir, "spanish_test.csv")

    ca_train_file_path = os.path.join(data_dir, "catalan_train.csv")
    ca_valid_file_path = os.path.join(data_dir, "catalan_val.csv")
    ca_test_file_path = os.path.join(data_dir, "catalan_test.csv")

    tokenizer = ""
   
    

    num_train_lines = 0
    max_seq_len = 500

    
    es_train_dataset, es_valid_dataset, es_test_dataset = get_datasets_main_CIC("Spanish", es_train_file_path, es_valid_file_path, es_test_file_path, tokenizer, num_train_lines, max_seq_len)
    ca_train_dataset, ca_valid_dataset, ca_test_dataset = get_datasets_main_CIC("Catalan", ca_train_file_path, ca_valid_file_path, ca_test_file_path, tokenizer, num_train_lines, max_seq_len)

    print(es_train_dataset[1])
    print(len(es_train_dataset))
    print(len(es_valid_dataset))
    print(len(es_test_dataset))

    print(len(ca_train_dataset))
    print(len(ca_valid_dataset))
    print(len(ca_test_dataset))

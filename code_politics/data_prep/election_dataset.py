
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


class ElectionDataset(Dataset):
    def __init__(self,
                 split,
                 src_train_file_path,
                 tgt_valid_file_path, 
                 tgt_test_file_path,
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
        self.num_labels = 3  # stance label 个数
        
        self.label_dict = {"FAVOR": 1, "FAVOUR": 1, "AGAINST": 0, "NONE": 2}



        if split == "train":
            cnt = 0 
            with open(src_train_file_path, 'r', encoding='utf-8') as f:
                print("fine")
                reader = csv.reader(f)
                header = next(reader)
                for i, line in enumerate(reader):
                
                    question = line[0]
                    self.raw_X_question.append(question)

                    comment = line[1]
                    self.raw_X_comment.append(comment)

                    self.X.append((question, comment[:self._max_seq_len])) 


                    label = line[2]
                    label_index = self.label_dict[label]
                    self.Y.append(label_index)

                    cnt += 1
                    if num_train_lines > 0 and cnt >= num_train_lines:
                        break
        
        if split == "valid":
            with open(tgt_valid_file_path, 'r', encoding='utf-8') as f:
                print("fine")
                reader = csv.reader(f)
                header = next(reader)
                cnt = 0
                for i, line in enumerate(reader):
                
                    question = line[0]
                    self.raw_X_question.append(question)

                    comment = line[1]
                    self.raw_X_comment.append(comment)

                    self.X.append((question, comment[:self._max_seq_len])) 


                    label = line[2]
                    label_index = self.label_dict[label]
                    self.Y.append(label_index)


                    cnt += 1
                    if num_train_lines > 0 and cnt >= num_train_lines:
                        break

            
        if split == "test":
            with open(tgt_test_file_path, 'r', encoding='utf-8') as f:
                print("fine")
                reader = csv.reader(f)
                header = next(reader)
                cnt = 0
                for i, line in enumerate(reader):
                
                    question = line[0]
                    self.raw_X_question.append(question)

                    comment = line[1]
                    self.raw_X_comment.append(comment)

                    self.X.append((question, comment[:self._max_seq_len])) 


                    label = line[2]
                    label_index = self.label_dict[label]
                    self.Y.append(label_index)


                    cnt += 1
                    if num_train_lines > 0 and cnt >= num_train_lines:
                        break


    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])   

        

def get_datasets_main_election(en_train_file_path,
                 fr_train_file_path,
                 fr_valid_file_path,
                 fr_test_file_path,
                 num_train_lines,
                 max_seq_len
                 ):
    # 按照每个target分别获取样本
    
    train_dataset = ElectionDataset("train", en_train_file_path, fr_valid_file_path, fr_test_file_path, num_train_lines, max_seq_len) 
    valid_dataset = ElectionDataset("valid", en_train_file_path, fr_valid_file_path, fr_test_file_path, num_train_lines, max_seq_len)
    test_dataset = ElectionDataset("test", en_train_file_path, fr_valid_file_path, fr_test_file_path, num_train_lines, max_seq_len)

    
    return train_dataset, valid_dataset, test_dataset



if __name__ == "__main__":
    data_dir = "./dataset_election/" # mention the work dir

    en_train_file_path = os.path.join(data_dir, "en_train.csv")
    fr_train_file_path = os.path.join(data_dir, "fr_train.csv")
    fr_valid_file_path = os.path.join(data_dir, "fr_valid.csv")
    fr_test_file_path = os.path.join(data_dir, "fr_test.csv")
    

    num_train_lines = 0
    max_seq_len = 500

    train_dataset, valid_dataset, test_dataset = get_datasets_main_election(en_train_file_path, fr_train_file_path, fr_valid_file_path, fr_test_file_path, num_train_lines, max_seq_len)
    
    print(len(train_dataset))
    print(len(valid_dataset))
    print(len(test_dataset))

    print(train_dataset[1])
    print(test_dataset[1])
    
    print("end")



    

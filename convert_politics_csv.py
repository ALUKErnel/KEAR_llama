import os

import pickle
import torch
from torch.utils.data import Dataset
import time
import jsonlines
import jsonlines

import csv 


def readin_jsonl(lang, split, settype, file_path, num_train_lines, max_seq_len, target_list, topic_dict):
    label_dict = {"FAVOR": 1, "AGAINST": 0}
    data = []
    with jsonlines.open(file_path, 'r') as inf:
        print("fine")
        cnt = 0 
        for i, answer in enumerate(inf):
            if split == "test":
                if answer["test_set"] != settype:
                    continue

                curlang = answer["language"]
                if curlang != lang:
                    continue
                    
                topic = answer["topic"]
                if topic not in topic_dict.keys():
                    continue

                question_id = answer["question_id"]
                if question_id not in target_list:
                    continue

                question = answer["question"]
                comment = answer["comment"]

                label = answer.get("label", None)
                # label_index = label_dict[label]
                
                data.append([question, comment[:max_seq_len], label])

                cnt += 1
                if num_train_lines > 0 and cnt >= num_train_lines:
                    break
            
            else:
                curlang = answer["language"]
                if curlang != lang:
                    continue
                    
                topic = answer["topic"]
                if topic not in topic_dict.keys():
                    continue

                question_id = answer["question_id"]
                if question_id not in target_list:
                    continue

                question = answer["question"]
                comment = answer["comment"]

                label = answer.get("label", None)
                # label_index = label_dict[label]
                
                data.append([question, comment[:max_seq_len], label])

                cnt += 1
                if num_train_lines > 0 and cnt >= num_train_lines:
                    break

            
    return data

def get_datasets_main(train_file_path, valid_file_path, test_file_path, num_train_lines, max_seq_len,
                      src_target_list, tgt_target_list, topic_dict):
    de_train = readin_jsonl('de', 'train', None, train_file_path, num_train_lines, max_seq_len, src_target_list, topic_dict)
    de_valid = readin_jsonl('de', 'valid', None, valid_file_path, num_train_lines, max_seq_len, src_target_list, topic_dict)
    de_test =  readin_jsonl('de', 'test','new_comments_defr', test_file_path, num_train_lines, max_seq_len, src_target_list, topic_dict)

    fr_train = readin_jsonl('fr', 'train', None, train_file_path, num_train_lines, max_seq_len, tgt_target_list, topic_dict)
    fr_valid = readin_jsonl('fr', 'valid', None, valid_file_path, num_train_lines, max_seq_len, tgt_target_list, topic_dict)
    fr_test = readin_jsonl('fr', 'test','new_comments_defr', test_file_path, num_train_lines, max_seq_len, tgt_target_list, topic_dict)

    return de_train, de_valid, de_test, fr_train, fr_valid, fr_test


    



def write_out(file_path, data, header):
    f = open(file_path, 'w', encoding='utf-8', newline="")
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)
    f.close()



if __name__ == "__main__":
    data_dir = "./dataset/" # mention the work dir

    tokenizer = ""

    train_file_path = os.path.join(data_dir, "train.jsonl")
    valid_file_path = os.path.join(data_dir, "valid.jsonl")
    test_file_path = os.path.join(data_dir, "test.jsonl")

    max_seq_len = 500 
    num_train_lines = 0


    # sub_datasets_list  = ["political","security"]
    # target_settings_list  = ["all", "partial", "none"]
    sub_datasets_list  = ["political"]

    for sub in sub_datasets_list:
        sub_dataset = sub
        if sub_dataset == "political":
            topic_dict = {"Foreign Policy": 4, "Immigration": 5}
            src_target_list = [15, 16, 17, 18, 19, 20, 35, 59, 60, 61, 62, 63, 64, 1449, 1452, 1453, 1493, 1495, 1496, 1497, 2715, 3391, 3427, 3428, 3429, 3430, 3431, 3468, 3469, 3470, 3471] 
            tgt_target_list = [15, 16, 17, 18, 19, 20, 35, 59, 60, 61, 62, 63, 64, 1449, 1452, 1453, 1493, 1495, 1496, 1497, 2715, 3391, 3427, 3428, 3429, 3430, 3431, 3468, 3469, 3470, 3471] 
             
        else:
            topic_dict = {"Security": 7, "Society": 8}

            src_target_list = [21, 22, 24, 25, 26, 53, 54, 55, 56, 57, 58, 1454, 1455, 1456, 1457, 1458, 1459, 1460, 1487, 1488, 1489, 1490, 1491, 1492, 2716, 3392, 3398, 3432, 3433, 3435, 3461, 3462]
            tgt_target_list = [21, 22, 24, 25, 26, 53, 54, 55, 56, 57, 58, 1454, 1455, 1456, 1457, 1458, 1459, 1460, 1487, 1488, 1489, 1490, 1491, 1492, 2716, 3392, 3398, 3432, 3433, 3435, 3461, 3462]
               
            
        
    
        de_train, de_valid, de_test, fr_train, fr_valid, fr_test = get_datasets_main(train_file_path, valid_file_path, test_file_path, num_train_lines, max_seq_len, src_target_list, tgt_target_list, topic_dict)

    print(len(de_train))
    print(len(de_valid))
    print(len(de_test))
    print(len(fr_train))
    print(len(fr_valid))
    print(len(fr_test))
    print(de_train[1])
    print(fr_test[1])



    header = ["Target", "Text", "Label"]
        
    write_out("./dataset/de_train_data.csv", de_train, header)
    write_out("./dataset/fr_test_data.csv", fr_test, header)






            



                




   
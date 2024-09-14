'''
Description: 
Version: 
Author: zrk
Date: 2023-08-14 15:53:07
LastEditors: zrk
LastEditTime: 2024-07-29 17:20:57
'''
import openai
import jsonlines
import os
import numpy as np
import time 
from data_prep.xstance_dataset import get_datasets_main
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import logging
from options_p import opt


if not os.path.exists(opt.save_file):
    os.makedirs(opt.save_file)
# logging.basicConfig(stream=sys.stderr, level=logging.DEBUG if opt.debug else logging.INFO)
logging.basicConfig(level=logging.INFO if opt.local_rank in [-1, 0] else logging.WARN)
log = logging.getLogger(__name__)
fh = logging.FileHandler(os.path.join(opt.save_file, 'log_politics_p5_temperature{}_translated_2-shot_{}_{}_3.txt'.format(opt.temperature, opt.index1, opt.index2)))
log.addHandler(fh)

index2label = {1: "Favor", 0: "Against"}


def get_metrics_f1(y_true, y_pred):
    average = 'macro'
    # f1 = f1_score(y_true, y_pred, average=average)
    f1_1 = f1_score(y_true == 1, y_pred == 1)
    log.info('favor f1: {}'.format(100 * f1_1))
    f1_0 = f1_score(y_true == 0, y_pred == 0)
    log.info('against f1: {}'.format(100 * f1_0))
    f1_avg = (f1_1 + f1_0) / 2
    log.info("classification report: {}\n".format(classification_report(y_true, y_pred, digits=4)))
    return f1_avg 


def form_template_label(test_data):
    test_templates = []
    for data in test_data:
        x, label = data
        question, comment = x
        # p1 
        # prompt_template = "TARGET: " + question + " TEXT: " + comment + " What is the attitude of TEXT toward TARGET? Give me a one-word answer. Select from 'Favor' and 'Against'."
        # What is the attitude of <text> toward <target>?

        # prompt_template =  "What is the attitude of '" + comment + "' towards '" + question + "'? Give me a one-word answer. Select from 'Favor' and 'Against'."
        # print(prompt_template)
        # p2

        # prompt_template = "TARGET: " + question + " TEXT: " + comment + " Quelle est l’attitude de <text> envers <target>? Give me a one-word answer. Select from 'Favor' and 'Against'." 
        # p3
        # prompt_template = "La position de '" + comment + "' envers '" + question + "' est [MASK]. Give me a one-word answer. Select from 'Favor' and 'Against'."
        # p4
        # prompt_template = "La position de '" + comment + "' envers '" + question + "' est? Give me a one-word answer. Select from 'Favor' and 'Against'."
        # p4
        # prompt_template = "La position de " + comment + " envers " + question + " est? Give me a one-word answer. Select from 'Favor' and 'Against'."
        # p5
        prompt_template = "La position de " + comment + " envers " + question + " est [MASK]. You can translate TEXT into English and determine the stance next. Give me a one-word answer. Select from 'Favor' and 'Against'."

        test_templates.append([prompt_template, label])
    
    return test_templates 


def get_2_example(dataset, index1, index2):
    example_templates = []
    for i, data in enumerate(dataset):
        if i not in [index1, index2]:
            continue

        x, label = data
        question, comment = x
        prompt_template = "La position de " + comment + " envers " + question + " est [MASK]. You can translate TEXT into English and determine the stance next. Give me a one-word answer. Select from 'Favor' and 'Against'."
        
        
        example_templates.append([prompt_template, index2label[label]])
    
    return example_templates 
    


def send_request(template):
    openai.api_key = "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"
    # 0613之后不更新
    completion = openai.ChatCompletion.create(
    model="gpt-4-1106-preview",
    messages=[
        {"role": "user", "content": template },
    ],
    temperature=opt.temperature,
    n=opt.n
    )
    predict_word = completion.choices[0].message.content 
     
    log.info("Response: {}".format(predict_word))
    return predict_word
  

# def send_request_1_shot(template, example_template, example_answer):
#     openai.api_key = "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"
#     # 0613之后不更新
#     completion = openai.ChatCompletion.create(
#     model="gpt-4-1106-preview",
#     messages=[
#         {"role": "user", "content": example_template },
#         {"role": "user", "content": example_answer },
#         {"role": "user", "content": template },
#     ],
#     temperature=opt.temperature,
#     n=opt.n
#     )
#     predict_word = completion.choices[0].message.content 
     
#     log.info("Response: {}".format(predict_word))
#     return predict_word

def send_request_2_shot(template, example_templates):
    example_len = len(example_templates)
    messages = []
    for i in range(example_len):
        messages.append({"role": "user", "content": example_templates[i][0]})
        messages.append({"role": "user", "content": example_templates[i][1]})
    
    messages.append({"role": "user", "content": template })
    openai.api_key = "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"
    # openai.api_key = "sk-proj-sCrr2VANRDMW3fXSBoCwT3BlbkFJ0EVDowOyWptBVlmpB1Ai" # personal
    # 0613之后不更新
    # 0613之后不更新
    completion = openai.ChatCompletion.create(
    model="gpt-4-1106-preview",
    messages=messages,
    temperature=opt.temperature,
    n=opt.n
    )
    predict_word = completion.choices[0].message.content 
     
    log.info("Response: {}".format(predict_word))
    return predict_word

          
def main():
    data_dir = "./dataset/" # mention the work dir

    tokenizer = ""

    train_file_path = os.path.join(data_dir, "train.jsonl")
    valid_file_path = os.path.join(data_dir, "valid.jsonl")
    test_file_path = os.path.join(data_dir, "test.jsonl")

    max_seq_len = 500 
    num_train_lines = 0

    # word_2_label = {"favor": 1,
    #                 "Favor": 1,
    #                 "FAVOR": 1,
    #                 "against": 0,
    #                 "Against": 0,
    #                 "AGAINST": 0}
    
    word_2_label = {"Favor": 1, "FAVOR": 1, "favor": 1, "In favor of": 1, "Favor.": 1, "FAVOR.": 1, "favor.": 1, "In favor of.": 1, 
              "Against": 0, "AGAINST": 0, "against": 0, "Against.": 0, "AGAINST.": 0, "against.": 0
              }


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
            
        de_targets, fr_targets, de_train, de_valid, de_test, fr_train, fr_valid, fr_test = get_datasets_main(train_file_path, valid_file_path, test_file_path, tokenizer, num_train_lines, max_seq_len, src_target_list, tgt_target_list, topic_dict)
        
    
        
        log.info("Dataset: {} ".format(sub_dataset))
        test_templates = form_template_label(fr_test)

        # example_question = "Soll die Schweiz ein Agrarfreihandelsabkommen mit der EU abschliessen?"
        # example_comment = "Wir können mit den Grossbetrieben in der EU nicht konkurrenzieren. Ein Beitritt wäre das Ende für viele von unseren Bauern."
        # example_template = "La position de " + example_comment  + " envers " + example_question + " est [MASK]. Select from 'Favor' and 'Against'."
        # example_answer = "Against"
        example_templates = get_2_example(de_train, opt.index1, opt.index2)
            
        y_true = []
        y_pred = [] 
        total = len(fr_test)
        unknown_num = 0
        correct = 0 
        for i, test_template in enumerate(test_templates[opt.startpoint:], start=opt.startpoint):
            if i % 50 == 0:
                time.sleep(5)
            log.info("Sample {} result: ".format(i))
            prompt_template = test_template[0]
            label = test_template[1]
            time.sleep(1)
            log.info(prompt_template)
            predict_word = send_request_2_shot(prompt_template, example_templates) 
            if predict_word not in word_2_label.keys():
                unknown_num += 1
                # log.info("response of test sample {}th is unknown".format(i))
                continue
            predict_label = word_2_label[predict_word]
            if predict_label == label:
                correct += 1
            y_pred.append(predict_label)
            y_true.append(label)

        log.info("num of unknown is {}".format(unknown_num))
        accuarcy = correct/total
            
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        f1 = get_metrics_f1(y_true, y_pred)
            
        log.info("{} ends! ".format(sub_dataset))
        log.info("acc on {} samples: {}".format(total, 100 * accuarcy))
        log.info("f1 on {} samples: {}".format(total, 100 * f1))
        
    
            

if __name__ == "__main__":
    
    main()



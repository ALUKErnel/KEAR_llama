'''
Description: 
Version: 
Author: zrk
Date: 2023-08-14 15:53:07
LastEditors: zrk
LastEditTime: 2024-09-14 20:17:49
'''
import openai
import jsonlines
import os
import numpy as np
import time 
from data_prep.vaxxstance_dataset_retrieve_res import get_datasets_main_vaxx
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import logging
from options_v import opt


if not os.path.exists(opt.save_file):
    os.makedirs(opt.save_file)
# logging.basicConfig(stream=sys.stderr, level=logging.DEBUG if opt.debug else logging.INFO)
logging.basicConfig(level=logging.INFO if opt.local_rank in [-1, 0] else logging.WARN)
log = logging.getLogger(__name__)
fh = logging.FileHandler(os.path.join(opt.save_file, 'log_vaxx_{}_gpt4_temp_{}.txt'.format(opt.template, opt.temperature)))
log.addHandler(fh)



def get_metrics_f1(y_true, y_pred):
    average = 'macro'
    # f1 = f1_score(y_true, y_pred, average=average)
    f1_1 = f1_score(y_true == 1, y_pred == 1)
    log.info('favor f1: {}'.format(100 * f1_1))
    f1_0 = f1_score(y_true == 0, y_pred == 0)
    log.info('against f1: {}'.format(100 * f1_0))
    f1_2 = f1_score(y_true == 2, y_pred == 2)
    log.info('none f1: {}'.format(100 * f1_2))
    f1_avg = (f1_1 + f1_0 + f1_2) / 3
    log.info("classification report: {}\n".format(classification_report(y_true, y_pred, digits=4)))
    return f1_avg 


def form_template_label(test_data):
    test_templates = []
    for data in test_data:
        x, label = data
        question, comment = x

        if opt.template == "p1":
            prompt_template = "TARGET: '{}' TEXT: '{}' What is the attitude of TEXT toward TARGET? Give me a one-word answer. Select from 'Favor', 'Against' and 'None'.".format(question, comment)
        elif opt.template == "p2":
            prompt_template = "What is the attitude of '{}' toward '{}'? Give me a one-word answer. Select from 'Favor', 'Against' and 'None'.".format(comment, question)
        elif opt.template == "p3":
            prompt_template = "The attitude of '{}' toward '{}' is [MASK]. Give me a one-word answer. Select from 'Favor', 'Against' and 'None'.".format(comment, question)
        elif opt.template == "p4":
            prompt_template = "TARGET: '{}' TEXT: '{}' Zein da TEXT-ek TARGETekiko duen jarrera? Emaidazu hitz bakarreko erantzuna. Hautatu 'Favor', 'Against' eta 'None'.".format(question, comment)
        elif opt.template == "p5":
            prompt_template = "Zein jarrera adierazten du '{}' k '{}'ekiko? Emaidazu hitz bakarreko erantzuna. Hautatu 'Favor', 'Against' eta 'None'.".format(comment, question)
        elif opt.template == "p6":
            prompt_template = "'{}'-ren jarrera '{}'-rekin [MASK] da. Emaidazu hitz bakarreko erantzuna. Hautatu 'Favor', 'Against' eta 'None'".format(comment, question)
        elif opt.template == "pmix1":
            prompt_template = "TARGET: '{}' TEXT: '{}' Note that the Basque sentences are the text for stance detection, and the following English sentences are supplementary knowledge. You can consider whether to refer to English supplementary knowledge depending on the situation. What is the attitude of TEXT toward TARGET? Give me a one-word answer. Select from 'Favor', 'Against' and 'None'.".format(question, comment)
        elif opt.template == "pmix2":
            prompt_template = "TARGET: '{}' TEXT: '{}' Note that the Basque sentences are the text for stance detection, and the following English sentences are supplementary knowledge. You can consider whether to refer to English supplementary knowledge depending on the situation. For example, if the Basque sentences show no stance, you can ignore the English supplementary knowledge at all. What is the attitude of TEXT toward TARGET? Give me a one-word answer. Select from 'Favor', 'Against' and 'None'.".format(question, comment)
        elif opt.template == "pmix3":
            prompt_template = "TARGET: '{}' TEXT: '{}' What is the attitude of TEXT toward TARGET? Give me a one-word answer. Select from 'Favor', 'Against' and 'None'. Note that in TEXT, the Basque sentences are the text for stance detection, and the following English sentences are supplementary knowledge. Please pay more attention on the Basque sentences. If the Basque sentences are not sufficient to determine the stance toward TARGET, you can further refer to English supplementary knowledge. For example, if the Basque sentences show 'None' stance, you can ignore the English supplementary knowledge at all.".format(question, comment)
        elif opt.template == "pmix4":
            prompt_template = "TARGET: '{}' TEXT: '{}' What is the attitude of TEXT toward TARGET? Give me a one-word answer. Select from 'Favor', 'Against' and 'None'. Note that in TEXT, the Basque sentences are the text for stance detection, and the following English sentences are supplementary knowledge. Please pay more attention on the Basque sentences. If the Basque sentences are not relevant to Vaccine, the answer is 'None'. If the Basque sentences are relevant to Vaccine, please determine the attitude. If the Basque sentences show 'Neutral' stance, you can ignore the English supplementary knowledge at all and output 'None'. If the Basque sentences are not sufficient to determine the stance toward TARGET, you can further refer to English supplementary knowledge.".format(question, comment)
        elif opt.template == "pmix5":
            prompt_template = "TARGET: '{}' TEXT: '{}' What is the attitude of TEXT toward TARGET? Give me a one-word answer. Select from 'Favor', 'Against' and 'None'. Note that in TEXT, the Basque sentences are the text for stance detection, and the following English sentences are supplementary knowledge. Ignore the English sentences. If the Basque sentences are not relevant to Vaccine, the answer is 'None'.".format(question, comment)
        elif opt.template == "pmix6": # ignore
            prompt_template = "TARGET: '{}' TEXT: '{}' Note that in TEXT, the Basque sentences are the text for stance detection, and the following English sentences are supplementary knowledge. Ignore the English sentences. What is the attitude of Basque sentences toward TARGET? Give me a one-word answer. Select from 'Favor', 'Against' and 'None'. If the Basque sentences are not relevant to Vaccine, the answer is 'None'.".format(question, comment)
        elif opt.template == "pmix7": # no ignore
            prompt_template = "TARGET: '{}' TEXT: '{}' Note that in TEXT, the Basque sentences are the text for stance detection, and the following English sentences are supplementary knowledge. What is the attitude of Basque sentences toward TARGET? Give me a one-word answer. Select from 'Favor', 'Against' and 'None'. If the Basque sentences are not relevant to Vaccine, the answer is 'None'.".format(question, comment)
        elif opt.template == "pmix8": # more detail about knowledge
            prompt_template = "TARGET: '{}' TEXT: '{}' Note that in TEXT, the Basque sentences are the text for stance detection, and the following English sentences are supplementary knowledge. What is the attitude of Basque sentences toward TARGET? Give me a one-word answer. Select from 'Favor', 'Against' and 'None'. If the Basque sentences are not relevant to Vaccine or express neutral stance toward TARGET, the answer is 'None'. If the Basque sentences are not sufficient to determine the stance toward TARGET, you can further refer to English supplementary knowledge.".format(question, comment)
        elif opt.template == "pmix9": # ignore again
            prompt_template = "TARGET: '{}' TEXT: '{}' Note that in TEXT, the Basque sentences are the text for stance detection, and the following English sentences are supplementary knowledge. What is the attitude of Basque sentences toward TARGET? Give me a one-word answer. Select from 'Favor', 'Against' and 'None'. If the Basque sentences are not relevant to Vaccine or express neutral stance toward TARGET, the answer is 'None'. Ignore the English supplementary knowledge at all.".format(question, comment)
        elif opt.template == "pmix10": # ignore again again
            prompt_template = "TARGET: '{}' TEXT: '{}' Note that in TEXT, the Basque sentences are the text for stance detection, and the following English sentences you can ignore. What is the attitude of Basque sentences toward TARGET? Give me a one-word answer. Select from 'Favor', 'Against' and 'None'.".format(question, comment)

        
        else:
            prompt_template = "None"


        print(prompt_template)
        
        test_templates.append([prompt_template, label])

    
    return test_templates 


def send_request(template):

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
  
          
def main():
    data_dir = "./dataset_vaxxstance/" # mention the work dir

    tokenizer = ""

    es_train_file_path = os.path.join(data_dir, "es_train_data.csv")
    es_valid_file_path = os.path.join(data_dir, "es_valid_data.csv")
    es_test_file_path = os.path.join(data_dir, "es_test_data.csv")

    eu_train_file_path = os.path.join(data_dir, "eu_train_data.csv")
    eu_valid_file_path = os.path.join(data_dir, "eu_valid_data.csv")
    eu_test_file_path = os.path.join(data_dir, "eu_test_data_retrieval_res.csv") 

    max_seq_len = 500 
    num_train_lines = 0

    word_2_label = {"Favor": 1, "FAVOR": 1, "favor": 1, "In favor of": 1, "Favor.": 1, "FAVOR.": 1, "favor.": 1, "In favor of.": 1, 
              "Against": 0, "AGAINST": 0, "against": 0, "Against.": 0, "AGAINST.": 0, "against.": 0,
              "Neutral": 2, "NEUTRAL": 2, "neutral": 2, "None": 2, "NONE": 2, "none": 2,
              "Neutral.": 2, "NEUTRAL.": 2, "neutral.": 2, "None.": 2, "NONE.": 2, "none.": 2,
              }

    datasets_list  = ["vaxx"]

    for sub in datasets_list:
        sub_dataset = sub
        es_train_dataset, es_valid_dataset, es_test_dataset = get_datasets_main_vaxx("es", es_train_file_path, es_valid_file_path, es_test_file_path, tokenizer, num_train_lines, max_seq_len)
        eu_train_dataset, eu_valid_dataset, eu_test_dataset = get_datasets_main_vaxx("eu", eu_train_file_path, eu_valid_file_path, eu_test_file_path, tokenizer, num_train_lines, max_seq_len)

        log.info("Dataset: {} ".format(sub_dataset))
        test_templates = form_template_label(eu_test_dataset)
            
        y_true = []
        y_pred = [] 
        total = 0
        unknown_num = 0
        correct = 0 
        for i, test_template in enumerate(test_templates[opt.startpoint: opt.endpoint], start=opt.startpoint):
            if i % 50 == 0:
                time.sleep(10)
            log.info("Sample {} result: ".format(i))
            prompt_template = test_template[0]
            label = test_template[1]
            log.info(prompt_template)
            predict_word = send_request(prompt_template) 
            if predict_word not in word_2_label.keys():
                # 改一下这里的逻辑
                # 如果是unknown的情况，
                unknown_num += 1
                # log.info("response of test sample {}th is unknown".format(i))
                continue
            predict_label = word_2_label[predict_word]
            if predict_label == label:
                correct += 1
            y_pred.append(predict_label)
            y_true.append(label)
            total += 1

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



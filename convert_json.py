'''
Description: 
Version: 
Author: zrk
Date: 2024-08-28 22:14:20
LastEditors: zrk
LastEditTime: 2024-09-02 22:58:39
'''

import csv
import json

# CSV文件路径
csv_file_path = "./dataset_vaxxstance/eu_test_data.csv"
# JSON文件路径
json_file_path = './dataset_vaxxstance/vaxxstance_llama.json'

# 用于存储转换后的数据
data = []

with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
    csv_reader = csv.DictReader(csvfile)
    for row in csv_reader:
        # 创建一个新的字典来保存转换后的数据
        entry = {
            "instruction": "What is the attitude of Text toward Target? Give me a one-word answer. Select from 'Favor', 'Against' and 'None'.",
            "input": "Target: Vaccine, Text: {}".format(row['text']),
            "output": row['label'].lower().title()
        }
        data.append(entry)

# 将数据写入JSON文件
with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
    json.dump(data, jsonfile, ensure_ascii=False, indent=4)

print(f"CSV file has been converted to JSON and saved to {json_file_path}")



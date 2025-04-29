import json
import os
import random
import sys
sys.path.append('./')
from data_process.list_record_cache import ListRecordCacher, ListRecordLoader

if __name__ == "__main__":
    # Example of how to package data. This image is from the first sample in the CORD training set (https://github.com/clovaai/cord) for reference only.
    # You can modify the code as needed.   
    
    img_path = './data_process/toy_data/receipt_00000.png'
    json_path = './data_process/toy_data/receipt_00000.json'
    package_output_path = './data_process/output/toy.lrc'
    
    os.makedirs(os.path.dirname(package_output_path), exist_ok=True)
    writer = ListRecordCacher(package_output_path)
    for _ in range(1000):
        with open(json_path, 'r') as f:
            json_info = json.load(f)
        texts, text_polys = [], []
        for line in json_info['valid_line']:
            x1s, y1s, x2s, y2s, x3s, y3s, x4s, y4s = [], [], [], [], [], [], [], []
            text = ''
            for word in line['words']:
                x1s.append(word['quad']['x1'])
                y1s.append(word['quad']['y1'])
                x2s.append(word['quad']['x2'])
                y2s.append(word['quad']['y2'])
                x3s.append(word['quad']['x3'])
                y3s.append(word['quad']['y3'])
                x4s.append(word['quad']['x4'])
                y4s.append(word['quad']['y4'])
                text = text + ' ' + word['text']
            x1, y1, x2, y2, x3, y3, x4, y4 = min(x1s), min(y1s), max(x2s), min(y2s), max(x3s), max(y3s), min(x4s), max(y4s) 
            texts.append(text)
            text_polys.append([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])
        ########### You should remove these lines. We keep them to simulate data diversity since we only use a single data. ################
        add_ratio = random.random() * 3
        texts = texts + texts * int(add_ratio) + texts[:int((add_ratio - int(add_ratio)) * len(texts))]
        text_polys = text_polys + text_polys * int(add_ratio) + text_polys[:int((add_ratio - int(add_ratio)) * len(text_polys))]
        ####################################################################
        format_data = {
            'image_path':img_path, 'origin_image_size':[json_info['meta']['image_size']['height'], json_info['meta']['image_size']['width']], 'resize_image_size':[json_info['meta']['image_size']['height'], json_info['meta']['image_size']['width']], 'text_polys':text_polys, 'texts':texts
        }
        writer.add_record(format_data)
    
    writer.close()

    # After packaging, you can read data as follows:
    reader = ListRecordLoader(package_output_path)
    for i in range(1000):
        data = reader.get_record(i)

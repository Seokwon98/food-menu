from transformers import BertTokenizerFast
from transformers import TFBertForSequenceClassification
from transformers import TextClassificationPipeline
import tensorflow as tf
import tensorflow as tf
import pandas as pd
import numpy as np
import random
include = ['포함해', '추천', '좋아', '먹고싶', '땡겨', '땡긴', '생각나', '먹을래', '어때']
exclude = ['빼고', '빼서', '제외', '싫', '안먹', '먹고싶지않', '부담스', '좋아하지않', '피하고싶']
randomize = ['아무거나', '랜덤']
data = pd.read_csv("home/ubuntu/flask_model.py", names=['menu', 'content', 'label'], encoding='cp949', header=0)
menu_names = data.menu.unique()


from flask import Flask

app = Flask(__name__)

import json

from flask import Response

from functools import wraps
    
path = "C:/Users/vmfhr/230921 메뉴추천 모델 불러오기/230920 메뉴추천모델" # 여기 모델 주소 입력
loaded_tokenizer = BertTokenizerFast.from_pretrained(path) 
loaded_model = TFBertForSequenceClassification.from_pretrained(path) 

def as_json(f):
    @wraps(f)

    def decorated_function(*args, **kwargs):
        res = f(*args, **kwargs)
        res = json.dumps(res, ensure_ascii=False).encode('utf8')
        return Response(res, content_type='application/json; charset=utf-8')
    
    return decorated_function

@app.route('/')
@as_json
def home():
    input_sentence = '데이트 할 때 뭐 먹지'
    text_classifier = TextClassificationPipeline(tokenizer=loaded_tokenizer,
                                              model=loaded_model,
                                              framework='tf',
                                              return_all_scores=True)
    
    
#     score_list = [i['score'] for i in text_classifier(input_sentence)[0]]
#     indices_of_max_values = np.argsort(score_list)[-10:][::-1]
    
#     # label index 를 data에서 찾아서 해당하는 메뉴명 저장
#     menu = []
#     for i in indices_of_max_values:
#         menu.append(data[data['label'] == i].iloc[0]['menu'])



# # 입력 문장에 메뉴명에 해당하는 단어가 있다면, 추가/배제/랜덤화 규칙 적용
#     nor_sentence = input_sentence.replace(' ', '')
#     for f in menu_names:
#         if f in input_sentence.replace(' ', ''):
#         # 추가
#             for x in include:
#                 if x in nor_sentence:
#                     menu.insert(0, f)
#         # 삭제
#             for y in exclude:
#                 if y in nor_sentence:
#                     menu = list(filter(lambda x:x!=f, menu))
# # 아무거나                
#     for z in randomize:
#         if z in nor_sentence:
#             menu = random.sample(list(menu_names),10)

#     return menu

    score_list = [i['score'] for i in text_classifier(input_sentence)[0]]
    indices_of_max_values = np.argsort(score_list)[-10:][::-1]
    menu = [data.iloc[i]['menu'] for i in indices_of_max_values][0]  # 첫 번째 메뉴 선택

    nor_sentence = input_sentence.replace(' ', '')
    menu_set = set(menu)  # 메뉴를 집합으로 변환하여 중복 확인

    # 규칙 적용
    for f in menu_names:
        if f in nor_sentence:
            for x in include:
                if x in nor_sentence and f not in menu_set:
                    menu = f
                    menu_set = {menu}  # 메뉴를 설정하고 집합 업데이트
            for y in exclude:
                if y in nor_sentence and f == menu:
                    menu = ''  # 메뉴 제거

    for z in randomize:
        if z in nor_sentence:
            random_menu = random.sample(list(menu_names), 10)
            if menu in random_menu:
                random_menu.remove(menu)  # 중복 메뉴 제거
            menu = random_menu[0]  # 첫 번째 무작위 메뉴 선택

    if not menu:
        menu = '기본 메뉴'  # 메뉴가 없을 경우 기본 메뉴 설정

    return menu


if __name__ == '__main__':
   app.run('0.0.0.0', port=5000, debug=True)

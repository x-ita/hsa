from fastapi import FastAPI
import torch
from transformers import MLukeTokenizer, LukeModel
import pickle

# インスタンス化
app = FastAPI()

# 埋め込みモデルの読み込み
with open('models/sentence-luke-japanese-base-lite', 'rb') as f:
    model = pickle.load(f)

# トップページ
@app.get('/')
def index():
    return {"Iris": 'iris_prediction'}

# POST が送信された時（入力）と予測値（出力）の定義
@app.post('/predict')
def make_predictions(features: iris):
    return({'prediction':model.encode(text)})

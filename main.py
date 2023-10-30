import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.embeddings import OpenAIEmbeddings
import pandas as pd
import numpy as np
import pickle

# OpenAI API KEY設定
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# チャンクDataFrame読み込み
chunk_df = pd.read_pickle('chunk_df.pkl')

# Vector DBアレイ読み込み
with open('vectordb_array.pkl', 'rb') as f:
  vectordb_array = pickle.load(f)
    
# インスタンス化
app = FastAPI()

# 入力するデータ型の定義
class input_text(BaseModel):
    text: str

# トップページ
@app.get('/')
def index():
    return {"Iris": 'iris_prediction'}

# POST が送信された時（入力）と予測値（出力）の定義
@app.post('/search_similar')
def search_similar(query: input_text):
    # cosine類似度を計算する関数(sklearnでも可)
    def cosine_similarity(matrix1, matrix2):
        # 各行列のL2ノルム（ユークリッド距離）を計算
        norm_matrix1 = np.linalg.norm(matrix1, axis=1, keepdims=True)
        norm_matrix2 = np.linalg.norm(matrix2, axis=1, keepdims=True)
        # ベクトルの内積を計算
        dot_product = np.dot(matrix1, matrix2.T)
        # コサイン類似度を計算
        return dot_product / (norm_matrix1 * norm_matrix2.T)

    query_embed_list = embeddings.embed_query(query.text)
    query_array = np.array(query_embed_list).reshape(1, 1536)
    similarity = cosine_similarity(query_array, vectordb_array)[0]
    sorted_df = chunk_df.assign(similarity=similarity).sort_values('similarity', ascending=False)
    search_results_json = sorted_df.head(3).to_json()
    
    return search_results_json

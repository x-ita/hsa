from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle

# 埋め込みモデルのダウンロード


# チャンクデータの読み込み
#chunk_df = pd.read_pickle('chunk_df.pkl')

# Vector DBの読み込み
#with open('vecdb.pkl', 'rb') as f:
#    vecdb = pickle.load(f) 

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
    def cosine_similarity(matrix1, matrix2):
        # 各行列のL2ノルム（ユークリッド距離）を計算
        norm_matrix1 = np.linalg.norm(matrix1, axis=1, keepdims=True)
        norm_matrix2 = np.linalg.norm(matrix2, axis=1, keepdims=True)
        # ベクトルの内積を計算
        dot_product = np.dot(matrix1, matrix2.T)
        # コサイン類似度を計算
        return dot_product / (norm_matrix1 * norm_matrix2.T)
    
#    similarity = cosine_similarity(model.encode([query.text]).detach().clone().numpy(), vecdb)[0]
#    sorted_df = chunk_df.assign(similarity=similarity).sort_values('similarity', ascending=False)
#    return sorted_df.head(3).to_json(orient='records')
    return pd.DataFrame({'a':['aaa', 'bbb'], 'b':[query.text, query.text]}).to_json(orient='records')

import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pandas as pd
import numpy as np
import pickle

# OpenAIEmbeddingsインスタンス作成
embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

# チャンクDataFrame読み込み
chunk_df = pd.read_pickle('chunk_df.pkl')

# 計算済みVector DBアレイ読み込み
with open('vectordb_array.pkl', 'rb') as f:
  vectordb_array = pickle.load(f)

# LLMChainインスタンス作成
prompt = PromptTemplate(
  template="""与えられたテキストの内容に基づいて質問に回答してください．### テキスト\n{context}\n### 質問:{question}""",
  input_variables=["context", "question"]
)
llm_chain = LLMChain(prompt=prompt, llm=OpenAI()) 

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
    # cosine類似度を計算する関数
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
    results_df = chunk_df.assign(similarity=similarity)
    # 類似度上位3件のみ
    results_df = results_df.sort_values('similarity', ascending=False).head(3)
    # チャンクに基づく質問応答
    ans_list = []
    for i in range(3):
      ans = llm_chain.run(context=response_df['chunk'].iloc[i], question=query.text)
      ans_list.append(ans)
    # 結果をJSONにして返す
    results_df = results_df.assign(answer=ans_list)    
    search_results_json = results_df.to_json()
    
    return search_results_json

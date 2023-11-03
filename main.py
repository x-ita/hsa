from fastapi import FastAPI
from pydantic import BaseModel
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.utils.math import cosine_similarity
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
chat_model = ChatOpenAI() # gpt3.5-turbo??
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "あなたは優秀なAIアシスタントです．"),
    ("human", "与えられたテキストの内容に基づいて質問に回答してください．\n" + \
#              "ただし情報が不十分な場合はわからないと回答すること．\n" + \
              "### テキスト\n{context}\n### 質問:{question}"),
])
llm_chain = LLMChain(
    llm=chat_model, 
    prompt=prompt_template
)

# FastAPIインスタンス化
app = FastAPI()

# 入力するデータ型の定義
class input_text(BaseModel):
    text: str

# トップページ
@app.get('/')
def index():
    return {"ファイル検索": '質問に対してファイルを検索しその内容に基づいて回答する'}

# POST が送信された時
@app.post('/search_qa')
def search_qa(query: input_text):
    # 入力文章をベクトル化
    query_embed_list = embeddings.embed_query(query.text)
    query_array = np.array(query_embed_list).reshape(1, 1536)
    # Vector DBに対して類似度を計算
    similarity = cosine_similarity(query_array, vectordb_array)[0]
    results_df = chunk_df.assign(similarity=similarity)
    # 類似度上位3件のみ
    results_df = results_df.sort_values('similarity', ascending=False).head(3)
    # 上位3件それぞれについてチャンクに基づく質問応答
    ans_list = []
    for i in range(3):
      ans = llm_chain.run(context=results_df['chunk'].iloc[i], question=query.text)
      ans_list.append(ans)
    # 結果をJSONにして返す
    results_df = results_df.assign(answer=ans_list)    
    results_json = results_df.to_json()
    
    return results_json

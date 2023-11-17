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
import json

# OpenAIEmbeddingsインスタンス作成
embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

# チャンクDataFrame読み込み
chunk_df = pd.read_pickle('summary_chunk_df.pkl')

# 検索対象の埋め込みベクトル（np.ndarray）読み込み
with open('summary_chunk_embeddings_array.pkl', 'rb') as f:
    embed_array = pickle.load(f)

# LLMChainインスタンス作成
llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "あなたは優秀なAIアシスタントです．"),
    ("human", "下記テキストに基づいて質問に回答してください．" + \
              "必要な情報がテキストに書かれていない場合はわからないと答えること．\n" + \
              "### テキスト\n{context}\n### 質問\n{question}"),
])
llm_chain = LLMChain(
    llm=llm, 
    prompt=prompt_template
)

# FastAPIインスタンス化
app = FastAPI()

# 入力するデータ型の定義
class question_kw(BaseModel):
    question: str
    kw: str

class context_question(BaseModel):
    context: str
    question: str

# トップページ
@app.get('/')
def index():
    return {"ファイル検索": '質問に対してファイルを検索しその内容に基づいて回答する'}

# POST が送信された時
@app.post('/vector_kw_search')
def vector_kw_search(query: question_kw):
    question = query.question
    kw = query.kw
    # キーワードを含むチャンクを選択
    if kw != '': # キーワード欄に入力がある場合はキーワード検索する
        tf = chunk_df['text'].str.contains(kw).to_numpy()    
        chunk_df_filtered = chunk_df[tf]
        embed_array_filtered = embed_array[tf, :]
    else: # キーワード欄が未入力の場合
      chunk_df_filtered = chunk_df.copy()
      embed_array_filtered = embed_array.copy()
    # 質問文をベクトル化
    query_embed_list = embeddings.embed_query(question)
    query_array = np.array(query_embed_list).reshape(1, 1536)
    # ベクトルストアに対してベクトル類似度を計算
    similarity = cosine_similarity(query_array, embed_array_filtered)[0]
    results_df = chunk_df_filtered.assign(similarity=similarity)
    # 類似度上位3件のみ
    results_df = results_df.sort_values('similarity', ascending=False).head(3)

    # 結果をJSONで返す
    return results_df.to_json()

@app.post('/llm_qa')
def llm_qa(query: context_question):
    answer_text = llm_chain.run(
        context=query.context,
        question=query.question
    )
    # 結果をJSONで返す
    results_json = json.dumps({'answer': answer_text})
    return results_json

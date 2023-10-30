import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import pickle
import json

# OpenAI API KEY設定
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Vector DB読み込み
#db_dir = './chroma_db/'
#vectordb = Chroma(persist_directory=db_dir, embedding_function=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))

# チャンクDocument読み込み
with open('docs.pkl', 'rb') as f:
    docs = pickle.load(f)

# Vector DB作成
vectordb = Chroma.from_documents(
    documents=docs[:5],
    embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
)

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
    search_results = vectordb.similarity_search_with_relevance_scores(query.text, k=3)
    search_results_json = json.dumps([
        {
            'title_author':res[0].metadata['title_author'],
            'chunk':res[0].page_content,
            'similarity':round(res[1], 3)
        } for res in search_results])
    return search_results_json

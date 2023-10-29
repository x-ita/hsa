import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# OpenAI API KEY設定
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Vector DB読み込み
db_dir = '/content/drive/MyDrive/aozorabunko/chroma_db/'
vectordb = Chroma(persist_directory=db_dir, embedding_function=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))

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
    search_results = vectordb.similarity_search_with_score(query.text)
#    similarity = cosine_similarity(model.encode([query.text]).detach().clone().numpy(), vecdb)[0]
#    sorted_df = chunk_df.assign(similarity=similarity).sort_values('similarity', ascending=False)
#    return sorted_df.head(3).to_json(orient='records')
    return pd.DataFrame({'a':['aaa', 'bbb'], 'b':[query.text, query.text]}).to_json(orient='records')

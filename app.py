import streamlit as st
import pandas as pd
import requests

fastapi_url = st.secrets['FASTAPI_URL']

st.title('文章でファイル検索')
st.markdown(
    '青空文庫の昔話・童話の中から，以下の10作品を検索対象ファイルにしました．  \n'
    '[一寸法師](https://www.aozora.gr.jp/cards/000329/files/43457_23938.html)，'
    '[花咲かじじい](https://www.aozora.gr.jp/cards/000329/files/3391.html)，'
    '[浦島太郎](https://www.aozora.gr.jp/cards/000329/files/3390_33153.html)，'
    '[猿かに合戦](https://www.aozora.gr.jp/cards/000329/files/18334_11947.html)，'
    '[桃太郎](https://www.aozora.gr.jp/cards/000329/files/18376_12100.html)，'
    '[瘤とり](https://www.aozora.gr.jp/cards/000329/files/43461_23937.html)，'
    '[金太郎](https://www.aozora.gr.jp/cards/000329/files/18337_11942.html)，'
    '[赤ずきんちゃん](https://www.aozora.gr.jp/cards/001091/files/42311_15546.html)，'
    '[マッチ売りの少女](https://www.aozora.gr.jp/cards/000019/files/194_23024.html)，'
    '[シンデレラ](https://www.aozora.gr.jp/cards/001239/files/46348_23182.html)'
    )

input_question = st.text_input('上記の昔話・童話の内容についての質問を入力してください．')
input_kw = st.text_input('キーワードを入力してください（任意）．')

input_dict = {
    'question': input_question,
    'kw': input_kw
　　}

if st.button('Submit'):
  # 類似度計算を実行し上位3件を取得(FastAPI)
  response = requests.post(fastapi_url, json=input_dict) # 引数jsonでなぜかdict型を渡す
  response_df = pd.read_json(response.json(), orient="records")
  # チャンクに基づく質問応答の表示
  for i, row in response_df.iterrows():
    st.write('\n\n回答 ' + str(i+1) + '：  \n' + row['answer'])
    st.write('\n\nファイル（作品）：\n' + row['title_author'])
    st.write('\n\n類似度：\n' + str(round(row['similarity'], 3)))
    st.write('\n\nテキスト：\n' + row['text'])
  

import streamlit as st
import pandas as pd
import requests

st.title('文章でファイル検索')
st.write('青空文庫の中谷宇吉郎の作品の中から，「雪の結晶」を含むものを検索対象ファイルにしました．')

text_input = st.text_input('テキストを入力')

text_input_json = {
  'text': text_input
}

if st.button('Submit'):
  # 類似度計算の実行
  response = requests.post("https://homsa.onrender.com/search_similar", json=text_input_json)
  response_df = pd.read_json(response.json(), orient="records")
  
  # 類似度計算結果上位3件の表示
  st.write('### Similar Files')
  st.dataframe(response_df)

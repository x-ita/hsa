import streamlit as st
import pandas as pd
import requests

st.title('Iris Classifier')

text_input = st.text_input('テキストを入力')

text_input_json = {
  'text': text_input
}

if st.button('Submit'):
  # 類似度計算の実行
  response = requests.post("https://homsa.onrender.com/search_similar", json=text_input_json)
  response_json = response.json()
  response_df = pd.read_json(response_json, orient="records")
  
  # 類似度計算結果上位3件の表示
  st.write('### Similar Files')
  st.dataframe(response_df)

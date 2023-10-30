import streamlit as st
import pandas as pd
import requests

st.title('Iris Classifier')

text_input = st.text_input('', '')
st.button('Submit')

text_input_json = {
  'query': text_input
}

# 類似度計算の実行
response = requests.post("http://localhost:8000/search_similar", json=text_input_json)
response_json = response.json()
response_df = pd.read_jsonresponse_jsonorient="records")

# 類似度計算結果上位3件の表示
st.write('## Similar Files')
st.dataframe(response_df)

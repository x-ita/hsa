import streamlit as st
#import pandas as pd
import requests

st.title('Iris Classifier')

st.text_input('', '')
st.button('Submit')

# 予測の実行
#response = requests.post("http://localhost:8000/predict", json=iris)
#prediction = response.json()["prediction"]

# 予測結果の表示
#st.write('## Prediction')
#st.write(prediction)

# 予測結果の出力
#st.write('## Result')
#st.write('このアイリスはきっと',str(targets[int(prediction)]),'です!')

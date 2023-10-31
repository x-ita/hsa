import streamlit as st
import pandas as pd
import requests

st.title('文章でファイル検索')
st.write('青空文庫の昔話・童話の中から，以下の18作品を検索対象ファイルにしました．  \nかちかち山，一寸法師，花咲かじじい，浦島太郎，舌切りすずめ，猿かに合戦，桃太郎，瘤とり，金太郎，瓜子姫子，文福茶がま，ジャックと豆の木（以上楠山正雄），星の銀貨，赤ずきんちゃん，おおかみと七ひきのこどもやぎ，ヘンゼルとグレーテル（以上グリム兄弟 楠山正雄訳），桃太郎（芥川龍之介），お伽草紙（太宰治）')

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

import streamlit as st
import pandas as pd
import requests

fastapi_url = st.secrets['FASTAPI_URL']

st.title('文章でファイル検索')
st.write('青空文庫の昔話・童話の中から，以下の18作品を検索対象ファイルにしました．  \n' + \
         'かちかち山，一寸法師，花咲かじじい，浦島太郎，舌切りすずめ，猿かに合戦，' + \
         '桃太郎，瘤とり，金太郎，瓜子姫子，文福茶がま，ジャックと豆の木（以上楠山正雄），' + \
         '星の銀貨，赤ずきんちゃん，おおかみと七ひきのこどもやぎ，ヘンゼルとグレーテル（以上グリム兄弟 楠山正雄訳），' + \
         '桃太郎（芥川龍之介），お伽草紙（太宰治）')

text_input = st.text_input('質問を入力してください．')

text_input_json = {
  'text': text_input
}

if st.button('Submit'):
  # 類似度計算を実行し上位3件を取得(FastAPI)
  response = requests.post(fastapi_url, json=text_input_json)
  response_df = pd.read_json(response.json(), orient="records")
  # チャンクに基づく質問応答の表示
  for i in range(3):
    st.write('\n\n回答' + str(i+1) + '：\n' + response_df['answer'].iloc[i])
    st.write('\n\nファイル（作品）' + str(i+1) + '：\n' + response_df['title_author'].iloc[i])
    st.write('\n\nテキスト' + str(i+1) + '：\n' + response_df['chunk'].iloc[i])
  

import streamlit as st
import pandas as pd
import requests

fastapi_url = st.secrets['FASTAPI_URL']

st.title('文章でファイル検索')
st.write('青空文庫の昔話・童話の中から，以下の10作品を検索対象ファイルにしました．  \n' + \
         '一寸法師，花咲かじじい，浦島太郎，猿かに合戦，桃太郎，瘤とり，金太郎（以上楠山正雄），' + \
         '赤ずきんちゃん（グリム兄弟 楠山正雄訳），' + \
         'マッチ売りの少女（ハンス・クリスチャン・アンデルセン 矢崎源九郎訳），' + \
         'シンデレラ（アンドルー・ラング 大久保ゆう訳）'
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
  for i, row in enumerate(response_df.itertuples()):
    st.write('\n\n回答 ' + str(i+1) + '：\n' + row.answer)
    st.write('\n\nファイル（作品）：\n' + row.title_author)
    st.write('\n\n類似度：\n' + str(round(row.similarity, 3)))
    st.write('\n\nテキスト：\n' + row.text)
  

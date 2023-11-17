import streamlit as st
import pandas as pd
import requests
import json

fastapi_url = st.secrets['FASTAPI_URL']

st.markdown(
    '### ファイル検索チャットボット  \n'
    '青空文庫の昔話・童話の中から，以下の10作品を検索対象ファイルにしました．  \n'
    '[一寸法師](https://www.aozora.gr.jp/cards/000329/files/43457_23938.html)，'
    '[浦島太郎](https://www.aozora.gr.jp/cards/000329/files/3390_33153.html)，'
    '[金太郎](https://www.aozora.gr.jp/cards/000329/files/18337_11942.html)，'
    '[瘤とり](https://www.aozora.gr.jp/cards/000329/files/43461_23937.html)，'
    '[猿かに合戦](https://www.aozora.gr.jp/cards/000329/files/18334_11947.html)，'
    '[花咲かじじい](https://www.aozora.gr.jp/cards/000329/files/3391.html)，'
    '[桃太郎](https://www.aozora.gr.jp/cards/000329/files/18376_12100.html)，'
    '[赤ずきんちゃん](https://www.aozora.gr.jp/cards/001091/files/42311_15546.html)，'
    '[シンデレラ](https://www.aozora.gr.jp/cards/001239/files/46348_23182.html)，'
    '[マッチ売りの少女](https://www.aozora.gr.jp/cards/000019/files/194_23024.html)  \n  \n'
    '上記の昔話・童話の内容についての質問を入力してください．'
    '本文の要約または本文を500字以内に分割したテキストをベクトル検索し，'
    '類似度上位3件について質問に対する回答を生成します．'
)

input_question = st.text_input(
    '例：「おじいさんが飼っていた犬の名前は？」「誰と誰が結婚しましたか？」'
)
input_kw = st.text_input(
    'キーワードを入力してください（任意）．'
    'キーワードを含むテキストのみベクトル検索の対象になります．'
)

input_question_kw = {
    'question': input_question,
    'kw': input_kw
}

if st.button('Submit'):
    # 類似度計算を実行し上位3件を取得(FastAPI)
    response = requests.post(fastapi_url + 'vector_kw_search', json=input_question_kw) # 引数jsonになぜかdict型を渡す
    response_df = pd.DataFrame(json.loads(response.json())).reset_index(drop=True)
    # チャンクに基づく質問応答の表示
    for i, row in response_df.iterrows():
        input_context = row['text']
        input_context_question = {
            'context': input_context,
            'question': input_question
        }
        response = requests.post(fastapi_url + 'llm_qa', json=input_context_question)
        answer_text = json.loads(response.json())['answer']
        st.markdown('##### 回答' + str(i+1) + ':  ' + answer_text)
        st.write('テキスト：\n' + input_context)
        st.write('ファイル（作品）：\n' + row['title_author'])
        st.write('類似度：\n' + str(round(row['similarity'], 3)))


  

from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import MLukeTokenizer, LukeModel
import pickle

# 埋め込みモデルのダウンロード（2回目以降はキャッシュから読み込み？）
# https://huggingface.co/sonoisa/sentence-luke-japanese-base-lite
class SentenceLukeJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = MLukeTokenizer.from_pretrained(model_name_or_path)
        self.model = LukeModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest",
                                           truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        return torch.stack(all_embeddings)


MODEL_NAME = "sonoisa/sentence-luke-japanese-base-lite"
#model = SentenceLukeJapanese(MODEL_NAME)

# チャンクデータの読み込み
#chunk_df = pd.read_pickle('chunk_df.pkl')

# Vector DBの読み込み
#with open('vecdb', 'rb') as f:
#    vecdb = pickle.load(f) 

def cosine_similarity(matrix1, matrix2):
    # 各行列のL2ノルム（ユークリッド距離）を計算
    norm_matrix1 = np.linalg.norm(matrix1, axis=1, keepdims=True)
    norm_matrix2 = np.linalg.norm(matrix2, axis=1, keepdims=True)
    # ベクトルの内積を計算
    dot_product = np.dot(matrix1, matrix2.T)
    # コサイン類似度を計算
    return dot_product / (norm_matrix1 * norm_matrix2.T)

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
@app.post('/text_similarity')
def text_similarity(input: input_text):
    return({'similar_text': input_text.text + '___ok'})

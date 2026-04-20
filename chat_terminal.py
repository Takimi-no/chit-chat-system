# コマンドライン引数を受けるためのもの
import argparse
# Pytorch：深層学習ライブラリ
import torch
# AutoModelForCausalLM：文章生成モデルを構築するためのもの
# BitsAndBytesConfig：4bit量子化の設定を行うためのもの
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# ベースモデルにLoRAを重ねるために利用
from peft import PeftModel

DEFAULT_BASE_MODEL = "llm-jp/llm-jp-3-13b-instruct3"

# history：会話履歴
# user_message：今回の入力
# 学習の形式でこの形式を利用
def build_input(history, user_message):
    parts = []
    for user,assistant in history:
        parts.append(f"User: {user}\nAssistant: {assistant}")
    parts.append(f"User: {user_message}\nAssistant:")
    # 複数の文字列(User:〜〜Assistant:〜〜)を\nを間に挟むことで1つの文字列にまとめている。
    return "\n".join(parts)

# tokenizerとモデルのロード
# adapter_pathからtokenizerを獲得
def load_model_and_tokenizer(adapter_path, base_model_name, load_in_4bit=True):
    # トークナイザー処理
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast = False)
    # eos_token：文の終わりを示すトークン
    # pad_token：長さを調整するためのトークン
    # pad_tokenがなければ、eos_tokenをpad_tokenの代わりに利用する
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # CUDA(GPUのチェック)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA の設定ミス。GPUが見つかっていない")
    
    # モデル処理
    # モデルを軽く読むための定番設定
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit = True,    # モデル4bit読み込み
            bnb_4bit_compute_dtype = torch.bfloat16,    # 計算は16bit
            bnb_4bit_use_double_quant = True,   # 効率化
            bnb_4bit_quant_type = "nf4",    # 4bit量子化
        )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,    # モデル指定
        quantization_config = quantization_config,  # 4bit読み込み
        device_map = "auto",    # GPU自動割り当て
        torch_dtype = torch.bfloat16,   # 計算型をbfloat16に
    )
    
    # 上のbaseモデルの設定に基づいて、モデル読み込み＋LoRA差分の適用
    model = PeftModel.from_pretrained(base_model, adapter_path)
    # 学習時のパラメータ調整などをやめて推論モードにする
    model.eval()
    return model, tokenizer
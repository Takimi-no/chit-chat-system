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
def build_input(history, user_message):
    parts = []
    for user,assistant in history:
        parts.append(f"User: {user}\nAssistant: {assistant}")
    parts.append(f"User: {user_message}\nAssistant:")
    # 複数の文字列(User:〜〜Assistant:〜〜)を\nを間に挟むことで1つの文字列にまとめている。
    return "\n".join(parts)
        
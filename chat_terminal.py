# コマンドライン引数を受けるためのもの
import argparse
# Pytorch：深層学習ライブラリ
import torch
# AutoModelForCausalLM：文章生成モデルを構築するためのもの
# BitsAndBytesConfig：4bit量子化の設定を行うためのもの
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# ベースモデルにLoRAを重ねるために利用
from peft import PeftModel


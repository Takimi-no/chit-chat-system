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

# トークン規制
def build_bad_words_ids(tokenizer):
    vocab = tokenizer.get_vocab()
    bad_words = []
    for tok, tid in vocab.items():
        if any("\uE000" <= c <= "\uF8FF" for c in tok):
            bad_words.append(tid)
    return bad_words

def clean_pred(pred:str) -> str:
    # 後処理：許容文字以外が出力されたらそこで打ち切る。
    
    out = []
    for c in pred:
        o = ord(c)
        if c in ("\n", "\t"):
            out.append(c)
            continue
        # ASCII
        if 0x20 <= o <= 0x7E:
            out.append(c)
            continue
        # ひらがな、カタカナ
        if 0x3040 <= o <= 0x30FF:
            out.append(c)
            continue
        # 漢字
        if 0x4E00 <= o <= 0x9FFF:
            out.append(c)
            continue
        # CJK記号、句読点
        if 0x3000 <= o <= 0x303F:
            out.append(c)
            continue
        # 全角英語・記号
        if o == 0x3000 or (0xFF01 <= o <= 0xFF60) or (0xFFE0 <= o <= 0xFFEE):
            out.append(c)
            continue
        # それ以外は打ち切る
        break
    return "".join(out)

# tokenizerとモデルのロード
# adapter_pathからtokenizerを獲得
def load_model_and_tokenizer(adapter_path, base_model_name, load_in_4bit=True):
    # トークナイザー処理
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast = True)
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

# 入力文字列をトークン化して、モデルのあるデバイスに移す
def generate_reply(
    model,  # load_model_and_tokenizer関数にて作成したmodel(事前学習モデル+LoRA)とtokenizer
    tokenizer,
    prompt, # 今回入力する文章を指す
    max_new_tokens = 128,   # 新しく最大何トークンまで生成するか
    temperature = 0.8,  # 返答のランダムさ
    top_p = 0.9,    # 確率の合計が0.9以上になる範囲を候補とする
    repetition_penalty = 1.1,   # 同じ単語や表現の繰り返しを抑える
):
    '''
    1. prompt(入力の文章)をtokenizerを通してID列化
    2. model.generate()で後続のトークン生成
    3. 入力部分を除いて、生成された文章だけ取り出す
    4. 取り出した文章を文字列に戻す
    5. 余計に生成されたUser以降を削除する
    '''
    # retun_tensors：結果をどのような形式で返すか。pt：Pytorch Tensor
    # .to()：作った入力をモデルがあるGPUデバイスに移動。無くても問題がないような気もするが、念のため。
    inputs = tokenizer(prompt,return_tensors = "pt").to(model.device)
    
    # with torch.no_grad：勾配を計算しないモード
    with torch.no_grad():
        # output_ids：出力のid列
        # model.generate：文章の続きを自動生成する関数(Decoder型ならでは)
        output_ids = model.generate(
            # pt型で取得したinput(pt型、辞書みたいなもの)の引数として展開している。
            # 入力文字列を1つのid列として展開している。
            # attentionmask：paddingの位置を示すもの
            **inputs,
            max_new_tokens = max_new_tokens,
            # 次単語を確率的に選ぶ
            do_sample = True,
            # ランダム具合の強さ
            temperature = temperature,
            top_p = top_p,
            repetition_penalty = repetition_penalty,
            pad_token_id = tokenizer.pad_token_id,
            eos_token_id = tokenizer.eos_token_id,
        )
        
    # output_idsは入力+出力が含まれる
    # transformerの使用でどうしても2次元になってしまう
    # gererated_idsに、入力の文字より後の要素(出力文)を代入
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    # skip_special_tokens：特殊トークンを明示しない設定
    # 出力id列をtokenizerを通して、文字列に変換
    text = tokenizer.decode(generated_ids, skip_special_tokens = True)
    
    # 例外処理：User：が出た時点で区切る
    # 出力発話以降にさらに追加で作成してしまう事象を抑える
    '''
    例)
    こんにちは！
    User: 元気？
    Assistant: 元気です！
    '''
    if "\nUser:" in text:
        text = text.split("\nUser:")[0]

    # 前後の空白・タブを削除して返す
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="/home/takisan/takisan/storage/model/chitchat_fordeim/mix_s2/epoch_2",
        help="LoRA adapter のパス。例: mix_s2, epoch_3, checkpoint-2052",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help="ベースモデル名",
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument(
        "--max_turns",
        type=int,
        default=6,
        help="履歴として残す最大ターン数",
    )
    parser.add_argument(
        "--no_4bit",
        action="store_true",
        help="4bit量子化を使わない（VRAM多めに必要）",
    )
    args = parser.parse_args()

    print("モデル読み込み中...")
    model, tokenizer = load_model_and_tokenizer(
        adapter_path=args.adapter_path,
        base_model_name=args.base_model,
        load_in_4bit=not args.no_4bit,
    )
    print("読み込み完了")
    print("終了: /exit")
    print("履歴リセット: /reset")
    print()

    history = []

    while True:
        try:
            user_message = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n終了します。")
            break

        if not user_message:
            continue

        if user_message == "/exit":
            print("終了します。")
            break

        if user_message == "/reset":
            history = []
            print("履歴をリセットしました。")
            continue

        prompt = build_input(history[-args.max_turns:], user_message)

        try:
            reply = generate_reply(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
        except RuntimeError as e:
            print(f"生成中にエラー: {e}")
            continue

        print(f"Bot > {reply}\n")
        history.append((user_message, reply))
        


if __name__ == "__main__":
    main()
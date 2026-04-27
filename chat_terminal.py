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
            bad_words.append([tid])
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
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast = False)
    # eos_token：文の終わりを示すトークン
    # pad_token：長さを調整するためのトークン
    # pad_tokenがなければ、eos_tokenをpad_tokenの代わりに利用する
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # 文章の左側にpad_tokenをつける設定
    
    bad_words_ids = build_bad_words_ids(tokenizer)
    print("private-useのトークン数:", len(bad_words_ids))
    
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
    return model, tokenizer,bad_words_ids

# 入力文字列をトークン化して、モデルのあるデバイスに移す
def generate_reply(
    model,  # load_model_and_tokenizer関数にて作成したmodel(事前学習モデル+LoRA)とtokenizer
    tokenizer,
    prompt, # 今回入力する文章を指す
    bad_words_ids = None,   # 生成させたくないトークンのIDリスト
    max_new_tokens = 128,   # 新しく最大何トークンまで生成するか
    repetition_penalty = 1.1,   # 同じ単語や表現の繰り返しを抑える
    no_repeat_ngram_size = 3,   # 同じn-gramの繰り返しを抑える(n=3なら、同じ3単語の連続を抑える
    debug = False,
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
    inputs = tokenizer(
        prompt,
        return_tensors = "pt",
        truncation = True,
        max_length = 512,
    )

    # llm-jp は token_type_ids を受け取らない
    if "token_type_ids" in inputs:
        inputs.pop("token_type_ids")

    inputs = inputs.to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    if repetition_penalty and repetition_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = repetition_penalty
    if no_repeat_ngram_size and no_repeat_ngram_size > 0:
        gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
    if bad_words_ids:
        gen_kwargs["bad_words_ids"] = bad_words_ids
    
    # with torch.no_grad：勾配を計算しないモード
    with torch.no_grad():
        # output_ids：出力のid列
        # model.generate：文章の続きを自動生成する関数(Decoder型ならでは)
        output_ids = model.generate(
            # pt型で取得したinput(pt型、辞書みたいなもの)の引数として展開している。
            # 入力文字列を1つのid列として展開している。
            # attentionmask：paddingの位置を示すもの
            **inputs,
            **gen_kwargs,
        )
        
    # 入力部分を除いて、新規生成分だけ取る
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    
    pred = tokenizer.decode(generated_ids, skip_special_tokens=True)

    if debug:
        print("==== FULL OUTPUT (repr) ====")
        print(repr(pred))
        print("================================")
        
    pred = pred.split("\nUser:",1)[0]

    # 念のため先頭の "Assistant:" も消す
    if pred.startswith("Assistant:"):
        pred = pred[len("Assistant:"):]

    # 異常文字が出たらそこで打ち切る
    pred = clean_pred(pred).strip()
    return pred


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
    # parser.add_argument("--temperature", type=float, default=0.8)
    # parser.add_argument("--top_p", type=float, default=0.9)
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
    model, tokenizer,bad_words_ids= load_model_and_tokenizer(
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
                # temperature=args.temperature,
                # top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                bad_words_ids=bad_words_ids,
                no_repeat_ngram_size=3,
                debug=False,
            )
        except RuntimeError as e:
            print(f"生成中にエラー: {e}")
            continue

        print(f"Bot > {reply}\n")
        history.append((user_message, reply))
        


if __name__ == "__main__":
    main()
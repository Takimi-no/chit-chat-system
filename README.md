# chit-chat-system
対話モデルのユーザ評価のためのwebサイトを構築する

## 内容補足：
### VLLM：
- Virtual LLM
    - 大規模言語モデルを高速で効率的に推論するためのオープンソースライブラリ
    - 推奨利用シーン：
      - AIサービス運用
      - データ処理
      - 計算資源(GPU)節約
      - 高負荷環境
    - 推論サーバとは
      - AIモデルをトレーニング段階から運用段階に前進させるのに役立つソフトウェア

## 実行コマンド：
1. サーバでモデル単体の動作確認
- chat_terminal.py
    CUDA_VISIBLE_DEVICES=0 python chat_terminal.py
2. vLLMで推論APIを立てる
export VLLM_API_KEY="my-demo-key"

vllm serve <MODEL> \
  --host 127.0.0.1 \
  --port 18000 \
  --api-key "$VLLM_API_KEY" \
  --generation-config vllm
1. 


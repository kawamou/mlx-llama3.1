# mlx-llama3.1

```sh
# 以降のコマンドはリポジトリルートで実行
rye sync

. .venv/bin/activate
python -m src.mlx_llama3_1.examples.llama3_1_generate_prompt

# or 

rye run python -m src.mlx_llama3_1.examples.llama3_1_generate_prompt
```
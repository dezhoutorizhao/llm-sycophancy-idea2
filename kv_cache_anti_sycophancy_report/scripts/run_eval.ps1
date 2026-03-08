$ErrorActionPreference = "Stop"
$Repo = "e:\谄媚KVcache_idea\Sycopancy Interpretability代码\sycophancy-interpretability"
Set-Location $Repo

python evaluation\evaluate_sycophancy_chat_hf.py `
  --model_path meta-llama/Llama-3.1-8B-Instruct `
  --batch_size 8 `
  --max_new_tokens 128 `
  --output_dir runs\eval_base

python evaluation\knockout_topk_sweep_hf.py `
  --model_path meta-llama/Llama-3.1-8B-Instruct `
  --results_path runs\results.pt `
  --topk 64 `
  --output_path runs\eval_kv_gate.json

$ErrorActionPreference = "Stop"
$Repo = "e:\谄媚KVcache_idea\Sycopancy Interpretability代码\sycophancy-interpretability"
Set-Location $Repo

python scripts\download_kv_cache_curated.py

python path_patching\path_patching_delta_ci_hf.py `
  --model_path meta-llama/Llama-3.1-8B-Instruct `
  --seed 0 `
  --output_dir runs\delta_ci

python path_patching\path_patching_delta_fg_hf.py `
  --model_path meta-llama/Llama-3.1-8B-Instruct `
  --seed 0 `
  --output_dir runs\delta_fg

python e:\谄媚KVcache_idea\kv_cache_anti_sycophancy_report\code\aggregate_delta_to_results.py `
  --ci runs\delta_ci\results.pt `
  --fg runs\delta_fg\results.pt `
  --out runs\results.pt

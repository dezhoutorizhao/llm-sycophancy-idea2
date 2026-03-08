$ErrorActionPreference = "Stop"

$Repo = "e:\谄媚KVcache_idea\Sycopancy Interpretability代码\sycophancy-interpretability"
$Model = "e:\谄媚KVcache_idea\Qwen\Qwen2.5-0.5B-Instruct"
$Data = "$Repo\evaluation\datasets\sycophancy_eval\multiple_choice.jsonl"
$OutRoot = "$Repo\evaluation\results\kv_gate_qwen05b"
$DeltaOut = "$Repo\path_patching\results_delta_ci\qwen05b_local"
$ResultsPt = "$DeltaOut\scores_delta.pt"

Set-Location $Repo
$env:TEMP = "e:\谄媚KVcache_idea\.pip_tmp"
$env:TMP = "e:\谄媚KVcache_idea\.pip_tmp"
$env:PIP_CACHE_DIR = "e:\谄媚KVcache_idea\.pip_cache"
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
New-Item -ItemType Directory -Force -Path $env:PIP_CACHE_DIR | Out-Null

$env:PYTHONPATH = "e:\谄媚KVcache_idea\.vendor\runtime_pkgs;e:\谄媚KVcache_idea\.vendor\modelscope_pkgs"

python path_patching\path_patching_delta_ci_hf.py `
  --model_path $Model `
  --data_path $Data `
  --batch_size 4 `
  --dtype float16 `
  --device_map auto `
  --max_samples 200 `
  --out_dir $DeltaOut

python evaluation\evaluate_sycophancy_chat_kv_gate_hf.py `
  --model_path $Model `
  --data_path $Data `
  --output_dir $OutRoot `
  --torch_dtype float16 `
  --device_map auto `
  --max_new_tokens 16 `
  --max_samples 200

python evaluation\score_mc_c2i_simple.py `
  --run_dir "$OutRoot\multiple_choice\Qwen2.5-0.5B-Instruct\baseline"

python evaluation\evaluate_sycophancy_chat_kv_gate_hf.py `
  --model_path $Model `
  --data_path $Data `
  --output_dir $OutRoot `
  --torch_dtype float16 `
  --device_map auto `
  --max_new_tokens 16 `
  --max_samples 200 `
  --results_pt $ResultsPt `
  --topk 32 `
  --gate_gamma 0.35 `
  --gate_window 32

python evaluation\score_mc_c2i_simple.py `
  --run_dir "$OutRoot\multiple_choice\Qwen2.5-0.5B-Instruct\kvgate_topk32_g0.35_w32"

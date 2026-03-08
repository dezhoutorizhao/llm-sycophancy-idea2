#!/bin/bash
###
 # @Author             : 陈蔚 (cy424151@alibaba-inc.com)
 # @Date               : 2024-10-15 18:36
 # @Last Modified By   : 陈蔚 (cy424151@alibaba-inc.com)
 # @Last Modified Date : 2024-10-15 19:20
 # @Description        : 
 # -------- 
 # Copyright (c) 2024 Alibaba Inc. 
### 

mkdir datasets

# MMLU

huggingface-cli download --repo-type dataset --resume-download cais/mmlu --local-dir datasets/mmlu --local-dir-use-symlinks False

# MATH-QA

curl -o datasets/MathQA.zip https://math-qa.github.io/math-QA/data/MathQA.zip
unzip datasets/MathQA.zip -d datasets/mathqa
rm -rf datasets/MathQA.zip

# AQUA

mkdir datasets/aqua
curl -o datasets/aqua/train.json https://raw.githubusercontent.com/google-deepmind/AQuA/refs/heads/master/train.json

# TrviaQA

huggingface-cli download --repo-type dataset --resume-download mandarjoshi/trivia_qa --local-dir datasets/trivia_qa --local-dir-use-symlinks False --include unfiltered.nocontext/*
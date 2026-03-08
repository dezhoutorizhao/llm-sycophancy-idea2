#!/bin/bash
###
 # @Author             : 陈蔚 (cy424151@alibaba-inc.com)
 # @Date               : 2024-10-15 20:38
 # @Last Modified By   : 陈蔚 (cy424151@alibaba-inc.com)
 # @Last Modified Date : 2024-10-15 21:15
 # @Description        : 
 # -------- 
 # Copyright (c) 2024 Alibaba Inc. 
### 

# We notice that a small fraction of data is enough
shuf -n 3840 datasets/all_scyophancy_mixed_instruction_data.jsonl >> datasets/scyophancy_mixed_instruction_data.jsonl

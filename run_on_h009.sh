#!/bin/bash
# 包装脚本：SSH 到 h009，激活环境，然后执行命令

# 获取要执行的命令（如果提供了参数）
if [ $# -eq 0 ]; then
    # 如果没有参数，只 SSH 并激活环境
    ssh h009 -t "bash -c 'cd /anvil/projects/x-cis250705/molmo_hf && source activate_env.sh && exec bash'"
else
    # 如果有参数，执行命令
    ssh h009 -t "bash -c 'cd /anvil/projects/x-cis250705/molmo_hf && source activate_env.sh && $@'"
fi


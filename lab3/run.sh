#!/bin/bash

# 定义要运行的 Python 文件路径
PYTHON_FILE="/old_home/lyt/zxj_workplaces/ailab/CodeWithDataset/lab3/bert+textcnn_best.py"

# 运行脚本 5 次
for i in {1..30}
do
    echo "Running iteration $i..."
    python3 $PYTHON_FILE
done

echo "All iterations completed."
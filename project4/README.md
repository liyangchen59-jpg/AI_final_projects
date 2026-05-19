# Project 4 README

This project fine-tunes `prajjwal1/bert-mini` on two GLUE tasks:

- **SST-2**: sentiment classification
- **MRPC**: paraphrase classification

## 1. File Structure

```text
project4/
├── README.md                    # project说明与运行方式
├── requirements.txt             # Python dependencies
├── sst2.py                      # SST-2训练脚本
├── mrpc.py                      # MRPC训练脚本
├── eval_sst2.py                 # SST-2评估脚本
├── eval_mrpc.py                 # MRPC评估脚本
├── AI_project4_225040037.pdf    # 最终报告PDF
├── bert-for-sst2/
│   ├── config.json              # SST-2模型配置
│   └── model.safetensors        # SST-2模型权重
├── bert-for-mrpc/
│   ├── config.json              # MRPC模型配置
│   └── model.safetensors        # MRPC模型权重
└── results/
    ├── sst2.png                 # SST-2运行截图
    ├── mrpc.png                 # MRPC运行截图
    ├── sst2_training_log.txt    # SST-2训练日志
    ├── mrpc_training_log.txt    # MRPC训练日志
    ├── sst2_eval_results.txt    # SST-2评估结果归档
    └── mrpc_eval_results.txt    # MRPC评估结果归档
```

## 2. Environment Setup
First
```bash
export HFexport HF_ENDPOINT=https://hf-mirror.com
```

Run in the project root directory:

```bash
pip install -r requirements.txt
```

The first run may download:
- GLUE datasets
- `bert-base-uncased`
- `prajjwal1/bert-mini`

## 3. How to Run

### 3.1 Train SST-2

```bash
python3 sst2.py
```

Output:
- terminal training log
- saved model in `bert-for-sst2/`

### 3.2 Train MRPC

```bash
python3 mrpc.py
```

Output:
- terminal training log
- saved model in `bert-for-mrpc/`

### 3.3 Evaluate SST-2

```bash
python3 eval_sst2.py
```

Output:
- terminal evaluation summary
- `sst2_eval_results.txt`

### 3.4 Evaluate MRPC

```bash
python3 eval_mrpc.py
```

Output:
- terminal evaluation summary
- `mrpc_eval_results.txt`

## 4. Notes

- Run all commands in the `project4/` root directory.
- The scripts use relative paths for loading and saving models.
- The evaluation scripts write `.txt` outputs to the current working directory.
- The `results/` folder stores archived logs, screenshots, and evaluation outputs.

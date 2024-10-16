# Replication Package of EXE-Reviewer

EXE-Reviewer: Towards EXplainable and Effective Review Comments Generation

## File Structure

- `Dataset`: contains the data files.
- `Model`: contains the pre-trained models.
- `Result`: contains the generated review comments and metric results.
- `Code`: contains the source code of EXE-Reviewer.

### Source Code Structure

```text
.
├── causal_utils.py               # Causal info extraction utilities
├── configs.py                    # Configurations and hyperparameters
├── evaluator/                    # BLEU evaluation scripts
├── models.py                     # Model definitions
├── run_finetune_msg_explain.py   # Fine-tuning script for EXE-Reviewer
├── run_test_msg_explain.py       # Testing script for EXE-Reviewer
├── sh/                           # Shell scripts
│   ├── calc_metrics.sh           # Script to calculate metrics
│   ├── finetune-msg.sh           # Script to run fine-tuning
│   ├── interface-msg.sh          # Interface script
│   └── test-msg.sh               # Script to run tests
├── total_eval.py                 # Aggregated evaluation script
└── utils.py                      # General utility functions

```

## How to Use

### Preparation

1. Install the required packages.

```shell
pip install -r requirements.txt
```

2. Download Dataset and Pre-trained Models.

See the `Dataset` and `Model` directories for details.

### Interface

The interface stage is to use code-pretrained model to generate model focuses.

```shell
cd Code/sh
# adjust the arguments in the *sh* scripts
bash interface-msg.sh
```

### Train

```shell
cd Code/sh
# adjust the arguments in the *sh* scripts
bash finetune-msg.sh
```

### Evaluate

```shell
cd Code/sh
# adjust the arguments in the *sh* scripts
bash test-msg.sh
```

After running the above commands, you will get the generated review comments.
```shell
# adjust the arguments in the *sh* scripts
bash calc_metrics.sh
```
Use the above command to calculate the metrics.
- effective metrics: BLEU-4, ROUGE-1, ROUGE-L
- explainable metrics: ERCR, HC-ERCR

## Licensing Information

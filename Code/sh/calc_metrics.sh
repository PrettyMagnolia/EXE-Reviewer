# Change the arguments as required:
#   causal_seq_model_path: the path of the causal model for sequence
#   stopwords_path: the path of the stopwords file
#   predict: the path of the prediction file
#   gold: the path of the gold file

python ./total_eval.py \
  --causal_seq_model_path ../model/unicausal-seq-baseline \
  --stopwords_path ../evaluator/stopwords.txt \
  --predict preds_topk10.txt \
  --gold golds.txt
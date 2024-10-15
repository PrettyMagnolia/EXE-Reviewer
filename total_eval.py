import argparse

import nltk
import torch
from rouge.rouge import Rouge

from causal_utils import load_model_and_tokenizer, seq_predict
from evaluator.smooth_bleu import computeMaps, bleuFromMaps


def read_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]

    return lines


def tokenize(texts):
    processed_texts = []
    for text in texts:
        tokenized_text = nltk.wordpunct_tokenize(text)
        processed_text = " ".join(tokenized_text)
        processed_texts.append(processed_text)
    return processed_texts


def remove_stopwords(texts):
    with open(stopwords_path, 'r') as f:
        stopwords = [line.strip() for line in f]
    processed_texts = []
    for text in texts:
        filtered_words = [word for word in text.split() if word not in stopwords]
        processed_text = " ".join(filtered_words)
        processed_texts.append(processed_text)

    return processed_texts


def preprocess(file_path: str, rmstop: bool):
    """
    read file, tokenize and remove stopwords if needed

    Args:
        file_path (str): file path.
        rmstop (bool): whether to remove stopwords.

    Returns:
        list: processed texts.
    """
    content = tokenize(read_txt(file_path))
    if rmstop:
        content = remove_stopwords(content)
    return content


def rouge_preprocess(predictions):
    """
    preprocess the predictions for rouge evaluation.
    change all '.' predictions to '', the rouge evaluation will ignore the empty predictions.

    Args:
        predictions (list): list of predictions.

    Returns:
        list: processed predictions.
    """
    for i in range(len(predictions)):
        pred = predictions[i]
        if all(char == '.' for char in pred):
            predictions[i] = ''
    return predictions


def eval_bleu(predict_file, gold_file, rmstop):
    """
    Evaluate the BLEU score.

    Args:
        predict_file (str): prediction file path.
        gold_file (str): gold file path.
        rmstop (bool): whether to remove stopwords.

    Returns:
        None
    """
    print('Evaluating the BLEU score...')
    predictions = preprocess(predict_file, rmstop)
    golds = preprocess(gold_file, rmstop)

    predictions = [str(i) + "\t" + pred.replace("\t", " ") for (i, pred) in enumerate(predictions)]
    golds = [str(i) + "\t" + gold.replace("\t", " ") for (i, gold) in enumerate(golds)]
    gold_map, prediction_map = computeMaps(predictions, golds)
    bleu = bleuFromMaps(gold_map, prediction_map)[0]
    print("BLEU ", 'w/o stopwords' if rmstop else 'w/ stopwords', ': ', bleu)


def eval_rouge(predict_file, gold_file, rmstop):
    """
    Evaluate the Rouge score.

    Args:
        predict_file (str): prediction file path.
        gold_file (str): gold file path.
        rmstop (bool): whether to remove stopwords.

    Returns:
        None
    """
    print('Evaluating the ROUGE score...')
    predictions = preprocess(predict_file, rmstop)
    golds = preprocess(gold_file, rmstop)

    predictions = rouge_preprocess(predictions)

    rouge = Rouge()
    rouge_score = rouge.get_scores(hyps=predictions, refs=golds, avg=True, ignore_empty=True)
    rouge_1 = rouge_score['rouge-1']['r']
    rouge_l = rouge_score['rouge-l']['f']

    print("ROUGE_1", 'w/o stopwords' if rmstop else 'w/ stopwords', ': ', rouge_1 * 100)
    print("ROUGE_L", 'w/o stopwords' if rmstop else 'w/ stopwords', ': ', rouge_l * 100)


def eval_explain(predict_file):
    """
    Evaluate the explain metric

    Args:
        predict_file (str): prediction file path.

    Returns:
        None
    """
    print('Start to evaluate the explainability...')
    model, tokenizer = load_model_and_tokenizer(model_type='seq', model_name_or_path=causal_seq_model_path,
                                                cache_dir=None)
    lines = read_txt(predict_file)
    logits = seq_predict(model, tokenizer, lines, local_rank)

    # count the amount of label 1
    predictions = torch.argmax(logits, dim=1)
    count_label_1 = (predictions == 1).sum().item()

    # count the amount of label 1 with confidence > 95%
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    label_1_probabilities = probabilities[:, 1]
    count_high_confidence_label_1 = ((predictions == 1) & (label_1_probabilities > 0.95)).sum().item()
    percent_high_confidence_label_1 = (count_high_confidence_label_1 / count_label_1) * 100 if count_label_1 > 0 else 0

    print('Total amount of label 1: ', count_label_1)
    print('Percent of label 1 with confidence > 95%: ', percent_high_confidence_label_1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model predictions')

    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    parser.add_argument('--causal_seq_model_path', type=str, default='', help='Path to the causal sequence model')
    parser.add_argument('--stopwords_path', type=str, default='./evaluator/stopwords.txt', help='Path to the stopwords file')
    parser.add_argument('--predict', type=str, default='', help='Path to the prediction file')
    parser.add_argument('--gold', type=str, default='', help='Path to the gold standard file')

    args = parser.parse_args()

    stopwords_path = args.stopwords_path
    causal_seq_model_path = args.causal_seq_model_path
    local_rank = args.local_rank

    # bleu
    eval_bleu(args.predict, args.gold, rmstop=False)
    eval_bleu(args.predict, args.gold, rmstop=True)

    # rouge
    eval_rouge(args.predict, args.gold, rmstop=False)
    eval_rouge(args.predict, args.gold, rmstop=True)

    # explain
    eval_explain(args.predict)

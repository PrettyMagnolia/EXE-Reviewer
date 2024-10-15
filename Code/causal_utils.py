import torch
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForTokenClassification


def load_model_and_tokenizer(model_type, model_name_or_path, cache_dir=None):
    """
    Load the pre-trained model and tokenizer based on the specified model type.

    Args:
        model_type (str): The type of model ('seq' for sequence classification, 'tok' for token classification).
        model_name_or_path (str): The path or name of the pre-trained model.
        cache_dir (str, optional): Directory to cache the downloaded pre-trained models.

    Returns:
        model: The loaded pre-trained model.
        tokenizer: The loaded tokenizer.

    Raises:
        ValueError: If an invalid model type is provided.
    """
    if model_type == "seq":
        config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config,
                                                                   cache_dir=cache_dir)
    elif model_type == "tok":
        config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        model = AutoModelForTokenClassification.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir, use_fast=True)
    return model, tokenizer


def preprocess_data(tokenizer, text, max_seq_length=128):
    """
    Preprocess the input text for the model by tokenizing and padding it.

    Args:
        tokenizer: The tokenizer for the model.
        text (str): The input text to preprocess.
        max_seq_length (int, optional): The maximum sequence length. Default is 128.

    Returns:
        dict: A dictionary containing tokenized input tensors.
    """
    tokenized_inputs = tokenizer(
        text,
        max_length=max_seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return tokenized_inputs


def remove_special_tokens(predictions, word_ids):
    """
    Remove special tokens from the predictions based on the word IDs.

    Args:
        predictions (list): The list of predicted tokens.
        word_ids (list): The list of word IDs corresponding to each token.

    Returns:
        list: The cleaned list of predictions without special tokens.
    """
    cleaned_preds = []
    for pred, word_id in zip(predictions, word_ids):
        if word_id is not None:
            cleaned_preds.append(pred)
    return cleaned_preds


def seq_predict(model, tokenizer, texts, device):
    """
    Perform sequence classification predictions on the input texts.

    Args:
        model: The pre-trained sequence classification model.
        tokenizer: The tokenizer for the model.
        texts (list of str): The list of input texts to classify.
        device (torch.device or int): The device (CPU/GPU) to run the model on.

    Returns:
        torch.Tensor: The logits for all input texts.
    """
    model.to(device)
    model.eval()
    all_logits = []

    for text in tqdm(texts):
        inputs = preprocess_data(tokenizer, text)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            all_logits.append(logits)

    # Stack all logits into a single tensor
    all_logits_tensor = torch.cat(all_logits, dim=0).to(device)

    return all_logits_tensor


def tok_predict(model, tokenizer, texts, device):
    """
    Perform token classification predictions on the input texts.

    Args:
        model: The pre-trained token classification model.
        tokenizer: The tokenizer for the model.
        texts (list of str): The list of input texts to classify.
        device (torch.device or int): The device (CPU/GPU) to run the model on.

    Returns:
        list: A list of predictions for each input text.
    """
    # Label mapping for the model
    label_list = ['B-C', 'B-E', 'I-C', 'I-E', 'O']  # Modify based on the actual label set
    id_to_label = {i: l for i, l in enumerate(label_list)}

    model.to(device)
    model.eval()

    predictions = []
    for text in texts:
        # Tokenize input text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128,
                           is_split_into_words=False)
        inputs.to(device)
        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the prediction results
        logits = outputs.logits
        predicted_token_class_ids = torch.argmax(logits, dim=-1).tolist()[0]
        word_ids = inputs.word_ids(batch_index=0)

        # Remove special tokens
        cleaned_preds = remove_special_tokens([id_to_label[id] for id in predicted_token_class_ids], word_ids)

        predictions.append(cleaned_preds)

    return predictions


if __name__ == "__main__":
    model_path = '/data/lyf/code/Code_Reviewer/3_Pretrained_Model/tok-baseline/'
    texts = [
        "I like you because you like me.",
        "This should not be included since there are no code changes."
    ]
    device = 1

    m, tok = load_model_and_tokenizer('tok', model_name_or_path)
    preds = tok_predict(m, tok, texts, device)
    for text, prediction in zip(texts, preds):
        print(f"Text: {text}\nPrediction: {prediction}")

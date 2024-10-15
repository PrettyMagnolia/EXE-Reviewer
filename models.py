import os
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import numpy as np
from utils import MyTokenizer
from transformers import (
    RobertaConfig,
    RobertaModel,
    RobertaTokenizer,
    BartConfig,
    BartForConditionalGeneration,
    BartTokenizer,
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
)
import logging
from causal_utils import seq_predict, tok_predict

logger = logging.getLogger(__name__)


class ReviewerModel(T5ForConditionalGeneration):

    def __init__(self, config):
        super().__init__(config)
        self.cls_head = nn.Linear(self.config.d_model, 2, bias=True)
        self.init()

    def init(self):
        nn.init.xavier_uniform_(self.lm_head.weight)
        factor = self.config.initializer_factor
        self.cls_head.weight.data.normal_(mean=0.0, \
                                          std=factor * ((self.config.d_model) ** -0.5))
        self.cls_head.bias.data.zero_()

    def forward(
            self, *argv, **kwargs
    ):
        r"""
        Doc from Huggingface transformers:
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``
        Returns:
        Examples::
            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')
            >>> # training
            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits
            >>> # inference
            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
            >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
            >>> # studies have shown that owning a dog is good for you.
        """
        if "cls" in kwargs:
            assert (
                    "input_ids" in kwargs and \
                    "labels" in kwargs and \
                    "attention_mask" in kwargs
            )
            return self.cls(
                input_ids=kwargs["input_ids"],
                labels=kwargs["labels"],
                attention_mask=kwargs["attention_mask"],
            )
        if "input_labels" in kwargs:
            assert (
                    "input_ids" in kwargs and \
                    "input_labels" in kwargs and \
                    "decoder_input_ids" in kwargs and \
                    "attention_mask" in kwargs and \
                    "decoder_attention_mask" in kwargs
            ), "Please give these arg keys."
            input_ids = kwargs["input_ids"]
            input_labels = kwargs["input_labels"]
            decoder_input_ids = kwargs["decoder_input_ids"]
            attention_mask = kwargs["attention_mask"]
            decoder_attention_mask = kwargs["decoder_attention_mask"]
            encoder_loss = kwargs.get("encoder_loss", True)

            # for explain loss
            explain_label = kwargs.get("explain_label", None)
            tokenizer = kwargs.get("tokenizer", None)
            seq_model = kwargs.get("seq_model", None)
            seq_tokenizer = kwargs.get("seq_tokenizer", None)
            tok_model = kwargs.get("tok_model", None)
            tok_tokenizer = kwargs.get("tok_tokenizer", None)
            local_rank = kwargs.get("local_rank", None)

            return self.review_forward(input_ids, input_labels, decoder_input_ids, attention_mask,
                                       decoder_attention_mask, encoder_loss,
                                       explain_label, tokenizer, seq_model, seq_tokenizer, tok_model, tok_tokenizer,
                                       local_rank)
        return super().forward(*argv, **kwargs)

    def cls(
            self,
            input_ids,
            labels,
            attention_mask,
    ):
        encoder_outputs = self.encoder( \
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            return_dict=False
        )
        hidden_states = encoder_outputs[0]
        first_hidden = hidden_states[:, 0, :]
        first_hidden = nn.Dropout(0.3)(first_hidden)
        logits = self.cls_head(first_hidden)
        loss_fct = CrossEntropyLoss()
        if labels != None:
            loss = loss_fct(logits, labels)
            return loss
        return logits

    def review_forward(
            self,
            input_ids,
            input_labels,
            decoder_input_ids,
            attention_mask,
            decoder_attention_mask,
            encoder_loss=True,
            explain_label=None,
            tokenizer=None,
            seq_model=None,
            seq_tokenizer=None,
            tok_model=None,
            tok_tokenizer=None,
            local_rank=None
    ):
        encoder_outputs = self.encoder( \
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            return_dict=False
        )
        hidden_states = encoder_outputs[0]
        decoder_inputs = self._shift_right(decoder_input_ids)
        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_inputs,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            output_attentions=False,
            return_dict=False
        )
        sequence_output = decoder_outputs[0]
        if self.config.tie_word_embeddings:  # this is True default
            sequence_output = sequence_output * (self.model_dim ** -0.5)
        if encoder_loss:
            # print(self.encoder.get_input_embeddings().weight.shape)
            cls_logits = nn.functional.linear(hidden_states, self.encoder.get_input_embeddings().weight)
            # cls_logits = self.cls_head(hidden_states)
        lm_logits = self.lm_head(sequence_output)
        if decoder_input_ids is not None:
            lm_loss_fct = CrossEntropyLoss(ignore_index=0)  # Warning: PAD_ID should be 0
            loss = 0
            lm_loss = lm_loss_fct(lm_logits.view(-1, lm_logits.size(-1)), decoder_input_ids.view(-1))
            loss += lm_loss
            if encoder_loss and input_labels is not None:
                cls_loss_fct = CrossEntropyLoss(ignore_index=-100)
                loss += cls_loss_fct(cls_logits.view(-1, cls_logits.size(-1)), input_labels.view(-1))
            # add explanatory loss
            if explain_label is not None:
                explain_loss_fct = CrossEntropyLoss(ignore_index=-100)
                output_texts = [tokenizer.decode(torch.argmax(logit, dim=-1)) for logit in lm_logits]
                explain_pred = seq_predict(seq_model, seq_tokenizer, output_texts, local_rank)
                # calculate explanatory loss with label 1(has explanatory)
                explanatory_loss = explain_loss_fct(explain_pred.view(-1, explain_pred.size(-1)),
                                                    torch.ones_like(explain_label.view(-1)))
                loss += explanatory_loss

                filtered_logits = []
                filtered_labels = []
                for text, logit, label_id, e_label_true, e_label_pred in zip(output_texts, lm_logits, decoder_input_ids,
                                                                             explain_label, explain_pred):
                    if e_label_true == 0 and torch.argmax(e_label_pred, dim=-1) == 1:
                        tok_pred = tok_predict(tok_model, tok_tokenizer, [text], local_rank)[0]
                        new_token = tokenizer.encode(tok_pred, return_tensors='pt')[0]

                        new_token_ids = new_token.tolist()
                        logit_ids = torch.argmax(logit, dim=-1)
                        mask = torch.tensor([logit_id in new_token_ids for logit_id in logit_ids], device=logit.device)

                        logit = logit[mask]
                        label_id = label_id[:logit.size(0)]

                    filtered_logits.append(logit)
                    filtered_labels.append(label_id)

                # correct the lm_loss
                if len(filtered_logits) != 0:
                    filtered_logits = torch.cat(filtered_logits, dim=0)
                    filtered_labels = torch.cat(filtered_labels, dim=0)
                    new_lm_loss = lm_loss_fct(filtered_logits.view(-1, filtered_logits.size(-1)),
                                              filtered_labels.view(-1))
                    loss = loss - lm_loss + new_lm_loss
            return loss

        return cls_logits, lm_logits


def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e6))


def build_or_load_gen_model(args):
    config_class, model_class, tokenizer_class = T5Config, ReviewerModel, RobertaTokenizer

    config = config_class.from_pretrained(args.model_name_or_path)

    # modify vocab_size
    if args.has_focus:
        config.vocab_size += 2  # add <true_focus> and <other_focus>

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path, config=config, ignore_mismatched_sizes=True)

    tokenizer.special_dict = {
        f"<e{i}>": tokenizer.get_vocab()[f"<e{i}>"] for i in range(99, -1, -1)
    }

    tokenizer.mask_id = tokenizer.get_vocab()["<mask>"]
    tokenizer.bos_id = tokenizer.get_vocab()["<s>"]
    tokenizer.pad_id = tokenizer.get_vocab()["<pad>"]
    tokenizer.eos_id = tokenizer.get_vocab()["</s>"]
    tokenizer.msg_id = tokenizer.get_vocab()["<msg>"]
    tokenizer.keep_id = tokenizer.get_vocab()["<keep>"]
    tokenizer.add_id = tokenizer.get_vocab()["<add>"]
    tokenizer.del_id = tokenizer.get_vocab()["<del>"]
    tokenizer.start_id = tokenizer.get_vocab()["<start>"]
    tokenizer.end_id = tokenizer.get_vocab()["<end>"]

    # add focus token
    tokenizer.add_tokens("<true_focus>")
    tokenizer.focus_id = tokenizer.get_vocab()["<true_focus>"]
    tokenizer.add_tokens("<other_focus>")
    tokenizer.focus_id = tokenizer.get_vocab()["<other_focus>"]

    logger.info(
        "Finish loading model [%s] from %s",
        get_model_size(model),
        args.model_name_or_path,
    )

    if args.load_model_path is not None:
        model_path = os.path.join(args.load_model_path, "pytorch_model.bin")
        logger.info("Reload model from {}".format(model_path))
        try:
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError:
            saved = model.cls_head
            model.cls_head = None
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.cls_head = saved
        model.to(args.local_rank)

    return config, model, tokenizer


if __name__ == '__main__':
    tokenizer_class = RobertaTokenizer
    model_name_or_path = r''
    tokenizer = tokenizer_class.from_pretrained(model_name_or_path)

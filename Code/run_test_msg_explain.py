import argparse
import json
import logging
import multiprocessing
import os
import time
import torch
import torch.distributed as dist
from configs import add_args, set_seed
from evaluator.smooth_bleu import bleu_fromstr
from models import build_or_load_gen_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from utils import CommentGenDataset, SimpleGenDataset, top_k_token_dict, merge_dict

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_loader(data_file, args, tokenizer, pool):
    def fn(features):
        return features

    logger.info(f"Start data file {data_file}.")
    if args.raw_input:
        dataset = SimpleGenDataset(tokenizer, pool, args, data_file)
    else:
        dataset = CommentGenDataset(tokenizer, pool, args, data_file)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=0, collate_fn=fn)
    logger.info(f"Finish data files {data_file}.")
    return dataset, sampler, dataloader


def eval_epoch_bleu(args, eval_dataloader, model, tokenizer):
    logger.info(f"  ***** Running bleu evaluation on {args.eval_file} *****")
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    if hasattr(model, "module"):
        model = model.module
    pred_ids, ex_ids, focuses = [], [], []

    for step, examples in tqdm(enumerate(eval_dataloader, 1)):
        source_ids = torch.tensor(
            [ex.source_ids for ex in examples], dtype=torch.long
        ).to(args.local_rank)
        ids = [ex.example_id for ex in examples]
        source_mask = source_ids.ne(tokenizer.pad_id)

        # use topk sample
        # if generate focus need output attentions
        outputs = model.generate(source_ids,
                                 attention_mask=source_mask,
                                 use_cache=True,
                                 do_sample=True,
                                 top_k=args.topk,
                                 max_length=args.max_target_length,
                                 output_attentions=args.generate_focus,
                                 return_dict_in_generate=args.generate_focus)
        preds = outputs.sequences
        cross_attentions = outputs.cross_attentions

        # decode and record the indices of special tokens
        special_token_indices = []
        decoded_tokens = []
        for i, token_id in enumerate(list(source_ids[0].cpu().numpy())):
            token = tokenizer.decode(token_id, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            if token.startswith("<") and token.endswith(">"):  # 检查是否为特殊标记
                special_token_indices.append(i)
            else:
                decoded_tokens.append(token)
        input_tokens = decoded_tokens

        if args.generate_focus:
            res_focus_dict = {}
            for i in range(len(cross_attentions)):
                # attentions when generate i-th token
                cur_attentions = cross_attentions[i]
                for j in range(len(cur_attentions)):
                    for k in range(cur_attentions[j].size(1)):
                        # print(cur_attentions[j][0, k, :])
                        all_attention = cur_attentions[j][0, k, :]

                        # filter special tokens
                        all_indices = list(range(all_attention.size(1)))
                        keep_indices = torch.tensor(list(set(all_indices) - set(special_token_indices))).to(
                            args.local_rank)
                        focus_attention = torch.index_select(all_attention, 1, keep_indices).to(args.local_rank)

                        # get top k tokens(attention layer)
                        top_values, top_indices = torch.topk(focus_attention, focus_attention.size(1))
                        top_tokens = [input_tokens[idx] for idx in top_indices[0].cpu().numpy()]
                        top_values = [top_value for top_value in top_values[0].cpu().numpy()]
                        top_tokens_dict = top_k_token_dict(top_tokens, top_values, k=args.focus_len, rm_symbol=True)
                        res_focus_dict = merge_dict(res_focus_dict, top_tokens_dict)

            # get top k tokens(whole sequence)
            res_focus = sorted(res_focus_dict, key=res_focus_dict.get, reverse=True)[:args.focus_len]
            focuses.append(res_focus)

        top_preds = list(preds.cpu().numpy())
        pred_ids.extend(top_preds)

    pred_nls = [tokenizer.decode(id[2:], skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in
                pred_ids]
    valid_file = args.eval_file
    golds = []
    with open(valid_file, "r") as f:
        for line in f:
            golds.append(json.loads(line)["msg"])
    golds = golds[:len(pred_nls)]
    with open(os.path.join(args.output_dir, "preds_topk{}.txt".format(args.topk)), "w", encoding="utf-8") as f:
        for pred in pred_nls:
            f.write(pred.strip() + "\n")
    with open(os.path.join(args.output_dir, "golds_{}.txt".format(args.check)), "w", encoding="utf-8") as f:
        for gold in golds:
            f.write(gold.strip() + "\n")

    if args.generate_focus:
        # write focus info to file
        with open(os.path.join(args.output_dir, args.focus_file_name), "w", encoding="utf-8") as f:
            for focus in focuses:
                f.write(','.join(focus) + '\n')

    bleu = bleu_fromstr(pred_nls, golds, rmstop=False)
    logger.warning(f"WithStop BLEU: {bleu}")
    bleu = bleu_fromstr(pred_nls, golds, rmstop=True)
    return bleu


def main(args):
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank() % args.gpu_per_node
    args.global_rank = local_rank + args.node_index * args.gpu_per_node
    args.local_rank = local_rank
    args.world_size = dist.get_world_size()
    logger.warning("Process rank: %s, global rank: %s, world size: %s, bs: %s",
                   args.local_rank, args.global_rank,
                   torch.distributed.get_world_size(),
                   args.eval_batch_size)

    torch.cuda.set_device(local_rank)
    set_seed(args)

    config, model, tokenizer = build_or_load_gen_model(args)
    model = DDP(model.cuda(), device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    pool = multiprocessing.Pool(args.cpu_count)
    data_file = args.eval_file
    set_seed(args)
    _, _, dataloader = get_loader(data_file, args, tokenizer, pool)  # WARNING: this is an iterator, to save memory

    model.eval()
    bleu = eval_epoch_bleu(args, dataloader, model, tokenizer)
    logger.warning(f"BLEU: {bleu}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    args.cpu_count = multiprocessing.cpu_count()

    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    logger.info(args)
    main(args)
    logger.info("Test finished.")

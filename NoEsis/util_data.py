"""Utility functions for data processing"""
import hashlib
import itertools
import random
from typing import Any, Dict, List, Tuple


import datasets
import numpy as np
import torch
import transformers

from NoEsis import constants, logger, util


def load_dataset(lang: str, split: str, use_dedup: bool) -> Tuple[datasets.Dataset, int]:
    """Loads the dataset either [Python, Java] or our custom Go dataset"""
    # #######################
    with open('/root/hf_token.txt', encoding='utf-8') as f:
        token = f.read().strip()
    # #######################

    if lang.strip().lower() == 'go':
        ds = datasets.load_dataset("anonymous/codexglue_golang_01", split=split, token=token)
    elif use_dedup:
        print(f"Loading deduped dataset for {lang}")
        assert lang in ['python', 'java'], (
            f"Only deduped datasets available for Python and Java, not {lang}")
        ds = datasets.load_dataset(f"anomymous/csn_dedup_{lang}_01", split=split, token=token)
    else:
        ds = datasets.load_dataset("code_x_glue_cc_code_completion_token", lang, split=split)

    # Cast all ids to int64
    ds = ds.cast_column('id', datasets.Value('int64'))
    return ds, constants.LANGUAGE_MAP[lang]


def provision_dataset(args, split: str, testtime: bool = False) -> datasets.Dataset:
    """Provisions the dataset for training"""
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.load)

    if util.has_multiple_datasets(args.datasets):
        logger.info(f'  ==> Interpreting [{args.datasets}] as multiple datasets')
        # interleave datasets
        ds_list = []
        for lang in util.iter_datasets_from_str(args.datasets):
            ds, lang_code = load_dataset(lang, split, use_dedup=args.use_dedup)
            def insert_lang(x):
                x['id'] = np.int64(x['id'])
                x['lang'] = lang_code  # pylint: disable=cell-var-from-loop
                return x
            ds = ds.map(insert_lang)
            ds_list.append(ds)

        ds = datasets.concatenate_datasets(ds_list)
    else:
        ds, lang_code = load_dataset(args.datasets, split, use_dedup=args.use_dedup)
        def insert_lang(x):
            x['id'] = np.int64(x['id'])
            x['lang'] = lang_code  # pylint: disable=cell-var-from-loop
            return x
        ds = ds.map(insert_lang)  # instance of datasets.arrow_dataset.Dataset

    def set_ignore_label(row):
        # CodeT5p uses -100 as ignore label
        return [(l if l != tokenizer.pad_token_id else -100) for l in row]

    def preprocess_function(examples):
        target = [' '.join(ex) for ex in examples["code"]]
        num_targets = len(target)

        target_tokens = tokenizer(target, truncation=False, verbose=False)

        model_inputs = {
            "decoder_input_ids": [[] for _ in range(num_targets)],
            "labels": [[] for _ in range(num_targets)],
            "decoder_attention_mask": [[] for _ in range(num_targets)],
            "expert_ids": [[] for _ in range(num_targets)],
            "doc_ids": [[] for _ in range(num_targets)],
        }
        for idx_target, target_ids in enumerate(target_tokens["input_ids"]):
            decoder_ids = target_ids.copy()
            num_blocks = int(np.ceil((len(target_ids)-1) / args.max_target_len))

            # if (not args.use_dedup) and (num_blocks > 2):
            #     # print(f"Skipping target {args.datasets}{args.use_dedup} with {num_blocks} blocks")
            #     continue
            # Make a document hash from examples["code"][idx_target]
            hash_object = hashlib.sha256(target[idx_target].encode()).hexdigest()
            doc_id = int(hash_object, 16) % 100000000  # Make 8 digit hashcode for the document

            for num_block in range(num_blocks):
                start, end = num_block*args.max_target_len, (num_block+1)*args.max_target_len+1

                if end <= len(target_ids):
                    # No padding needed
                    decoder_ids = target_ids[start:end]
                    decoder_attention_mask = [1] * args.max_target_len
                else:
                    # Padding needed
                    num_available_tokens = len(target_ids) - start
                    decoder_ids = target_ids[start:] + [tokenizer.pad_token_id] * (
                        args.max_target_len - num_available_tokens + 1)
                    decoder_attention_mask = (
                        [1] * (num_available_tokens - 1)
                        +[0] * (args.max_target_len - num_available_tokens + 1))

                labels = set_ignore_label(decoder_ids.copy()[1:])

                assert len(decoder_ids) == args.max_target_len + 1, (
                    f"{len(decoder_ids)} != {args.max_target_len + 1}")
                assert len(labels) == args.max_target_len, (
                    f"{len(labels)} != {args.max_target_len}")
                assert len(decoder_attention_mask) == args.max_target_len, (
                    f"{len(decoder_attention_mask)} != {args.max_target_len}")
                model_inputs["decoder_input_ids"][idx_target].append(decoder_ids[:-1])
                model_inputs["labels"][idx_target].append(labels)
                model_inputs["decoder_attention_mask"][idx_target].append(decoder_attention_mask)
                model_inputs["expert_ids"][idx_target].append(examples["lang"][idx_target])
                model_inputs["doc_ids"][idx_target].append(doc_id)

        if testtime:
            for key in model_inputs.keys():
                model_inputs[key] = list(itertools.chain.from_iterable(model_inputs[key]))

        return model_inputs


    return ds.map(
        preprocess_function,
        batched=True,
        remove_columns=ds.column_names,
        num_proc=32,
        load_from_cache_file=False)


def custom_collate_fn(batch: List[Dict[str, Any]]):
    """Collates the batch.

    For each row in batch, select a random block and apply to all keys."""
    collated_batch = {}

    num_blocks = [len(batch[i]["decoder_input_ids"]) for i in range(len(batch))]
    block_id = [np.random.choice(num) for num in num_blocks]

    for key in batch[0].keys():
        collated_batch[key] = torch.tensor(
            [batch[i][key][block_id[i]] for i in range(len(batch))], dtype=torch.long)
    return collated_batch

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : 陈蔚 (weichen.cw@zju.edu.cn)
Date               : 2024-10-09 10:19
Last Modified By   : 陈蔚 (weichen.cw@zju.edu.cn)
Last Modified Date : 2024-10-14 00:17
Description        : JSON dataset.
-------- 
Copyright (c) 2024 Wei Chen. 
'''

import json
import logging
import os
import concurrent.futures
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from typing import List, Literal, Optional, Tuple

import torch
import torch.distributed
from addict import Dict
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from utils import IGNORE_INDEX, logging_rank

logger = logging.getLogger("DeepSpeed")


def build_dataset_json(args: Namespace,
                       tokenizer: PreTrainedTokenizer) -> Dataset:
    """Build dataset from json file.

    Parameters
    ----------
    args : Namespace
        Arguments from argparse.
    tokenizer : PreTrainedTokenizer
        Tokenizer used to tokenize data.

    Returns
    -------
    Dataset
        Tokenized dataset.
    """
    train_dataset = JsonDataset(args, tokenizer)
    return train_dataset


class JsonDataset(Dataset):
    """
    JSON dataset.
    """

    def __init__(self, args: Namespace,
                 tokenizer: PreTrainedTokenizer) -> None:
        """Build dataset from json file.

        Parameters
        ----------
        args : Namespace
            Arguments from argparse.
        tokenizer : PreTrainedTokenizer
            Tokenizer used to tokenize data.
        """
        
        super().__init__()

        self.args = args
        self.tokenizer = tokenizer
        self.rank, self.world_size = self.get_rank_and_word_size()
        self.load_data()

    def get_rank_and_word_size(self) -> Tuple[int, int]:
        """Get rank and world size in distributed training.

        Returns
        -------
        Tuple[int, int]
            Rank and world size.
        """
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        return rank, world_size

    def logging_message(self, msg) -> None:
        """Logging message on rank 0.

        Parameters
        ----------
        msg : _type_
            Message to log.
        """
        if self.rank == 0:
            logger.info(msg)
        return

    def load_data(self) -> None:
        """Load data from json file. If not exists, tokenize data and save to cache.
        """
        cache_path = self.get_cache_path()

        if not os.path.exists(cache_path) and self.rank == 0:
            with open(self.args.data_path, 'r', encoding='utf-8') as file:
                data = [json.loads(line) for line in file]

            self.logging_message(f"Tokenizing data and saving to {cache_path}")
            self.tokenized_data = self._tokenize_data(
                data,
                self.tokenizer,
                self.args.max_seq_length,
                self.args.padding,
                self.args.data_type,
                train_on_prompt=self.args.train_on_prompt,
                train_on_last_n_assistant_messages=getattr(
                    self.args, "train_on_last_n_assistant_messages", 1),
            )

            torch.save(self.tokenized_data, cache_path)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        logger.info(f"Loading tokenized data from {cache_path}.")
        # Prefer the safer `weights_only=True` when available (PyTorch >= 2.4),
        # but fall back for older versions / non-weight objects (our cache is trusted).
        try:
            self.tokenized_data = torch.load(cache_path, weights_only=True)
        except Exception:
            self.tokenized_data = torch.load(cache_path)

    def get_cache_path(self) -> str:
        """Get the name of the cache file.

        Returns
        -------
        str
            The path of the cache file.
        """
        assert self.args.cache_dir is not None, "Cache directory must be specified."
        
        dataset_name = os.path.basename(self.args.data_path)
        tokenizer_name = type(self.tokenizer).__name__
        model_type = self.args.model_type
        max_seq_length = self.args.max_seq_length
        padding = self.args.padding
        train_on_prompt = self.args.train_on_prompt
        train_on_last_n_assistant_messages = getattr(
            self.args, "train_on_last_n_assistant_messages", 1)
        
        cache_path = os.path.join(
            self.args.cache_dir,
            f"{dataset_name}.cache_{model_type}_{tokenizer_name}_{max_seq_length}_{padding}_{train_on_prompt}_{train_on_last_n_assistant_messages}.pt"
        )
        return cache_path

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return self.tokenized_data[i]

    @staticmethod
    def _tokenize_data(data: List[dict],
                       tokenizer: PreTrainedTokenizer,
                       max_seq_length: int,
                       padding: bool,
                       data_type: Literal['pretraining', 'instruction_tuning'],
                       num_workers: int = 16,
                       **kwargs) -> List[Dict[str, List[int]]]:
        """Tokenize a list of data in parallel.

        Parameters
        ----------
        data : List[dict]
            Data to tokenize.
        tokenizer : PreTrainedTokenizer
            Tokenizer used to tokenize data.
        max_seq_length : int
            Max sequence length.
        padding : bool
            Whether to pad the sequence.
        data_type : Literal[&#39;pretraining&#39;, &#39;instruction_tuning&#39;]
            The type of data.
        num_workers : int, optional
            Number of workers in parallel, by default 16

        Returns
        -------
        List[Dict[str, List[int]]]
            Tokenized data, each element is a dict with keys:
                - index: int, index of the datapoint.
                - input_ids: List[int], tokenized input ids.
                - labels: List[int], tokenized labels.
                - attention_mask: List[int], attention mask.
        """

        futures = []
        tokenized_data = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for index, datapoint in enumerate(data):
                future = executor.submit(JsonDataset._tokenize_datapoint,
                                         datapoint, index, tokenizer,
                                         max_seq_length, padding, data_type,
                                         **kwargs)
                futures.append(future)
                if index % 10000 == 0:
                    logging_rank(
                        f"Prepare Tokenizing {index} / {len(data)} datapoints."
                    )

            logging_rank(f"Prepare tokenizing {len(data)} datapoints.")

            index = 0
            for future in concurrent.futures.as_completed(futures):
                if index % 10000 == 0:
                    logging_rank(
                        f"Complete Tokenizing {index} / {len(data)} datapoints."
                    )
                result = future.result()
                if result is not None:
                    tokenized_data.append(result)
                index += 1

            logging_rank(f"Finish tokenizing {len(data)} datapoints.")

        tokenized_data = sorted(tokenized_data, key=lambda x: x['index'])
        return tokenized_data

    @staticmethod
    def _tokenize_datapoint(datapoint: List[dict], index: int,
                            tokenizer: PreTrainedTokenizer,
                            max_seq_length: int, padding: bool,
                            data_type: Literal['pretraining',
                                               'instruction_tuning'],
                            **kwargs) -> Dict[str, List[int]]:
        """Tokenize a single datapoint.

        Parameters
        ----------
        datapoint : List[dict]
            Datapoint to tokenize.
        index : int
            Index of the datapoint.
        tokenizer : PreTrainedTokenizer
            Tokenizer used to tokenize data.
        max_seq_length : int
            Max sequence length.
        padding : bool
            Whether to pad the sequence.
        data_type : Literal[&#39;pretraining&#39;, &#39;instruction_tuning&#39;]
            The type of data.

        Returns
        -------
        Dict[str, List[int]]
            Tokenized result of a single datapoint, which is a dict with keys:
                - index: int, index of the datapoint.
                - input_ids: List[int], tokenized input ids.
                - labels: List[int], tokenized labels.
                - attention_mask: List[int], attention mask.

        Raises
        ------
        ValueError
            Data type not supported.
        """

        if data_type == 'instruction_tuning':
            input_ids, labels = JsonDataset._tokenize_datapoint_instruction_tuning(
                datapoint, tokenizer, **kwargs)
        elif data_type == 'pretraining':
            input_ids, labels = JsonDataset._tokenize_datapoint_pretraining(
                datapoint, tokenizer, **kwargs)
        else:
            raise ValueError(
                f"data_type must be either 'instruction_tuning' or 'pretraining, got {data_type}."
            )

        # Skip empty datapoints.
        if len(input_ids) == 0 or len(labels) == 0 or all(x == IGNORE_INDEX
                                                          for x in labels):
            return None

        attention_mask = [1] * len(input_ids)
        max_seq_length = max_seq_length

        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]
            labels = labels[:max_seq_length]
            attention_mask = attention_mask[:max_seq_length]

        if padding:
            padding_length = max_seq_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
            labels = labels + [IGNORE_INDEX] * padding_length
            attention_mask = attention_mask + [0] * padding_length

        # BUG: Using torch.tensor in Datacolator will be slower than list,
        # so we use list here.
        
        # input_ids = torch.tensor(input_ids, dtype=torch.int)
        # labels = torch.tensor(labels, dtype=torch.int)
        # attention_mask = torch.tensor(attention_mask, dtype=torch.bool)

        tokenized_datapoint = {
            'index': index,
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }

        return tokenized_datapoint

    @staticmethod
    def _tokenize_datapoint_instruction_tuning(
            datapoint: List[dict],
            tokenizer: PreTrainedTokenizer,
            train_on_prompt: bool = True,
            train_on_last_n_assistant_messages: int = 1,
            **kwargs) -> Tuple[List[int], List[int]]:
        """Tokenize a single datapoint for instruction tuning.

        Parameters
        ----------
        datapoint : List[dict]
            Datapoint to tokenize.
        tokenizer : PreTrainedTokenizer
            Tokenizer used to tokenize data.
        train_on_prompt : bool, optional
            Whether to backward loss and train on dialogues before the last, by default True

        Returns
        -------
        Tuple[List[int], List[int]]
            Input ids and labels.
        """

        assert 'messages' in datapoint, "Datapoint must contain 'messages' key."
        messages = datapoint['messages']

        # Make sure the last message is from assistant
        while messages[-1]["role"] != "assistant":
            _ = messages.pop()

        assert hasattr(tokenizer, 'apply_chat_template'
                       ), "Tokenizer must have .apply_chat_template() method."
        assert hasattr(
            tokenizer,
            'chat_template'), "Tokenizer must have .chat_template attribute."

        # NOTE: Some tokenizers/templates are not prefix-consistent under repeated
        # apply_chat_template(..., tokenize=True) calls on sub-dialogues (e.g., rare
        # newline quirks or template-specific boundary effects). The original code
        # asserted prefix equality to guard label alignment; however, that breaks
        # training for certain models (e.g., Mistral) even when the rendered text is
        # correct.
        #
        # We keep the fast path when the prefix checks pass; otherwise we fall back
        # to a robust (text-delta) construction that guarantees label alignment by
        # building token ids from successive rendered-text deltas.

        # Select which assistant messages are labeled.
        #
        # - If train_on_prompt=True: label all assistant messages.
        # - Else: label only the last N assistant messages (N>=1), which lets us
        #   align supervision to late turns (e.g., challenge response + final answer)
        #   without training on earlier, possibly incorrect, answers.
        assistant_msg_idxs = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
        if train_on_prompt:
            labeled_assistant_idxs = set(assistant_msg_idxs)
        else:
            try:
                n_last = int(train_on_last_n_assistant_messages)
            except Exception:
                n_last = 1
            n_last = max(1, n_last)
            labeled_assistant_idxs = set(assistant_msg_idxs[-n_last:])

        # Prefer HF-provided assistant token mask when available (Transformers>=4.46).
        # This avoids relying on chat-template prefix-consistency across sub-dialogues,
        # which is known to break for some models/templates (notably Mistral).
        try:
            enc = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_assistant_tokens_mask=True,
            )
        except TypeError:
            enc = None

        if enc is not None and "assistant_tokens_mask" in enc:
            input_ids = enc["input_ids"]
            assistant_mask = enc["assistant_tokens_mask"]

            # Normalize to flat python lists.
            if hasattr(input_ids, "tolist"):
                input_ids = input_ids.tolist()
            if hasattr(assistant_mask, "tolist"):
                assistant_mask = assistant_mask.tolist()
            if input_ids and isinstance(input_ids[0], list):
                input_ids = input_ids[0]
            if assistant_mask and isinstance(assistant_mask[0], list):
                assistant_mask = assistant_mask[0]

            if len(input_ids) == len(assistant_mask) and len(input_ids) > 0:
                labels = [IGNORE_INDEX] * len(input_ids)

                # Each contiguous 1-run corresponds to one assistant message span.
                spans = []
                start = None
                for idx, flag in enumerate(assistant_mask):
                    flag = bool(flag)
                    if flag and start is None:
                        start = idx
                    elif (not flag) and start is not None:
                        spans.append((start, idx))
                        start = None
                if start is not None:
                    spans.append((start, len(assistant_mask)))

                if spans:
                    chosen_spans = spans if train_on_prompt else spans[-max(1, int(train_on_last_n_assistant_messages)):]
                    for a, b in chosen_spans:
                        labels[a:b] = input_ids[a:b]

                    input_ids += [tokenizer.eos_token_id]
                    labels += [tokenizer.eos_token_id]
                    return input_ids, labels

        def _to_flat_list(x: object) -> List[int]:
            # `apply_chat_template(..., tokenize=True)` typically returns List[int],
            # but some implementations may return a 1xT nested list or a tensor.
            if hasattr(x, "tolist"):
                x = x.tolist()
            if x and isinstance(x[0], list):
                x = x[0]
            return x

        def _lcp(a: List[int], b: List[int]) -> int:
            """Longest common prefix length (in tokens)."""
            n = min(len(a), len(b))
            i = 0
            while i < n and a[i] == b[i]:
                i += 1
            return i

        def _try_lcp_labeling() -> Tuple[Optional[List[int]], Optional[List[int]]]:
            """Robust labeling via token-level longest-common-prefix alignment.

            This path does not require that sub-dialogue tokenizations are strict
            prefixes of the full dialogue tokenization (an assumption that can be
            violated by some chat templates, notably Mistral).
            """
            full_ids = _to_flat_list(
                tokenizer.apply_chat_template(messages,
                                              add_generation_prompt=False,
                                              tokenize=True))
            if not full_ids:
                return None, None

            labels = [IGNORE_INDEX] * len(full_ids)

            for i, message in enumerate(messages):
                if message.get("role") != "assistant":
                    continue

                # End index: align the tokenization of messages[:i+1] to the full dialogue.
                with_ids = _to_flat_list(
                    tokenizer.apply_chat_template(messages[:i + 1],
                                                  add_generation_prompt=False,
                                                  tokenize=True))
                label_end = _lcp(with_ids, full_ids)

                # Start index: pick the prefix rendering (gen prompt on/off) that aligns best.
                best_start = 0
                for gen in (True, False):
                    pref_ids = _to_flat_list(
                        tokenizer.apply_chat_template(messages[:i],
                                                      add_generation_prompt=gen,
                                                      tokenize=True))
                    best_start = max(best_start, _lcp(pref_ids, full_ids))

                label_start = best_start
                if label_end <= label_start:
                    continue

                if i in labeled_assistant_idxs:
                    labels[label_start:label_end] = full_ids[label_start:label_end]

            if all(x == IGNORE_INDEX for x in labels):
                return None, None
            return full_ids, labels

        def _try_prefix_labeling(
                *,
                prefix_add_generation_prompt: bool
        ) -> Tuple[Optional[List[int]], Optional[List[int]]]:
            """Fast path: label spans by token-length diffs on sub-dialogues.

            Returns (input_ids, labels) if prefix checks succeed; (None, None) otherwise.
            """
            input_ids = tokenizer.apply_chat_template(messages,
                                                      add_generation_prompt=False,
                                                      tokenize=True)
            labels = [IGNORE_INDEX] * len(input_ids)

            for i, message in enumerate(messages):
                if message.get("role") != "assistant":
                    continue

                input_ids_without_target = tokenizer.apply_chat_template(
                    messages[:i],
                    add_generation_prompt=prefix_add_generation_prompt,
                    tokenize=True)
                input_ids_with_target = tokenizer.apply_chat_template(
                    messages[:i + 1],
                    add_generation_prompt=False,
                    tokenize=True)

                label_start = len(input_ids_without_target)
                label_end = len(input_ids_with_target)

                # Prefix checks (allow a one-token boundary quirk by dropping the last token).
                if input_ids_without_target[:-1] != input_ids[:label_start - 1]:
                    return None, None
                if input_ids_with_target != input_ids[:label_end]:
                    return None, None

                if i in labeled_assistant_idxs:
                    labels[label_start:label_end] = input_ids[label_start:label_end]

            return input_ids, labels

        # Try the standard "generation prompt" prefix logic first (works for many chat templates).
        # If that fails (notably for some Mistral templates), try again without generation prompt.
        input_ids, labels = _try_prefix_labeling(prefix_add_generation_prompt=True)
        if input_ids is None:
            input_ids, labels = _try_prefix_labeling(prefix_add_generation_prompt=False)
        if input_ids is None:
            input_ids, labels = _try_lcp_labeling()
        if input_ids is None:
            # Robust path: build token ids incrementally from rendered-text deltas.
            # This avoids relying on tokenizer/template prefix-consistency in token space.
            input_ids = []
            labels = []
            prev_rendered = ""

            for i, message in enumerate(messages):
                if message.get("role") != "assistant":
                    continue

                # For robust text-delta construction, we render both prefixes with
                # add_generation_prompt=False to ensure prefix-consistency in text space.
                # Some templates (notably Mistral) do not guarantee that a "generation
                # prompt" rendering is a strict prefix of a later rendering that
                # includes the assistant message.
                rendered_wo = tokenizer.apply_chat_template(
                    messages[:i], add_generation_prompt=False, tokenize=False)
                if not rendered_wo.startswith(prev_rendered):
                    # Skip extremely rare datapoints where the chat template rendering is not
                    # prefix-consistent between prefixes. Returning an empty sample keeps the
                    # training run robust without risking label misalignment.
                    return [], []

                delta_prompt = rendered_wo[len(prev_rendered):]
                prompt_ids = tokenizer.encode(delta_prompt, add_special_tokens=False)
                input_ids += prompt_ids
                labels += [IGNORE_INDEX] * len(prompt_ids)

                rendered_with = tokenizer.apply_chat_template(
                    messages[:i + 1], add_generation_prompt=False, tokenize=False)
                if not rendered_with.startswith(rendered_wo):
                    # Skip extremely rare datapoints where the chat template rendering is not
                    # prefix-consistent between prefixes. Returning an empty sample keeps the
                    # training run robust without risking label misalignment.
                    return [], []

                delta_target = rendered_with[len(rendered_wo):]
                target_ids = tokenizer.encode(delta_target, add_special_tokens=False)
                input_ids += target_ids
                if i in labeled_assistant_idxs:
                    labels += target_ids
                else:
                    labels += [IGNORE_INDEX] * len(target_ids)

                prev_rendered = rendered_with

        input_ids += [tokenizer.eos_token_id]
        labels += [tokenizer.eos_token_id]

        return input_ids, labels

    @staticmethod
    def _tokenize_datapoint_pretraining(
            datapoint: List[dict], tokenizer: PreTrainedTokenizer,
            **kwargs) -> Tuple[List[int], List[int]]:
        """Tokenize a single datapoint for pretraining.

        Parameters
        ----------
        datapoint : List[dict]
            Datapoint to tokenize.
        tokenizer : PreTrainedTokenizer
            Tokenizer used to tokenize data.

        Returns
        -------
        Tuple[List[int], List[int]]
            Input ids and labels.
        """

        assert 'text' in datapoint, "Datapoint must contain 'text' key."
        text = datapoint['text']

        input_ids = tokenizer(text).input_ids + [tokenizer.eos_token_id]
        labels = input_ids.copy()

        return input_ids, labels

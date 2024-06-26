# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import os
import torch
from functools import partial
from typing import Union
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core import mpu
from megatron.core.enums import ModelType
import megatron.model
from megatron.core.models.gpt import GPTModel
from megatron.training import pretrain
from megatron.core.transformer.spec_utils import import_module
from megatron.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    average_losses_across_data_parallel_group
)
from megatron.arguments import core_transformer_config_from_args
from tasks.gpt_chat.data import ChatDataset
# from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec


def get_tasks_args(parser):
    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title='tasks')

    group.add_argument('--task', type=str, required=True,
                       help='Task name.')
    group.add_argument('--epochs', type=int, default=None,
                       help='Number of finetunning epochs. Zero results in '
                       'evaluation only.')
    group.add_argument('--pretrained-checkpoint', type=str, default=None,
                       help='Pretrained checkpoint used for finetunning.')
    group.add_argument('--keep-last', action='store_true',
                       help='Keep the last batch (maybe incomplete) in'
                       'the data loader')
    group.add_argument('--train-data', nargs='+', default=None,
                       help='Whitespace separated paths or corpora names '
                       'for training.')
    group.add_argument('--valid-data', nargs='*', default=None,
                       help='path(s) to the validation data.')
    group.add_argument('--overlapping-eval', type=int, default=32,
                       help='Sliding window for overlapping evaluation.')
    group.add_argument('--strict-lambada', action='store_true',
                       help='Use more difficult formulation of lambada.')
    # Retriever args
    group.add_argument('--qa-data-dev', type=str, default=None,
                       help='Path to the QA dataset dev file.')
    group.add_argument('--qa-data-test', type=str, default=None,
                       help='Path to the QA dataset test file.')

    # Faiss arguments for retriever
    group.add_argument('--faiss-use-gpu', action='store_true',
                       help='Whether create the FaissMIPSIndex on GPU')
    group.add_argument('--faiss-match', type=str, default='string', \
                        choices=['regex', 'string'], help="Answer matching '\
                        'logic type")
    group.add_argument('--faiss-topk-retrievals', type=int, default=100,
                       help='Number of blocks to use as top-k during retrieval')

    # finetune for retriever
    group.add_argument('--eval-micro-batch-size', type=int, default=None,
                       help='Eval Batch size per model instance (local batch '
                            'size). Global batch size is local batch size '
                            'times data parallel size.')
    group.add_argument('--train-with-neg', action='store_true',
                       help='Whether to use negative examples during model '
                        'training')
    group.add_argument('--train-hard-neg', type=int, default=0,
                       help='Number of hard negative exmaples to use during '
                        'training')


    # parameters for Av.rank validation method
    # Following options/arguments have been taken directly from DPR codebase
    group.add_argument('--val-av-rank-hard-neg', type=int, default=30,
                        help='Av.rank validation: how many hard negatives to'
                        ' take from each question pool')
    group.add_argument('--val-av-rank-other-neg', type=int, default=30,
                        help='Av.rank validation: how many other negatives to'
                        ' take from each question pool')


    return parser


def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.model.GPTModel]:
    """Builds the model.

    If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.model.GPTModel]: The returned model
    """
    args = get_args()

    print_rank_0('building GPT model ...')
    config = core_transformer_config_from_args(get_args())

    if args.use_mcore_models:
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            # transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm)
            assert False, "Not implemented yet"

        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent
        )
    else:
        assert(args.context_parallel_size == 1), "Context parallelism is only supported with Megatron Core!"

        model = megatron.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process
        )

    return model


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator) 

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    return batch['tokens'], batch['labels'], batch['loss_mask'], batch['attention_mask'], batch['position_ids']


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
    """    
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    if args.context_parallel_size > 1:
        loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)])
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
        loss = loss[0] / loss[1]
    else:
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan(), (
            f'Rank {global_rank}: found NaN in local forward loss calculation. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss * args.context_parallel_size, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    # batch['tokens'], batch['labels'], batch['loss_mask'], batch['attention_mask'], batch['position_ids']
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def is_dataset_built_on_rank():
    return (mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()) and mpu.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    return GPTDatasetConfig(
        is_built_on_rank=is_dataset_built_on_rank,
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=args.data_path,
        blend_per_split=[args.train_data_path, args.valid_data_path, args.test_data_path],
        split=args.split,
        path_to_cache=args.data_cache_path,
        mock=args.mock_data,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Provide train and validation datasets."""
    args = get_args()
    tokenizer = get_tokenizer()

    train_dataset = ChatDataset('training', args.train_data,
                                tokenizer, args.seq_length, pad_to_max_length=True)
    valid_dataset = ChatDataset('validation', args.valid_data,
                                tokenizer, args.seq_length, pad_to_max_length=True)
    test_dataset = ChatDataset('test', args.valid_data,
                                tokenizer, args.seq_length, pad_to_max_length=True)

    return train_dataset, valid_dataset, test_dataset


def main():
    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             extra_args_provider=get_tasks_args)


if __name__ == "__main__":
    main()
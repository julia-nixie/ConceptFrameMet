import os
import sys
import pickle
import random
import copy
import numpy as np

import torch
import torch.nn as nn

from tqdm import tqdm, trange
from collections import OrderedDict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup

from utils import Config, Logger, make_log_dir
from modeling import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification_SPV,
    AutoModelForSequenceClassification_MIP,
    AutoModelForSequenceClassification_SPV_MIP,
    FrameMelBert,
    FrameLogitsMelBert,
    MultiTaskMelbert,
    SourceLogitsMelBert,
    FrameSourceLogitsMelBert
)
from modeling_qa_source import SourceLogitsMelBert_QA
from modeling_qa_simple import SimpleSourceQAMelBert
from modeling_qa_soft_confidence import SoftConfidenceSourceQAMelBert
from modeling_qa_adaptive import AdaptiveSourceQAMelBert
from modeling_qa_hidden import HiddenStateSourceQAMelBert
from modeling_qa_frame import SimpleFrameQAMelBert
from model import FrameFinder
from run_classifier_dataset_utils import processors, output_modes, compute_metrics
from data_loader import load_train_data, load_train_data_kf, load_test_data, load_dev_data, load_frame_data
from debug_frames_sources import load_label_mappings, debug_sample
from pprint import pprint

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
ARGS_NAME = "training_args.bin"


def validate_labels(label_ids, num_labels, step, logger):
    """Validate that all labels are in the valid range [0, num_labels)
    
    This prevents the CUDA error: Assertion `t >= 0 && t < n_classes` failed
    """
    min_label = label_ids.min().item()
    max_label = label_ids.max().item()
    
    if min_label < 0 or max_label >= num_labels:
        logger.error(f"❌ Invalid labels at step {step}!")
        logger.error(f"   Label range: [{min_label}, {max_label}]")
        logger.error(f"   Expected range: [0, {num_labels-1}]")
        invalid_mask = (label_ids < 0) | (label_ids >= num_labels)
        logger.error(f"   Invalid indices: {torch.where(invalid_mask)[0].tolist()}")
        logger.error(f"   Invalid values: {label_ids[invalid_mask].tolist()}")
        
        # Clamp labels to valid range as emergency fix
        logger.warning(f"   ⚠️  Clamping labels to valid range [0, {num_labels-1}]")
        label_ids = torch.clamp(label_ids, 0, num_labels - 1)
    
    return label_ids


def main():
    # read configs

    # apply system arguments if exist
    argv = sys.argv[1:]
    # main_conf_path="/user/HS502/yl02706/MetaphorFrame/"
    main_conf_path="./"
    config = Config(main_conf_path=main_conf_path)
    print(argv)
    if len(argv) > 0:
        cmd_arg = OrderedDict()
        argvs = " ".join(sys.argv[1:]).split(" ")
        i = 0
        while i < len(argvs):
            arg_name = argvs[i].strip("-")
            # Check if this is a boolean flag (no value follows or next item is another flag)
            if i + 1 >= len(argvs) or argvs[i + 1].startswith("--"):
                cmd_arg[arg_name] = True
                i += 1
            else:
                arg_value = argvs[i + 1]
                cmd_arg[arg_name] = arg_value
                i += 2
        config.update_params(cmd_arg)

    args = config
    # pprint(args.__dict__)

    # logger
    if "saves" in args.bert_model:
        log_dir = args.bert_model
        logger = Logger(log_dir)
        config = Config(main_conf_path=log_dir)
        
        # Save command-line arguments before loading saved config
        cmd_line_args = {}
        argv = sys.argv[1:]
        if len(argv) > 0:
            argvs = " ".join(sys.argv[1:]).split(" ")
            i = 0
            while i < len(argvs):
                arg_name = argvs[i].strip("-")
                # Check if this is a boolean flag (no value follows or next item is another flag)
                if i + 1 >= len(argvs) or argvs[i + 1].startswith("--"):
                    cmd_line_args[arg_name] = True
                    i += 1
                else:
                    arg_value = argvs[i + 1]
                    cmd_line_args[arg_name] = arg_value
                    i += 2
        
        # Update args with saved config
        args.__dict__.update(config.__dict__)
        
        # When loading from saves, default to NOT training (evaluation mode)
        # unless explicitly specified in command line
        if 'do_train' not in cmd_line_args:
            args.do_train = False
        
        # Override with command-line arguments (these take precedence)
        for key, value in cmd_line_args.items():
            # Convert string "False"/"True" to boolean
            if isinstance(value, str):
                if value.lower() == "false":
                    value = False
                elif value.lower() == "true":
                    value = True
            setattr(args, key, value)
        
        args.log_dir = log_dir
    else:
        if not os.path.exists(args.logging_dir):
            os.mkdir(args.logging_dir)
        # Include seed in the model directory name
        model_dir_name = f"{args.bert_model}_seed{args.seed}"
        log_dir = make_log_dir(os.path.join(args.logging_dir, model_dir_name))
        logger = Logger(log_dir)
        config.save(log_dir)
        args.logging_dir = log_dir
        args.log_dir = log_dir


    # set CUDA devices
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    logger.info("device: {} n_gpu: {}".format(device, args.n_gpu))

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # get dataset and processor
    task_name = args.task_name.lower()
    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()
    args.num_labels = len(label_list)

    # build tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    if args.multitask:
        frame_tokenizer = AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case, add_prefix_space=True)

    model = load_pretrained_model(args, tokenizer)
    model = model.to(device)

    # Clear CUDA cache and check memory after loading models
    if args.n_gpu > 0:
        torch.cuda.empty_cache()
        logger.info(f"Models loaded. GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated")

    ########### Training ###########
    # VUA-18 / VUA-20
    if args.do_train and args.task_name == "vua":
        train_dataloader = load_train_data(
            args, logger, processor, task_name, label_list, tokenizer, output_mode
        )
        if args.multitask:
            assert args.model_type == "FrameMelbert", "Multitask only works with FrameBERT"
            train_frame_dl, eval_frame_dl = load_frame_data(frame_tokenizer, args, melbert_data_size = len(train_dataloader.dataset))
        else:
            train_frame_dl, eval_frame_dl = None, None
        model, best_result = run_train(
            args,
            logger,
            model,
            train_dataloader,
            processor,
            task_name,
            label_list,
            tokenizer,
            output_mode,
            train_frame_dl=train_frame_dl
        )

    # TroFi / MOH-X (K-fold)
    elif args.do_train and args.task_name == "trofi":
        # For k-fold, we train all folds together epoch by epoch
        model, best_result = run_train_kfold(
            args,
            logger,
            model,
            processor,
            task_name,
            label_list,
            tokenizer,
            output_mode,
        )
        
        logger.info(f"-----Best Averaged Result-----")
        for key in sorted(best_result.keys()):
            logger.info(f"  {key} = {str(best_result[key])}")

    # Load trained model
    if "saves" in args.bert_model:
        model = load_trained_model(args, model, tokenizer)

    ########### Inference ###########
    # VUA-18 / VUA-20
    if (args.do_eval or args.do_test) and task_name == "vua":
        # if test data is genre or POS tag data
        if ("genre" in args.data_dir) or ("pos" in args.data_dir):
            if "genre" in args.data_dir:
                targets = ["acad", "conv", "fict", "news"]
            elif "pos" in args.data_dir:
                targets = ["adj", "adv", "noun", "verb"]
            orig_data_dir = args.data_dir
            for idx, target in tqdm(enumerate(targets)):
                logger.info(f"====================== Evaluating {target} =====================")
                args.data_dir = os.path.join(orig_data_dir, target)
                all_guids, eval_dataloader, eval_examples = load_test_data(
                    args, logger, processor, task_name, label_list, tokenizer, output_mode
                )
                run_eval(args, logger, model, eval_dataloader, all_guids, task_name, processor, write_detailed=True, eval_examples=eval_examples)
        else:
            all_guids, eval_dataloader, eval_examples = load_test_data(
                args, logger, processor, task_name, label_list, tokenizer, output_mode
            )
            run_eval(args, logger, model, eval_dataloader, all_guids, task_name, processor, write_detailed=True, eval_examples=eval_examples)

    # TroFi / MOH-X (K-fold)
    elif (args.do_eval or args.do_test) and args.task_name == "trofi":
        logger.info(f"***** Evaluating with {args.data_dir}")
        k_result = []
        for k in tqdm(range(10), desc="K-fold"):
            all_guids, eval_dataloader, eval_examples = load_test_data(
                args, logger, processor, task_name, label_list, tokenizer, output_mode, k
            )
            result = run_eval(args, logger, model, eval_dataloader, all_guids, task_name, processor, write_detailed=True, k=k, eval_examples=eval_examples)
            k_result.append(result)

        # Calculate average result
        avg_result = copy.deepcopy(k_result[0])
        for result in k_result[1:]:
            for k, v in result.items():
                avg_result[k] += v
        for k, v in avg_result.items():
            avg_result[k] /= len(k_result)

        logger.info(f"-----Averge Result-----")
        for key in sorted(avg_result.keys()):
            logger.info(f"  {key} = {str(avg_result[key])}")
    logger.info(f"Saved to {logger.log_dir}")


def run_train(
    args,
    logger,
    model,
    train_dataloader,
    processor,
    task_name,
    label_list,
    tokenizer,
    output_mode,
    train_frame_dl=None,
    k=None,
):
    tr_loss = 0
    num_train_optimization_steps = len(train_dataloader) * args.num_train_epoch

    # Prepare optimizer, scheduler
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    if args.lr_schedule != False or args.lr_schedule.lower() != "none":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(args.warmup_epoch * len(train_dataloader)),
            num_training_steps=num_train_optimization_steps,
        )

    logger.info("***** Running training *****")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  Num steps = { num_train_optimization_steps}")

    # Clear CUDA cache before training
    if args.n_gpu > 0:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        logger.info(f"GPU Memory before training: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # Run training
    model.train()
    max_val_f1 = -1
    max_result = {}
    
    # Early stopping
    patience = getattr(args, 'early_stopping_patience', 20)  # Default 3 epochs
    epochs_without_improvement = 0
    logger.info(f"  Early stopping patience = {patience} epochs")
    logger.info(f"  ⚠️  MODEL SELECTION CRITERION: Binary F1 (positive class F1)")
    
    # Load label mappings for debugging
    frame_id2label, source_id2label = load_label_mappings()
    if frame_id2label:
        logger.info(f"  Loaded {len(frame_id2label)} frame labels for debugging")
    if source_id2label:
        logger.info(f"  Loaded {len(source_id2label)} source labels for debugging")
    
    # Debug counter for first 3 samples
    debug_sample_count = 0
    
    for epoch in trange(int(args.num_train_epoch), desc="Epoch"):
        tr_loss = 0
        
        # Zip dataloaders if using multitask learning (must be done per epoch)
        epoch_dataloader = zip(train_dataloader, train_frame_dl) if train_frame_dl is not None else train_dataloader
        
        for step, batch in enumerate(tqdm(epoch_dataloader, desc="Iteration")):
            # move batch data to gpuf
            if train_frame_dl is not None:
                batch, frame_batch = batch
                frame_batch = tuple(t.to(args.device) for t in frame_batch)
                (frame_attention_mask, frame_labels, frame_input_ids, frame_token_type) = frame_batch

            batch = tuple(t.to(args.device) for t in batch)

            if args.model_type in ["MELBERT_MIP", "MELBERT", "FrameMelbert", "SourceLogitsMelbert", "SourceLogitsMelBert_QA", "FrameSourceLogitsMelbert", "SimpleSourceQAMelBert", "HiddenStateSourceQAMelBert", "SoftConfidenceSourceQAMelBert", "SimpleFrameQAMelBert", "AdaptiveSourceQAMelBert"]:
                if args.spvmask or args.spvmaskcls:
                    (
                        input_ids,
                        input_mask,
                        segment_ids,
                        label_ids,
                        input_ids_2,
                        input_mask_2,
                        segment_ids_2,
                        input_with_mask_ids
                    ) = batch
                else:
                    (
                        input_ids,
                        input_mask,
                        segment_ids,
                        label_ids,
                        input_ids_2,
                        input_mask_2,
                        segment_ids_2,
                    ) = batch
                    input_with_mask_ids=None
            else:
                input_ids, input_mask, segment_ids, label_ids = batch

            # DEBUG: Show frame/source predictions for first 3 training samples
            if epoch == 0 and debug_sample_count < 3 and args.model_type in ["FrameMelbert", "SourceLogitsMelbert", "SourceLogitsMelBert_QA", "FrameSourceLogitsMelbert", "SimpleSourceQAMelBert", "SimpleFrameQAMelBert", "SoftConfidenceSourceQAMelBert", "AdaptiveSourceQAMelBert"]:
                for i in range(min(3 - debug_sample_count, input_ids.size(0))):
                    
                    
                    
                    if args.model_type == "SimpleSourceQAMelBert":
                     
                        # Special debug for SimpleSourceQAMelBert - show predicted source
                     
                        logger.info(f"\n{'='*80}")
                     
                        logger.info(f"🎯 DEBUG: SimpleSourceQAMelBert - Training Sample {debug_sample_count + 1}")
                     
                        logger.info(f"{'='*80}")
                     
                        
                     
                        # Decode sentence and target
                     
                        sentence = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                     
                        target_positions = (segment_ids[i] == 1).nonzero(as_tuple=True)[0]
                     
                        if len(target_positions) > 0:
                     
                            target_tokens = input_ids[i][target_positions]
                     
                            target_word = tokenizer.decode(target_tokens, skip_special_tokens=True)
                     
                        else:
                     
                            target_word = "unknown"
                     
                        
                     
                        logger.info(f"  Sentence: {sentence}")
                     
                        logger.info(f"  Target word: '{target_word}'")
                     
                        
                     
                        # Get predicted source from model (with details)
                     
                        with torch.no_grad():
                     
                            # Temporarily set model to eval mode for clean prediction
                     
                            model.eval()
                     
                            predicted_sources, source_embedding = model.predict_source_with_details(
                     
                                input_ids[i:i+1], 
                     
                                (segment_ids[i:i+1] == 1),
                     
                                input_mask[i:i+1]
                     
                            )
                     
                            model.train()
                     
                        
                     
                        # Extract predicted source for this sample
                     
                        source_id, source_name, confidence = predicted_sources[0]
                     
                        logger.info(f"  🎯 Predicted source: '{source_name}' (ID: {source_id}, confidence: {confidence:.4f})")
                     
                        logger.info(f"  ✓ Source embedding shape: {source_embedding.shape}")
                     
                        logger.info(f"{'='*80}\n")
                    

                    elif args.model_type == "SimpleFrameQAMelBert":

                        # Special debug for SimpleFrameQAMelBert - show predicted frame

                        logger.info(f"\n{'='*80}")

                        logger.info(f"🎯 DEBUG: SimpleFrameQAMelBert - Training Sample {debug_sample_count + 1}")

                        logger.info(f"{'='*80}")

                        # Decode sentence and target

                        sentence = tokenizer.decode(input_ids[i], skip_special_tokens=True)

                        target_positions = (segment_ids[i] == 1).nonzero(as_tuple=True)[0]

                        if len(target_positions) > 0:

                            target_tokens = input_ids[i][target_positions]

                            target_word = tokenizer.decode(target_tokens, skip_special_tokens=True)

                        else:

                            target_word = "unknown"

                        logger.info(f" Sentence: {sentence}")

                        logger.info(f" Target word: '{target_word}'")

                        # Get predicted frame from model (with details)

                        with torch.no_grad():

                        # Temporarily set model to eval mode for clean prediction

                            model.eval()

                            predicted_frames, frame_embedding = model.predict_frame_with_details(

                            input_ids[i:i+1],

                            (segment_ids[i:i+1] == 1),

                            input_mask[i:i+1]

                            )

                            model.train()

                            # Extract predicted frame for this sample

                        frame_id, frame_name, confidence = predicted_frames[0]

                        logger.info(f" 🎯 Predicted frame: '{frame_name}' (ID: {frame_id}, confidence: {confidence:.4f})")

                        logger.info(f" ✓ Frame embedding shape: {frame_embedding.shape}")

                        logger.info(f"{'='*80}\n")
                     
                    else:
                     
                        debug_sample(logger, model, input_ids, segment_ids, tokenizer, i, 
                     
                                    frame_id2label, source_id2label, mode="train",
                     
                                    input_ids_2=input_ids_2, segment_ids_2=segment_ids_2)
                     
                    debug_sample_count += 1
                     
                    if debug_sample_count >= 3:
                     
                        break


                
                     

                     
            # Validate labels to prevent CUDA assertion error
                     
            label_ids = validate_labels(label_ids, args.num_labels, step, logger)
                     
            
                     
            # compute loss values
                     
            try:
                     
                if args.model_type in ["BERT_SEQ", "BERT_BASE", "MELBERT_SPV"]:
                     
                    logits = model(
                     
                        input_ids,
                     
                        target_mask=(segment_ids == 1),
                     
                        token_type_ids=segment_ids,
                     
                        attention_mask=input_mask,
                     
                    )
                     
                    loss_fct = nn.NLLLoss(weight=torch.Tensor([1, args.class_weight]).to(args.device))
                     
                    loss = loss_fct(logits.view(-1, args.num_labels), label_ids.view(-1))
                     
                elif args.model_type in ["MELBERT_MIP", "MELBERT", "FrameMelbert", "SourceLogitsMelbert", "SourceLogitsMelBert_QA", "FrameSourceLogitsMelbert", "SimpleSourceQAMelBert", "SimpleFrameQAMelBert", "HiddenStateSourceQAMelBert", "SoftConfidenceSourceQAMelBert", "AdaptiveSourceQAMelBert"]:
                     
                    if train_frame_dl is not None:
                     
                        logits, frame_loss = model(
                     
                            input_ids,
                     
                            input_ids_2,
                     
                            target_mask=(segment_ids == 1),
                     
                            target_mask_2=segment_ids_2,
                     
                            attention_mask_2=input_mask_2,
                     
                            frame_input_ids = frame_input_ids,
                     
                            frame_attention_mask = frame_attention_mask,
                     
                            frame_token_type = frame_token_type,
                     
                            frame_labels = frame_labels,
                     
                            token_type_ids=segment_ids,
                     
                            attention_mask=input_mask,
                     
                            input_with_mask_ids=input_with_mask_ids
                     
                        )
                     
                    else:
                     
                        logits = model(
                     
                            input_ids,
                     
                            input_ids_2,
                     
                            target_mask=(segment_ids == 1),
                     
                            target_mask_2=segment_ids_2,
                     
                            attention_mask_2=input_mask_2,
                     
                            token_type_ids=segment_ids,
                     
                            attention_mask=input_mask,
                     
                            input_with_mask_ids=input_with_mask_ids
                     
                        )
                     
                    loss_fct = nn.NLLLoss(weight=torch.Tensor([1, args.class_weight]).to(args.device))
                     
                    loss = loss_fct(logits.view(-1, args.num_labels), label_ids.view(-1))
                     
                    if train_frame_dl is not None:
                     
                        loss += frame_loss
                     

                     
                # average loss if on multi-gpu.
                     
                if args.n_gpu > 1:
                     
                    loss = loss.mean()
                     

                     
                loss.backward()
                     
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                     
                optimizer.step()
                     

                     
                if args.lr_schedule != False or args.lr_schedule.lower() != "none":
                     
                    scheduler.step()
                     

                     
                optimizer.zero_grad()
                     

                     
                tr_loss += loss.item()
                     
                
                     
            except RuntimeError as e:
                     
                if "CUDA" in str(e) or "cuBLAS" in str(e) or "Assertion" in str(e):
                     
                    logger.error(f"❌ CUDA/Assertion error at step {step}: {e}")
                     
                    logger.info("⚠️  Skipping batch and clearing CUDA cache...")
                     
                    torch.cuda.empty_cache()
                     
                    optimizer.zero_grad()
                     
                    continue
                     
                else:
                     
                    raise e
                     

                     
        cur_lr = optimizer.param_groups[0]["lr"]
                     
        logger.info(f"[epoch {epoch+1}] ,lr: {cur_lr} ,tr_loss: {tr_loss}")
                     

                     
        # evaluate on DEV set during training
                     
        if args.do_eval:
                     
            # For TroFi k-fold, use test fold as validation set
                     
            if task_name == "trofi" and k is not None:
                     
                all_guids, eval_dataloader, eval_examples = load_test_data(
                     
                    args, logger, processor, task_name, label_list, tokenizer, output_mode, k
                     
                )
                     
            else:
                     
                all_guids, eval_dataloader, eval_examples = load_dev_data(
                     
                    args, logger, processor, task_name, label_list, tokenizer, output_mode, k
                     
                )
                     
            result = run_eval(args, logger, model, eval_dataloader, all_guids, task_name, processor, epoch=epoch+1, eval_examples=eval_examples)
                     

                     
            # Always save checkpoint for this epoch
                     
            epoch_save_dir = os.path.join(args.log_dir, f"epoch_{epoch+1}")
                     
            os.makedirs(epoch_save_dir, exist_ok=True)
                     
            original_log_dir = args.log_dir
                     
            args.log_dir = epoch_save_dir
                     
            save_model(args, model, tokenizer)
                     
            args.log_dir = original_log_dir
                     
            logger.info(f"  ✓ Epoch {epoch+1} checkpoint saved to {epoch_save_dir}")
                     
            
                     
            # Run TEST on multiple datasets after each checkpoint
                     
            test_datasets = ['VUA18', 'VUA20', 'MOH-X', 'trofi']
                     
            logger.info(f"\n{'='*80}")
                     
            logger.info(f"Testing Epoch {epoch+1} on all test sets")
                     
            logger.info(f"{'='*80}")
                     
            
                     
            for test_dataset in test_datasets:
                     
                test_data_dir = f'data_all/{test_dataset}'
                     
                try:
                     
                    # Save original data_dir and temporarily switch to test dataset
                     
                    original_data_dir = args.data_dir
                     
                    args.data_dir = test_data_dir
                     
                    
                     
                    # Determine the task name and processor for this dataset
                     
                    if test_dataset in ['VUA18', 'VUA20']:
                     
                        test_task_name = 'vua'
                     
                        test_processor = processors[test_task_name]()
                     
                    elif test_dataset in ['MOH-X', 'trofi']:
                     
                        test_task_name = 'trofi'
                     
                        test_processor = processors[test_task_name]()
                     
                    
                     
                    logger.info(f"\n📊 Testing on {test_dataset}...")
                     
                    
                     
                    # For k-fold datasets (MOH-X, trofi), we need to evaluate all folds
                     
                    if test_dataset in ['MOH-X', 'trofi']:
                     
                        fold_results = []
                     
                        for k in range(10):
                     
                            all_test_guids, test_dataloader, test_examples = load_test_data(
                     
                                args, logger, test_processor, test_task_name, label_list, tokenizer, output_mode, k
                     
                            )
                     
                            # Write test outputs into the epoch checkpoint directory
                     
                            original_log_dir = args.log_dir
                     
                            args.log_dir = epoch_save_dir
                     
                            result = run_eval(
                     
                                args,
                     
                                logger,
                     
                                model,
                     
                                test_dataloader,
                     
                                all_test_guids,
                     
                                test_task_name,
                     
                                test_processor,
                     
                                write_to_file=True,
                     
                                write_detailed=True,
                     
                                epoch=epoch+1,
                     
                                k=k
                     
                            )
                     
                            fold_results.append(result)
                     
                            args.log_dir = original_log_dir
                     
                        
                     
                        # Calculate and log average results for trofi
                     
                        avg_result = {}
                     
                        for key in fold_results[0].keys():
                     
                            avg_result[key] = np.mean([r[key] for r in fold_results])
                     
                        
                     
                        logger.info(f"  ✓ {test_dataset} - Avg F1: {avg_result['f1']:.4f}, Avg Macro F1: {avg_result['f1_macro']:.4f}")
                     
                        
                     
                        # Write averaged results to file
                     
                        results_file = os.path.join(epoch_save_dir, f"test_results_{test_dataset}_avg.txt")
                     
                        with open(results_file, "w") as f:
                     
                            f.write(f"Epoch {epoch+1} - {test_dataset} Average Results (10-fold)\n")
                     
                            f.write(f"{'='*50}\n")
                     
                            for key in sorted(avg_result.keys()):
                     
                                f.write(f"  {key} = {avg_result[key]:.4f}\n")
                     
                    else:
                     
                        # For VUA datasets (single test set)
                     
                        all_test_guids, test_dataloader, test_examples = load_test_data(
                     
                            args, logger, test_processor, test_task_name, label_list, tokenizer, output_mode
                     
                        )
                     
                        # Write test outputs into the epoch checkpoint directory
                     
                        original_log_dir = args.log_dir
                     
                        args.log_dir = epoch_save_dir
                     
                        result = run_eval(
                     
                            args,
                     
                            logger,
                     
                            model,
                     
                            test_dataloader,
                     
                            all_test_guids,
                     
                            test_task_name,
                     
                            test_processor,
                     
                            write_to_file=True,
                     
                            write_detailed=True,
                     
                            epoch=epoch+1
                     
                        )
                     
                        args.log_dir = original_log_dir
                     
                        logger.info(f"  ✓ {test_dataset} - F1: {result['f1']:.4f}, Macro F1: {result['f1_macro']:.4f}")
                     
                    
                     
                    # Restore original data_dir
                     
                    args.data_dir = original_data_dir
                     
                    
                     
                except Exception as e:
                     
                    logger.info(f"  ⚠️ Error testing on {test_dataset}: {e}")
                     
                    # Restore original data_dir even on error
                     
                    args.data_dir = original_data_dir
                     
            
                     
            logger.info(f"\n{'='*80}")
                     
            logger.info(f"Completed testing epoch {epoch+1} on all datasets")
                     
            logger.info(f"{'='*80}\n")
                     

                     
            # Save best model based on Binary F1 score
                     
            if result["f1"] > max_val_f1:
                     
                logger.info(f"  🎯 EPOCH {epoch+1}: New best Binary F1: {result['f1']:.4f} (previous best: {max_val_f1:.4f})")
                     
                logger.info(f"  📊 MACRO F1: {result['f1_macro']:.4f} | Micro F1: {result['f1_micro']:.4f}")
                     
                logger.info(f"  📝 Saving model (this will OVERWRITE previous best model)")
                     
                max_val_f1 = result["f1"]
                     
                max_result = result
                     
                max_result['best_epoch'] = epoch + 1
                     
                epochs_without_improvement = 0
                     
                save_model(args, model, tokenizer)
                     
                
                     
                # Save best epoch info to file
                     
                best_epoch_file = os.path.join(args.log_dir, "best_epoch.txt")
                     
                with open(best_epoch_file, "w") as f:
                     
                    f.write(f"Best Epoch: {epoch + 1}\n")
                     
                    f.write(f"Best Binary F1: {max_val_f1:.4f}\n")
                     
                    f.write(f"MACRO F1: {result['f1_macro']:.4f}\n")
                     
                    f.write(f"Micro F1: {result['f1_micro']:.4f}\n")
                     
                    f.write(f"Precision (macro): {result['precision_macro']:.4f}\n")
                     
                    f.write(f"Recall (macro): {result['recall_macro']:.4f}\n")
                     
                
                     
                logger.info(f"  ✓ Model from EPOCH {epoch+1} saved to {args.log_dir}")
                     
            else:
                     
                epochs_without_improvement += 1
                     
                logger.info(f"  ⚠️  EPOCH {epoch+1}: No improvement (Binary F1={result['f1']:.4f}, best={max_val_f1:.4f})")
                     
                logger.info(f"  ⏳ Patience: {epochs_without_improvement}/{patience}")
                     
                
                     
                # Early stopping
                     
                if epochs_without_improvement >= patience:
                     
                    logger.info(f"  🛑 Early stopping triggered after {epoch + 1} epochs")
                     
                    logger.info(f"  📊 Best Binary F1 score: {max_val_f1:.4f} from EPOCH {max_result.get('best_epoch', 'unknown')}")
                     
                    break
                     
        
                     
        if args.do_shuffle_eval:
                     
            all_guids, eval_dataloader, eval_examples = load_dev_data(
                     
                args, logger, processor, task_name, label_list, tokenizer, output_mode, k
                     
            )
                     
            model.args.shuffle_concepts_in_batch = True
                     
            logger.info("^^^^^^^^ Shuffle eval ^^^^^^^ ")
                     
            result = run_eval(args, logger, model, eval_dataloader, all_guids, task_name, processor)
                     
            model.args.shuffle_concepts_in_batch = False
                     

                     
    logger.info(f"-----Best Result-----")
                     
    for key in sorted(max_result.keys()):
                     
        logger.info(f"  {key} = {str(max_result[key])}")
                     

                     
    return model, max_result
                     

                     

                     
def run_eval(args, logger, model, eval_dataloader, all_guids, task_name, processor=None, epoch=None, return_preds=False, write_to_file=True, write_detailed=False, k=None, eval_examples=None):
                     
    model.eval()
                     

                     
    eval_loss = 0
                     
    nb_eval_steps = 0
                     
    preds = []
                     
    pred_guids = []
                     
    out_label_ids = None
                     

                     
    for eval_batch in tqdm(eval_dataloader, desc="Evaluating"):
                     
        eval_batch = tuple(t.to(args.device) for t in eval_batch)
                     

                     
        if args.model_type in ["MELBERT_MIP", "MELBERT", "FrameMelbert", "SourceLogitsMelbert", "SourceLogitsMelBert_QA", "FrameSourceLogitsMelbert", "SimpleSourceQAMelBert", "SimpleFrameQAMelBert", "HiddenStateSourceQAMelBert", "SoftConfidenceSourceQAMelBert", "AdaptiveSourceQAMelBert"]:
                     
            if args.spvmask or args.spvmaskcls:
                     
                (
                     
                    input_ids,
                     
                    input_mask,
                     
                    segment_ids,
                     
                    label_ids,
                     
                    idx,
                     
                    input_ids_2,
                     
                    input_mask_2,
                     
                    segment_ids_2,
                     
                    input_with_mask_ids,
                     
                ) = eval_batch
                     
            else:
                     
                (
                     
                    input_ids,
                     
                    input_mask,
                     
                    segment_ids,
                     
                    label_ids,
                     
                    idx,
                     
                    input_ids_2,
                     
                    input_mask_2,
                     
                    segment_ids_2,
                     
                ) = eval_batch
                     
                input_with_mask_ids=None
                     
        else:
                     
            input_ids, input_mask, segment_ids, label_ids, idx = eval_batch
                     

                     
        with torch.no_grad():
                     
            # compute loss values
                     
            if args.model_type in ["BERT_BASE", "BERT_SEQ", "MELBERT_SPV"]:
                     
                logits = model(
                     
                    input_ids,
                     
                    target_mask=(segment_ids == 1),
                     
                    token_type_ids=segment_ids,
                     
                    attention_mask=input_mask,
                     
                )
                     
                loss_fct = nn.NLLLoss()
                     
                tmp_eval_loss = loss_fct(logits.view(-1, args.num_labels), label_ids.view(-1))
                     
                eval_loss += tmp_eval_loss.mean().item()
                     
                nb_eval_steps += 1
                     

                     
                if len(preds) == 0:
                     
                    preds.append(logits.detach().cpu().numpy())
                     
                    pred_guids.append([all_guids[i] for i in idx])
                     
                    out_label_ids = label_ids.detach().cpu().numpy()
                     
                else:
                     
                    preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)
                     
                    pred_guids[0].extend([all_guids[i] for i in idx])
                     
                    out_label_ids = np.append(
                     
                        out_label_ids, label_ids.detach().cpu().numpy(), axis=0
                     
                    )
                     

                     
            elif args.model_type in ["MELBERT_MIP", "MELBERT", "FrameMelbert", "SourceLogitsMelbert", "SourceLogitsMelBert_QA", "FrameSourceLogitsMelbert", "SimpleSourceQAMelBert", "SimpleFrameQAMelBert", "HiddenStateSourceQAMelBert", "SoftConfidenceSourceQAMelBert", "AdaptiveSourceQAMelBert"]:
                     
                logits = model(
                     
                    input_ids,
                     
                    input_ids_2,
                     
                    target_mask=(segment_ids == 1),
                     
                    target_mask_2=segment_ids_2,
                     
                    attention_mask_2=input_mask_2,
                     
                    token_type_ids=segment_ids,
                     
                    attention_mask=input_mask,
                     
                    input_with_mask_ids=input_with_mask_ids
                     
                )
                     
                loss_fct = nn.NLLLoss()
                     
                tmp_eval_loss = loss_fct(logits.view(-1, args.num_labels), label_ids.view(-1))
                     
                eval_loss += tmp_eval_loss.mean().item()
                     
                nb_eval_steps += 1
                     

                     
                if len(preds) == 0:
                     
                    preds.append(logits.detach().cpu().numpy())
                     
                    pred_guids.append([all_guids[i] for i in idx])
                     
                    out_label_ids = label_ids.detach().cpu().numpy()
                     
                else:
                     
                    preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)
                     
                    pred_guids[0].extend([all_guids[i] for i in idx])
                     
                    out_label_ids = np.append(
                     
                        out_label_ids, label_ids.detach().cpu().numpy(), axis=0
                     
                    )
                     

                     
    eval_loss = eval_loss / nb_eval_steps
                     
    preds = preds[0]
                     
    preds = np.argmax(preds, axis=1)
                     
    print(preds, out_label_ids)
                     

                     
    # compute metrics
                     
    result = compute_metrics(preds, out_label_ids)
                     

                     
    for key in sorted(result.keys()):
                     
        logger.info(f"  {key} = {str(result[key])}")
                     

                     
    # Extract dataset name from data_dir for output filenames
                     
    dataset_name = os.path.basename(args.data_dir.rstrip('/'))
                     
    
                     
    # Write results to file
                     
    if write_to_file:
                     
        # Use different filename for test results vs eval (dev) results
                     
        if write_detailed:
                     
            # This is test set evaluation (final evaluation)
                     
            results_file = os.path.join(args.log_dir, f"test_results_{dataset_name}.txt")
                     
            results_type = "Test Results"
                     
        else:
                     
            # This is dev set evaluation (during training)
                     
            results_file = os.path.join(args.log_dir, f"eval_results_{dataset_name}.txt")
                     
            results_type = "Evaluation Results"
                     
        
                     
        with open(results_file, "a") as f:
                     
            f.write(f"\n{'='*50}\n")
                     
            if epoch is not None:
                     
                f.write(f"{results_type} - {task_name} (Epoch {epoch})\n")
                     
            else:
                     
                f.write(f"{results_type} - {task_name}\n")
                     
            f.write(f"Data directory: {args.data_dir}\n")
                     
            f.write(f"Dataset: {dataset_name}\n")
                     
            if epoch is not None:
                     
                f.write(f"Epoch: {epoch}\n")
                     
            f.write(f"{'='*50}\n")
                     
            for key in sorted(result.keys()):
                     
                f.write(f"  {key} = {str(result[key])}\n")
                     
            f.write(f"  eval_loss = {eval_loss}\n")
                     
            f.write(f"\n")
                     
    
                     
    # Write detailed predictions ONLY for test set (not during training)
                     
    if write_detailed:
                     
        # Write basic predictions
                     
        predictions_file = os.path.join(args.log_dir, f"predictions_{dataset_name}.txt")
                     
        with open(predictions_file, "w") as f:
                     
            f.write("guid\tprediction\tlabel\n")
                     
            for guid, pred, label in zip(pred_guids[0], preds, out_label_ids):
                     
                f.write(f"{guid}\t{pred}\t{label}\n")
                     
        
                     
        # Write detailed predictions with original data + frame/source predictions
                     
        import pandas as pd
                     
        import json
                     
        detailed_predictions = []
                     
        
                     
        # Get frame and source predictions if available
                     
        frame_predictions_list = []
                     
        source_predictions_list = []
                     
        
                     
        # Load label mappings
                     
        frame_id2label = None
                     
        source_id2label = None
                     
        
                     
        if args.model_type in ["FrameMelbert", "FrameSourceLogitsMelbert"]:
                     
            logger.info("Extracting frame predictions from FrameFinder...")
                     
            try:
                     
                with open('frame_finder/frame_labels.json', 'r') as f:
                     
                    frame_label2id = json.load(f)
                     
                frame_id2label = {v: k for k, v in frame_label2id.items()}
                     
                logger.info(f"  Loaded {len(frame_id2label)} frame labels")
                     
            except Exception as e:
                     
                logger.warning(f"  Could not load frame labels: {e}")
                     
            
                     
        if args.model_type in ["SourceLogitsMelbert", "FrameSourceLogitsMelbert"]:
                     
            logger.info("Extracting source predictions from SourceFinder...")
                     
            try:
                     
                with open('source_finder/source_labels.json', 'r') as f:
                     
                    source_label2id = json.load(f)
                     
                source_id2label = {v: k for k, v in source_label2id.items()}
                     
                logger.info(f"  Loaded {len(source_id2label)} source labels")
                     
            except Exception as e:
                     
                logger.warning(f"  Could not load source labels: {e}")
                     
        
                     
        # Extract frame/source predictions by re-running through models
                     
        if frame_id2label is not None or source_id2label is not None:
                     
            logger.info("Extracting frame and/or source domain predictions...")
                     
            model.eval()
                     
            
                     
            # Debug counter for first 3 test samples
                     
            debug_eval_count = 0
                     
            
                     
            for eval_batch in tqdm(eval_dataloader, desc="Extracting predictions"):
                     
                eval_batch = tuple(t.to(args.device) for t in eval_batch)
                     
                
                     
                if args.model_type in ["MELBERT_MIP", "MELBERT", "FrameMelbert", "SourceLogitsMelbert", "FrameSourceLogitsMelbert"]:
                     
                    if args.spvmask or args.spvmaskcls:
                     
                        (input_ids, input_mask, segment_ids, label_ids, idx, 
                     
                         input_ids_2, input_mask_2, segment_ids_2, input_with_mask_ids) = eval_batch
                     
                    else:
                     
                        (input_ids, input_mask, segment_ids, label_ids, idx, 
                     
                         input_ids_2, input_mask_2, segment_ids_2) = eval_batch
                     
                else:
                     
                    input_ids, input_mask, segment_ids, label_ids, idx = eval_batch
                     
                
                     
                # DEBUG: Show frame/source predictions for first 3 test samples
                     
                if debug_eval_count < 3 and args.model_type in ["FrameMelbert", "SourceLogitsMelbert", "SourceLogitsMelBert_QA", "FrameSourceLogitsMelbert", "SimpleSourceQAMelBert", "SimpleFrameQAMelBert", "SoftConfidenceSourceQAMelBert", "AdaptiveSourceQAMelBert"]:
                     
                    from transformers import AutoTokenizer
                     
                    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
                     
                    for i in range(min(3 - debug_eval_count, input_ids.size(0))):
                     
                        debug_sample(logger, model, input_ids, segment_ids, tokenizer, i, 
                     
                                    frame_id2label, source_id2label, mode="eval",
                     
                                    input_ids_2=input_ids_2, segment_ids_2=segment_ids_2)
                     
                        debug_eval_count += 1
                     
                        if debug_eval_count >= 3:
                     
                            break
                     
                
                     
                with torch.no_grad():
                     
                    # Get frame predictions if model has frame_encoder
                     
                    if hasattr(model, 'frame_encoder') and frame_id2label is not None:
                     
                        frame_outputs = model.frame_encoder(
                     
                            input_ids,
                     
                            token_type_ids=(segment_ids == 1).int(),
                     
                            attention_mask=input_mask
                     
                        )
                     
                        
                     
                        # Check if frame_outputs has logits attribute (FrameFinder/FrameLogitsMelbert)
                     
                        # vs regular AutoModel (FrameMelbert with embeddings)
                     
                        if hasattr(frame_outputs, 'logits'):
                     
                            frame_logits = frame_outputs.logits
                     
                            frame_preds_batch = torch.argmax(frame_logits, dim=-1)
                     
                            
                     
                            # Extract predictions for target words only
                     
                            target_mask = (segment_ids == 1)
                     
                            for i in range(input_ids.size(0)):
                     
                                target_indices = torch.where(target_mask[i])[0]
                     
                                if len(target_indices) > 0:
                     
                                    # Get prediction for first target word
                     
                                    target_idx = target_indices[0].item()
                     
                                    frame_pred_id = frame_preds_batch[i, target_idx].item()
                     
                                    frame_pred = frame_id2label.get(frame_pred_id, f"UNK_{frame_pred_id}")
                     
                                    frame_predictions_list.append(frame_pred)
                     
                                else:
                     
                                    frame_predictions_list.append("_")
                     
                        else:
                     
                            # Frame encoder is AutoModel (returns embeddings, not logits)
                     
                            # Skip frame predictions for this model type
                     
                            logger.info("  Skipping frame predictions - model uses frame embeddings, not logits")
                     
                            frame_id2label = None  # Prevent further processing
                     
                    
                     
                    # Get source predictions if model has source_encoder
                     
                    if hasattr(model, 'source_encoder') and source_id2label is not None:
                     
                        source_outputs = model.source_encoder(
                     
                            input_ids,
                     
                            token_type_ids=(segment_ids == 1).int(),
                     
                            attention_mask=input_mask
                     
                        )
                     
                        source_logits = source_outputs.logits
                     
                        source_preds_batch = torch.argmax(source_logits, dim=-1)
                     
                        
                     
                        # Extract predictions for target words only
                     
                        target_mask = (segment_ids == 1)
                     
                        for i in range(input_ids.size(0)):
                     
                            target_indices = torch.where(target_mask[i])[0]
                     
                            if len(target_indices) > 0:
                     
                                # Get prediction for first target word
                     
                                target_idx = target_indices[0].item()
                     
                                source_pred_id = source_preds_batch[i, target_idx].item()
                     
                                source_pred = source_id2label.get(source_pred_id, f"UNK_{source_pred_id}")
                     
                                source_predictions_list.append(source_pred)
                     
                            else:
                     
                                source_predictions_list.append("O")
                     
        
                     
        # Use eval_examples if provided, otherwise load from processor
                     
        if eval_examples is None:
                     
            logger.info("Loading original test data for detailed output...")
                     
            if k is not None:
                     
                eval_examples = processor.get_test_examples(args.data_dir, k)
                     
            else:
                     
                eval_examples = processor.get_test_examples(args.data_dir)
                     
        
                     
        # Create guid-to-example mapping for efficient lookup
                     
        guid_to_example = {ex.guid: ex for ex in eval_examples}
                     
        
                     
        # Compile detailed results
                     
        for i, (guid, pred, label) in enumerate(zip(pred_guids[0], preds, out_label_ids)):
                     
            # Find matching example using dictionary lookup
                     
            example = guid_to_example.get(guid, None)
                     
            
                     
            entry = {
                     
                'guid': guid,
                     
                'text': example.text_a if example else '',
                     
                'target_word': example.text_b if example else '',
                     
                'metaphor_prediction': int(pred),
                     
                'metaphor_label': int(label)
                     
            }
                     
            
                     
            # Add frame prediction if available
                     
            if frame_predictions_list and i < len(frame_predictions_list):
                     
                entry['frame_prediction'] = frame_predictions_list[i]
                     
            
                     
            # Add source prediction if available
                     
            if source_predictions_list and i < len(source_predictions_list):
                     
                entry['source_prediction'] = source_predictions_list[i]
                     
            
                     
            detailed_predictions.append(entry)
                     
        
                     
        # Save as CSV
                     
        df = pd.DataFrame(detailed_predictions)
                     
        df.to_csv(os.path.join(args.log_dir, f"predictions_detailed_{dataset_name}.csv"), index=False)
                     
        
                     
        # Save as JSON
                     
        with open(os.path.join(args.log_dir, f"predictions_detailed_{dataset_name}.json"), "w") as f:
                     
            json.dump(detailed_predictions, f, indent=2)
                     
        
                     
        logger.info(f"Predictions written to {predictions_file}")
                     
        logger.info(f"Detailed predictions written to predictions_detailed_{dataset_name}.csv and predictions_detailed_{dataset_name}.json")
                     

                     
    if return_preds:
                     
        return preds
                     
    return result
                     

                     

                     
def run_train_kfold(
                     
    args,
                     
    logger,
                     
    model,
                     
    processor,
                     
    task_name,
                     
    label_list,
                     
    tokenizer,
                     
    output_mode,
                     
):
                     
    """Train with k-fold cross-validation: evaluate all folds after each epoch"""
                     
    
                     
    # Prepare k models and dataloaders
                     
    models = []
                     
    train_dataloaders = []
                     
    optimizers = []
                     
    schedulers = []
                     
    
                     
    logger.info(f"***** K-Fold Cross-Validation Training ({args.kfold} folds) *****")
                     
    logger.info(f"  Batch size = {args.train_batch_size}")
                     
    
                     
    # Initialize model, optimizer, scheduler for each fold
                     
    for k in range(args.kfold):
                     
        # Load fresh model for this fold
                     
        fold_model = load_pretrained_model(args, tokenizer)
                     
        fold_model = fold_model.to(args.device)
                     
        
                     
        # Load training data for this fold
                     
        fold_train_dataloader = load_train_data(
                     
            args, logger, processor, task_name, label_list, tokenizer, output_mode, k
                     
        )
                     
        
                     
        # Prepare optimizer
                     
        param_optimizer = list(fold_model.named_parameters())
                     
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
                     
        optimizer_grouped_parameters = [
                     
            {
                     
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                     
                "weight_decay": 0.01,
                     
            },
                     
            {
                     
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                     
                "weight_decay": 0.0,
                     
            },
                     
        ]
                     
        fold_optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
                     
        
                     
        # Prepare scheduler
                     
        num_train_optimization_steps = len(fold_train_dataloader) * args.num_train_epoch
                     
        if args.lr_schedule != False or args.lr_schedule.lower() != "none":
                     
            fold_scheduler = get_linear_schedule_with_warmup(
                     
                fold_optimizer,
                     
                num_warmup_steps=int(args.warmup_epoch * len(fold_train_dataloader)),
                     
                num_training_steps=num_train_optimization_steps,
                     
            )
                     
        else:
                     
            fold_scheduler = None
                     
        
                     
        models.append(fold_model)
                     
        train_dataloaders.append(fold_train_dataloader)
                     
        optimizers.append(fold_optimizer)
                     
        schedulers.append(fold_scheduler)
                     
    
                     
    # Early stopping
                     
    patience = getattr(args, 'early_stopping_patience', 3)
                     
    epochs_without_improvement = 0
                     
    max_avg_f1 = -1
                     
    best_result = {}
                     
    logger.info(f"  Early stopping patience = {patience} epochs")
                     
    logger.info(f"  ⚠️  MODEL SELECTION CRITERION: MACRO F1 (averaged across both classes)")
                     
    
                     
    # Train epoch by epoch
                     
    for epoch in trange(int(args.num_train_epoch), desc="Epoch"):
                     
        logger.info(f"\n{'='*80}")
                     
        logger.info(f"Epoch {epoch + 1}/{args.num_train_epoch}")
                     
        logger.info(f"{'='*80}")
                     
        
                     
        # Train all folds for this epoch
                     
        for k in range(args.kfold):
                     
            models[k].train()
                     
            tr_loss = 0
                     
            
                     
            for step, batch in enumerate(tqdm(train_dataloaders[k], desc=f"Training Fold {k}")):
                     
                batch = tuple(t.to(args.device) for t in batch)
                     
                
                     
                if args.model_type in ["MELBERT_MIP", "MELBERT", "FrameMelbert", "SourceLogitsMelbert", "SourceLogitsMelBert_QA", "FrameSourceLogitsMelbert"]:
                     
                    if args.spvmask or args.spvmaskcls:
                     
                        (input_ids, input_mask, segment_ids, label_ids,
                     
                         input_ids_2, input_mask_2, segment_ids_2, input_with_mask_ids) = batch
                     
                    else:
                     
                        (input_ids, input_mask, segment_ids, label_ids,
                     
                         input_ids_2, input_mask_2, segment_ids_2) = batch
                     
                        input_with_mask_ids = None
                     
                else:
                     
                    input_ids, input_mask, segment_ids, label_ids = batch
                     
                
                     
                # Forward pass
                     
                if args.model_type in ["BERT_SEQ", "BERT_BASE", "MELBERT_SPV"]:
                     
                    logits = models[k](input_ids, target_mask=(segment_ids == 1),
                     
                                      token_type_ids=segment_ids, attention_mask=input_mask)
                     
                elif args.model_type in ["MELBERT_MIP", "MELBERT", "FrameMelbert", "SourceLogitsMelbert", "SourceLogitsMelBert_QA", "FrameSourceLogitsMelbert"]:
                     
                    logits = models[k](input_ids, input_ids_2, target_mask=(segment_ids == 1),
                     
                                      target_mask_2=segment_ids_2, attention_mask_2=input_mask_2,
                     
                                      token_type_ids=segment_ids, attention_mask=input_mask,
                     
                                      input_with_mask_ids=input_with_mask_ids)
                     
                
                     
                loss_fct = nn.NLLLoss(weight=torch.Tensor([1, args.class_weight]).to(args.device))
                     
                loss = loss_fct(logits.view(-1, args.num_labels), label_ids.view(-1))
                     
                
                     
                if args.n_gpu > 1:
                     
                    loss = loss.mean()
                     
                
                     
                loss.backward()
                     
                torch.nn.utils.clip_grad_norm_(models[k].parameters(), 1.0)
                     
                optimizers[k].step()
                     
                
                     
                if schedulers[k] is not None:
                     
                    schedulers[k].step()
                     
                
                     
                optimizers[k].zero_grad()
                     
                tr_loss += loss.item()
                     
            
                     
            logger.info(f"  Fold {k}: train_loss = {tr_loss:.4f}")
                     
        
                     
        # Evaluate all folds after this epoch
                     
        logger.info(f"\nEvaluating all folds after epoch {epoch + 1}...")
                     
        fold_results = []
                     
        
                     
        for k in range(args.kfold):
                     
            # Load test data for this fold (used as validation)
                     
            all_guids, eval_dataloader = load_test_data(
                     
                args, logger, processor, task_name, label_list, tokenizer, output_mode, k
                     
            )
                     
            result = run_eval(args, logger, models[k], eval_dataloader, all_guids, task_name, processor,
                     
                            write_to_file=False, write_detailed=False)
                     
            fold_results.append(result)
                     
            logger.info(f"  Fold {k}: F1 = {result['f1']:.4f}")
                     
        
                     
        # Calculate average MACRO F1 across all folds
                     
        avg_f1_macro = np.mean([r['f1_macro'] for r in fold_results])
                     
        avg_result = {}
                     
        for key in fold_results[0].keys():
                     
            avg_result[key] = np.mean([r[key] for r in fold_results])
                     
        
                     
        logger.info(f"\n  Average MACRO F1 across all folds: {avg_f1_macro:.4f}")
                     
        logger.info(f"  Average Binary F1: {avg_result['f1']:.4f} | Average Micro F1: {avg_result['f1_micro']:.4f}")
                     
        
                     
        # Check if this is the best epoch (based on MACRO F1)
                     
        if avg_f1_macro > max_avg_f1:
                     
            logger.info(f"  🎯 New best average MACRO F1: {avg_f1_macro:.4f} (previous: {max_avg_f1:.4f})")
                     
            max_avg_f1 = avg_f1_macro
                     
            best_result = avg_result
                     
            epochs_without_improvement = 0
                     
            
                     
            # Save all fold models
                     
            for k in range(args.kfold):
                     
                fold_save_dir = os.path.join(args.log_dir, f"fold_{k}")
                     
                os.makedirs(fold_save_dir, exist_ok=True)
                     
                
                     
                # Temporarily change log_dir to save to fold-specific directory
                     
                original_log_dir = args.log_dir
                     
                args.log_dir = fold_save_dir
                     
                save_model(args, models[k], tokenizer)
                     
                args.log_dir = original_log_dir
                     
            
                     
            logger.info(f"  ✓ All fold models saved to {args.log_dir}/fold_*")
                     
        else:
                     
            epochs_without_improvement += 1
                     
            logger.info(f"  ⚠️  No improvement (MACRO F1={avg_f1_macro:.4f}, best={max_avg_f1:.4f})")
                     
            logger.info(f"  ⏳ Patience: {epochs_without_improvement}/{patience}")
                     
            
                     
            # Early stopping
                     
            if epochs_without_improvement >= patience:
                     
                logger.info(f"  🛑 Early stopping triggered after {epoch + 1} epochs")
                     
                logger.info(f"  📊 Best average MACRO F1 score: {max_avg_f1:.4f}")
                     
                break
                     
    
                     
    logger.info(f"\n{'='*80}")
                     
    logger.info(f"Training completed!")
                     
    logger.info(f"Best average MACRO F1: {max_avg_f1:.4f}")
                     
    logger.info(f"{'='*80}\n")
                     
    
                     
    # Return the first fold's model (for compatibility, though all folds are saved)
                     
    return models[0], best_result
                     

                     

                     
def load_pretrained_model(args, tokenizer=None):
                     
    # Pretrained Model
                     
    bert = AutoModel.from_pretrained(args.bert_model)
                     
    config = bert.config
                     
    config.type_vocab_size = 4
                     
    if "albert" in args.bert_model:
                     
        bert.embeddings.token_type_embeddings = nn.Embedding(
                     
            config.type_vocab_size, config.embedding_size
                     
        )
                     
    else:
                     
        bert.embeddings.token_type_embeddings = nn.Embedding(
                     
            config.type_vocab_size, config.hidden_size
                     
        )
                     
    bert._init_weights(bert.embeddings.token_type_embeddings)
                     

                     
    # Additional Layers
                     
    if args.model_type in ["BERT_BASE"]:
                     
        model = AutoModelForSequenceClassification(
                     
            args=args, Model=bert, config=config, num_labels=args.num_labels
                     
        )
                     
    if args.model_type == "BERT_SEQ":
                     
        model = AutoModelForTokenClassification(
                     
            args=args, Model=bert, config=config, num_labels=args.num_labels
                     
        )
                     
    if args.model_type == "MELBERT_SPV":
                     
        model = AutoModelForSequenceClassification_SPV(
                     
            args=args, Model=bert, config=config, num_labels=args.num_labels
                     
        )
                     
    if args.model_type == "MELBERT_MIP":
                     
        model = AutoModelForSequenceClassification_MIP(
                     
            args=args, Model=bert, config=config, num_labels=args.num_labels
                     
        )
                     
    if args.model_type == "MELBERT":
                     
        model = AutoModelForSequenceClassification_SPV_MIP(
                     
            args=args, Model=bert, config=config, num_labels=args.num_labels
                     
        )
                     
    if args.model_type == "FrameMelbert":
                     
        if args.frame_logits:
                     
            frame_model = FrameFinder.from_pretrained(args.frame_model, type_vocab_size=2)
                     
            model = FrameLogitsMelBert(
                     
                args=args, Model=bert, config=config, Frame_Model=frame_model, num_labels=args.num_labels
                     
            )
                     
        elif args.multitask:
                     
            frame_model = FrameFinder.from_pretrained(args.frame_model, type_vocab_size=2)
                     
            model = MultiTaskMelbert(
                     
                args=args, Model=bert, config=config, Frame_Model=frame_model, num_labels=args.num_labels
                     
            )
                     
        else:
                     
            frame_model = AutoModel.from_pretrained(args.frame_model, type_vocab_size=2, add_pooling_layer=False)
                     
            model = FrameMelBert(
                     
                args=args, Model=bert, config=config, Frame_Model=frame_model, num_labels=args.num_labels
                     
            )
                     
    if args.model_type == "SourceLogitsMelbert":
                     
        from transformers import RobertaForTokenClassification
                     
        print("\n" + "="*80)
                     
        print("🎯 Loading SOURCE PREDICTION MODEL (INSTEAD of Frame Model)")
                     
        print(f"   Source Model Path: {args.source_model}")
                     
        print(f"   Source Classes: 99 metaphor source domains")
                     
        print("="*80 + "\n")
                     
        source_model = RobertaForTokenClassification.from_pretrained(args.source_model, type_vocab_size=2)
                     
        model = SourceLogitsMelBert(
                     
            args=args, Model=bert, config=config, Source_Model=source_model, num_labels=args.num_labels
                     
        )
                     
        print("✓ SourceLogitsMelBert initialized - Using SOURCE logits ONLY (no frame logits)")
                     
        print(f"  - SPV input dim: {config.hidden_size * 2 + 2 * 100}")
                     
        print(f"  - MIP input dim: {config.hidden_size * 2 + 2 * 100}\n")
                     
    if args.model_type == "SourceLogitsMelBert_QA":
                     
        from transformers import RobertaForTokenClassification, AutoTokenizer
                     
        import sys
                     
        import os
                     
        # Add source_finder to path to import FrameAwareSourcePredictor
                     
        source_finder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'source_finder')
                     
        if source_finder_path not in sys.path:
                     
            sys.path.insert(0, source_finder_path)
                     
        
                     
        print("\n" + "="*80)
                     
        print("🎯 Loading SOURCE PREDICTION MODEL with QA-style for ISOLATED predictions")
                     
        print(f"   Mask Model (CONTEXT): {args.source_mask_model}")
                     
        print(f"   QA Model (ISOLATED): {args.source_qa_model}")
                     
        print(f"   Source Classes: 99 metaphor source domains")
                     
        print("="*80 + "\n")
                     
        
                     
        # Load mask-based model for CONTEXT predictions
                     
        source_mask_model = RobertaForTokenClassification.from_pretrained(args.source_mask_model, type_vocab_size=2)
                     
        source_mask_tokenizer = AutoTokenizer.from_pretrained(args.source_mask_model, do_lower_case=args.do_lower_case)
                     
        print("✓ Mask-based source model loaded (for CONTEXT)")
                     
        
                     
        # Load QA-style model for ISOLATED predictions using the custom FrameAwareSourcePredictor
                     
        from source_finder_qa_frame_integrated import FrameAwareSourcePredictor
                     
        from transformers import RobertaConfig
                     
        import json
                     
        
                     
        # Load config and model_config.json to get num_frames and num_sources
                     
        qa_config = RobertaConfig.from_pretrained(args.source_qa_model)
                     
        model_config_path = os.path.join(args.source_qa_model, 'model_config.json')
                     
        
                     
        if os.path.exists(model_config_path):
                     
            with open(model_config_path, 'r') as f:
                     
                model_config = json.load(f)
                     
            num_frames_qa = model_config.get('num_frames', 797)
                     
            num_sources_qa = model_config.get('num_sources', 100)
                     
        else:
                     
            # Fallback defaults
                     
            num_frames_qa = 797
                     
            num_sources_qa = qa_config.num_labels  # Should be 100
                     
        
                     
        # Create model instance
                     
        source_qa_model = FrameAwareSourcePredictor(
                     
            config=qa_config,
                     
            num_frames=num_frames_qa,
                     
            num_sources=num_sources_qa,
                     
            frame_model_path=None  # Frame features not needed for inference
                     
        )
                     
        
                     
        # Load trained weights
                     
        
                     
        state_dict_path = os.path.join(args.source_qa_model, 'pytorch_model.bin')
                     
        state_dict = torch.load(state_dict_path, map_location='cpu')
                     
        # Use strict=False to ignore missing position_ids (these are auto-generated buffers)
                     
        missing_keys, unexpected_keys = source_qa_model.load_state_dict(state_dict, strict=False)
                     
        if missing_keys:
                     
            print(f"  ⚠️  Missing keys (will be auto-initialized): {missing_keys}")
                     
        if unexpected_keys:
                     
            print(f"  ⚠️  Unexpected keys (ignored): {unexpected_keys}")
                     
        source_qa_model.eval()
                     
        
                     
        source_qa_tokenizer = AutoTokenizer.from_pretrained(args.source_qa_model, do_lower_case=args.do_lower_case, use_fast=False)
                     
        print("✓ QA-style source model loaded (for ISOLATED) - FrameAwareSourcePredictor")
                     
        print(f"  Model config: {num_frames_qa} frames, {num_sources_qa} sources")
                     
        
                     
        model = SourceLogitsMelBert_QA(
                     
            args=args, 
                     
            Model=bert, 
                     
            config=config, 
                     
            Source_Mask_Model=source_mask_model,
                     
            Source_QA_Model=source_qa_model,
                     
            source_mask_tokenizer=source_mask_tokenizer,
                     
            source_qa_tokenizer=source_qa_tokenizer,
                     
            melbert_tokenizer=tokenizer,
                     
            num_labels=args.num_labels
                     
        )
                     
        print("✓ SourceLogitsMelBert_QA initialized")
                     
        print("  - CONTEXT: Mask-based source prediction")
                     
        print("  - ISOLATED: QA-style source prediction [CLS] target [SEP] sentence [SEP]")
                     
        print(f"  - SPV input dim: {config.hidden_size * 2 + 2 * source_qa_model.config.num_labels}")
                     
        print(f"  - MIP input dim: {config.hidden_size * 2 + 2 * source_qa_model.config.num_labels}\n")
                     
    if args.model_type == "FrameSourceLogitsMelbert":
                     
        from transformers import RobertaForTokenClassification
                     
        print("\n" + "="*80)
                     
        print("🎯 Loading BOTH Frame Model AND Source Prediction Model")
                     
        print(f"   Frame Model Path: {args.frame_model}")
                     
        print(f"   Source Model Path: {args.source_model}")
                     
        print(f"   Frame Classes: 797 | Source Classes: 99")
                     
        print("="*80 + "\n")
                     
        frame_model = FrameFinder.from_pretrained(args.frame_model, type_vocab_size=2)
                     
        source_model = RobertaForTokenClassification.from_pretrained(args.source_model, type_vocab_size=2)
                     
        model = FrameSourceLogitsMelBert(
                     
            args=args, Model=bert, config=config, Frame_Model=frame_model, Source_Model=source_model, num_labels=args.num_labels
                     
        )
                     
        print("✓ FrameSourceLogitsMelBert initialized - Using BOTH Frame + Source logits")
                     
        print(f"  - SPV input dim: {config.hidden_size * 2 + 2 * 797 + 2 * 100}")
                     
        print(f"  - MIP input dim: {config.hidden_size * 2 + 2 * 797 + 2 * 100}\n")
                     
    
                     
    if args.model_type == "SimpleSourceQAMelBert":
                     
        from transformers import AutoTokenizer
                     
        from source_finder.source_finder_qa_frame_integrated import FrameAwareSourcePredictor
                     
        from transformers import RobertaConfig
                     
        import json
                     
        import os
                     
        
                     
        print("\n" + "="*80)
                     
        print("🎯 Loading SIMPLE QA-STYLE SOURCE PREDICTION MODEL")
                     
        print(f"   QA Model Path: {args.source_qa_model}")
                     
        print(f"   Approach: Predict TOP source → Encode source word → Replace isolated target")
                     
        print("="*80 + "\n")
                     
        
                     
        # Load QA-style model for source prediction
                     
        qa_config = RobertaConfig.from_pretrained(args.source_qa_model)
                     
        model_config_path = os.path.join(args.source_qa_model, 'model_config.json')
                     
        
                     
        if os.path.exists(model_config_path):
                     
            with open(model_config_path, 'r') as f:
                     
                model_config = json.load(f)
                     
            num_frames_qa = model_config.get('num_frames', 797)
                     
            num_sources_qa = model_config.get('num_sources', 100)
                     
        else:
                     
            num_frames_qa = 797
                     
            num_sources_qa = qa_config.num_labels
                     
        
                     
        # Create QA model instance
                     
        source_qa_model = FrameAwareSourcePredictor(
                     
            config=qa_config,
                     
            num_frames=num_frames_qa,
                     
            num_sources=num_sources_qa,
                     
            frame_model_path=None
                     
        )
                     
        
                     
        # Load trained weights
                     
        state_dict_path = os.path.join(args.source_qa_model, 'pytorch_model.bin')
                     
        state_dict = torch.load(state_dict_path, map_location='cpu')
                     
        missing_keys, unexpected_keys = source_qa_model.load_state_dict(state_dict, strict=False)
                     
        if missing_keys:
                     
            print(f"  ⚠️  Missing keys (will be auto-initialized): {missing_keys}")
                     
        if unexpected_keys:
                     
            print(f"  ⚠️  Unexpected keys (ignored): {unexpected_keys}")
                     
        source_qa_model.eval()
                     
        
                     
        source_qa_tokenizer = AutoTokenizer.from_pretrained(args.source_qa_model, do_lower_case=args.do_lower_case, use_fast=False)
                     
        print("✓ QA-style source model loaded")
                     
        
                     
        model = SimpleSourceQAMelBert(
                     
            args=args,
                     
            Model=bert,
                     
            config=config,
                     
            Source_QA_Model=source_qa_model,
                     
            source_qa_tokenizer=source_qa_tokenizer,
                     
            melbert_tokenizer=tokenizer,
                     
            num_labels=args.num_labels
                     
        )
                     
        print("✓ SimpleSourceQAMelBert initialized")
                     
        print("  - Predicts: TOP source domain (e.g., 'MACHINE', 'JOURNEY')")
                     
        print("  - Encodes: Predicted source word using encoder 2")
                     
        print("  - Replaces: isolated target with source embedding")
                     
        print(f"  - SPV input dim: {config.hidden_size * 2}")
                     
        print(f"  - MIP input dim: {config.hidden_size * 2}\n")



    if args.model_type  == "SoftConfidenceSourceQAMelBert":
                     
        from transformers import AutoTokenizer
                     
        from source_finder.source_finder_qa_frame_integrated import FrameAwareSourcePredictor
                     
        from transformers import RobertaConfig
                     
        import json
                     
        import os
                     
        
                     
        print("\n" + "="*80)
                     
        print("🎯 Loading SIMPLE QA-STYLE SOURCE PREDICTION MODEL")
                     
        print(f"   QA Model Path: {args.source_qa_model}")
                     
        print(f"   Approach: Predict TOP source → Encode source word → Replace isolated target")
                     
        print("="*80 + "\n")
                     
        
                     
        # Load QA-style model for source prediction
                     
        qa_config = RobertaConfig.from_pretrained(args.source_qa_model)
                     
        model_config_path = os.path.join(args.source_qa_model, 'model_config.json')
                     
        
                     
        if os.path.exists(model_config_path):
                     
            with open(model_config_path, 'r') as f:
                     
                model_config = json.load(f)
                     
            num_frames_qa = model_config.get('num_frames', 797)
                     
            num_sources_qa = model_config.get('num_sources', 100)
                     
        else:
                     
            num_frames_qa = 797
                     
            num_sources_qa = qa_config.num_labels
                     
        
                     
        # Create QA model instance
                     
        source_qa_model = FrameAwareSourcePredictor(
                     
            config=qa_config,
                     
            num_frames=num_frames_qa,
                     
            num_sources=num_sources_qa,
                     
            frame_model_path=None
                     
        )
                     
        
                     
        # Load trained weights
                     
        state_dict_path = os.path.join(args.source_qa_model, 'pytorch_model.bin')
                     
        state_dict = torch.load(state_dict_path, map_location='cpu')
                     
        missing_keys, unexpected_keys = source_qa_model.load_state_dict(state_dict, strict=False)
                     
        if missing_keys:
                     
            print(f"  ⚠️  Missing keys (will be auto-initialized): {missing_keys}")
                     
        if unexpected_keys:
                     
            print(f"  ⚠️  Unexpected keys (ignored): {unexpected_keys}")
                     
        source_qa_model.eval()
                     
        
                     
        source_qa_tokenizer = AutoTokenizer.from_pretrained(args.source_qa_model, do_lower_case=args.do_lower_case, use_fast=False)
                     
        print("✓ QA-style source model loaded")
                     
        
                     
        model = SoftConfidenceSourceQAMelBert(
                     
            args=args,
                     
            Model=bert,
                     
            config=config,
                     
            Source_QA_Model=source_qa_model,
                     
            source_qa_tokenizer=source_qa_tokenizer,
                     
            melbert_tokenizer=tokenizer,
                     
            num_labels=args.num_labels
                     
        )
                     
        print("✓ SoftConfidenceSourceQAMelBert initialized")
                     
        print("  - Predicts: TOP source domain (e.g., 'MACHINE', 'JOURNEY')")
                     
        print("  - Encodes: Predicted source word using encoder 2")
                     
        print("  - Replaces: isolated target with source embedding")
                     
        print(f"  - SPV input dim: {config.hidden_size * 2}")
                     
        print(f"  - MIP input dim: {config.hidden_size * 2}\n")


    if args.model_type == "AdaptiveSourceQAMelBert":
        from transformers import AutoTokenizer
        from source_finder.source_finder_qa_frame_integrated import FrameAwareSourcePredictor
        from transformers import RobertaConfig
        import json
        import os
    
        print("\n" + "="*80)
        print("🎯 Loading ADAPTIVE QA-STYLE SOURCE PREDICTION MODEL")
        print(f"   QA Model Path: {args.source_qa_model}")
        print(f"   Blend Mode: {getattr(args, 'source_blend_mode', 'replacement')}")
        print(f"   Use Mode: {getattr(args, 'source_use_mode', 'all')}")
        print("="*80 + "\n")
    
        # Load QA-style model
        qa_config = RobertaConfig.from_pretrained(args.source_qa_model)
        model_config_path = os.path.join(args.source_qa_model, 'model_config.json')
        
        if os.path.exists(model_config_path):
            with open(model_config_path, 'r') as f:
                model_config = json.load(f)
            num_frames_qa = model_config.get('num_frames', 797)
            num_sources_qa = model_config.get('num_sources', 100)
        else:
            num_frames_qa = 797
            num_sources_qa = qa_config.num_labels
        
        source_qa_model = FrameAwareSourcePredictor(
            config=qa_config,
            num_frames=num_frames_qa,
            num_sources=num_sources_qa,
            frame_model_path=args.frame_model
        )

        # Validate frame model path exists
        if not hasattr(args, 'frame_model') or args.frame_model is None:
            raise ValueError("AdaptiveSourceQAMelBert with frame-aware source model requires --frame_model argument")

        print(f"   Frame Model Path: {args.frame_model}")

        
        state_dict_path = os.path.join(args.source_qa_model, 'pytorch_model.bin')
        state_dict = torch.load(state_dict_path, map_location='cpu')
        missing_keys, unexpected_keys = source_qa_model.load_state_dict(state_dict, strict=False)
        source_qa_model.eval()
        
        print("loaded weights")
        source_qa_tokenizer = AutoTokenizer.from_pretrained(args.source_qa_model, do_lower_case=args.do_lower_case, use_fast=False)
        
        model = AdaptiveSourceQAMelBert(
            args=args,
            Model=bert,
            config=config,
            Source_QA_Model=source_qa_model,
            source_qa_tokenizer=source_qa_tokenizer,
            melbert_tokenizer=tokenizer,
            num_labels=args.num_labels
        )
        print("✓ AdaptiveSourceQAMelBert initialized with configurable blending")
                     
    
                     
    if args.model_type == "HiddenStateSourceQAMelBert":
                     
        from transformers import AutoTokenizer
                     
        from source_finder.source_finder_qa_frame_integrated import FrameAwareSourcePredictor
                     
        from transformers import RobertaConfig
                     
        import json
                     
        import os
                     
        
                     
        print("\n" + "="*80)
                     
        print("🎯 Loading HIDDEN STATE QA-STYLE SOURCE PREDICTION MODEL")
                     
        print(f"   QA Model Path: {args.source_qa_model}")
                     
        print(f"   Approach: Use QA model's hidden states → Project → Use as source embedding")
                     
        print(f"   Benefits: Richer semantics, more efficient, no information loss from argmax")
                     
        print("="*80 + "\n")
                     
        
                     
        # Load QA-style model for source prediction
                     
        qa_config = RobertaConfig.from_pretrained(args.source_qa_model)
                     
        model_config_path = os.path.join(args.source_qa_model, 'model_config.json')
                     
        
                     
        if os.path.exists(model_config_path):
                     
            with open(model_config_path, 'r') as f:
                     
                model_config = json.load(f)
                     
            num_frames_qa = model_config.get('num_frames', 797)
                     
            num_sources_qa = model_config.get('num_sources', 100)
                     
        else:
                     
            num_frames_qa = 797
                     
            num_sources_qa = qa_config.num_labels
                     
        
                     
        # Create QA model instance
                     
        source_qa_model = FrameAwareSourcePredictor(
                     
            config=qa_config,
                     
            num_frames=num_frames_qa,
                     
            num_sources=num_sources_qa,
                     
            frame_model_path=None
                     
        )
                     
        
                     
        # Load trained weights
                     
        state_dict_path = os.path.join(args.source_qa_model, 'pytorch_model.bin')
                     
        state_dict = torch.load(state_dict_path, map_location='cpu')
                     
        missing_keys, unexpected_keys = source_qa_model.load_state_dict(state_dict, strict=False)
                     
        if missing_keys:
                     
            print(f"  ⚠️  Missing keys (will be auto-initialized): {missing_keys}")
                     
        if unexpected_keys:
                     
            print(f"  ⚠️  Unexpected keys (ignored): {unexpected_keys}")
                     
        source_qa_model.eval()
                     
        
                     
        source_qa_tokenizer = AutoTokenizer.from_pretrained(args.source_qa_model, do_lower_case=args.do_lower_case, use_fast=False)
                     
        print("✓ QA-style source model loaded")
                     
        
                     
        model = HiddenStateSourceQAMelBert(
                     
            args=args,
                     
            Model=bert,
                     
            config=config,
                     
            Source_QA_Model=source_qa_model,
                     
            source_qa_tokenizer=source_qa_tokenizer,
                     
            melbert_tokenizer=tokenizer,
                     
            num_labels=args.num_labels
                     
        )
                     

                     
    
                     
    if args.model_type == "SimpleFrameQAMelBert":
                     
        from transformers import AutoTokenizer

        from transformers import RobertaForSequenceClassification
        import os
                     

                     
        
                     
        print("\n" + "="*80)
                     
        print("🎯 Loading SIMPLE QA-STYLE FRAME PREDICTION MODEL")
                     
        print(f"   QA Model Path: {args.frame_qa_model}")
                     
        print(f"   Approach: Predict TOP frame → Encode frame word → Replace isolated target")
                     
        print("="*80 + "\n")
                     
        
                     
        # Load QA-style model for frame prediction
                     
        from transformers import RobertaConfig
        frame_qa_config = RobertaConfig.from_pretrained(args.frame_qa_model)
        frame_qa_model = RobertaForSequenceClassification(frame_qa_config)
        state_dict_path = os.path.join(args.frame_qa_model, 'pytorch_model.bin')
        state_dict = torch.load(state_dict_path, map_location='cpu')
        missing_keys, unexpected_keys = frame_qa_model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"  ⚠️  Missing keys (will be auto-initialized): {missing_keys}")
        if unexpected_keys:
            print(f"  ⚠️  Unexpected keys (ignored): {unexpected_keys}")

        frame_qa_model.eval()
                     
        
                     
        frame_qa_tokenizer = AutoTokenizer.from_pretrained(args.frame_qa_model, do_lower_case=args.do_lower_case, use_fast=False)
                     
        print("✓ QA-style frame model loaded")
                     
        
                     
        model = SimpleFrameQAMelBert(
                     
            args=args,
                     
            Model=bert,
                     
            config=config,
                     
            Frame_QA_Model=frame_qa_model,
                     
            frame_qa_tokenizer=frame_qa_tokenizer,
                     
            melbert_tokenizer=tokenizer,
                     
            num_labels=args.num_labels
                     
        )
                     
        print("✓ SimpleFrameQAMelBert initialized")
                     
        print("  - Predicts: TOP frame (e.g., 'Communicate_categorization', 'Judgment')")
                     
        print("  - Encodes: Predicted frame word using encoder 2")
                     
        print("  - Replaces: isolated target with frame embedding")
                     
        print(f"  - SPV input dim: {config.hidden_size * 2}")
                     
        print(f"  - MIP input dim: {config.hidden_size * 2}\n")
                     
    model.to(args.device)
                     
    if args.n_gpu > 1 and not args.no_cuda:
                     
        model = torch.nn.DataParallel(model)
                     
    return model
                     

                     

                     
def save_model(args, model, tokenizer):
                     
    model_to_save = (
                     
        model.module if hasattr(model, "module") else model
                     
    )  # Only save the model it-self
                     

                     
    # If we save using the predefined names, we can load using `from_pretrained`
                     
    output_model_file = os.path.join(args.log_dir, WEIGHTS_NAME)
                     
    output_config_file = os.path.join(args.log_dir, CONFIG_NAME)
                     

                     
    torch.save(model_to_save.state_dict(), output_model_file)
                     
    model_to_save.config.to_json_file(output_config_file)
                     
    tokenizer.save_vocabulary(args.log_dir)
                     

                     
    # Good practice: save your training arguments together with the trained model
                     
    output_args_file = os.path.join(args.log_dir, ARGS_NAME)
                     
    torch.save(args, output_args_file)
                     

                     

                     
def load_trained_model(args, model, tokenizer):
                     
    # If we save using the predefined names, we can load using `from_pretrained`
                     
    output_model_file = os.path.join(args.log_dir, WEIGHTS_NAME)
                     

                     
    if hasattr(model, "module"):
                     
        model.module.load_state_dict(torch.load(output_model_file))
                     
    else:
                     
        model.load_state_dict(torch.load(output_model_file))
                     

                     
    return model
                     

                     

                     
if __name__ == "__main__":
                     
    main()

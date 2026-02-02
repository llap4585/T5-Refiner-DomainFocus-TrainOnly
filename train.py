# -*- coding: utf-8 -*-
# @Author: llap4585
# @Project: T5-Refiner-DomainFocus-TrainOnly
# @License: Apache-2.0
# @GitHub: https://github.com/llap4585/T5-Refiner-DomainFocus-TrainOnly
'''
Equipment List:
GPU: NVIDIA RTX 3060 Laptop (6GB)
Memory: 64GB DDR4 (upgraded prior to the price increase😄😆)
'''

import os
import time

from datasets import load_dataset
from transformers import TrainerCallback
from transformers import EarlyStoppingCallback
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments

class DelayedEarlyStopping(EarlyStoppingCallback):
    #6,000 Warmup Steps. Observation Window: Across 5 evaluation steps**************************************************************
    def __init__(self, start_step=6000, patience=5, threshold=0.001):
        super().__init__(early_stopping_patience=patience, early_stopping_threshold=threshold)
        self.start_step = start_step

    def on_evaluate(self, args, state, control, **kwargs):
        if state.global_step < self.start_step:
            return control 
        return super().on_evaluate(args, state, control, **kwargs)
    
class SafeDetailedProgressCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print("Start!")

    def on_step_end(self, args, state, control, **kwargs):

        if state.global_step % args.logging_steps == 0 and state.global_step > 0:
            elapsed = time.time() - self.start_time
            steps_done = state.global_step
            total_steps = state.max_steps
            percent = steps_done / total_steps * 100
            eta = elapsed / steps_done * (total_steps - steps_done) if steps_done > 0 else 0


            logs = state.log_history[-1] if state.log_history else {}
            loss = logs.get('loss', None)
            lr = logs.get('learning_rate', None)

            loss_str = f"{loss:.4f}" if loss is not None else "-"
            lr_str = f"{lr:.6f}" if lr is not None else "-"

            print(f"[Progress] Epoch: {state.epoch:.2f}/{args.num_train_epochs}, "
                  f"Step: {steps_done}/{total_steps} ({percent:.2f}%), "
                  f"Loss: {loss_str}, LR: {lr_str}, "
                  f"Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):

        elapsed = time.time() - self.start_time if self.start_time else 0.0
        if metrics is None:
            print("[EVAL] no metrics returned.")
            return

        metric_strs = []
        for k, v in metrics.items():
            try:
                metric_strs.append(f"{k}: {v:.6f}")
            except:
                metric_strs.append(f"{k}: {v}")
        print(f"[EVAL] Step {state.global_step}, Epoch {state.epoch:.2f}, Elapsed {elapsed:.1f}s, " + ", ".join(metric_strs))




# Data preprocessing function
def preprocess_function(examples):
    inputs = examples["inputs"]
    targets = examples["targets"]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

if __name__ == "__main__":
    # Model Path
    model_path = r" "
    # Load Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    # Output Path
    OUTPUT_DIR=model_path
    
    # Dataset Path
    train_file = "train.jsonl"
    valid_file = "valid.jsonl"

    # Load Dataset
    dataset = load_dataset("json", data_files={"train": train_file, "validation": valid_file})

    # Maximum Sequence Length
    max_input_length = 512
    max_target_length = 256
    
    # Apply Data Preprocessing
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    
    # Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./t5-finetuned",   #Model Save Path
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,
        load_best_model_at_end=True,
        
        metric_for_best_model="eval_loss",     
        greater_is_better=False, # Save Best Model based on Loss
        logging_steps=100, #Reporting Frequency
        per_device_train_batch_size=2,
    	gradient_accumulation_steps=8,
        per_device_eval_batch_size=2,

        num_train_epochs=5,
        predict_with_generate=True,
        
        fp16=True,  # Configurable by Device （Reduce Memory）
        gradient_checkpointing=True,
        group_by_length=True,
        dataloader_num_workers=2,
        
        
        learning_rate=5e-5,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",

    )
    

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
       
        callbacks=[SafeDetailedProgressCallback(),DelayedEarlyStopping()] 
    )
    

    try:
    
        trainer.train()
        #trainer.train(resume_from_checkpoint=True) #（Resume Training from Checkpoint）
    except KeyboardInterrupt:
        print("\n⚠️ Training Interruption...Save Latest Model")
        trainer.save_model(os.path.join(model_path, "./t5-finetuned/latest_model_interrupted"))
        trainer.save_model(os.path.join(model_path, "./t5-finetuned/best_model_interrupted"))
    
        raise
    trainer.save_model(os.path.join(OUTPUT_DIR, "./t5-finetuned/best_model"))
    
    print("✅ Save Latest Model")
    trainer.save_model(os.path.join(OUTPUT_DIR, "./t5-finetuned/latest_model"))

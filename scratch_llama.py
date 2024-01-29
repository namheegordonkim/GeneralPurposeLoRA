import glob
import os
from argparse import ArgumentParser

import numpy as np
import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig

from utils import get_mnist_data, EmpiricalEvalCallback, MNISTDatasetMaker


def main(args):

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    x_test, x_train, y_test, y_train = get_mnist_data(args.mnist_dir)

    dataset_maker = MNISTDatasetMaker()
    dataset_maker.setup(x_train)
    train_dataset = dataset_maker.make(x_train, y_train)
    valid_dataset = dataset_maker.make(x_test, y_test)
    eval_every = 10

    batch_size = 256
    micro_batch_size = 32
    gradient_accumulation_steps = batch_size // micro_batch_size

    print(f"Running {args.eval_name}, fine-tune Llama-7B")

    base_model = 'decapoda-research/llama-7b-hf'
    device_map = 'auto'

    # Initializing a LLaMA style configuration
    vocab_size = 12
    hidden_size = 256
    configuration = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=1024,
        num_hidden_layers=16,
        num_attention_heads=16,
        eos_token_id=1,
        bos_token_id=0,
    )
    # Initializing a model from the llama-7b style configuration
    model = LlamaForCausalLM(configuration)

    # START OVERHAUL
    model.config.vocab_size = vocab_size

    # Logit production layer overhaul
    old_head = model.lm_head
    new_head = torch.nn.Linear(hidden_size, model.config.vocab_size, bias=False, device=old_head.parameters().__next__().device)
    model.lm_head = new_head

    # Token embedding layer overhaul
    old_embed_tokens = model.base_model.embed_tokens
    new_embed_tokens = torch.nn.Embedding(model.config.vocab_size + 2000, hidden_size, device=old_embed_tokens.parameters().__next__().device)
    model.base_model.embed_tokens = new_embed_tokens

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"  # Allow batched inference

    empirical_eval_callback = EmpiricalEvalCallback(eval_every)

    trainer = transformers.Trainer(
        callbacks=[empirical_eval_callback],
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        args=transformers.TrainingArguments(
            seed=seed,
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=np.inf,
            num_train_epochs=8000,
            learning_rate=3e-4,
            warmup_steps=100,
            lr_scheduler_type="linear",
            logging_dir=f"logdir/scratch",
            fp16=True,
            use_cpu=False,
            logging_steps=1,
            optim="adamw_torch",
            adam_beta1=0.9,
            adam_beta2=0.98,
            weight_decay=1e-2,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=100,
            save_steps=200,
            output_dir=f"train_output/scratch",
            save_total_limit=3,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=None,
            group_by_length=False,
            report_to=["tensorboard"],
            run_name=None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    with torch.autocast("cuda"):
        if args.continue_yes:
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mnist_dir", type=str, help="The directory containing the MNIST dataset in raw -ubyte format", required=True)
    args = parser.parse_args()

    main(args)

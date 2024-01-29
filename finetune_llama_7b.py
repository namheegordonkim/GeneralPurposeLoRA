from argparse import ArgumentParser
from typing import List

import numpy as np
import torch
import transformers
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)
from transformers import LlamaForCausalLM, LlamaTokenizer

from utils import get_mnist_data, MNISTDatasetMaker, EmpiricalEvalCallback


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
    micro_batch_size = args.micro_batch_size  # Must change this to accommodate for available GPU memory
    gradient_accumulation_steps = batch_size // micro_batch_size

    base_model = 'decapoda-research/llama-7b-hf'
    device_map = 'auto'

    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.50
    lora_target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    # Initialize the model by loading from a pretrained snapshot and LoRA-fying it.
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    # START OVERHAUL
    model.config.vocab_size = 12  # 10 digit classes, 0 for padding and 1 for <EOS>

    # Logit production layer overhaul
    old_head = model.lm_head
    new_head = torch.nn.Linear(4096, model.config.vocab_size, bias=False, device=old_head.parameters().__next__().device)  # 4096 is the final layer size of the base model (constant)
    model.lm_head = new_head

    # Token embedding layer overhaul
    old_embed_tokens = model.base_model.embed_tokens
    new_embed_tokens = torch.nn.Embedding(model.config.vocab_size + 2000, 4096, device=old_embed_tokens.parameters().__next__().device)  # 2000 words in input language vocabulary on top of 12 in output language vocabulary
    model.base_model.embed_tokens = new_embed_tokens
    # END OVERHAUL

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        modules_to_save=["embed_tokens", "lm_head", "rotary_emb", "input_layernorm", "post_attention_layernorm", "norm"],  # This disables LoRA attaching to these modules; you get gradients for these along with LoRA training
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

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
            logging_dir=f"logdir/finetune",
            fp16=True,
            use_cpu=False,  # change if using CPU
            logging_steps=1,
            optim="adamw_torch",
            adam_beta1=0.9,
            adam_beta2=0.98,
            weight_decay=1e-2,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=100,
            save_steps=200,
            output_dir=f"train_output/finetune",
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
        trainer.train()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mnist_dir", type=str, help="The directory containing the MNIST dataset in raw -ubyte format", required=True)
    parser.add_argument("--micro_batch_size", type=int, help="The number of samples to compute gradient, to be accumulated at mini-batch level", default=64)
    args = parser.parse_args()

    main(args)

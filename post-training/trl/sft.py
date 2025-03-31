# Copyright 2025 The root-agent Team. All rights reserved.


from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from trl import ModelConfig, ScriptArguments, SFTConfig, SFTTrainer, TrlParser
from trl.trainer import SFTTrainer


def add_special_tokens(tokenizer, model_name_or_path: str):
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    n_place_holder = model_config.vocab_size - tokenizer.vocab_size
    print(f"there are {n_place_holder} reseved tokens in the model for customization")
    additional_special_tokens = {"additional_special_tokens": ["<think>", "</think>"]}
    tokenizer.add_special_tokens(additional_special_tokens)
    return tokenizer


def main(script_args, training_args, model_args):

    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer = add_special_tokens(tokenizer, model_args.model_name_or_path)

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)

    # training
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(
            dataset[script_args.dataset_test_split]
            if training_args.eval_strategy != "no"
            else None
        ),
        processing_class=tokenizer,
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)

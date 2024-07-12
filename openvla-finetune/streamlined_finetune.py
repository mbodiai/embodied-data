import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Dict
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import wandb
from datasets import load_dataset

from embdata.sample import Sample
from embdata.episode import Episode, VisionMotorStep, ImageTask
from embdata.trajectory import Trajectory
from embdata.motion.control import HandControl
from tokenizer import ActionTokenizer

@dataclass
class FinetuneConfig:
    vla_path: str = "openvla/openvla-7b"
    dataset_name: str = "mbodiai/xarm_overfit"
    split: str = "train"
    run_root_dir: Path = Path("runs")
    batch_size: int = 16
    max_steps: int = 50_000
    save_steps: int = 5000
    learning_rate: float = 5e-6
    grad_accumulation_steps: int = 1
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    use_quantization: bool = False
    wandb_project: str = "openvla"
    wandb_entity: str = None
    image_augmentation: bool = False
    lr_scheduler_type: str = "constant"

class StreamlinedDataset(Sample):
    def __init__(self, dataset_name: str, split: str):
        self.dataset = load_dataset(dataset_name, split=split, streaming=True)

    def __iter__(self):
        for data in self.dataset:
            yield self.process_sample(data)

    def process_sample(self, data: Dict[str, Any]) -> VisionMotorStep:
        observation = ImageTask(
            image=data["observation"]["image"],
            task=data["observation"]["instruction"]
        )
        action = HandControl(
            pose=data["action"]["pose"],
            grasp=data["action"]["grasp"]
        )
        return VisionMotorStep(observation=observation, action=action)

class StreamlinedTrainer(Trainer):
    def __init__(self, action_tokenizer: ActionTokenizer, **kwargs):
        super().__init__(**kwargs)
        self.action_tokenizer = action_tokenizer

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(
            pixel_values=inputs.get("pixel_values"),
            input_ids=inputs.get("input_ids"),
            labels=labels,
        )
        loss = outputs.loss

        if not return_outputs:
            return loss

        return loss, outputs

def prepare_sample_for_model(sample: VisionMotorStep, tokenizer: Any, image_processor: Any, action_tokenizer: ActionTokenizer):
    pixel_values = image_processor(sample.observation.image).pixel_values
    input_ids = tokenizer(sample.observation.task, add_special_tokens=True, return_tensors="pt").input_ids
    labels = action_tokenizer(sample.action.numpy())
    
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "labels": labels
    }

def main(cfg: FinetuneConfig):
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(cfg.vla_path, torch_dtype=torch.bfloat16, trust_remote_code=True)

    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    dataset = StreamlinedDataset(cfg.dataset_name, cfg.split)

    def collate_fn(samples):
        return {
            key: torch.stack([sample[key] for sample in samples])
            for key in samples[0].keys()
        }

    training_args = TrainingArguments(
        output_dir=cfg.run_root_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accumulation_steps,
        max_steps=cfg.max_steps,
        save_steps=cfg.save_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        logging_dir=cfg.run_root_dir / "logs",
        logging_steps=10,
        report_to="wandb",
    )

    trainer = StreamlinedTrainer(
        model=vla,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        tokenizer=processor.tokenizer,
        action_tokenizer=action_tokenizer,
    )

    wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity)
    trainer.train()

if __name__ == "__main__":
    cfg = FinetuneConfig()
    main(cfg)

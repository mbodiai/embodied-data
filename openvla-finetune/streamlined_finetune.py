import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Dict
import torch
import numpy as np
from transformers import AutoModelForVision2Seq, AutoProcessor, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import wandb
from datasets import load_dataset
from torchvision import transforms

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

class StreamlinedDataset(Episode):
    def __init__(self, dataset_name: str, split: str, image_augmentation: bool = False):
        super().__init__(load_dataset(dataset_name, split=split, streaming=True))
        self.image_augmentation = image_augmentation
        if self.image_augmentation:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomApply([
                    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                ]),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.2, 2.0), value=0, inplace=False),
                transforms.ToPILImage()
            ])
        
        # Calculate dataset statistics using Trajectory
        self.action_trajectory = self.trajectory(field="action")
        self.dataset_statistics = self.action_trajectory.stats()

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        return self.action_trajectory.transform("minmax", min=-1, max=1).numpy()

    def process_sample(self, data: Dict[str, Any]) -> VisionMotorStep:
        image = data["observation"]["image"]
        if self.image_augmentation:
            image = self.transform(image)
        
        observation = ImageTask(
            image=image,
            task=data["observation"]["instruction"]
        )
        action = np.array([
            data["action"]["pose"]["x"],
            data["action"]["pose"]["y"],
            data["action"]["pose"]["z"],
            data["action"]["pose"]["roll"],
            data["action"]["pose"]["pitch"],
            data["action"]["pose"]["yaw"],
            data["action"]["grasp"],
        ])
        normalized_action = self.normalize_action(action)
        action = HandControl(
            pose=normalized_action[:6],
            grasp=normalized_action[6]
        )
        return VisionMotorStep(observation=observation, action=action)

    def __iter__(self):
        return self.map(self.process_sample).iter()

class StreamlinedTrainer(Trainer):
    def __init__(self, action_tokenizer: ActionTokenizer, processor: Any, **kwargs):
        super().__init__(**kwargs)
        self.action_tokenizer = action_tokenizer
        self.processor = processor

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

    def get_train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def data_collator(self, samples: List[VisionMotorStep]) -> Dict[str, torch.Tensor]:
        batch = Sample.pack_from(samples)
        
        pixel_values = self.processor.image_processor(batch.observation.image, return_tensors="pt").pixel_values
        input_ids = self.processor.tokenizer(batch.observation.task, padding=True, return_tensors="pt").input_ids
        labels = self.action_tokenizer(batch.action.numpy())
        
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": labels
        }

    def on_train_begin(self, args, state, control, **kwargs):
        super().on_train_begin(args, state, control, **kwargs)
        # Log dataset statistics
        wandb.config.update({
            "dataset_stats": self.train_dataset.dataset_statistics,
        })

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
    dataset = StreamlinedDataset(cfg.dataset_name, cfg.split, image_augmentation=cfg.image_augmentation)

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
        tokenizer=processor.tokenizer,
        action_tokenizer=action_tokenizer,
        processor=processor,
    )

    wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity)
    trainer.train()

if __name__ == "__main__":
    cfg = FinetuneConfig()
    main(cfg)

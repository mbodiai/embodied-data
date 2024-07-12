import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Sequence, Type
import numpy as np
import torch
from torch.utils.data import IterableDataset
from PIL import Image
from datasets import load_dataset
import wandb
from torchvision import transforms

from prompter import PromptBuilder, PurePromptBuilder
from tokenizer import ActionTokenizer

IGNORE_INDEX = -100

# def calculate_dataset_statistics(dataset):
#     # TODO: Use mbodied_data
#     episode = Episode(steps=[])
#     for data in dataset:
#         episode.append(TimeStep(observation=data["observation"], action=data["action"]))
#     traj = episode.trajectory()
#     return{
#         "min": traj.min().to("np"),
#         "max": traj.max().to("np"),
#         "mean": traj.mean().to("np"),
#         "q01": traj.q01().to("np"),
#         "q99": traj.q99().to("np"),
#         "std": traj.std().to("np"),
#     }

from embdata.trajectory import Trajectory

class XarmDataset(IterableDataset):
    def __init__(
        self,
        dataset_name: str,
        split: str,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: Any,
        image_transform: Callable[[Image.Image], torch.Tensor],
        prompt_builder_fn: Type[PromptBuilder],
        image_augmentation = False,
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        self.dataset = load_dataset(
            dataset_name, split=split, streaming=True, trust_remote_code=True
        )
        
        # Calculate dataset statistics using Trajectory
        actions = [np.array(list(sample['action']['pose'].values()) + [sample['action']['grasp']]) for sample in self.dataset]
        trajectory = Trajectory(actions)
        self.dataset_statistics = trajectory.stats()
        
        self.image_augmentation = image_augmentation
        if self.image_augmentation:
            transform = [
                transforms.ToTensor(),
                transforms.RandomApply([
                    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                    # transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                ])
            ]
            transform.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.2, 2.0), value=0, inplace=False))
            self.transform = transforms.Compose(transform)

    def normalize_action(self, action):
        q01 = self.dataset_statistics.q01()[:-1]
        q99 = self.dataset_statistics.q99()[:-1]
        normalized_action = 2 * (action[:-1] - q01) / (q99 - q01 + 1e-8) - 1
        normalized_action = np.clip(normalized_action, -1, 1)
        # Append the grasp value without normalization
        normalized_action = np.append(normalized_action, action[-1])
        return normalized_action

    def __iter__(self):
        for data in self.dataset:
            image = data["observation"]["image"]
            if self.image_augmentation:
                image = self.transform(image)
                image = transforms.ToPILImage()(image)
            instruction = data["observation"]["instruction"]
            action = np.array(
                [
                    data["action"]["pose"]["x"],
                    data["action"]["pose"]["y"],
                    data["action"]["pose"]["z"],
                    data["action"]["pose"]["roll"],
                    data["action"]["pose"]["pitch"],
                    data["action"]["pose"]["yaw"],
                    data["action"]["grasp"],
                ]
            )
            # Normalize the action
            normalized_action = self.normalize_action(action)

            prompt_builder = self.prompt_builder_fn("openvla")
            conversation = [
                {
                    "from": "human",
                    "value": f"What action should the robot take to {instruction}?",
                },
                {"from": "gpt", "value": self.action_tokenizer(normalized_action)},
            ]
            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])

            input_ids = self.base_tokenizer(
                prompt_builder.get_prompt(), add_special_tokens=True
            ).input_ids
            labels = list(input_ids)

            input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
            pixel_values = self.image_transform(image).to(torch.bfloat16)

            labels[: -(len(normalized_action) + 1)] = IGNORE_INDEX

            yield dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)

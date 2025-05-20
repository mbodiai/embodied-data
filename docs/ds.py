from datasets import load_dataset
from embdata.episode import Episode, Sample, Trajectory, Image
from transformers import GPT2LMHeadModel, CLIPModel, CLIPProcessor, AutoTokenizer
from embdata.utils.import_utils import smart_import
torch =smart_import("torch")
nn = smart_import("torch.nn")

# Download the dataset
dataset = load_dataset("mbodiai/oxe_taco_play")

# Function to flatten and process a single example
def process_example(example):
    flattened = Sample(example).flatten(
        to={
            "image": "data.pickle.steps.observation.image.bytes",
            "instruction": "data.pickle.steps.observation.natural_language_instruction",
            "action": "data.pickle.steps.action",
            "reward": "data.pickle.steps.reward",
            "is_terminal": "data.pickle.steps.is_terminal"
        }
    )
    return flattened

# Process the entire dataset
processed_dataset = dataset.map(process_example, remove_columns=dataset["train"].column_names)

# Create an episode from the first example
first_example = processed_dataset["train"][0]
episode = Episode()

for i in range(len(first_example["image"])):
    step = Sample(
        image=Image(base64=first_example["image"][i]),
        instruction=first_example["instruction"][i],
        action=first_example["action"][i],
        reward=first_example["reward"][i],
        is_terminal=first_example["is_terminal"][i]
    )
    episode.append(step)

# Clean the data using Trajectory
action_trajectory = episode.trajectory(field="action")
cleaned_trajectory = action_trajectory.low_pass_filter(cutoff_freq=2)

# Visualize the episode and save the trajectory plot
episode.show()
cleaned_trajectory.save("cleaned_action_trajectory.png")

## 2. Define a GPT2-CLIP Model

class GPT2CLIP(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.fusion = nn.Linear(self.gpt2.config.hidden_size + self.clip.config.projection_dim, 512)
        self.action_head = nn.Linear(512, num_actions)
        
    def forward(self, input_ids, attention_mask, pixel_values):
        text_features = self.gpt2(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, -1, :]
        image_features = self.clip.get_image_features(pixel_values=pixel_values)
        fused_features = torch.cat([text_features, image_features], dim=1)
        fused_features = self.fusion(fused_features)
        action_logits = self.action_head(fused_features)
        return action_logits

## 3. Prepare Data for Training

gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def prepare_batch(examples):
    gpt2_inputs = gpt2_tokenizer(examples["instruction"], padding=True, truncation=True, return_tensors="pt")
    clip_inputs = clip_processor(images=[Image(base64=img).pil for img in examples["image"]], return_tensors="pt")
    actions = torch.tensor(examples["action"])
    return {
        "input_ids": gpt2_inputs.input_ids,
        "attention_mask": gpt2_inputs.attention_mask,
        "pixel_values": clip_inputs.pixel_values,
        "labels": actions
    }

train_dataset = processed_dataset["train"].map(prepare_batch, batched=True, remove_columns=processed_dataset["train"].column_names)

## 4. Training Loop

model = GPT2CLIP(num_actions=len(first_example["action"][0]))
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
loss_fn = nn.MSELoss()

num_epochs = 5
batch_size = 16

for epoch in range(num_epochs):
    for i in range(0, len(train_dataset), batch_size):
        batch = train_dataset[i:i+batch_size]
        optimizer.zero_grad()
        outputs = model(batch["input_ids"], batch["attention_mask"], batch["pixel_values"])
        loss = loss_fn(outputs, batch["labels"])
        loss.backward()
        optimizer.step()
        


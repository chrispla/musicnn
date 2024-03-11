import argparse
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm

import wandb
from config import config
from datasets import MTAT
from model import Musicnn

# Initialize wandb
wandb.init(project="musicnn")

parser = argparse.ArgumentParser(description="Training.")
for k, v in config.items():
    parser.add_argument(f"--{k}", default=v)
args = parser.parse_args()

model = Musicnn(
    args=args,
    y_input_dim=args.n_mels,
    timbral_k_height=[0.4, 0.7],
    temporal_k_width=[32, 64, 128],
    filter_factor=1.6,
    pool_type="temporal",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = MTAT(root="./data/mtat/", args=args)
loader = dataset.get_dataloader(batch_size=args.batch_size, shuffle=True)

loss_function = nn.BCELoss()
optimizer = Adam(
    model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
)
# Adaptive learning rate
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=args.lr_patience
)
model.to(device)

run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_dir = Path("./ckpt") / run_name
checkpoint_dir.mkdir(exist_ok=True, parents=True)

for epoch in range(args.epochs):
    model.train()
    running_loss = 0.0
    pbar = tqdm(enumerate(loader), total=len(loader), leave=False)
    for i, data in pbar:
        inputs, labels = data

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pbar.set_description(f"Epoch {epoch+1}, Loss: {running_loss/(i+1):.4f}")

        # Log metrics to wandb
        wandb.log({"Loss": running_loss / (i + 1)})
        wandb.log({"Learning Rate": optimizer.param_groups[0]["lr"]})

    scheduler.step(running_loss)

    # save model
    if (epoch + 1) % 50 == 0:
        torch.save(
            model.state_dict(),
            checkpoint_dir / f"model_{epoch}_loss_{format(running_loss, '.3f')}.pth",
        )

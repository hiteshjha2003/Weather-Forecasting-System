import os
import sys
import argparse
import json
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models_trainings.dataset import TimeSeriesDataset
from models_trainings.transformer import AdaptiveDropPatch
from models_trainings.config import *

# Paths
dataset_path = "/Users/hiteshj/Desktop/Python/my-apps/Weather-Forecasting-System/weatherHistory.csv"
model_path = "/Users/hiteshj/Desktop/Python/my-apps/Weather-Forecasting-System/models/adaptive_drop_patch_model.pth"

# def train_model(train_loader, val_loader, input_dim, target_dim):
#     model = AdaptiveDropPatch(input_dim, SEQ_LEN, PRED_LEN, PATCH_SIZE, EMBED_DIM, target_dim, NHEAD, NUM_LAYERS).to(DEVICE)
#     optimizer = torch.optim.Adam(model.parameters(), lr=LR)
#     criterion = torch.nn.MSELoss()

#     best_val_loss = float('inf')
#     wait = 0
#     mse_list = []

#     for epoch in range(EPOCHS):
#         model.train()
#         total_train_loss = 0
#         for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
#             x, y = x.to(DEVICE), y.to(DEVICE)
#             optimizer.zero_grad()
#             out, _, _ = model(x)
#             loss = criterion(out, y)
#             loss.backward()
#             optimizer.step()
#             total_train_loss += loss.item()

#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for x, y in val_loader:
#                 x, y = x.to(DEVICE), y.to(DEVICE)
#                 out, _, _ = model(x)
#                 val_loss += criterion(out, y).item()
#         val_loss /= len(val_loader)
#         mse_list.append(val_loss)

#         print(f"\nEpoch {epoch+1}: Train Loss = {total_train_loss:.4f}, Val MSE = {val_loss:.4f}")

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save(model.state_dict(), model_path)
#             print(f"✅ Best model saved at epoch {epoch+1} with Val MSE {val_loss:.4f}")
#             wait = 0
#         else:
#             wait += 1
#             if wait >= PATIENCE:
#                 print(f"⏹️ Early stopping at epoch {epoch+1}. No improvement for {PATIENCE} epochs.")
#                 break

#     if os.path.exists(model_path):
#         model.load_state_dict(torch.load(model_path, map_location=DEVICE))
#         print(f"✅ Best model loaded from {model_path}")
#     else:
#         print("⚠️ Warning: Model file not found. Training might not have improved.")

#     return model, mse_list
def train_model(data_path, targets, epochs, batch_size, learning_rate, device, save_path):
    from models_trainings.dataset import TimeSeriesDataset
    from models_trainings.transformer import AdaptiveDropPatch
    import json

    dataset = TimeSeriesDataset(data_path, seq_len=96, pred_len=24, target_columns=targets)
    input_dim = dataset[0][0].shape[1]
    target_dim = dataset[0][1].shape[1]

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = AdaptiveDropPatch(input_dim, 96, 24, 24, 128, target_dim, 8, 4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out, _, _ = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out, _, _ = model(x)
                val_loss += criterion(out, y).item()
        avg_loss = val_loss / len(val_loader)
        losses.append(avg_loss)
        st.write(f"Epoch {epoch+1}/{epochs}, Val Loss: {avg_loss:.4f}")

    # Save model + meta
    torch.save(model.state_dict(), save_path)
    meta = {
        "input_dim": input_dim,
        "patch_size": 24,
        "target_columns": targets,
        "seq_len": 96,
        "pred_len": 24,
        "embed_dim": 128,
        "nhead": 8,
        "num_layers": 4
    }
    with open(os.path.splitext(save_path)[0] + "_meta.json", "w") as f:
        json.dump(meta, f, indent=4)
    return model, meta, losses


if __name__ == "__main__":
    # CLI parser
    parser = argparse.ArgumentParser(description="Train Weather Forecasting Transformer")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["Temperature (C)"],
        help="List of target columns to forecast (e.g., --targets 'Temperature (C)' 'Humidity')"
    )
    args = parser.parse_args()
    targets = args.targets

    print(f"📌 Forecast targets: {targets}")

    dataset = TimeSeriesDataset(dataset_path, SEQ_LEN, PRED_LEN, target_columns=targets)
    input_dim = dataset[0][0].shape[1]
    target_dim = dataset[0][1].shape[1]

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model, mse = train_model(train_loader, val_loader, input_dim, target_dim)

    # Save metadata (AFTER input_dim and targets are defined)
    metadata = {
        "input_dim": input_dim,
        "patch_size": PATCH_SIZE,
        "target_columns": targets,
        "seq_len": SEQ_LEN,
        "pred_len": PRED_LEN,
        "embed_dim": EMBED_DIM,
        "nhead": NHEAD,
        "num_layers": NUM_LAYERS
    }
    meta_path = os.path.splitext(model_path)[0] + "_meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"✅ Metadata saved to: {meta_path}")

    # Plot MSE
    plt.plot(mse)
    plt.title("Validation MSE per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"🎉 Training complete. Final model saved to: {model_path}")
    print(f"📊 Metadata saved to: {meta_path}")
# forecast_cli.py

import os
import sys
import argparse
import datetime
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models_trainings.dataset import TimeSeriesDataset
from models_trainings.transformer import AdaptiveDropPatch

DEFAULT_MODEL_PATH = "/Users/hiteshj/Desktop/Python/my-apps/Weather-Forecasting-System/models/adaptive_drop_patch_model.pth"
DEFAULT_DATA_PATH = "/Users/hiteshj/Desktop/Python/my-apps/Weather-Forecasting-System/weatherHistory.csv"

def load_model(model_path, device):
    import json
    meta_path = os.path.splitext(model_path)[0] + "_meta.json"
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    model = AdaptiveDropPatch(
        input_dim=meta["input_dim"],
        seq_len=meta["seq_len"],
        pred_len=meta["pred_len"],
        patch_size=meta["patch_size"],
        embed_dim=meta["embed_dim"],
        target_dim=len(meta["target_columns"]),
        num_heads=meta["nhead"],
        num_layers=meta["num_layers"]
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, meta

def forecast_temperature(model, meta, data_path, input_days, device):
    dataset = TimeSeriesDataset(data_path, meta["seq_len"], meta["pred_len"], target_columns=meta["target_columns"])
    sample_x, _ = dataset[0]

    val_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    x, _ = next(iter(val_loader))
    x = x.to(device)

    with torch.no_grad():
        preds, _, _ = model(x)

    pred_len = meta["pred_len"]
    target_columns = meta["target_columns"]

    input_days = min(input_days, pred_len)
    preds_np = preds[0, :input_days].cpu().numpy()

    # Ensure 2D
    if preds_np.ndim == 1:
        preds_np = preds_np.reshape(-1, 1)

    dates = [datetime.date.today() + datetime.timedelta(days=i) for i in range(input_days)]
    df = pd.DataFrame(preds_np, columns=target_columns)
    df.insert(0, "Date", dates)

    print("🧪 Forecast DataFrame columns:", df.columns.tolist())  # Debug line

    return df

def plot_forecast(df, targets, save_path=None):
    plt.figure(figsize=(10, 5))
    for col in targets:
        plt.plot(df["Date"], df[col], marker='o', label=col)
    plt.title("Forecast")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"📈 Forecast plot saved to {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Run weather forecasting from CLI")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to trained model .pth file")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_PATH, help="Path to input CSV data file")
    parser.add_argument("--days", type=int, default=7, help="Number of days to forecast (max 24)")
    parser.add_argument("--csv-out", type=str, help="Path to save forecast CSV")
    parser.add_argument("--plot", type=str, help="Path to save forecast plot (PNG)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU (even if CUDA is available)")

    args = parser.parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    print(f"🔍 Using device: {device}")
    print(f"📁 Loading model from: {args.model}")
    print(f"📄 Forecasting from dataset: {args.data}")

    model, meta = load_model(args.model, device)
    forecast_df = forecast_temperature(model, meta, args.data, args.days, device)

    print("\n📊 Forecast Results:\n")
    print(forecast_df)

    if args.csv_out:
        forecast_df.to_csv(args.csv_out, index=False)
        print(f"📁 Forecast saved to: {args.csv_out}")

    if args.plot:
        plot_forecast(forecast_df, meta["target_columns"], save_path=args.plot)
    else:
        plot_forecast(forecast_df, meta["target_columns"])

if __name__ == "__main__":
    main()

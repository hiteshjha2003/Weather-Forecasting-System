import os
import sys
import torch
import streamlit as st
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models_trainings.forecast_cli import load_model, forecast_temperature

st.set_page_config(layout="centered")
st.title("🌤️ Weather Forecasting with Transformer")

DEFAULT_MODEL_PATH = "/Users/hiteshj/Desktop/Python/my-apps/Weather-Forecasting-System/models/adaptive_drop_patch_model.pth"
DEFAULT_DATA_PATH = "/Users/hiteshj/Desktop/Python/my-apps/Weather-Forecasting-System/weatherHistory.csv"

# Select device (optional)
use_cpu = st.checkbox("Force CPU", value=True)
device = torch.device("cpu" if use_cpu or not torch.cuda.is_available() else "cuda")
st.write(f"Using device: {device}")

# Select targets and forecast days
available_targets = ["Temperature (C)"]
selected_targets = st.multiselect("Choose variables to forecast:", available_targets, default=["Temperature (C)"])
input_days = st.slider("Number of days to forecast", 1, 24, 7)

@st.cache(allow_output_mutation=True)
def load_trained_model(model_path, device):
    model, meta = load_model(model_path, device)
    return model, meta

model, meta = load_trained_model(DEFAULT_MODEL_PATH, device)

# If user changed targets, update meta to match (optional)
# For now, we assume the model predicts those targets only.

if st.button("Generate Forecast"):
    with st.spinner("Generating forecast..."):
        forecast_df = forecast_temperature(
            model=model,
            meta=meta,
            data_path=DEFAULT_DATA_PATH,
            input_days=input_days,
            device=device
        )

        # Filter columns if user selected a subset (if model predicts multiple)
        forecast_df = forecast_df[["Date"] + selected_targets]

        st.success("Forecast generated!")
        st.write("📈 Forecast Results")
        st.dataframe(forecast_df)

        # Plot each selected variable
        fig, ax = plt.subplots()
        for target in selected_targets:
            ax.plot(forecast_df["Date"], forecast_df[target], label=target, marker='o')
        ax.set_title("Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        st.pyplot(fig)

st.markdown("---")
st.markdown("Made with ❤️ by [Kulshum]")

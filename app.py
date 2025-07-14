import os
import sys
import glob
import uuid
import json
import torch
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from PIL import Image
from torch.utils.data import DataLoader
import plotly.graph_objs as go
# --------------------------------------------------------------------
# Project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models_trainings.dataset import TimeSeriesDataset
from models_trainings.transformer import AdaptiveDropPatch
from models_trainings.forecast_cli import forecast_temperature, load_model
from models_trainings.train_model import train_model  # optional if used
from models_trainings.config import *
from data_loading import load_data
from data_preprocesssing import preprocess_data
from eda import perform_eda, get_data_info
from sql_connection import connect_to_db, retrieve_data
# --------------------------------------------------------------------

st.set_page_config(page_title="🌤️ Weather Forecasting", layout="wide")

# --- Constants and device setup ---
DEFAULT_MODEL_PATH = "models/adaptive_drop_patch_model.pth"
DEFAULT_DATA_PATH = "weatherHistory.csv"
device = torch.device("cpu" if (st.sidebar.checkbox("Force CPU", value=True) or not torch.cuda.is_available()) else "cuda")

# --- Sidebar navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📈 Train Model", "🌤️ Forecasting", "🗃️ Database Access", "📊 Exploratory Data Analysis"])

# --- Home Page ---
if page == "🏠 Home":
    st.title("Weather Forecasting App")
    st.markdown("""
    Welcome to the Weather Forecasting App!  
    This app uses a transformer-based model (`AdaptiveDropPatch`) to forecast weather features.  
    Use the sidebar to train a model, forecast, explore the database, or run EDA.
    """)

# --- Load Model Helper ---
@st.cache_resource
def load_trained_forecast_model(model_path, device):
    try:
        model, meta = load_model(model_path, device)
        return model, meta
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None

# --- Train Model Page ---
if page == "📈 Train Model":
    st.title("📈 Train New Forecasting Model")

    data_path = st.text_input("Path to Training CSV", DEFAULT_DATA_PATH)
    targets = st.multiselect("Target Features", ["Temperature", "Humidity", "WindSpeed"], default=["Temperature"])
    epochs = st.number_input("Epochs", 1, 100, 10)
    batch_size = st.number_input("Batch Size", 1, 128, 32)
    learning_rate = st.number_input("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f")
    default_filename = f"models/adaptive_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    save_path = st.text_input("Save Model Path", default_filename)

    if st.button("Start Training"):
        if not os.path.exists(data_path):
            st.error("Data path does not exist.")
        else:
            with st.spinner("Training model..."):
                try:
                    dataset = TimeSeriesDataset(data_path, seq_len=SEQ_LEN, pred_len=PRED_LEN, target_columns=targets)
                    st.write(f"🧪 Total samples: {len(dataset)}")
                    input_dim = dataset[0][0].shape[1]
                    target_dim = dataset[0][1].shape[1]
                    train_size = int(0.8 * len(dataset))
                    val_size = len(dataset) - train_size
                    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size)

                    model = AdaptiveDropPatch(input_dim, SEQ_LEN, PRED_LEN, PATCH_SIZE, EMBED_DIM, target_dim, NHEAD, NUM_LAYERS).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                    criterion = torch.nn.MSELoss()

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
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                            total_loss += loss.item()

                        val_loss = 0
                        model.eval()
                        with torch.no_grad():
                            for x, y in val_loader:
                                x, y = x.to(device), y.to(device)
                                out, _, _ = model(x)
                                val_loss += criterion(out, y).item()
                        avg_val = val_loss / len(val_loader)
                        losses.append(avg_val)
                        st.write(f"Epoch {epoch+1}/{epochs}, Val Loss: {avg_val:.4f}")

                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(model.state_dict(), save_path)
                    meta = {
                        "input_dim": input_dim, "patch_size": PATCH_SIZE,
                        "target_columns": targets, "seq_len": SEQ_LEN,
                        "pred_len": PRED_LEN, "embed_dim": EMBED_DIM,
                        "nhead": NHEAD, "num_layers": NUM_LAYERS
                    }
                    with open(os.path.splitext(save_path)[0] + "_meta.json", "w") as f:
                        json.dump(meta, f, indent=4)
                    st.success(f"Model trained and saved to {save_path}")
                    # Plot losses
                    fig, ax = plt.subplots()
                    ax.plot(losses, marker='o')
                    ax.set_title("Validation Loss")
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Loss")
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Training failed: {e}")

# --- Forecasting Page ---

#     st.title("🌤️ Forecast Weather")

#     model_path = st.text_input("Model Path", DEFAULT_MODEL_PATH)
#     input_days = st.slider("Days to Forecast", 1, PRED_LEN, min(PRED_LEN, 7))
#     selected_targets = st.multiselect("Select Targets", ["Temperature", "Humidity", "WindSpeed"], default=["Temperature"])

#     if st.button("Load Model"):
#         with st.spinner("Loading model..."):
#             model, meta = load_trained_forecast_model(model_path, device)
#             if model:
#                 st.success("Model loaded successfully!")
#                 st.session_state.forecast_model = model
#                 st.session_state.forecast_meta = meta
#             else:
#                 st.error("Failed to load model.")

#     if "forecast_model" in st.session_state:
#         if st.button("Generate Forecast"):
#             with st.spinner("Forecasting..."):
#                 try:
#                     df_f = forecast_temperature(
#                         model=st.session_state.forecast_model,
#                         meta=st.session_state.forecast_meta,
#                         data_path=DEFAULT_DATA_PATH,
#                         input_days=input_days,
#                         device=device
#                     )

#                     # Map target names
#                     col_map = {
#                         "Temperature": next((c for c in df_f.columns if "Temperature" in c), None),
#                         "Humidity": next((c for c in df_f.columns if "Humidity" in c), None),
#                         "WindSpeed": next((c for c in df_f.columns if "Wind" in c), None)
#                     }
#                     selected_cols = ["Date"] + [col_map[t] for t in selected_targets if col_map[t]]
#                     df_out = df_f[selected_cols]

#                     st.success("Forecast generated!")
#                     st.dataframe(df_out)

#                     fig, ax = plt.subplots(figsize=(10, 6))
#                     for t in selected_targets:
#                         if col_map[t]:
#                             ax.plot(df_out["Date"], df_out[col_map[t]], label=t, marker='o')
#                     ax.set_title("Forecast")
#                     ax.legend()
#                     st.pyplot(fig)
#                 except Exception as e:
#                     st.error(f"Forecasting failed: {e}")
# --- Forecasting Page ---
if page == "🌤️ Forecasting":
    st.title("🌤️ Forecast Weather")

    model_path = st.text_input("Model Path", DEFAULT_MODEL_PATH)
    input_days = st.slider("Days to Forecast", 1, PRED_LEN, min(PRED_LEN, 7))
    selected_targets = st.multiselect("Select Targets", ["Temperature", "Humidity", "WindSpeed"], default=["Temperature"])

    if st.button("Load Model"):
        with st.spinner("Loading model..."):
            model, meta = load_trained_forecast_model(model_path, device)
            if model:
                st.success("Model loaded successfully!")
                st.session_state.forecast_model = model
                st.session_state.forecast_meta = meta
            else:
                st.error("Failed to load model.")

    if "forecast_model" in st.session_state:
        if st.button("Generate Forecast"):
            with st.spinner("Forecasting..."):
                try:
                    df_f = forecast_temperature(
                        model=st.session_state.forecast_model,
                        meta=st.session_state.forecast_meta,
                        data_path=DEFAULT_DATA_PATH,
                        input_days=input_days,
                        device=device
                    )

                    # Map target columns for plotting
                    col_map = {
                        "Temperature": next((c for c in df_f.columns if "Temperature" in c), None),
                        "Humidity": next((c for c in df_f.columns if "Humidity" in c), None),
                        "WindSpeed": next((c for c in df_f.columns if "Wind" in c), None)
                    }
                    selected_cols = ["Date"] + [col_map[t] for t in selected_targets if col_map[t]]
                    df_out = df_f[selected_cols]

                    st.success("Forecast generated!")
                    st.dataframe(df_out)

                    # Create Plotly figure
                    fig = go.Figure()
                    for t in selected_targets:
                        if col_map[t]:
                            fig.add_trace(go.Scatter(
                                x=df_out["Date"],
                                y=df_out[col_map[t]],
                                mode="lines+markers",
                                name=t,
                                marker=dict(size=6),
                                line=dict(width=2)
                            ))

                    fig.update_layout(
                        title="Weather Forecast",
                        xaxis_title="Date",
                        yaxis_title="Value",
                        legend_title="Features",
                        hovermode="x unified",
                        template="plotly_dark",
                        margin=dict(l=40, r=40, t=40, b=40)
                    )

                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Forecasting failed: {e}")
# --- SQL Database Access Page ---
elif page == "🗃️ Database Access":
    st.header("🗃️ View Weather Data from SQL Server")

    with st.expander("🔑 Enter SQL Connection Details"):
        server = st.text_input("Server")
        database = st.text_input("Database")
        connect_btn = st.button("🔌 Connect")

    if connect_btn and all([server, database]):
        try:
            conn = connect_to_db(server, database)
            st.success("Connected to SQL Server")
            with st.spinner("Fetching data..."):
                df = retrieve_data(conn)
                st.dataframe(df.head(50))
        except Exception as e:
            st.error(f"Connection or retrieval failed: {e}")

# --- Exploratory Data Analysis Page ---
elif page == "📊 Exploratory Data Analysis":
    st.header("📊 Exploratory Data Analysis (EDA)")
    st.markdown("Analyze trends, missing values, outliers, and more.")

    output_dir = "eda_plots"
    plots_exist = os.path.exists(output_dir) and bool(glob.glob(f"{output_dir}/*.png"))

    if plots_exist:
        st.info("📁 Existing EDA plots found.")
        if st.button("🧹 Delete and regenerate"):
            for f in glob.glob(os.path.join(output_dir, "*.png")):
                os.remove(f)
            os.rmdir(output_dir)
            plots_exist = False

    if not plots_exist:
        if st.button("🚀 Run EDA"):
            try:
                df_raw = load_data(DEFAULT_DATA_PATH)
                df_proc, _ = preprocess_data(df_raw)

                with st.spinner("Data summary..."):
                    get_data_info(df_proc)
                with st.spinner("Creating visualizations..."):
                    os.makedirs(output_dir, exist_ok=True)
                    perform_eda(df_proc, output_dir)
                st.success("EDA completed!")
            except Exception as e:
                st.error(f"EDA failed: {e}")

    if os.path.exists(output_dir):
        images = sorted(glob.glob(os.path.join(output_dir, "*.png")))
        if images:
            st.subheader("📊 Visualizations")
            col1, col2 = st.columns(2)
            for idx, img_path in enumerate(images):
                with (col1 if idx % 2 == 0 else col2):
                    st.image(Image.open(img_path), caption=os.path.basename(img_path), use_container_width=True)
        else:
            st.warning("No plots found in 'eda_plots'.")

# --- Footer ---
st.markdown("---")
st.markdown("Made with ❤️ by Kulshum")

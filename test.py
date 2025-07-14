import os
def check_model_files():
    model_dir = "models/"
    required_files = [
        "adaptive_drop_patch_model.pth",
        "temp_model.pkl",
        "hum_model.pkl",
        "rain_model.pkl",
        "preprocessor.pkl"
    ]
    missing = [f for f in required_files if not os.path.isfile(os.path.join(model_dir, f))]
    if missing:
        print(f"⚠️ The following required model files are missing in `models/` folder:\n\n- " + "\n- ".join(missing))
    else:
        print("✅ All required model files are present.")
    return missing

check_model_files()
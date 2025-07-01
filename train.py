import torch
import pandas as pd
import os
import logging
from src.load_data import fetch_stock_data
from src.feature_engineering import load_data_pooled
from model.dual_transformer import *
from scripts.train_test_dual_transformer import *

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Config:
    batch_size = 32
    num_epochs = 50
    patience = 10
    learning_rate = 1e-5
    min_learning_rate = 1e-6

def main():
    data_path = r'./data'
    os.makedirs(data_path, exist_ok=True)

    stocks_to_train = [
        # Banks
        "CTG", "MBB", "SHB", "TCB", "STB", "VIB", "TPB", "BID", "VCB",

        # Securities
        "SSI", "VND", "VCI", "HCM", "TVS",

        # Real Estate & Construction
        "KDH", "DXG", "CTD", "NLG", "SJS", "HDC", "DPG", "VHM", "PDR",

        # Industrials & Materials
        "HPG", "HSG", "CSV", "DGC", "DPM", "PLC", "DHA", "NKG", "POM",

        # Oil, Gas & Utilities
        "GAS", "PLX", "POW", "PVS", "PVD", "BSR", "PVT", "NT2",

        # Consumption & Conglomerates
        "VNM", "MSN", "VIC", "SAB", "MWG",

        # Tech & Retail
        "FPT", "DGW", "FRT", "CMG",

        # Pharma
        "DHG", "TRA"
    ]

    pooled_data_filename = "pooled_stock_data.csv"
    pooled_data_path = os.path.join(data_path, pooled_data_filename)

    if not os.path.exists(pooled_data_path):
        logging.info("Pooled data file not found. Fetching from API...")
        fetch_stock_data(
            data_path=data_path,
            stocks=stocks_to_train,
            start_date="2015-01-01",
            end_date="2025-06-09",
            output_filename=pooled_data_filename
        )

    logging.info(f"Loading pooled data from {pooled_data_path}")
    df = pd.read_csv(pooled_data_path)


    X_train, y_train, X_valid, y_valid, X_test, y_test, y_mean, y_std, X_mean, X_std, train_symbols, val_symbols, test_symbols = load_data_pooled(df, seq_len=64, pred_len=5, stride=1, level=2, train_stocks=40, val_stocks=5, test_stocks=5, clip_value=3.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === Init model ===
    input_dim = X_train.shape[2]  # Number of features
    wavelet_levels = X_train.shape[-1]  # Number of wavelet levels (e.g. 4)
    model = CleanWaveletTransformer(
            input_dim=input_dim,
            wavelet_levels=wavelet_levels,
            d_model=64,
            nhead=2,
            num_layers=3,
            drop_out=0.5,
            pred_len=y_train.shape[1]  # Usually 5
        )

    # === Train & Test ===
    config = Config()
    model, grad_history = train_wavelet_transformer(model, X_train, y_train, X_valid, y_valid, X_test, y_test, config, device)
    all_preds, all_targets = test_wavelet_model(df, model, X_test, y_test, config, device, "trained_wavelet_model.pt")

if __name__ == "__main__":
    main()
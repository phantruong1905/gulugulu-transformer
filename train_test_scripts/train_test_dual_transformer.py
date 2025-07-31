import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import math
from loss.tdl import TanhDirectionalLoss


def plot_test_results(y_pred, y_true, title_prefix="Test", smooth_scale=5.0, chunk_size=300, save_path="example_trend_prediction.png"):
    if hasattr(y_pred, "detach"):
        y_pred = y_pred.detach().cpu().numpy()
    if hasattr(y_true, "detach"):
        y_true = y_true.detach().cpu().numpy()

    tgt_len = y_pred.shape[1]
    num_samples = y_pred.shape[0]
    num_chunks = int(np.ceil(num_samples / chunk_size))

    fig, axs = plt.subplots(num_chunks * tgt_len, 1, figsize=(14, 3 * tgt_len * num_chunks), sharex=False)

    if num_chunks * tgt_len == 1:
        axs = [axs]

    for chunk in range(num_chunks):
        start = chunk * chunk_size
        end = min((chunk + 1) * chunk_size, num_samples)

        for i in range(tgt_len):
            ax_idx = chunk * tgt_len + i
            axs[ax_idx].plot(y_true[start:end, i], label="True", color='black', linewidth=1)
            axs[ax_idx].plot(y_pred[start:end, i], label="Predicted", color='blue', alpha=0.8)
            axs[ax_idx].set_title(f"{title_prefix} Chunk {chunk+1} - Step {i+1}", fontsize=10)
            axs[ax_idx].legend(loc='upper right', fontsize=7)
            axs[ax_idx].grid(True)

    axs[-1].set_xlabel("Test Sample Index")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    # Directional accuracy
    pred_sign = (np.tanh(smooth_scale * y_pred) > 0).astype(int)
    true_sign = (np.tanh(smooth_scale * y_true) > 0).astype(int)
    per_step_da = (pred_sign == true_sign).mean(axis=0)

    plt.figure(figsize=(8, 4))
    plt.bar(range(1, tgt_len + 1), per_step_da, color='skyblue', edgecolor='black')
    plt.ylim(0, 1)
    plt.xticks(range(1, tgt_len + 1))
    plt.xlabel("Prediction Step")
    plt.ylabel("Directional Accuracy")
    plt.title(f"{title_prefix} Directional Accuracy per Step")
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{title_prefix}_directional_accuracy.png", dpi=300)
    plt.close()


def get_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def get_layerwise_gradients(model):
    layer_grads = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            norm = param.grad.data.norm(2).item()
            layer_grads[name] = norm
    return layer_grads


def check_vanishing_gradients(grad_history, threshold=1e-6):
    """Check which layers have vanishing gradients (below threshold for all epochs)."""
    vanishing_layers = []
    for name, norms in grad_history.items():
        if all(x == 0 for x in norms) or any(np.isnan(norms)):
            continue
        if all(norm < threshold for norm in norms):
            vanishing_layers.append((name, np.mean(norms)))
    return vanishing_layers


def plot_layerwise_gradients(grad_history):
    plt.figure(figsize=(14, 6))
    for name, norms in grad_history.items():
        if any(np.isnan(norms)) or all(x == 0 for x in norms):
            continue
        plt.plot(norms, label=name)
    plt.title("Layerwise Gradient Norms per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("L2 Norm (log scale)")
    plt.yscale("log")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def check_exploding_gradients(grad_history, threshold=10.0, min_epochs_ratio=0.8):
    """Check which layers have exploding gradients (above threshold for most epochs)."""
    exploding_layers = []
    num_epochs = len(next(iter(grad_history.values())))
    min_epochs = int(num_epochs * min_epochs_ratio)
    for name, norms in grad_history.items():
        if all(x == 0 for x in norms) or any(np.isnan(norms)):
            continue
        exceed_count = sum(1 for norm in norms if norm > threshold)
        if exceed_count >= min_epochs:
            exploding_layers.append((name, np.mean(norms)))
    return exploding_layers


def normalize_gradients(model, target_norm=5.0):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    if total_norm > target_norm:
        scale = target_norm / (total_norm + 1e-6)
        for p in model.parameters():
            if p.grad is not None:
                p.grad *= scale
    return total_norm


def directional_accuracy(y_pred, y_true):
    return ((torch.sign(y_pred) == torch.sign(y_true)).float().mean().item())


def train_model(model, X_train, y_train, X_valid, y_valid, config, device,
                patience=50, save_path=None):
    """Train the model for direct prediction."""
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=6e-5)
    plateau_scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3,
        threshold=1e-4, min_lr=config.min_learning_rate, verbose=True
    )

    criterion = TanhDirectionalLoss()

    # Create data loaders - X should be [batch, seq_len, input_dim, wavelet_levels]
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                     torch.tensor(y_train, dtype=torch.float32)),
        batch_size=config.batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        TensorDataset(torch.tensor(X_valid, dtype=torch.float32),
                     torch.tensor(y_valid, dtype=torch.float32)),
        batch_size=config.batch_size
    )

    best_val_loss = float('inf')
    patience_counter = 0
    layer_grad_history = {}

    print("🚀 Starting training...")

    for epoch in range(config.num_epochs):  # Use single epoch count from config
        # Update loss function epoch
        criterion.update_epoch(epoch)

        model.train()
        train_loss, train_da = 0, 0
        epoch_layer_grads = {}

        for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            y_pred = model(X)  # Direct call without mode parameter
            loss = criterion(y_pred, y)
            loss.backward()

            grad_norm = normalize_gradients(model, target_norm=5.0)

            if grad_norm < 1e-3:
                print("⚠️ Vanishing gradient warning (grad norm < 1e-3)")
            elif grad_norm > 100:
                print("⚠️ Exploding gradient warning (grad norm > 100)")

            optimizer.step()

            # Track layer gradients
            batch_layer_grads = get_layerwise_gradients(model)
            for name, norm in batch_layer_grads.items():
                if name not in epoch_layer_grads:
                    epoch_layer_grads[name] = []
                epoch_layer_grads[name].append(norm)

            train_loss += loss.item()
            train_da += directional_accuracy(y_pred, y)

        # Update gradient history
        for name, values in epoch_layer_grads.items():
            if name not in layer_grad_history:
                layer_grad_history[name] = []
            layer_grad_history[name].append(np.mean(values))

        train_loss /= len(train_loader)
        train_da /= len(train_loader)

        # Validation
        model.eval()
        val_loss, val_da = 0, 0
        with torch.no_grad():
            for X, y in valid_loader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)  # Direct call without mode parameter
                val_loss += criterion(y_pred, y).item()
                val_da += directional_accuracy(y_pred, y)

        val_loss /= len(valid_loader)
        val_da /= len(valid_loader)

        plateau_scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"📘 Epoch {epoch+1:03d} | LR: {current_lr:.2e} | "
              f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
              f"Train DA: {train_da:.3f} | Val DA: {val_da:.3f} | Grad Norm: {grad_norm:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
                print("✅ Saved new best model.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⛔ Early stopping.")
                break

    # Load best model if save_path was provided
    if save_path:
        model.load_state_dict(torch.load(save_path))

    # Gradient analysis
    vanishing_layers = check_vanishing_gradients(layer_grad_history, threshold=1e-6)
    if vanishing_layers:
        print("\nLayers with Vanishing Gradients (mean norm < 1e-6):")
        for name, mean_norm in vanishing_layers:
            print(f"  {name}: Mean Gradient Norm = {mean_norm:.2e}")
    else:
        print("\nNo layers with vanishing gradients (mean norm < 1e-6).")

    exploding_layers = check_exploding_gradients(layer_grad_history, threshold=10.0)
    if exploding_layers:
        print("\nLayers with Exploding Gradients (mean norm > 10.0 in most epochs):")
        for name, mean_norm in exploding_layers:
            print(f"  {name}: Mean Gradient Norm = {mean_norm:.2e}")
    else:
        print("\nNo layers with exploding gradients (mean norm > 10.0 in most epochs).")

    plot_layerwise_gradients(layer_grad_history)

    return model, layer_grad_history


def train_wavelet_transformer(model, X_train, y_train, X_valid, y_valid, X_test, y_test,
                             config, device):
    """Complete training pipeline - simplified for direct prediction."""

    print("=" * 60)
    print("TRAINING WAVELET TRANSFORMER")
    print("=" * 60)

    model, grad_history = train_model(
        model, X_train, y_train, X_valid, y_valid, config, device,
        patience=config.patience if hasattr(config, 'patience') else 50,
        save_path="trained_wavelet_model.pt"
    )

    return model, grad_history


def test_wavelet_model(df, model, X_test, y_test, config, device, checkpoint_path="trained_wavelet_model.pt"):
    #print("📦 Loading model checkpoint...")
    model = model.to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    directional_criterion = TanhDirectionalLoss()
    mse_criterion = nn.MSELoss()

    total_dir_loss = 0.0
    total_mse_loss = 0.0
    total_da = 0.0

    #print(f"🧪 Testing on {len(test_dataset)} samples")

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, y in tqdm(test_loader, desc="Testing", leave=False):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            dir_loss = directional_criterion(y_pred, y)
            mse_loss = mse_criterion(y_pred, y)
            da = directional_accuracy(y_pred, y)

            total_dir_loss += dir_loss.item()
            total_mse_loss += mse_loss.item()
            total_da += da

            all_preds.append(y_pred.cpu())
            all_targets.append(y.cpu())

    avg_dir_loss = total_dir_loss / len(test_loader)
    avg_mse_loss = total_mse_loss / len(test_loader)
    avg_da = total_da / len(test_loader)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    output_std = torch.std(all_preds).item()

    # print(f"\n📊 Final Test Results:")
    # print(f"Directional Loss: {avg_dir_loss:.6f}")
    # print(f"MSE Loss:         {avg_mse_loss:.6f}")
    # print(f"Directional Acc.: {avg_da:.3f}")
    # print(f"Output Std:       {output_std:.4f}")
    # print(f"HELOOOO:       {all_preds.shape}")
    # print(all_preds[0])

    # Plot results
    plot_test_results(y_pred=all_preds, y_true=all_targets)

    # Simple plotting for first time step only
    num_samples, tgt_len = all_preds.shape  # e.g., (280, 10)

    # Get the last num_samples of actual close prices (these are the starting points)
    historical_prices = df['Adj Close'].values[-num_samples:]

    # Take only the first time step predictions and convert to percentage
    first_step_preds = all_preds[:, 0].numpy() / 100.0  # Convert to actual percentage

    # Calculate predicted prices (each based on previous day's actual price)
    predicted_prices = np.zeros(num_samples)
    for i in range(num_samples):
        if i == 0:
            # For first prediction, we need a base price (could be the last known price before test period)
            base_price = historical_prices[0]  # or use the last training price
        else:
            base_price = historical_prices[i - 1]  # Previous day's actual price

        predicted_prices[i] = base_price * (1 + first_step_preds[i])

    # Plot
    plt.figure(figsize=(14, 8))
    plt.plot(range(num_samples), historical_prices,
             label="Historical Close Price", color='black', linewidth=1.5)
    plt.plot(range(num_samples), predicted_prices,
             label="Predicted Cumulative Price (1-step)", color='blue', linewidth=1.5)

    plt.title("Historical vs Predicted Cumulative Price - First Time Step")
    plt.xlabel("Time Index")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("example_1_step_prediction.png", dpi=300)
    # plt.show()

    return all_preds, all_targets


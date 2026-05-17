"""
FDTD Solver with CNN Prediction - Hybrid FDTD-ML Approach
With Enhanced Initial Wave Amplitude
"""

import numpy as np
import time
import warnings
import sys
import os
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg')

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("CuPy available, using GPU acceleration")
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not installed, using CPU (slower)")
    import numpy as cp

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
    print("PyTorch available")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU device: {torch.cuda.get_device_name(0)}")
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not installed, CNN prediction will be disabled")

import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Microsoft YaHei', 'SimHei']
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 100

warnings.filterwarnings("ignore", category=DeprecationWarning)


class ProgressDisplay:
    def __init__(self, total_steps, desc="Simulating", unit="step"):
        self.total_steps = total_steps
        self.desc = desc
        self.unit = unit
        self.start_time = None
        self.current_step = 0

        if TQDM_AVAILABLE:
            self.pbar = tqdm(total=total_steps, desc=desc, unit=unit)
        else:
            self.pbar = None

    def start(self):
        self.start_time = time.time()
        self.current_step = 0
        if self.pbar:
            self.pbar.reset()
        else:
            self._print_progress(0)

    def update(self, n=1):
        self.current_step += n
        if self.pbar:
            self.pbar.update(n)
        else:
            if self.current_step % max(1, self.total_steps // 20) == 0:
                self._print_progress(self.current_step)

    def _print_progress(self, step):
        if step == 0:
            return
        elapsed = time.time() - self.start_time
        steps_per_sec = step / elapsed
        remaining = (self.total_steps - step) / steps_per_sec if steps_per_sec > 0 else 0
        percent = step / self.total_steps * 100
        print(f"  {self.desc}: {step}/{self.total_steps} ({percent:.1f}%) | "
              f"Speed: {steps_per_sec:.0f} {self.unit}/s | Remaining: {remaining:.0f}s")

    def close(self):
        if self.pbar:
            self.pbar.close()
        else:
            elapsed = time.time() - self.start_time
            print(f"\n {self.desc} completed in {elapsed:.2f}s")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.close()


class TimeSeriesDataset(Dataset):
    """Dataset for time series prediction - STRICTLY maintains temporal order"""

    def __init__(self, wavefield_sequence, input_timesteps=5):
        self.input_timesteps = input_timesteps
        self.data = []
        self.targets = []

        print(f"  Creating time-series dataset (preserving temporal order)...")
        print(f"  Input timesteps: {input_timesteps}")
        print(f"  Total wavefields: {len(wavefield_sequence)}")

        for i in range(len(wavefield_sequence) - input_timesteps - 1):
            input_data = wavefield_sequence[i:i+input_timesteps]
            target_data = wavefield_sequence[i+input_timesteps]
            self.data.append(input_data)
            self.targets.append(target_data)

        self.n_samples = len(self.data)
        print(f"  Created {self.n_samples} temporal samples (t=0 to t={self.n_samples-1})")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.data[idx])
        y = torch.FloatTensor(self.targets[idx])
        return x, y


class Simple2DCNN(nn.Module):
    def __init__(self, input_channels, hidden_channels=64):
        super(Simple2DCNN, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)

        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)

        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_channels)

        self.conv4 = nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x shape: [batch, input_timesteps, Nx, Ny]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        return x.squeeze(1)  # [batch, Nx, Ny]


class FDTD3D:
    def __init__(self, Lx, Ly, Lz, dx, c=343, dt=None, use_gpu=True, name="FDTD"):
        self.name = name
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.xp = cp if self.use_gpu else np

        self.Nx = int(Lx / dx) + 1
        self.Ny = int(Ly / dx) + 1
        self.Nz = int(Lz / dx) + 1
        self.dx = dx
        self.c = c
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz

        self.lambda_cfl = 1.0 / np.sqrt(3)

        # Slightly larger dt for better stability with strong sources
        if dt is None:
            self.dt = 0.9 * self.lambda_cfl * dx / c  # 90% of CFL for safety
        else:
            self.dt = dt

        self.lambda2 = (c * self.dt / dx) ** 2
        self.n_points = self.Nx * self.Ny * self.Nz

        self.coeff_self = 2 - 6 * self.lambda2
        self.coeff_neighbor = self.lambda2

        self.allocate_memory()
        self.boundary_type = 'rigid'
        self.step_count = 0
        self.wavefield_history = []
        self.slice_history = []
        self.total_time = 0

        print(f"\n {self.name} Solver initialized:")
        print(f"  Grid: {self.Nx} x {self.Ny} x {self.Nz} = {self.n_points:,} points")
        print(f"  dx: {dx*1000:.2f} mm, dt: {self.dt*1e9:.2f} ns")
        print(f"  CFL factor: {c * self.dt / dx:.3f} (max: {self.lambda_cfl:.3f})")

    def allocate_memory(self):
        shape = (self.Nx, self.Ny, self.Nz)
        if self.use_gpu:
            self.p_prev = self.xp.zeros(shape, dtype=self.xp.float32)
            self.p_curr = self.xp.zeros(shape, dtype=self.xp.float32)
            self.p_next = self.xp.zeros(shape, dtype=self.xp.float32)
        else:
            self.p_prev = np.zeros(shape, dtype=np.float32)
            self.p_curr = np.zeros(shape, dtype=np.float32)
            self.p_next = np.zeros(shape, dtype=np.float32)

    def to_numpy(self, array):
        if self.use_gpu:
            return self.xp.asnumpy(array)
        return array

    def enhanced_ricker_source(self, fc, duration, amplitude=1.0):
        """
        Enhanced Ricker wavelet with higher amplitude
        Ricker wavelet = (1 - 2π²f²(t-t0)²) exp(-π²f²(t-t0)²)
        """
        t = np.arange(0, duration, self.dt)
        tau = 1.0 / (np.pi * fc)
        t0 = 1.5 * tau
        source = amplitude * (1 - 2 * np.pi**2 * fc**2 * (t - t0)**2) * np.exp(-np.pi**2 * fc**2 * (t - t0)**2)
        return source.astype(np.float32)

    def initial_gaussian_bump(self, center_x=None, center_y=None, width=5, amplitude=2.0):
        """
        Set initial Gaussian pressure bump for immediate large amplitude
        """
        if center_x is None:
            center_x = self.Nx // 2
        if center_y is None:
            center_y = self.Ny // 2

        central_z = self.Nz // 2

        x = np.arange(self.Nx)
        y = np.arange(self.Ny)
        X, Y = np.meshgrid(x, y)

        bump = amplitude * np.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * width**2))

        for iz in range(self.Nz):
            if self.use_gpu:
                self.p_curr[:, :, iz] = self.xp.asarray(bump.T)
            else:
                self.p_curr[:, :, iz] = bump.T

        print(f"  Initial Gaussian bump applied: amplitude={amplitude}, width={width} grid points")
        return bump

    def reset(self):
        if self.use_gpu:
            self.p_prev.fill(0)
            self.p_curr.fill(0)
            self.p_next.fill(0)
        else:
            self.p_prev.fill(0)
            self.p_curr.fill(0)
            self.p_next.fill(0)
        self.step_count = 0
        self.wavefield_history = []
        self.slice_history = []

    def step(self):
        """Single timestep update using standard FDTD"""
        S = (self.p_curr[2:, 1:-1, 1:-1] +
             self.p_curr[:-2, 1:-1, 1:-1] +
             self.p_curr[1:-1, 2:, 1:-1] +
             self.p_curr[1:-1, :-2, 1:-1] +
             self.p_curr[1:-1, 1:-1, 2:] +
             self.p_curr[1:-1, 1:-1, :-2])

        self.p_next[1:-1, 1:-1, 1:-1] = (self.coeff_self * self.p_curr[1:-1, 1:-1, 1:-1]
                                         + self.coeff_neighbor * S
                                         - self.p_prev[1:-1, 1:-1, 1:-1])

        # Neumann boundary conditions
        self.p_curr[0, :, :] = self.p_curr[1, :, :]
        self.p_curr[-1, :, :] = self.p_curr[-2, :, :]
        self.p_curr[:, 0, :] = self.p_curr[:, 1, :]
        self.p_curr[:, -1, :] = self.p_curr[:, -2, :]
        self.p_curr[:, :, 0] = self.p_curr[:, :, 1]
        self.p_curr[:, :, -1] = self.p_curr[:, :, -2]

        self.p_prev, self.p_curr, self.p_next = self.p_curr, self.p_next, self.p_prev
        self.step_count += 1

    def get_central_slice(self):
        """Get the central z-slice of current wavefield"""
        field = self.to_numpy(self.p_curr)
        central_z = self.Nz // 2
        return field[:, :, central_z]

    def run_pure_fdtd(self, n_steps, source_position=None, source_waveform=None,
                      receiver_positions=None, save_history=True, verbose=True,
                      initial_bump=True, bump_amplitude=2.0):
        """Run pure FDTD simulation with enhanced initial waves"""
        receiver_signals = {}
        if receiver_positions is not None:
            for name in receiver_positions:
                receiver_signals[name] = []

        if verbose:
            print(f"\n {self.name}: Pure FDTD Simulation (Enhanced Initial Waves)")
            print(f"  Total steps: {n_steps}")
            print(f"  Initial Gaussian bump: {'ON' if initial_bump else 'OFF'} (amplitude={bump_amplitude})")

        self.reset()

        if initial_bump:
            self.initial_gaussian_bump(amplitude=bump_amplitude, width=8)

        if save_history:
            self.wavefield_history = []
            self.slice_history = []
            initial_field = self.to_numpy(self.p_curr).copy()
            self.wavefield_history.append(initial_field)
            self.slice_history.append(initial_field[:, :, self.Nz//2])

        start_time = time.time()

        with ProgressDisplay(n_steps, desc=f"  {self.name} stepping", unit="step") as prog:
            for step in range(n_steps):
                if source_position is not None and source_waveform is not None:
                    if step < len(source_waveform):
                        ix, iy, iz = source_position
                        self.p_curr[ix, iy, iz] += source_waveform[step]
                        if step < 50:
                            if ix+1 < self.Nx:
                                self.p_curr[ix+1, iy, iz] += source_waveform[step] * 0.5
                            if ix-1 >= 0:
                                self.p_curr[ix-1, iy, iz] += source_waveform[step] * 0.5
                            if iy+1 < self.Ny:
                                self.p_curr[ix, iy+1, iz] += source_waveform[step] * 0.5
                            if iy-1 >= 0:
                                self.p_curr[ix, iy-1, iz] += source_waveform[step] * 0.5

                self.step()

                if save_history:
                    if step % max(1, n_steps // 100) == 0 or step == n_steps - 1:
                        current_field = self.to_numpy(self.p_curr).copy()
                        self.wavefield_history.append(current_field)
                        self.slice_history.append(current_field[:, :, self.Nz//2])

                if receiver_positions is not None:
                    for name, pos in receiver_positions.items():
                        ix, iy, iz = pos
                        val = self.p_curr[ix, iy, iz]
                        if self.use_gpu:
                            val = float(val.item())
                        else:
                            val = float(val)
                        receiver_signals[name].append(val)

                prog.update()

        elapsed = time.time() - start_time
        self.total_time = elapsed

        if verbose:
            print(f"\n   Training data generated!")
            print(f"    Time: {elapsed:.2f}s")
            print(f"    Speed: {n_steps/elapsed:.0f} steps/s")
            print(f"    Recorded {len(self.slice_history)} slices")

            if receiver_positions and 'center' in receiver_signals:
                center_signal = np.array(receiver_signals['center'])
                print(f"    Signal amplitude: max={np.max(np.abs(center_signal)):.4f} Pa")

        return receiver_signals, elapsed

    def train_prediction_model(self, input_timesteps=5, n_epochs=3000, batch_size=16,
                               learning_rate=1e-3, hidden_channels=64, device='cuda'):
        """Train CNN model on wavefield history"""

        if not TORCH_AVAILABLE:
            print(" PyTorch not available")
            return None, None, None

        if len(self.slice_history) < input_timesteps + 10:
            print(f" Insufficient history: {len(self.slice_history)} < {input_timesteps + 10}")
            return None, None, None

        print(f"\n Training CNN Model (For Autoregressive Prediction)")
        print(f"  Input timesteps: {input_timesteps}")
        print(f"  Total snapshots: {len(self.slice_history)}")

        wavefield_2d = np.array(self.slice_history)
        print(f"  Slice shape: {wavefield_2d.shape}")

        dataset = TimeSeriesDataset(wavefield_2d, input_timesteps)

        n_samples = len(dataset)
        split_idx = int(0.2 * n_samples)

        train_indices = list(range(0, split_idx))
        val_indices = list(range(split_idx, n_samples))

        print(f"\n  Temporal Data Split:")
        print(f"    Total samples: {n_samples}")
        print(f"    Training: {len(train_indices)} samples (time steps 0-{split_idx-1})")
        print(f"    Validation: {len(val_indices)} samples (time steps {split_idx}-{n_samples-1})")

        class TemporalSubset(Dataset):
            def __init__(self, original_dataset, indices):
                self.original = original_dataset
                self.indices = indices

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, idx):
                return self.original[self.indices[idx]]

        train_dataset = TemporalSubset(dataset, train_indices)
        val_dataset = TemporalSubset(dataset, val_indices)

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=False, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=False, drop_last=True)

        model = Simple2DCNN(input_channels=input_timesteps, hidden_channels=hidden_channels)
        model = model.to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        print(f"\n  Training Configuration:")
        print(f"    Device: {device}")
        print(f"    Epochs: {n_epochs}")
        print(f"    Batch size: {batch_size}")
        print(f"    Shuffle: False (temporal order preserved)")

        train_losses = []
        val_losses = []

        epoch_range = tqdm(range(n_epochs), desc="  Training") if TQDM_AVAILABLE else range(n_epochs)

        for epoch in epoch_range:
            model.train()
            train_loss = 0.0
            n_train_batches = 0

            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                n_train_batches += 1

            train_loss /= n_train_batches
            train_losses.append(train_loss)

            model.eval()
            val_loss = 0.0
            n_val_batches = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    n_val_batches += 1

            val_loss /= n_val_batches
            val_losses.append(val_loss)

            scheduler.step(val_loss)

            if not TQDM_AVAILABLE and (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss:.6e} | Val Loss: {val_loss:.6e}")

        print(f"\n   Training completed!")
        print(f"    Final validation loss: {val_losses[-1]:.6e}")

        return model, train_losses, val_losses

    def run_autoregressive_prediction(self, total_steps, fdtd_init_steps, input_timesteps,
                                       cnn_model, source_position=None, source_waveform=None,
                                       receiver_positions=None, device='cuda', verbose=True,
                                       initial_bump=True, bump_amplitude=2.0):
        """Run autoregressive prediction with enhanced initial waves"""

        if cnn_model is None:
            print("No CNN model provided")
            return None, None, None, None, None

        receiver_signals = {}
        if receiver_positions is not None:
            for name in receiver_positions:
                receiver_signals[name] = []

        if verbose:
            print(f"\n▶ Autoregressive Prediction Mode")
            print(f"  Total steps: {total_steps}")
            print(f"  FDTD init steps: {fdtd_init_steps}")
            print(f"  Input timesteps: {input_timesteps}")
            print(f"  CNN prediction steps: {total_steps - fdtd_init_steps}")
            print(f"  Initial Gaussian bump: amplitude={bump_amplitude}")

        self.reset()

        if initial_bump:
            self.initial_gaussian_bump(amplitude=bump_amplitude, width=8)

        predicted_slice_history = []
        actual_slice_history = []
        prediction_buffer = []
        prediction_errors = []

        cnn_model.eval()
        cnn_model = cnn_model.to(device)

        start_time = time.time()

        # Phase 1: Pure FDTD to build initial history
        if verbose:
            print(f"\n  Phase 1: FDTD initialization (steps 0 to {fdtd_init_steps-1})")

        initial_slice = self.get_central_slice()
        actual_slice_history.append(initial_slice)
        prediction_buffer.append(initial_slice)

        for step in range(fdtd_init_steps):
            if source_position is not None and source_waveform is not None:
                if step < len(source_waveform):
                    ix, iy, iz = source_position
                    self.p_curr[ix, iy, iz] += source_waveform[step]
                    if step < 50:
                        if ix+1 < self.Nx:
                            self.p_curr[ix+1, iy, iz] += source_waveform[step] * 0.5
                        if ix-1 >= 0:
                            self.p_curr[ix-1, iy, iz] += source_waveform[step] * 0.5

            self.step()

            current_slice = self.get_central_slice()
            actual_slice_history.append(current_slice)

            prediction_buffer.append(current_slice)
            if len(prediction_buffer) > input_timesteps:
                prediction_buffer.pop(0)

            if receiver_positions is not None:
                for name, pos in receiver_positions.items():
                    ix, iy, iz = pos
                    val = self.p_curr[ix, iy, iz]
                    if self.use_gpu:
                        val = float(val.item())
                    else:
                        val = float(val)
                    receiver_signals[name].append(val)

            if verbose and (step + 1) % max(1, fdtd_init_steps // 10) == 0:
                print(f"    FDTD step {step+1}/{fdtd_init_steps}")

        # Phase 2: Autoregressive CNN prediction
        if verbose:
            print(f"\n  Phase 2: Autoregressive CNN prediction")

        for step in range(fdtd_init_steps, total_steps):
            if len(prediction_buffer) >= input_timesteps:
                input_slices = np.stack(prediction_buffer[-input_timesteps:], axis=0)
                input_tensor = torch.FloatTensor(input_slices).unsqueeze(0).to(device)

                with torch.no_grad():
                    predicted_slice = cnn_model(input_tensor).squeeze(0).cpu().numpy()

                predicted_slice_history.append(predicted_slice)

                prediction_buffer.append(predicted_slice)
                if len(prediction_buffer) > input_timesteps:
                    prediction_buffer.pop(0)

                if source_position is not None and source_waveform is not None:
                    if step < len(source_waveform):
                        ix, iy, iz = source_position
                        self.p_curr[ix, iy, iz] += source_waveform[step]

                self.step()
                actual_slice = self.get_central_slice()
                actual_slice_history.append(actual_slice)

                error = np.mean((predicted_slice - actual_slice) ** 2)
                prediction_errors.append(error)

                if receiver_positions is not None:
                    for name, pos in receiver_positions.items():
                        ix, iy, iz = pos
                        val = self.p_curr[ix, iy, iz]
                        if self.use_gpu:
                            val = float(val.item())
                        else:
                            val = float(val)
                        receiver_signals[name].append(val)
            else:
                print(f"    Warning: Not enough history at step {step}")
                break

            if verbose and (step - fdtd_init_steps + 1) % max(1, (total_steps - fdtd_init_steps) // 10) == 0:
                steps_done = step - fdtd_init_steps + 1
                total_pred_steps = total_steps - fdtd_init_steps
                print(f"    Prediction step {steps_done}/{total_pred_steps} | "
                      f"Current MSE: {error:.6e}")

        elapsed = time.time() - start_time

        if verbose:
            print(f"\n  Autoregressive prediction completed!")
            print(f"    Time: {elapsed:.2f}s")
            print(f"    Total predictions: {len(predicted_slice_history)}")
            if prediction_errors:
                print(f"    Avg prediction MSE: {np.mean(prediction_errors):.6e}")

        return receiver_signals, elapsed, actual_slice_history, predicted_slice_history, prediction_errors


def save_figure(fig, filename, dpi=150):
    """Safely save figure and close to free memory"""
    try:
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"   Saved: {filename}")
    except Exception as e:
        print(f"   Could not save {filename}: {e}")
    finally:
        plt.close(fig)


def run_autoregressive_demo():
    """Run complete demo with enhanced initial waves"""
    if not TORCH_AVAILABLE:
        print("\n PyTorch not available. Please install: pip install torch")
        return

    # Simulation parameters
    Lx, Ly, Lz = 0.5, 0.5, 0.5
    dx = 0.01
    c = 343
    fc = 100
    duration = 0.005

    total_steps = 300
    fdtd_init_steps = 60
    input_timesteps = 5

    # Enhanced source amplitude
    source_amplitude = 3.0
    bump_amplitude = 2.5

    print(f"\nSimulation Parameters:")
    print(f"  Domain: {Lx} x {Ly} x {Lz} m")
    print(f"  Grid: {int(Lx/dx)+1} x {int(Ly/dx)+1} x {int(Lz/dx)+1}")
    print(f"  Total steps: {total_steps}")
    print(f"  FDTD initialization steps: {fdtd_init_steps}")
    print(f"  Source amplitude: {source_amplitude}")
    print(f"  Initial bump amplitude: {bump_amplitude}")

    # Create solver and generate training data
    solver = FDTD3D(Lx, Ly, Lz, dx, c, use_gpu=CUPY_AVAILABLE, name="FDTD")

    source_pos = (solver.Nx//2, solver.Ny//2, solver.Nz//2)
    source_waveform = solver.enhanced_ricker_source(fc, duration, amplitude=source_amplitude)

    receivers = {'center': source_pos, 'offset': (solver.Nx//2 + 5, solver.Ny//2, solver.Nz//2)}

    # Generate training data
    print("\n" + "="*50)
    print("Step 1: Generating Training Data")
    print("   (With Enhanced Initial Waves)")
    print("="*50)

    signals, sim_time = solver.run_pure_fdtd(
        n_steps=total_steps,
        source_position=source_pos,
        source_waveform=source_waveform,
        receiver_positions=receivers,
        save_history=True,
        verbose=True,
        initial_bump=True,
        bump_amplitude=bump_amplitude
    )

    # Train CNN model
    print("\n" + "="*50)
    print("Step 2: Training CNN Model")
    print("="*50)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cnn_model, train_losses, val_losses = solver.train_prediction_model(
        input_timesteps=input_timesteps,
        n_epochs=3000,
        batch_size=16,
        learning_rate=1e-4,
        hidden_channels=64,
        device=device
    )

    if cnn_model is None:
        print("\n CNN training failed")
        return

    # 注意：已移除 training_loss.png 的生成代码

    # Run autoregressive prediction
    print("\n" + "="*50)
    print("Step 3: Autoregressive Prediction")
    print("="*50)

    solver_ar = FDTD3D(Lx, Ly, Lz, dx, c, use_gpu=CUPY_AVAILABLE, name="AR-Predictor")

    ar_signals, ar_time, actual_slices, predicted_slices, prediction_errors = solver_ar.run_autoregressive_prediction(
        total_steps=total_steps,
        fdtd_init_steps=fdtd_init_steps,
        input_timesteps=input_timesteps,
        cnn_model=cnn_model,
        source_position=source_pos,
        source_waveform=source_waveform,
        receiver_positions=receivers,
        device=device,
        verbose=True,
        initial_bump=True,
        bump_amplitude=bump_amplitude
    )

    # Compare results
    print("\n" + "="*50)
    print("Step 4: Results Comparison")
    print("="*50)

    fdtd_signal = np.array(signals['center'])
    ar_signal = np.array(ar_signals['center'])

    t = np.arange(len(fdtd_signal)) * solver.dt

    min_len = min(len(fdtd_signal), len(ar_signal))
    diff = np.abs(fdtd_signal[:min_len] - ar_signal[:min_len])

    print(f"\n  Error Metrics:")
    print(f"    Max absolute error: {np.max(diff):.6e} Pa")
    print(f"    Mean absolute error: {np.mean(diff):.6e} Pa")
    print(f"    RMS error: {np.sqrt(np.mean(diff**2)):.6e} Pa")

    print(f"\n  Signal Statistics:")
    print(f"    Max FDTD amplitude: {np.max(np.abs(fdtd_signal)):.4f} Pa")
    print(f"    Max AR amplitude: {np.max(np.abs(ar_signal)):.4f} Pa")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Full signal comparison
    ax1 = axes[0, 0]
    ax1.plot(t, fdtd_signal, 'b-', label='FDTD Ground Truth', alpha=0.8, linewidth=1)
    ax1.plot(t[:len(ar_signal)], ar_signal, 'r--', label='Autoregressive CNN', alpha=0.8, linewidth=1)
    ax1.axvline(x=t[fdtd_init_steps-1], color='g', linestyle=':', linewidth=2,
                label=f'Prediction starts (step {fdtd_init_steps})')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Pressure [Pa]')
    ax1.set_title('FDTD vs FDTD+CNN (Enhanced Initial Waves)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. Zoomed view of early high-amplitude region
    ax2 = axes[0, 1]
    zoom_steps = min(fdtd_init_steps, len(t))
    ax2.plot(t[:zoom_steps], fdtd_signal[:zoom_steps], 'b-', label='FDTD', alpha=0.8)
    ax2.plot(t[:zoom_steps], ar_signal[:zoom_steps], 'r--', label='CNN Prediction', alpha=0.8)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Pressure [Pa]')
    ax2.set_title(f'Zoomed: First {zoom_steps} steps ')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. Prediction error
    ax3 = axes[1, 0]
    pred_start_idx = fdtd_init_steps
    if pred_start_idx < len(diff):
        error_t = t[pred_start_idx:len(diff)]
        ax3.semilogy(error_t, diff[pred_start_idx:], 'orange', linewidth=1)
        ax3.axhline(y=np.mean(diff[pred_start_idx:]), color='r', linestyle='--',
                   label=f'Mean error: {np.mean(diff[pred_start_idx:]):.2e}')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Absolute Error [Pa]')
    ax3.set_title('Prediction Error (log scale)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4. Signal envelope (shows amplitude decay)
    ax4 = axes[1, 1]
    from scipy.signal import hilbert
    analytic_signal = hilbert(fdtd_signal)
    envelope = np.abs(analytic_signal)
    ax4.plot(t, envelope, 'purple', linewidth=1)
    ax4.axvline(x=t[fdtd_init_steps-1], color='g', linestyle=':', linewidth=1)
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Envelope Amplitude [Pa]')
    ax4.set_title('Signal Envelope (Shows Amplitude Decay After Initial Peak)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, 'autoregressive_results_enhanced.png')


    print("\n" + "="*70)
    print("DEMO COMPLETED")
    print("="*70)
    print(f"""
    Summary:
    ─────────────────────────────────────────────────────────────────
    Enhanced Features:
    - Initial Gaussian bump amplitude: {bump_amplitude} Pa
    - Source amplitude: {source_amplitude}
    - Max FDTD signal amplitude: {np.max(np.abs(fdtd_signal)):.4f} Pa
    
    Performance:
    ─────────────────────────────────────────────────────────────────
    Training data generation:    {sim_time:.2f}s
    Autoregressive prediction:   {ar_time:.2f}s
    
    """)

    return solver, solver_ar, signals, ar_signal, cnn_model, prediction_errors


# ========== Main Entry Point ==========
if __name__ == "__main__":


    result = run_autoregressive_demo()

    print("\n Done!")
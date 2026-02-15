"""
Shock Detection Module (core/shock.py)
Detects scene cuts and extreme motion events by monitoring Luma MAD
and warp error spikes. Triggers temporal reset on shock events.
"""
import torch
import numpy as np


class ShockDetector:
    """Lightweight frame-level shock/scene-cut detector."""

    def __init__(self, mad_threshold=30.0, err_spike_ratio=1.5, window=10):
        """
        Args:
            mad_threshold: Luma MAD above this = instant shock
            err_spike_ratio: warp_error > ratio * rolling_avg = shock
            window: rolling average window size
        """
        self.mad_threshold = mad_threshold
        self.err_spike_ratio = err_spike_ratio
        self.window = window
        self._err_history = []

    def reset(self):
        self._err_history.clear()

    def check(self, gray1_np, gray2_np, warp_error=None):
        """
        Check for shock event between two consecutive grayscale frames.

        Args:
            gray1_np: (H, W) uint8 previous frame luma
            gray2_np: (H, W) uint8 current frame luma
            warp_error: float, current frame's warp error (optional)

        Returns:
            is_shock: bool
            reason: str or None
        """
        # Luma MAD
        mad = float(np.abs(gray1_np.astype(np.float32) - gray2_np.astype(np.float32)).mean())

        if mad > self.mad_threshold:
            self._err_history.clear()
            return True, f"MAD={mad:.1f} > {self.mad_threshold}"

        # Warp error spike
        if warp_error is not None and len(self._err_history) >= 3:
            rolling_avg = np.mean(self._err_history[-self.window:])
            if warp_error > self.err_spike_ratio * rolling_avg:
                self._err_history.clear()
                return True, f"err={warp_error:.4f} > {self.err_spike_ratio}x avg={rolling_avg:.4f}"

        # Update history
        if warp_error is not None:
            self._err_history.append(warp_error)
            if len(self._err_history) > self.window * 2:
                self._err_history = self._err_history[-self.window:]

        return False, None

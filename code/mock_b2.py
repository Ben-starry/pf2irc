"""Large mock run for the null hypothesis b = 2 (Sect. 6.3): python3 mock_b2.py N seed"""
import sys, numpy as np
N, SEED = int(sys.argv[1]), int(sys.argv[2])
sys.argv = [sys.argv[0], "0"]
import robustness as R          # runs the data part once (fast) with 0 mocks
R.rng = np.random.default_rng(SEED)
s2 = np.sqrt(max(R.lrc.var()-4*R.lv.var(), 0.05**2))
M = np.array([R.mock(2.0, s2)[0] for _ in range(N)])
np.save(f"mock_b2_{SEED}.npy", M)
print(f"seed {SEED}: N={N} median {np.median(M):.3f} frac<=0.97 {(M<=0.97).mean():.4f} min {M.min():.3f}")

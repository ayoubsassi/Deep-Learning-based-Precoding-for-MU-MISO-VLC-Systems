import argparse
from pathlib import Path

import joblib
import matlab.engine
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import time

from R_func import R_func

BANDWIDTH = 1e8
EE = 1.60217662e-19
RHO_K = 0.4
XI = 10.93
A_PD_AREA = 1e-4
Q_INDEX = 1.5
THETA_C = np.deg2rad(60)
A_K = Q_INDEX**2 * A_PD_AREA / (np.sin(THETA_C) ** 2)
I_AMP = 5e-12
P_DB = np.arange(15, 31, 1)
P_LINEAR = 10 ** ((P_DB - 30) / 10)

DEFAULT_MODEL = Path("mixed_masked_model_16k(123456).h5")
DEFAULT_SCALER = Path("SS_scaler_mixed_masked_16k(123456).pkl")
DEFAULT_DATA = Path("mixed_dataset_test_split.pkl")

def call_cvx_in_matlab(eng, t, M, K, sigma, p, H):
    sigma_col = sigma.reshape(-1, 1)
    p_col = p.reshape(-1, 1)
    W, d, status = eng.cvx_func_bisection_2(
        matlab.double([t]),
        matlab.double([M]),
        matlab.double([K]),
        matlab.double(sigma_col.tolist()),
        matlab.double(p_col.tolist()),
        matlab.double(H.tolist()),
        nargout=3,
    )
    return np.array(W), np.array(d), status


def compute_sigma(H, Pn):
    Ps = Pn * np.sum(H, axis=0)
    ambient = 2 * EE * RHO_K * XI * A_K * 2 * np.pi * (1 - np.cos(THETA_C)) * BANDWIDTH
    sigma = np.zeros(H.shape[1])
    for i in range(H.shape[1]):
        sigma[i] = 2 * EE * Ps[i] * BANDWIDTH + ambient + I_AMP**2 * BANDWIDTH
    return sigma


def compute_sinr(H, W, sigma):
    K = H.shape[1]
    SINR = np.zeros(K)
    for k in range(K):
        h_k = H[:, k]
        w_k = W[:, k]
        W_k = np.delete(W, k, axis=1)
        Sum = W_k @ W_k.T
        num = (h_k @ w_k) ** 2
        den = h_k @ (Sum @ h_k) + sigma[k]
        SINR[k] = num / den if den > 0 else 0.0
    return SINR


def enforce_power_constraint(W, Pn):
    for m in range(W.shape[0]):
        s = np.sum(np.abs(W[m]))
        if s > Pn and s > 0:
            W[m] = W[m] / s * Pn
    return W


def format_matrix(M):
    return np.array2string(M, precision=6, suppress_small=True)


def format_pairs(p_db, values):
    return ", ".join(f"({int(p)}, {v:.2f})" for p, v in zip(p_db, values))


def select_indices(df, num_samples, target_k, seed):
    if target_k is not None:
        mask = df["mask"].apply(lambda m: sum(m) == target_k)
        candidates = df.index[mask].tolist()
        if not candidates:
            raise ValueError(f"No samples found with K={target_k}.")
    else:
        candidates = df.index.tolist()
    if num_samples is None or num_samples <= 0 or num_samples > len(candidates):
        selected = candidates
    else:
        rng = np.random.default_rng(seed)
        selected = rng.choice(candidates, size=num_samples, replace=False)
    return list(selected)


def plot_agg_curves(p_db, tuples_list, ylabel, title, fname):
    plt.figure(figsize=(8, 5))
    for vals, label, color in tuples_list:
        plt.semilogy(p_db, vals, color, label=label)
    plt.xlabel("Pn [dBm]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both")
    plt.xlim(P_DB[0], P_DB[-1])
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname)
    plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ZF/OLP/ML rates on dataset samples using CVX OLP."
    )
  
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--scaler-path", type=Path, default=DEFAULT_SCALER)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--target-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("test_rates_summary_cvx.txt"))
    args = parser.parse_args()

    df = pd.read_pickle(args.data_path)
    indices = select_indices(df, args.num_samples, args.target_k, args.seed)
    if not indices:
        raise ValueError("No samples selected. Adjust --num-samples or --target-k.")

    scaler = joblib.load(args.scaler_path)
    model = tf.keras.models.load_model(args.model_path.as_posix(), compile=False)

    eng = matlab.engine.start_matlab()
    eng.addpath("cvx_func_bisection_2.m", nargout=0)
    eng.addpath(eng.genpath(r"C:\Users\ayoubsa\Desktop\cvx"), nargout=0)

    rate_zf_sum = np.zeros(len(P_DB))
    rate_olp_sum = np.zeros(len(P_DB))
    rate_ml_sum = np.zeros(len(P_DB))
    max_zf_sum = np.zeros(len(P_DB))
    max_olp_sum = np.zeros(len(P_DB))
    max_ml_sum = np.zeros(len(P_DB))
    zf_time = 0.0
    olp_time = 0.0
    ml_time = 0.0
    time_count = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as report:
        report.write(
            f"Dataset: {args.data_path}\nModel: {args.model_path}\nScaler: {args.scaler_path}\n"
            f"Samples evaluated: {len(indices)}\nTarget K: {args.target_k or 'All'}\n\n"
        )

        for sample_no, idx in enumerate(indices, 1):
            row = df.loc[idx]
            H_full = np.array(row["H"])
            mask = np.array(row["mask"], dtype=float)
            active_idx = np.where(mask > 0)[0]
            H_active = H_full[:, active_idx]
            K = H_active.shape[1]
            M = H_active.shape[0]
            label_active = np.array(row["label"])[:, active_idx]
            Pn_orig = row["power"]

            H_vec = H_full.flatten()
            mask_vec = np.tile(mask, H_full.shape[0]).reshape(1, -1)

            rate_zf = np.zeros(len(P_LINEAR))
            rate_olp = np.zeros(len(P_LINEAR))
            rate_ml = np.zeros(len(P_LINEAR))
            max_zf = np.zeros(len(P_LINEAR))
            max_olp = np.zeros(len(P_LINEAR))
            max_ml = np.zeros(len(P_LINEAR))

            idx_ref = np.argmin(np.abs(P_LINEAR - Pn_orig))
            W_zf_ref = None
            W_olp_ref = None
            W_ml_ref = None

            for i, Pn in enumerate(P_LINEAR):
                sigma = compute_sigma(H_active, Pn)

                zf_start = time.perf_counter()
                gram = H_active.T @ H_active
                C = H_active @ np.linalg.pinv(gram)
                A = np.abs(C) @ np.diag(np.sqrt(sigma))
                vec = Pn / (A @ np.ones((K, 1)))
                mu = np.min(vec)
                gamma = mu * np.sqrt(sigma)
                W_zf = C @ np.diag(gamma)
                zf_time += time.perf_counter() - zf_start

                sinr_zf = compute_sinr(H_active, W_zf, sigma)
                log_rate_zf = BANDWIDTH * np.log2(1 + sinr_zf)
                rate_zf[i] = (1 / K) * np.sum(log_rate_zf)
                max_zf[i] = np.min(log_rate_zf)

                olp_start = time.perf_counter()
                p_vec = Pn * np.ones(M)
                W_init = H_active.copy()
                norms = np.linalg.norm(W_init, ord=1, axis=1, keepdims=True)
                val = np.max(norms)
                W_olp = W_init / val * Pn if val > 0 else W_init
                it = 0
                t_aux = 1e-6
                t_works = 1e-10
                W_works = W_olp
                while abs(t_works - t_aux) / (t_works + 1e-12) >= 1e-3 or it <= 1:
                    it += 1
                    t_aux = t_works
                    t_lower = t_works
                    t_upper = 1e5
                    tol = 0.01
                    while (t_upper - t_lower) / (t_lower + 1e-12) > tol:
                        t_test = (t_upper + t_lower) / 2.0
                        Wz, d, status = call_cvx_in_matlab(eng, t_test, M, K, sigma, p_vec, H_active)
                        if str(status) != "Solved":
                            t_upper = t_test
                        else:
                            W_works = Wz
                            t_works = t_test
                            t_lower = t_test
                    W_olp = W_works
                olp_time += time.perf_counter() - olp_start

                sinr_olp = compute_sinr(H_active, W_olp, sigma)
                log_rate_olp = BANDWIDTH * np.log2(1 + sinr_olp)
                rate_olp[i] = (1 / K) * np.sum(log_rate_olp)
                max_olp[i] = np.min(log_rate_olp)

                ml_start = time.perf_counter()
                x_in = np.concatenate([H_vec, [Pn]]).reshape(1, -1)
                x_scaled = scaler.transform(x_in).reshape(1, -1, 1)
                pred_flat = model.predict([x_scaled, mask_vec], verbose=0)[0]
                W_full = pred_flat.reshape(H_full.shape)
                W_ml = W_full[:, active_idx]
                W_ml = enforce_power_constraint(W_ml, Pn)
                ml_time += time.perf_counter() - ml_start

                sinr_ml = compute_sinr(H_active, W_ml, sigma)
                log_rate_ml = BANDWIDTH * np.log2(1 + sinr_ml)
                rate_ml[i] = (1 / K) * np.sum(log_rate_ml)
                max_ml[i] = np.min(log_rate_ml)

                if i == idx_ref:
                    W_zf_ref = W_zf.copy()
                    W_olp_ref = W_olp.copy()
                    W_ml_ref = W_ml.copy()
                time_count += 1

            rate_zf_sum += rate_zf
            rate_olp_sum += rate_olp
            rate_ml_sum += rate_ml
            max_zf_sum += max_zf
            max_olp_sum += max_olp
            max_ml_sum += max_ml

            report.write(f"=== Sample {sample_no} (idx={idx}, K={K}, Pn_orig={Pn_orig:.4f} W) ===\n")
            report.write(f"H_full (shape={H_full.shape}):\n{format_matrix(H_full)}\n")
            report.write(f"H_active (mask=1 cols):\n{format_matrix(H_active)}\n")
            report.write(f"Stored label (active cols):\n{format_matrix(label_active)}\n")
            report.write(f"W_ZF @ Pn~orig:\n{format_matrix(W_zf_ref)}\n")
            report.write(f"W_OLP @ Pn~orig:\n{format_matrix(W_olp_ref)}\n")
            report.write(f"W_ML @ Pn~orig:\n{format_matrix(W_ml_ref)}\n")
            report.write("Average rate tuples:\n")
            report.write(f"  ZF : {format_pairs(P_DB, rate_zf)}\n")
            report.write(f"  OLP: {format_pairs(P_DB, rate_olp)}\n")
            report.write(f"  ML : {format_pairs(P_DB, rate_ml)}\n")
            report.write("Max-min rate tuples:\n")
            report.write(f"  ZF : {format_pairs(P_DB, max_zf)}\n")
            report.write(f"  OLP: {format_pairs(P_DB, max_olp)}\n")
            report.write(f"  ML : {format_pairs(P_DB, max_ml)}\n\n")

        n = len(indices)
        avg_rate_zf = rate_zf_sum / n
        avg_rate_olp = rate_olp_sum / n
        avg_rate_ml = rate_ml_sum / n
        avg_max_zf = max_zf_sum / n
        avg_max_olp = max_olp_sum / n
        avg_max_ml = max_ml_sum / n

        report.write("=== Aggregate averages across evaluated samples ===\n")
        report.write(f"Average rate - ZF : {format_pairs(P_DB, avg_rate_zf)}\n")
        report.write(f"Average rate - OLP: {format_pairs(P_DB, avg_rate_olp)}\n")
        report.write(f"Average rate - ML : {format_pairs(P_DB, avg_rate_ml)}\n")
        report.write(f"Max-min rate - ZF : {format_pairs(P_DB, avg_max_zf)}\n")
        report.write(f"Max-min rate - OLP: {format_pairs(P_DB, avg_max_olp)}\n")
        report.write(f"Max-min rate - ML : {format_pairs(P_DB, avg_max_ml)}\n")

        if time_count:
            report.write("\n=== Timing summary (CVX path) ===\n")
            report.write(f"ZF avg per Pn (ms): {1e3 * zf_time / time_count:.4f}\n")
            report.write(f"OLP avg per Pn (ms): {1e3 * olp_time / time_count:.4f}\n")
            report.write(f"ML avg per Pn (ms): {1e3 * ml_time / time_count:.4f}\n")
            report.write(f"ZF avg per sweep (ms): {1e3 * zf_time / len(indices):.4f}\n")
            report.write(f"OLP avg per sweep (ms): {1e3 * olp_time / len(indices):.4f}\n")
            report.write(f"ML avg per sweep (ms): {1e3 * ml_time / len(indices):.4f}\n")

    eng.quit()

    plot_agg_curves(
        P_DB,
        [
            (avg_rate_zf, "ZF", "brown"),
            (avg_rate_olp, "OLP", "magenta"),
            (avg_rate_ml, "ML", "blue"),
        ],
        ylabel="Average rate [bits/sec]",
        title="Average rate per UE (aggregated)",
        fname="aggregate_avg_rate.png",
    )
    plot_agg_curves(
        P_DB,
        [
            (avg_max_zf, "ZF", "brown"),
            (avg_max_olp, "OLP", "magenta"),
            (avg_max_ml, "ML", "blue"),
        ],
        ylabel="Max-min rate [bits/sec]",
        title="Max-min rate (aggregated)",
        fname="aggregate_maxmin_rate.png",
    )

    print(f"Evaluation finished. Detailed log saved to {args.output}")
    print("Average rate tuples (all samples):")
    print(f"  ZF : {format_pairs(P_DB, avg_rate_zf)}")
    print(f"  OLP: {format_pairs(P_DB, avg_rate_olp)}")
    print(f"  ML : {format_pairs(P_DB, avg_rate_ml)}")
    print("Average max-min tuples (all samples):")
    print(f"  ZF : {format_pairs(P_DB, avg_max_zf)}")
    print(f"  OLP: {format_pairs(P_DB, avg_max_olp)}")
    print(f"  ML : {format_pairs(P_DB, avg_max_ml)}")
    if time_count:
        print("\nTiming summary (CVX path):")
        print(f"  ZF avg per Pn (ms): {1e3 * zf_time / time_count:.4f}")
        print(f"  OLP avg per Pn (ms): {1e3 * olp_time / time_count:.4f}")
        print(f"  ML avg per Pn (ms): {1e3 * ml_time / time_count:.4f}")
        print(f"  ZF avg per sweep (ms): {1e3 * zf_time / len(indices):.4f}")
        print(f"  OLP avg per sweep (ms): {1e3 * olp_time / len(indices):.4f}")
        print(f"  ML avg per sweep (ms): {1e3 * ml_time / len(indices):.4f}")


if __name__ == "__main__":
    main()

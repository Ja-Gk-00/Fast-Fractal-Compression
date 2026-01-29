import pandas as pd
import json
import re
import matplotlib.pyplot as plt
import numpy as np


def visualize_bench_result(df, features):
    if isinstance(features, str):
        features_list = [features]
    elif isinstance(features, list):
        features_list = features[:2]
    else:
        print("Error: Argument 'features' must be string or list type.")
        return

    for feature in features_list:
        if feature not in df.columns:
            print(f"Warning: Feature '{feature}' was not found in the data.")
            continue

        df_agg = df.groupby(feature, as_index=False).mean(numeric_only=True)
        df_agg = df_agg.sort_values(by=feature)
        df_agg[feature] = df_agg[feature].astype(str)

        # Bar chart for compression time
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(df_agg[feature], df_agg["ffc_mean_s"], color="#4682B4", edgecolor="#08306B", linewidth=1.5)
        ax.set_xlabel(f"Feature: {feature}")
        ax.set_ylabel("Mean time [s]")
        ax.set_title(f"Impact of {feature.upper()} on mean compression time")
        plt.tight_layout()
        plt.show()

        # Line chart for compression ratio
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df_agg[feature], df_agg["ffc_compression_ratio"], marker='o', linewidth=4, markersize=12, color="#2E4053", label="FFC Compression Ratio")
        
        if "jpeg_compression_ratio" in df.columns:
            jpeg_avg = df["jpeg_compression_ratio"].mean()
            ax.axhline(y=jpeg_avg, linestyle="--", color="#FF4B4B", label=f"JPEG Avg ({jpeg_avg:.2f})")
            ax.legend()

        ax.set_xlabel(f"Feature: {feature}")
        ax.set_ylabel("Compression ratio")
        ax.set_title(f"Impact of {feature.upper()} on compression ratio")
        plt.tight_layout()
        plt.show()

    if len(features_list) == 2:
        f1, f2 = features_list

        df_heat = df.groupby([f1, f2], as_index=False).mean(numeric_only=True)
        df_pivot = df_heat.pivot(index=f1, columns=f2, values="ffc_mean_s")

        df_pivot = df_pivot.sort_index(axis=0).sort_index(axis=1)
        df_pivot.index = df_pivot.index.astype(str)
        df_pivot.columns = df_pivot.columns.astype(str)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(df_pivot.values, cmap="viridis", aspect="auto")
        ax.set_xticks(np.arange(len(df_pivot.columns)))
        ax.set_yticks(np.arange(len(df_pivot.index)))
        ax.set_xticklabels(df_pivot.columns)
        ax.set_yticklabels(df_pivot.index)
        ax.set_xlabel(f2)
        ax.set_ylabel(f1)
        ax.set_title(f"Mean compression time: {f1} vs {f2}")
        
        # Add text annotations
        for i in range(len(df_pivot.index)):
            for j in range(len(df_pivot.columns)):
                value = df_pivot.values[i, j]
                if not np.isnan(value):
                    text = ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="white", fontsize=8)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Mean time [s]")
        plt.tight_layout()
        plt.show()


def extract_fractal_benchmarks_extended(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = {}

    for bench in data["benchmarks"]:
        fullname = bench["fullname"]

        match = re.search(r"\[(.*)\]", fullname)
        if not match:
            continue
        config_key = match.group(1)

        if config_key not in results:
            results[config_key] = {"config_raw": config_key}

        stats = bench.get("stats", {})
        extra = bench.get("extra_info", {})

        if "ffc_encode_speed" in fullname:
            results[config_key].update(
                {
                    "ffc_compression_ratio": extra.get("ffc_compression_ratio"),
                    "ffc_mean_s": stats.get("mean"),
                    "ffc_stddev_s": stats.get("stddev"),
                }
            )
        elif "jpeg_encode_speed" in fullname:
            results[config_key].update(
                {
                    "jpeg_compression_ratio": extra.get("jpeg_compression_ratio"),
                    "jpeg_mean_s": stats.get("mean"),
                    "jpeg_stddev_s": stats.get("stddev"),
                }
            )

    df_base = pd.DataFrame(list(results.values()))

    def parse_params_extended(config_str):
        params = {}
        parts = config_str.split("__")

        if len(parts) >= 2:
            params["case_name"] = parts[0]
            params["image_name"] = parts[1]

        for part in parts:
            if "=" in part:
                k, v = part.split("=")
                if v.lower() == "true":
                    v = True
                elif v.lower() == "false":
                    v = False
                else:
                    try:
                        v = float(v) if "." in v else int(v)
                    except ValueError:
                        pass
                params[k] = v
        return params

    params_df = df_base["config_raw"].apply(
        lambda x: pd.Series(parse_params_extended(x))
    )

    final_df = pd.concat([params_df, df_base.drop(columns=["config_raw"])], axis=1)

    cols = ["image_name", "case_name"] + [
        c for c in final_df.columns if c not in ["image_name", "case_name"]
    ]
    return final_df[cols]

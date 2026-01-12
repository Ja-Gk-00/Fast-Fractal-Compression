import pandas as pd
import json
import re
import seaborn as sns
import matplotlib.pyplot as plt


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


def visualize_benchmark_results(df, features, mode="both"):
    sns.set_theme(style="whitegrid")

    jpeg_ratio_val = None
    if "jpeg_compression_ratio" in df.columns:
        jpeg_ratio_val = df["jpeg_compression_ratio"].mean()

    for feature in features:
        if feature not in df.columns:
            print(f"Warning: Feature '{feature}' was not found in data.")
            continue

        plot_df = df.sort_values(by=feature)

        if mode in ["time", "both"]:
            plt.figure(figsize=(10, 5))
            sns.barplot(
                data=plot_df,
                x=feature,
                y="ffc_mean_s",
                color="skyblue",
                edgecolor="black",
            )
            plt.title(f"Impact of {feature} on mean compression time")
            plt.ylabel("Mean time [s]")
            plt.xlabel(feature)
            plt.tight_layout()
            plt.show()

        if mode in ["ratio", "both"]:
            plt.figure(figsize=(10, 5))
            sns.lineplot(
                data=plot_df,
                x=feature,
                y="ffc_compression_ratio",
                marker="o",
                color="coral",
                linewidth=2,
                label="FFC (Fractal)",
            )

            if jpeg_ratio_val is not None:
                plt.axhline(
                    y=jpeg_ratio_val,
                    color="red",
                    linestyle="--",
                    linewidth=1.5,
                    label=f"JPEG Baseline ({jpeg_ratio_val:.2f})",
                )

            if (
                plot_df["ffc_compression_ratio"].max()
                / (plot_df["ffc_compression_ratio"].min() + 1e-6)
                > 10
            ):
                plt.yscale("log")
                plt.ylabel("Compression ratio (log)")
            else:
                plt.ylabel("Compression ratio")

            plt.title(f"Impact of {feature} on compression ratio")
            plt.xlabel(feature)
            plt.legend()
            plt.tight_layout()
            plt.show()

import pandas as pd
import json
import re
import plotly.express as px


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

        fig_time = px.bar(
            df_agg,
            x=feature,
            y="ffc_mean_s",
            title=f"Impact of {feature.upper()} on mean compression time",
            labels={feature: f"Feature: {feature}", "ffc_mean_s": "Mean time [s]"},
            template="plotly_white",
        )
        fig_time.update_traces(
            marker_color="#4682B4",
            marker_line_color="rgb(8,48,107)",
            marker_line_width=1.5,
        )
        fig_time.update_layout(xaxis={"type": "category"})
        fig_time.show()

        fig_ratio = px.line(
            df_agg,
            x=feature,
            y="ffc_compression_ratio",
            markers=True,
            title=f"Impact of {feature.upper()} on compression ratio",
            labels={"ffc_compression_ratio": "Compression ratio"},
            template="plotly_white",
        )

        if "jpeg_compression_ratio" in df.columns:
            jpeg_avg = df["jpeg_compression_ratio"].mean()
            fig_ratio.add_hline(
                y=jpeg_avg,
                line_dash="dash",
                line_color="#FF4B4B",
                annotation_text=f"JPEG Avg ({jpeg_avg:.2f})",
                annotation_position="bottom right",
            )

        fig_ratio.update_xaxes(type="category")
        fig_ratio.update_traces(line_width=4, marker_size=12, line_color="#2E4053")
        fig_ratio.show()

    if len(features_list) == 2:
        f1, f2 = features_list

        df_heat = df.groupby([f1, f2], as_index=False).mean(numeric_only=True)
        df_pivot = df_heat.pivot(index=f1, columns=f2, values="ffc_mean_s")

        df_pivot = df_pivot.sort_index(axis=0).sort_index(axis=1)
        df_pivot.index = df_pivot.index.astype(str)
        df_pivot.columns = df_pivot.columns.astype(str)

        fig_heat = px.imshow(
            df_pivot,
            labels=dict(x=f2, y=f1, color="Mean time [s]"),
            x=df_pivot.columns,
            y=df_pivot.index,
            aspect="auto",
            color_continuous_scale="Viridis",
            title=f"Mean compression time: {f1} vs {f2}",
            template="plotly_white",
            text_auto=".3f",
        )
        fig_heat.update_layout(xaxis={"type": "category"}, yaxis={"type": "category"})
        fig_heat.show()


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

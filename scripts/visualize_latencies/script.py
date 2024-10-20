import json
from pathlib import Path

import matplotlib.pyplot as plt

# Load data from Json file
filepath = "/remote/vast0/tran/workspace/Open-Sora/save/20240924.191222/latencies.json"
with open(filepath, "r") as file:
    data = json.load(file)

resolutions = list(data)
times = list(data[resolutions[0]])

# Directory to save the figures
output_dir = Path("charts")
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Time change in order of resolution and time
plt.figure(figsize=(10, 6))
for res in resolutions:
    if data[res]:
        end2end_times = [data[res][t]["end2end"] for t in times if t in data[res]]
        plt.plot(times[: len(end2end_times)], end2end_times, marker="o", label=res)
plt.title("End-to-End Time Change by Resolution and Time")
plt.xlabel("Duration")
plt.ylabel("End-to-End Time (s)")
plt.legend()
plt.grid(True)
plt.savefig(output_dir / "time_change_by_resolution_and_time.png")
plt.close()

# 2. Time proportion (with 3 components vs end2end)
for res in resolutions:
    if data[res]:
        plt.figure(figsize=(10, 6))
        durations = [t for t in times if t in data[res]]
        end2end_times = [data[res][t]["end2end"] for t in durations]
        backbone_times = [data[res][t]["backbone"] for t in durations]
        text_encoder_times = [data[res][t]["text_encoder"] for t in durations]
        image_encoder_times = [data[res][t]["image_encoder"] for t in durations]

        plt.plot(durations, end2end_times, marker="o", label="End-to-End")
        plt.plot(durations, backbone_times, marker="x", label="Backbone")
        plt.plot(durations, text_encoder_times, marker="s", label="Text Encoder")
        plt.plot(durations, image_encoder_times, marker="d", label="Image Encoder")

        plt.title(f"Time Proportion for {res}")
        plt.xlabel("Duration")
        plt.ylabel("Time (s)")
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / f"time_proportion_{res}.png")
        plt.close()

# 3. Summary chart for displaying all data efficiently
plt.figure(figsize=(12, 8))
for res in resolutions:
    if data[res]:
        durations = [t for t in times if t in data[res]]
        end2end_times = [data[res][t]["end2end"] for t in durations]
        backbone_times = [data[res][t]["backbone"] for t in durations]
        text_encoder_times = [data[res][t]["text_encoder"] for t in durations]
        image_encoder_times = [data[res][t]["image_encoder"] for t in durations]

        plt.plot(durations, end2end_times, marker="o", label=f"{res} End-to-End")
        plt.plot(durations, backbone_times, marker="x", linestyle="--", label=f"{res} Backbone")
        plt.plot(durations, text_encoder_times, marker="s", linestyle=":", label=f"{res} Text Encoder")
        plt.plot(durations, image_encoder_times, marker="d", linestyle="-.", label=f"{res} Image Encoder")

plt.title("Summary of All Data")
plt.xlabel("Duration")
plt.ylabel("Time (s)")
plt.legend()
plt.grid(True)
plt.savefig(output_dir / "summary_of_all_data.png")
plt.close()

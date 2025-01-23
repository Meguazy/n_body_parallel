import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV file
data_file = "../data/performance_data.csv"
df = pd.read_csv(data_file)

# Plot 1: Elapsed Time
plt.figure(figsize=(8, 6))
plt.plot(df["process_number"], df["elapsed_time"], marker='o', label="Elapsed Time")
plt.title("Elapsed Time vs Process Number")
plt.xlabel("Process Number")
plt.ylabel("Elapsed Time (seconds)")
plt.grid(True)
plt.legend()
plt.savefig("../plots/elapsed_time.png")
plt.close()

# Plot 2: Speed Up
plt.figure(figsize=(8, 6))
plt.plot(df["process_number"], df["speed_up"], marker='o', color='green', label="Speed Up")
plt.title("Speed Up vs Process Number")
plt.xlabel("Process Number")
plt.ylabel("Speed Up")
plt.grid(True)
plt.legend()
plt.savefig("../plots/speed_up.png")
plt.close()

# Plot 3: Efficiency
plt.figure(figsize=(8, 6))
plt.plot(df["process_number"], df["efficiency"], marker='o', color='red', label="Efficiency")
plt.title("Efficiency vs Process Number")
plt.xlabel("Process Number")
plt.ylabel("Efficiency")
plt.grid(True)
plt.legend()
plt.savefig("../plots/efficiency.png")
plt.close()

# Plot 4: Combined Plot
plt.figure(figsize=(10, 8))
plt.plot(df["process_number"], df["speed_up"], marker='o', label="Speed Up", color='green')
plt.plot(df["process_number"], df["efficiency"], marker='o', label="Efficiency", color='red')
plt.title("Speed Up, and Efficiency vs Process Number")
plt.xlabel("Process Number")
plt.ylabel("Values")
plt.grid(True)
plt.legend()
plt.savefig("../plots/combined_plot.png")
plt.close()

print("Plots saved successfully in ../data folder.")

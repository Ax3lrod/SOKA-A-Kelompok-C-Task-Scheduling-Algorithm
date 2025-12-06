import pandas as pd

NUM_FILES = 10
SUMMARY_CSV = "summary_avg_start_time.csv"
summary_data = []

for i in range(1, NUM_FILES + 1):
    filename = f"shc_random_stratified_{i}.csv"
    try:
        df = pd.read_csv(filename)
        if "start_time" not in df.columns:
            print(f"File {filename} tidak punya kolom 'start_time', dilewati.")
            continue
        
        avg_start_time = df["start_time"].mean()
        summary_data.append({
            "Percobaan Ke-": i,
            "Average Start Time": avg_start_time
        })
        print(f"{filename}: Average Start Time = {avg_start_time:.4f}")
    except FileNotFoundError:
        print(f"File {filename} tidak ditemukan, dilewati.")

# Buat DataFrame summary
summary_df = pd.DataFrame(summary_data)

# Simpan ke CSV
summary_df.to_csv(SUMMARY_CSV, index=False)
print(f"\nSummary Average Start Time disimpan di '{SUMMARY_CSV}'")

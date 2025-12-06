import asyncio
import httpx
import time
from datetime import datetime
import csv
import pandas as pd
import sys
import os
from dotenv import load_dotenv
from collections import namedtuple
from cloudy_gsa_algorithm import cloudy_gsa_scheduler
from shc_algo import stochastic_hill_climb

# --- Konfigurasi Lingkungan ---

load_dotenv()

VM_SPECS = {
    'vm1': {'ip': os.getenv("VM1_IP"), 'cpu': 1, 'ram_gb': 1},
    'vm2': {'ip': os.getenv("VM2_IP"), 'cpu': 2, 'ram_gb': 2},
    'vm3': {'ip': os.getenv("VM3_IP"), 'cpu': 4, 'ram_gb': 4},
    'vm4': {'ip': os.getenv("VM4_IP"), 'cpu': 8, 'ram_gb': 4},
}

VM_PORT = 5000
DATASET_FILE = 'random_stratified.txt'
BASE_RESULTS_FILE = 'cgsa_random_stratified'
GSA_ITERATIONS = 1000
TOTAL_RUNS = 10  # Jumlah pengulangan test

VM = namedtuple('VM', ['name', 'ip', 'cpu_cores', 'ram_gb'])
Task = namedtuple('Task', ['id', 'name', 'index', 'cpu_load'])

# --- Fungsi Helper & Definisi Task ---

def get_task_load(index: int):
    cpu_load = (index * index * 10000)
    return cpu_load

def load_tasks(dataset_path: str) -> list[Task]:
    if not os.path.exists(dataset_path):
        print(f"Error: File dataset '{dataset_path}' tidak ditemukan.", file=sys.stderr)
        sys.exit(1)
        
    tasks = []
    with open(dataset_path, 'r') as f:
        for i, line in enumerate(f):
            try:
                index = int(line.strip())
                if not 1 <= index <= 10:
                    continue
                
                cpu_load = get_task_load(index)
                task_name = f"task-{index}-{i}"
                tasks.append(Task(
                    id=i,
                    name=task_name,
                    index=index,
                    cpu_load=cpu_load,
                ))
            except ValueError:
                pass
    
    return tasks

# --- Eksekutor Tugas Asinkron ---

async def execute_task_on_vm(task: Task, vm: VM, client: httpx.AsyncClient, 
                            vm_semaphore: asyncio.Semaphore, results_list: list):
    url = f"http://{vm.ip}:{VM_PORT}/task/{task.index}"
    task_start_time = None
    task_finish_time = None
    task_exec_time = -1.0
    task_wait_time = -1.0
    
    wait_start_mono = time.monotonic()
    
    try:
        async with vm_semaphore:
            task_wait_time = time.monotonic() - wait_start_mono
            
            # Catat waktu mulai
            task_start_mono = time.monotonic()
            task_start_time = datetime.now()
            
            # Kirim request GET
            response = await client.get(url, timeout=300.0)
            response.raise_for_status()
            
            # Catat waktu selesai
            task_finish_time = datetime.now()
            task_exec_time = time.monotonic() - task_start_mono
            
    except Exception as e:
        print(f"Error pada {task.name} di {vm.name}: {e}", file=sys.stderr)
        
    finally:
        if task_start_time is None:
            task_start_time = datetime.now()
        if task_finish_time is None:
            task_finish_time = datetime.now()
            
        results_list.append({
            "index": task.id,
            "task_name": task.name,
            "vm_assigned": vm.name,
            "start_time": task_start_time,
            "exec_time": task_exec_time,
            "finish_time": task_finish_time,
            "wait_time": task_wait_time
        })

# --- Fungsi Paska-Proses & Metrik ---

def write_results_to_csv(results_list: list, run_id: int):
    """Menyimpan hasil eksekusi ke file CSV dengan ID run tertentu."""
    if not results_list:
        return

    filename = f"{BASE_RESULTS_FILE}_{run_id}.csv"
    results_list.sort(key=lambda x: x['index'])

    headers = ["index", "task_name", "vm_assigned", "start_time", "exec_time", "finish_time", "wait_time"]
    
    formatted_results = []
    min_start = min(item['start_time'] for item in results_list)
    for r in results_list:
        new_r = r.copy()
        new_r['start_time'] = (r['start_time'] - min_start).total_seconds()
        new_r['finish_time'] = (r['finish_time'] - min_start).total_seconds()
        formatted_results.append(new_r)

    try:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(formatted_results)
    except IOError as e:
        print(f"Error menulis ke CSV {filename}: {e}", file=sys.stderr)

def calculate_metrics(results_list: list, vms: list[VM], total_schedule_time: float) -> dict:
    """
    Menghitung metrik dan mengembalikan dictionary berisi nilai-nilai metrik.
    Tidak lagi melakukan print (print dilakukan di main loop).
    """
    try:
        df = pd.DataFrame(results_list)
    except pd.errors.EmptyDataError:
        return {}

    # Konversi kolom waktu
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['finish_time'] = pd.to_datetime(df['finish_time'])
    
    success_df = df[df['exec_time'] > 0].copy()
    
    if success_df.empty:
        return {}

    num_tasks = len(success_df)
    
    # Hitung metrik
    total_cpu_time = success_df['exec_time'].sum()
    total_wait_time = success_df['wait_time'].sum()
    
    avg_exec_time = success_df['exec_time'].mean()
    avg_wait_time = success_df['wait_time'].mean()
    
    # Makespan & Throughput
    makespan = total_schedule_time 
    throughput = num_tasks / makespan if makespan > 0 else 0
    
    # Imbalance Degree
    vm_exec_times = success_df.groupby('vm_assigned')['exec_time'].sum()
    max_load = vm_exec_times.max()
    min_load = vm_exec_times.min()
    avg_load = vm_exec_times.mean()
    imbalance_degree = (max_load - min_load) / avg_load if avg_load > 0 else 0
    
    # Resource Utilization
    total_cores = sum(vm.cpu_cores for vm in vms)
    total_available_cpu_time = makespan * total_cores
    resource_utilization = total_cpu_time / total_available_cpu_time if total_available_cpu_time > 0 else 0

    return {
        "Makespan": makespan,
        "Throughput": throughput,
        "Total CPU Time": total_cpu_time,
        "Total Wait Time": total_wait_time,
        "Avg Execution Time": avg_exec_time,
        "Avg Wait Time": avg_wait_time,
        "Imbalance Degree": imbalance_degree,
        "Resource Utilization": resource_utilization
    }

# --- Fungsi Eksekusi Satu Kali ---

async def run_single_test(run_id: int, tasks: list[Task], vms: list[VM]) -> dict:
    print(f"\n--- Memulai Test Run ke-{run_id} ---")
    
    tasks_dict = {task.id: task for task in tasks}
    vms_dict = {vm.name: vm for vm in vms}

    # 2. Jalankan Algoritma Penjadwalan (CloudyGSA)
    best_assignment = cloudy_gsa_scheduler(tasks, vms, iterations=GSA_ITERATIONS)
    
    # 2. Siapkan Eksekusi
    results_list = []
    vm_semaphores = {vm.name: asyncio.Semaphore(vm.cpu_cores) for vm in vms}
    
    async with httpx.AsyncClient() as client:
        all_task_coroutines = []
        for task_id, vm_name in best_assignment.items():
            task = tasks_dict[task_id]
            vm = vms_dict[vm_name]
            sem = vm_semaphores[vm_name]
            all_task_coroutines.append(
                execute_task_on_vm(task, vm, client, sem, results_list)
            )
            
        print(f"Run {run_id}: Mengeksekusi {len(all_task_coroutines)} tugas...")
        
        start_time = time.monotonic()
        await asyncio.gather(*all_task_coroutines)
        end_time = time.monotonic()
        
        total_time = end_time - start_time
        print(f"Run {run_id} selesai dalam {total_time:.4f} detik.")

    # 3. Simpan & Hitung
    write_results_to_csv(results_list, run_id)
    metrics = calculate_metrics(results_list, vms, total_time)
    
    # Tampilkan metrik singkat untuk run ini
    if metrics:
        print(f"Run {run_id} Metrics -> Makespan: {metrics['Makespan']:.2f}s, Utilization: {metrics['Resource Utilization']:.2%}")
    
    return metrics

# --- Fungsi Main Utama ---

async def main():
    print(f"Mempersiapkan {TOTAL_RUNS} kali pengujian...")
    
    # Load Data
    vms = [VM(name, spec['ip'], spec['cpu'], spec['ram_gb']) for name, spec in VM_SPECS.items()]
    tasks = load_tasks(DATASET_FILE)
    
    if not tasks:
        print("Tidak ada tugas. Keluar.")
        return

    all_run_metrics = []

    # Loop Eksekusi
    for i in range(1, TOTAL_RUNS + 1):
        metrics = await run_single_test(i, tasks, vms)
        if metrics:
            all_run_metrics.append(metrics)
        
        # Optional: Jeda sedikit antar run agar port benar-benar bersih
        if i < TOTAL_RUNS:
            await asyncio.sleep(2)

    # Hitung Rata-rata
    print("\n" + "="*40)
    print(f"HASIL AKHIR RATA-RATA DARI {TOTAL_RUNS} RUN")
    print("="*40)

    if all_run_metrics:
        df_metrics = pd.DataFrame(all_run_metrics)
        avg_metrics = df_metrics.mean()
        
        # Tampilkan Tabel
        print(df_metrics)
        print("-" * 40)
        
        print("\nRATA-RATA METRIK:")
        for key, value in avg_metrics.items():
            if "Utilization" in key:
                print(f"{key:<25}: {value:.4%}")
            else:
                print(f"{key:<25}: {value:.4f}")
                
        # Simpan Summary ke CSV
        df_metrics.loc['Average'] = avg_metrics
        df_metrics.to_csv('summary_metrics_10_runs_cgsa_random_stratified.csv', index=True)
        print(f"\nRingkasan metrik disimpan ke 'summary_metrics_10_runs.csv'")
        
    else:
        print("Gagal mendapatkan metrik dari pengujian.")

if __name__ == "__main__":
    asyncio.run(main())
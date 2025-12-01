import random
import copy
from collections import namedtuple

VM = namedtuple('VM', ['name', 'ip', 'cpu_cores', 'ram_gb'])
Task = namedtuple('Task', ['id', 'name', 'index', 'cpu_load', 'ram_mb'])

# --- Helper Function ---
def get_initial_loads(solution: dict, tasks_dict: dict, vms_dict: dict) -> dict:
    """Menghitung beban awal semua VM berdasarkan solusi saat ini."""
    vm_loads = {name: 0.0 for name in vms_dict}
    for task_id, vm_name in solution.items():
        task = tasks_dict[task_id]
        vm = vms_dict[vm_name]
        vm_loads[vm_name] += (task.cpu_load / vm.cpu_cores)
    return vm_loads

# --- Algoritma Stochastic Hill Climbing dengan Restart ---

def stochastic_hill_climb(tasks: list[Task], vms: list[VM], iterations: int = 1000, restarts: int = 5) -> dict:
    """
    SHC dengan Random Restart.
    Restarts berguna untuk mencegah algoritma terjebak di 'Local Optima'.
    """
    print(f"Memulai SHC ({iterations} iterasi per restart, {restarts} restarts)...")
    
    vms_dict = {vm.name: vm for vm in vms}
    tasks_dict = {task.id: task for task in tasks}
    vm_names = list(vms_dict.keys())

    # Variable untuk menyimpan solusi terbaik absolut dari semua restart
    global_best_solution = None
    global_best_makespan = float('inf')

    for r in range(restarts):
        # 1. Inisialisasi Solusi Acak (Setiap restart mulai dari nol/random baru)
        current_solution = {task.id: random.choice(vm_names) for task in tasks}
        
        # Hitung beban awal (cukup sekali di awal loop)
        current_vm_loads = get_initial_loads(current_solution, tasks_dict, vms_dict)
        current_makespan = max(current_vm_loads.values()) # Cost Function

        # Simpan state lokal terbaik
        local_best_solution = current_solution.copy()
        local_best_makespan = current_makespan

        # 2. Iterasi Hill Climbing
        for i in range(iterations):
            # --- Buat Tetangga & Evaluasi Cepat (Delta Evaluation) ---
            # Alih-alih copy dict penuh dan hitung ulang semua, kita simulasi saja.
            
            # A. Pilih tugas dan target VM baru
            task_id_to_move = random.choice(list(tasks_dict.keys()))
            current_vm_name = current_solution[task_id_to_move]
            new_vm_name = random.choice([v for v in vm_names if v != current_vm_name])
            
            task = tasks_dict[task_id_to_move]
            
            # B. Hitung perubahan beban jika tugas dipindah
            # Load saat ini
            load_vm_old = current_vm_loads[current_vm_name]
            load_vm_new = current_vm_loads[new_vm_name]
            
            # Hitung load tugas
            cost_on_old = task.cpu_load / vms_dict[current_vm_name].cpu_cores
            cost_on_new = task.cpu_load / vms_dict[new_vm_name].cpu_cores
            
            # Prediksi load baru
            pred_load_vm_old = load_vm_old - cost_on_old
            pred_load_vm_new = load_vm_new + cost_on_new
            
            # Cek apakah pemindahan ini mengurangi makespan global?
            # Makespan ditentukan oleh VM dengan load TERTINGGI.
            # Kita perlu cek max load dari SELURUH VM jika perubahan ini terjadi.
            
            # (Untuk akurasi 100%, kita copy load dict sementara)
            temp_loads = current_vm_loads.copy()
            temp_loads[current_vm_name] = pred_load_vm_old
            temp_loads[new_vm_name] = pred_load_vm_new
            new_makespan = max(temp_loads.values())
            
            # --- Keputusan (Greedy) ---
            if new_makespan < local_best_makespan:
                # Terima solusi baru
                current_solution[task_id_to_move] = new_vm_name
                current_vm_loads = temp_loads # Update beban VM
                local_best_makespan = new_makespan
                local_best_solution = current_solution.copy() # Simpan backup

        print(f"  [Restart {r+1}] Makespan Terbaik: {local_best_makespan:.2f}")

        # Bandingkan hasil restart ini dengan global best
        if local_best_makespan < global_best_makespan:
            global_best_makespan = local_best_makespan
            global_best_solution = local_best_solution.copy()

    print(f"SHC Selesai. Global Makespan Terbaik: {global_best_makespan:.2f}")
    return global_best_solution
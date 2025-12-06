from collections import namedtuple

# --- Definisi Tipe Data (Sama persis) ---
VM = namedtuple('VM', ['name', 'ip', 'cpu_cores', 'ram_gb'])
Task = namedtuple('Task', ['id', 'name', 'index', 'cpu_load', 'ram_mb'])

# --- Helper Function (Untuk menghitung skor akhir saja) ---
def get_final_makespan(solution: dict, tasks_dict: dict, vms_dict: dict) -> float:
    """Menghitung total beban (makespan) dari hasil pembagian tugas."""
    vm_loads = {name: 0.0 for name in vms_dict}
    for task_id, vm_name in solution.items():
        task = tasks_dict[task_id]
        vm = vms_dict[vm_name]
        # Rumus beban: load / cores
        vm_loads[vm_name] += (task.cpu_load / vm.cpu_cores)
    
    # Makespan adalah VM dengan beban terberat
    return max(vm_loads.values())

# --- Algoritma Round Robin ---

def round_robin_algorithm(tasks: list[Task], vms: list[VM]) -> dict:
    """
    Algoritma Round Robin (RR).
    Membagi tugas secara berurutan (indeks % jumlah_vm).
    Deterministik, tidak butuh iterasi atau restart.
    """
    print(f"Memulai Algoritma Round Robin untuk {len(tasks)} tugas...")
    
    # Setup mapping
    vms_dict = {vm.name: vm for vm in vms}
    tasks_dict = {task.id: task for task in tasks}
    vm_names = list(vms_dict.keys())
    num_vms = len(vm_names)

    # Dictionary untuk menyimpan hasil assignment
    final_solution = {}

    # --- Logika Inti Round Robin ---
    for i, task in enumerate(tasks):
        # Gunakan operator Modulo (%) untuk memutar index
        # 0, 1, 2, 3 -> kembali ke 0, 1, ...
        target_vm_index = i % num_vms
        target_vm_name = vm_names[target_vm_index]
        
        # Tetapkan tugas ke VM tersebut
        final_solution[task.id] = target_vm_name

    # --- Evaluasi Hasil (Hanya untuk Info) ---
    final_makespan = get_final_makespan(final_solution, tasks_dict, vms_dict)
    
    print(f"Round Robin Selesai.")
    print(f"Makespan Akhir: {final_makespan:.2f}")
    
    return final_solution
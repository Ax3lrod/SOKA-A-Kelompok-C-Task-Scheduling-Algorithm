import numpy as np
import math
from collections import namedtuple
from typing import Tuple

# Definisi Tipe (bisa disesuaikan jika perlu)
VM = namedtuple('VM', ['name', 'ip', 'cpu_cores', 'ram_gb'])
Task = namedtuple('Task', ['id', 'name', 'index', 'cpu_load'])

# Konstanta Algoritma
POP_SIZE = 40
G0 = 100.0
ALPHA = 20.0
EPS = 1e-12
INERTIA = 0.7
VELOCITY_CLAMP = 2.0

# Bobot untuk Fungsi Fitness
# Bobot yang lebih tinggi pada DOI akan memaksa algoritma untuk memprioritaskan load balancing
W_MAKESPAN = 1.0  # Bobot untuk Makespan
W_DOI = 0.8       # Bobot untuk Degree of Imbalance (DOI)

# Fungsi Fitness (Fungsi Biaya) yang Dioptimalkan

def _evaluate_fitness(solution: np.ndarray, tasks_dict: dict, vms_dict: dict, vm_map: list) -> float:
    """
    [OPTIMASI #1] Fungsi Biaya (Cost Function) yang Dioptimalkan.
    Fokus pada dua tujuan utama:
    1. Makespan (waktu selesai maksimum).
    2. Degree of Imbalance (DOI) untuk memaksimalkan utilisasi.
    """
    n_vms = len(vm_map)
    
    # Menghitung total waktu eksekusi (beban kerja) pada setiap VM
    vm_total_load_time = np.zeros(n_vms)
    
    for task_idx, vm_assigned_idx in enumerate(solution):
        task = tasks_dict[task_idx]
        vm_name = vm_map[int(vm_assigned_idx)]
        vm = vms_dict[vm_name]
        
        exec_time = task.cpu_load / vm.cpu_cores
        vm_total_load_time[int(vm_assigned_idx)] += exec_time
        
    # 1. Menghitung Makespan
    makespan = np.max(vm_total_load_time)
    
    # 2. Menghitung Degree of Imbalance (DOI)
    max_load = np.max(vm_total_load_time)
    min_load = np.min(vm_total_load_time)
    avg_load = np.mean(vm_total_load_time)
    
    imbalance_degree = (max_load - min_load) / (avg_load + EPS)
    
    # Skor fitness gabungan dengan bobot
    fitness_score = (W_MAKESPAN * makespan) + (W_DOI * imbalance_degree * makespan)
    # DOI dikalikan makespan agar skalanya sebanding
    
    return fitness_score

# Fungsi Helper

def _compute_mass(fitness: np.ndarray):
    """Menghitung massa setiap partikel berdasarkan nilai fitness-nya."""
    best, worst = np.min(fitness), np.max(fitness)
    if (worst - best) < EPS:
        return np.ones(POP_SIZE) / POP_SIZE
    raw_mass = (fitness - best) / (worst - best) # Inversi karena kita ingin meminimalkan
    raw_mass = 1 - raw_mass
    total_mass = np.sum(raw_mass)
    return raw_mass / (total_mass + EPS)

def _map_to_solution(position: np.ndarray, n_vms: int) -> np.ndarray:
    """Membulatkan posisi kontinu ke solusi diskrit (indeks VM)."""
    return np.clip(np.round(position), 0, n_vms - 1).astype(int)
# Mekanisme Local Search yang Lebih Cerdas

def _local_search_improvement(solution: np.ndarray, tasks_dict: dict, vms_dict: dict, vm_map: list) -> Tuple[np.ndarray, float]:
    """
    Mencoba memindahkan satu tugas dari VM yang paling sibuk
    ke VM yang paling tidak sibuk untuk mengurangi imbalance.
    """
    n_vms = len(vm_map)
    current_fitness = _evaluate_fitness(solution, tasks_dict, vms_dict, vm_map)
    
    # Hitung beban per VM
    vm_loads = np.zeros(n_vms)
    for task_idx, vm_idx in enumerate(solution):
        task = tasks_dict[task_idx]
        vm = vms_dict[vm_map[int(vm_idx)]]
        vm_loads[int(vm_idx)] += task.cpu_load / vm.cpu_cores

    # Temukan VM paling sibuk dan paling tidak sibuk
    most_loaded_vm_idx = np.argmax(vm_loads)
    least_loaded_vm_idx = np.argmin(vm_loads)

    if most_loaded_vm_idx == least_loaded_vm_idx:
        return solution, current_fitness # Sudah seimbang

    # Cari tugas di VM paling sibuk yang bisa dipindahkan
    tasks_on_most_loaded_vm = [i for i, vm_idx in enumerate(solution) if vm_idx == most_loaded_vm_idx]
    
    if not tasks_on_most_loaded_vm:
        return solution, current_fitness

    # Coba pindahkan setiap tugas dan lihat apakah hasilnya membaik
    best_solution = solution.copy()
    best_fitness = current_fitness

    for task_to_move in tasks_on_most_loaded_vm:
        new_solution = solution.copy()
        new_solution[task_to_move] = least_loaded_vm_idx
        new_fitness = _evaluate_fitness(new_solution, tasks_dict, vms_dict, vm_map)
        
        if new_fitness < best_fitness:
            best_fitness = new_fitness
            best_solution = new_solution
    
    return best_solution, best_fitness


# Algoritma Utama CloudyGSA

def cloudy_gsa_scheduler(tasks: list[Task], vms: list[VM], iterations: int = 1000) -> dict:
    """
    Menjalankan algoritma Cloudy Gravitational Search Algorithm (Cloudy-GSA)
    dengan Elitisme dan Local Search yang Dioptimalkan.
    """
    print(f"Memulai Cloudy-GSA (Optimized Fitness & Local Search, {iterations} iterasi)...")

    n_tasks = len(tasks)
    n_vms = len(vms)

    vms_dict = {vm.name: vm for vm in vms}
    tasks_dict = {task.id: task for task in tasks}
    vm_map = [vm.name for vm in vms]

    # Inisialisasi posisi dan kecepatan
    pos = np.random.rand(POP_SIZE, n_tasks) * (n_vms - 1 + EPS)
    vel = np.zeros((POP_SIZE, n_tasks))

    # Evaluasi fitness awal
    solutions = [_map_to_solution(p, n_vms) for p in pos]
    fitness = np.array([_evaluate_fitness(s, tasks_dict, vms_dict, vm_map) for s in solutions])

    # Simpan solusi terbaik global (gbest)
    gbest_idx = np.argmin(fitness)
    gbest_val = fitness[gbest_idx]
    gbest_pos = pos[gbest_idx].copy()
    
    print(f"Estimasi Fitness Awal (Acak): {gbest_val:.2f}")

    # Iterasi utama
    for t in range(iterations):
        gbest_pos_before_update = gbest_pos.copy()
        gbest_val_before_update = gbest_val

        mass = _compute_mass(fitness)
        G = G0 * math.exp(-ALPHA * (t / iterations))

        force = np.zeros((POP_SIZE, n_tasks))
        for i in range(POP_SIZE):
            for j in range(POP_SIZE):
                if i == j: continue
                if mass[j] > mass[i]: # Hanya partikel yg lebih baik yg menarik
                    diff = pos[j] - pos[i]
                    dist = np.linalg.norm(diff) + EPS
                    f_magnitude = G * (mass[j] / (dist**2))
                    force[i] += np.random.rand(n_tasks) * f_magnitude * diff

        accel = force / (mass[:, np.newaxis] + EPS)
        
        vel = INERTIA * vel + accel
        vel = np.clip(vel, -VELOCITY_CLAMP, VELOCITY_CLAMP)
        pos += vel
        pos = np.clip(pos, 0, n_vms - 1)

        # Evaluasi ulang fitness dan terapkan Local Search
        for p in range(POP_SIZE):
            solution = _map_to_solution(pos[p], n_vms)
            
            # Terapkan Local Search Cerdas
            improved_solution, improved_fitness = _local_search_improvement(solution, tasks_dict, vms_dict, vm_map)
            
            if improved_fitness < fitness[p]:
                fitness[p] = improved_fitness
                # Update posisi kontinu agar konsisten
                for task_id in range(n_tasks):
                    pos[p, task_id] = improved_solution[task_id]
            else:
                fitness[p] = _evaluate_fitness(solution, tasks_dict, vms_dict, vm_map)
        
        # Elitisme: Ganti partikel terburuk dengan gbest dari iterasi sebelumnya
        worst_idx = np.argmax(fitness)
        if fitness[worst_idx] > gbest_val_before_update:
            pos[worst_idx] = gbest_pos_before_update
            vel[worst_idx].fill(0) # Reset kecepatan
            fitness[worst_idx] = gbest_val_before_update

        # Update gbest global
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < gbest_val:
            gbest_val = fitness[current_best_idx]
            gbest_pos = pos[current_best_idx].copy()
            if t % 50 == 0:
                print(f"Iterasi {t}: Fitness Baru Terbaik: {gbest_val:.2f}")

    print(f"Cloudy-GSA Selesai. Fitness Terbaik: {gbest_val:.2f}")

    # Bentuk hasil akhir
    best_solution_discrete = _map_to_solution(gbest_pos, n_vms)
    best_assignment = {
        task.id: vm_map[best_solution_discrete[i]]
        for i, task in enumerate(tasks)
    }
        
    return best_assignment
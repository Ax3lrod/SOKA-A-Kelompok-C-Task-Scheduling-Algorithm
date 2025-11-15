
import numpy as np
import math
from collections import namedtuple

# Definisi Tipe
VM = namedtuple('VM', ['name', 'ip', 'cpu_cores', 'ram_gb'])
Task = namedtuple('Task', ['id', 'name', 'index', 'cpu_load'])

# --- Konstanta Algoritma ---
POP_SIZE = 50  # Sedikit menambah populasi untuk eksplorasi lebih baik
G0 = 100.0
ALPHA = 20.0
EPS = 1e-12
INERTIA = 0.7
VELOCITY_CLAMP = 2.0
MUTATION_RATE = 0.02 # [OPTIMASI #3] Probabilitas mutasi per partikel
LOCAL_SEARCH_IMPROVEMENT_RATE = 0.1 # Hanya partikel terbaik yang dioptimalkan local search

# --- [OPTIMASI #1] Bobot untuk Fitness Function Baru ---
W_MAKESPAN = 1.0
W_STD_DEV = 1.5 # Bobot lebih tinggi pada standar deviasi untuk memprioritaskan keseimbangan

# --- Fungsi Fitness (Fungsi Biaya) yang Dioptimalkan ---

def _get_vm_loads(solution: np.ndarray, tasks_dict: dict, vms_dict: dict, vm_map: list) -> np.ndarray:
    """Helper untuk menghitung total beban waktu eksekusi pada setiap VM."""
    n_vms = len(vm_map)
    vm_total_load_time = np.zeros(n_vms)
    
    for task_idx, vm_assigned_idx in enumerate(solution):
        task = tasks_dict[task_idx]
        vm_name = vm_map[int(vm_assigned_idx)]
        vm = vms_dict[vm_name]
        
        exec_time = task.cpu_load / vm.cpu_cores
        vm_total_load_time[int(vm_assigned_idx)] += exec_time
    
    return vm_total_load_time

def _evaluate_fitness(solution: np.ndarray, tasks_dict: dict, vms_dict: dict, vm_map: list) -> float:
    """
    [OPTIMASI #1] Fitness Function Baru dengan Standar Deviasi.
    Tujuan: Minimalkan Makespan DAN sebaran beban kerja antar VM.
    """
    vm_loads = _get_vm_loads(solution, tasks_dict, vms_dict, vm_map)
    
    makespan = np.max(vm_loads)
    load_std_dev = np.std(vm_loads) # Standar deviasi dari beban kerja
    
    # Skor fitness gabungan dengan bobot
    fitness_score = (W_MAKESPAN * makespan) + (W_STD_DEV * load_std_dev)
    
    return fitness_score

# --- Fungsi Helper ---

def _compute_mass(fitness: np.ndarray):
    """Menghitung massa partikel berdasarkan fitness."""
    best, worst = np.min(fitness), np.max(fitness)
    if (worst - best) < EPS:
        return np.ones_like(fitness) / len(fitness)
    
    raw_mass = (worst - fitness) / (worst - best)
    total_mass = np.sum(raw_mass)
    return raw_mass / (total_mass + EPS)

def _map_to_solution(position: np.ndarray, n_vms: int) -> np.ndarray:
    """Membulatkan posisi kontinu ke solusi diskrit."""
    return np.clip(np.round(position), 0, n_vms - 1).astype(int)

# --- [OPTIMASI #2] Mekanisme Local Search yang Lebih Kuat ---

def _local_search_swap(solution: np.ndarray, tasks_dict: dict, vms_dict: dict, vm_map: list) -> np.ndarray:
    """
    Mencoba menukar satu tugas dari VM paling sibuk dengan satu tugas
    dari VM paling tidak sibuk untuk menyeimbangkan beban.
    """
    vm_loads = _get_vm_loads(solution, tasks_dict, vms_dict, vm_map)
    
    most_loaded_vm_idx = np.argmax(vm_loads)
    least_loaded_vm_idx = np.argmin(vm_loads)

    if most_loaded_vm_idx == least_loaded_vm_idx:
        return solution

    tasks_on_most_loaded = [i for i, vm_idx in enumerate(solution) if vm_idx == most_loaded_vm_idx]
    tasks_on_least_loaded = [i for i, vm_idx in enumerate(solution) if vm_idx == least_loaded_vm_idx]

    if not tasks_on_most_loaded or not tasks_on_least_loaded:
        return solution

    best_solution = solution.copy()
    current_fitness = _evaluate_fitness(solution, tasks_dict, vms_dict, vm_map)

    # Cari pertukaran terbaik
    for task_idx1 in tasks_on_most_loaded:
        for task_idx2 in tasks_on_least_loaded:
            new_solution = solution.copy()
            # Lakukan swap
            new_solution[task_idx1], new_solution[task_idx2] = new_solution[task_idx2], new_solution[task_idx1]
            
            new_fitness = _evaluate_fitness(new_solution, tasks_dict, vms_dict, vm_map)
            
            if new_fitness < current_fitness:
                current_fitness = new_fitness
                best_solution = new_solution
    
    return best_solution


# --- Algoritma Utama CloudyGSA ---

def cloudy_gsa_scheduler(tasks: list[Task], vms: list[VM], iterations: int = 1000) -> dict:
    """
    Menjalankan algoritma Cloudy-GSA dengan Fitness Function berbasis
    Standar Deviasi, Local Search Swap, dan Mutasi.
    """
    print(f"Memulai Cloudy-GSA (V2 - StdDev Fitness, Swap, Mutation, {iterations} iterasi)...")

    n_tasks = len(tasks)
    n_vms = len(vms)

    vms_dict = {vm.name: vm for vm in vms}
    tasks_dict = {task.id: task for task in tasks}
    vm_map = [vm.name for vm in vms]

    pos = np.random.rand(POP_SIZE, n_tasks) * (n_vms - 1 + EPS)
    vel = np.zeros((POP_SIZE, n_tasks))

    fitness = np.array([
        _evaluate_fitness(_map_to_solution(p, n_vms), tasks_dict, vms_dict, vm_map)
        for p in pos
    ])

    gbest_idx = np.argmin(fitness)
    gbest_val = fitness[gbest_idx]
    gbest_pos = pos[gbest_idx].copy()
    
    print(f"Estimasi Fitness Awal (Acak): {gbest_val:.2f}")

    for t in range(iterations):
        gbest_pos_before_update = gbest_pos.copy()
        gbest_val_before_update = gbest_val

        mass = _compute_mass(fitness)
        G = G0 * math.exp(-ALPHA * (t / iterations))

        force = np.zeros((POP_SIZE, n_tasks))
        for i in range(POP_SIZE):
            for j in range(POP_SIZE):
                if i != j:
                    diff = pos[j] - pos[i]
                    dist = np.linalg.norm(diff) + EPS
                    f = G * (mass[i] * mass[j] / (dist**2)) * diff
                    force[i] += np.random.rand(n_tasks) * f
        
        accel = force / (mass[:, np.newaxis] + EPS)
        vel = INERTIA * vel + accel
        vel = np.clip(vel, -VELOCITY_CLAMP, VELOCITY_CLAMP)
        pos += vel
        pos = np.clip(pos, 0, n_vms - 1)

        # Evaluasi ulang, terapkan Local Search dan Mutasi
        sorted_indices = np.argsort(fitness)
        for i, p_idx in enumerate(sorted_indices):
            # [OPTIMASI #2] Terapkan local search hanya pada sebagian kecil partikel terbaik
            if i < int(POP_SIZE * LOCAL_SEARCH_IMPROVEMENT_RATE):
                current_solution = _map_to_solution(pos[p_idx], n_vms)
                improved_solution = _local_search_swap(current_solution, tasks_dict, vms_dict, vm_map)
                pos[p_idx] = improved_solution.astype(float) # Update posisi

            # [OPTIMASI #3] Terapkan mutasi pada sebagian kecil partikel (selain yg terbaik)
            if i > 0 and np.random.rand() < MUTATION_RATE:
                task_to_mutate = np.random.randint(n_tasks)
                new_vm = np.random.randint(n_vms)
                pos[p_idx, task_to_mutate] = new_vm
            
            fitness[p_idx] = _evaluate_fitness(_map_to_solution(pos[p_idx], n_vms), tasks_dict, vms_dict, vm_map)
        
        # Elitisme
        worst_idx = np.argmax(fitness)
        if fitness[worst_idx] > gbest_val_before_update:
            pos[worst_idx] = gbest_pos_before_update
            vel[worst_idx].fill(0)
            fitness[worst_idx] = gbest_val_before_update

        # Update gbest global
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < gbest_val:
            gbest_val = fitness[current_best_idx]
            gbest_pos = pos[current_best_idx].copy()
            if t > 0 and t % 100 == 0:
                print(f"Iterasi {t}: Fitness Baru Terbaik: {gbest_val:.2f}")

    print(f"Cloudy-GSA Selesai. Fitness Terbaik: {gbest_val:.2f}")

    best_solution_discrete = _map_to_solution(gbest_pos, n_vms)
    best_assignment = {
        task.id: vm_map[best_solution_discrete[i]]
        for i, task in enumerate(tasks)
    }
        
    return best_assignment

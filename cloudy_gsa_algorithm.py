
import numpy as np
import math
from collections import namedtuple

# Definisi Tipe
VM = namedtuple('VM', ['name', 'ip', 'cpu_cores', 'ram_gb'])
Task = namedtuple('Task', ['id', 'name', 'index', 'cpu_load'])

# --- Konstanta Algoritma ---
POP_SIZE = 60 # Meningkatkan populasi untuk pencarian yang lebih komprehensif
G0 = 100.0
ALPHA = 20.0
EPS = 1e-12
MUTATION_RATE = 0.01
LOCAL_SEARCH_CANDIDATES = 7

# [OPTIMASI #2] Konstanta untuk Inersia Adaptif
INERTIA_MAX = 0.9
INERTIA_MIN = 0.4

# [OPTIMASI #1] Bobot untuk Fitness Function Terpadu
W_MAKESPAN = 1.0   # Fokus utama pada kecepatan
W_STD_DEV = 1.7    # Penalti SANGAT TINGGI untuk ketidakseimbangan
W_UTILIZATION = 1.2 # Penalti TINGGI untuk utilisasi yang rendah

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
    [OPTIMASI #1] Fitness Function Terpadu: Makespan dengan Penalti untuk
    Imbalance dan Utilisasi Rendah.
    """
    vm_loads = _get_vm_loads(solution, tasks_dict, vms_dict, vm_map)
    n_vms = len(vm_map)
    
    # Komponen Utama: Makespan
    makespan = np.max(vm_loads)
    
    # Penalti 1: Ketidakseimbangan (Imbalance)
    load_std_dev = np.std(vm_loads)
    imbalance_penalty = W_STD_DEV * load_std_dev
    
    # Penalti 2: Utilisasi Rendah (Low Utilization)
    total_cpu_time = np.sum(vm_loads)
    total_available_time = n_vms * makespan
    
    resource_utilization = total_cpu_time / (total_available_time + EPS)
    
    # Penalti adalah kebalikan dari utilisasi, diskalakan dengan makespan
    utilization_penalty = W_UTILIZATION * (1.0 - resource_utilization) * makespan
    
    # Fitness Total
    fitness_score = (W_MAKESPAN * makespan) + imbalance_penalty + utilization_penalty
    
    return fitness_score

# --- Fungsi Helper ---

def _compute_mass(fitness: np.ndarray):
    best, worst = np.min(fitness), np.max(fitness)
    if (worst - best) < EPS: return np.ones_like(fitness) / len(fitness)
    raw_mass = (worst - fitness) / (worst - best); return raw_mass / (np.sum(raw_mass) + EPS)

def _map_to_solution(position: np.ndarray, n_vms: int):
    return np.clip(np.round(position), 0, n_vms - 1).astype(int)

# --- Mekanisme Local Search Cerdas ---
def _intelligent_local_search(solution: np.ndarray, tasks_dict: dict, vms_dict: dict, vm_map: list) -> np.ndarray:
    current_fitness = _evaluate_fitness(solution, tasks_dict, vms_dict, vm_map)
    best_solution = solution.copy()

    for _ in range(2):
        vm_loads = _get_vm_loads(best_solution, tasks_dict, vms_dict, vm_map)
        most_loaded_vm_idx, least_loaded_vm_idx = np.argmax(vm_loads), np.argmin(vm_loads)
        if most_loaded_vm_idx == least_loaded_vm_idx: break
        tasks_on_most = [i for i, vm in enumerate(best_solution) if vm == most_loaded_vm_idx]
        if not tasks_on_most: break
        
        # Coba 'Move' tugas paling ringan dari VM tersibuk
        task_loads = {i: tasks_dict[i].cpu_load for i in tasks_on_most}
        task_to_move = min(task_loads, key=task_loads.get)
        
        temp_solution = best_solution.copy()
        temp_solution[task_to_move] = least_loaded_vm_idx
        new_fitness = _evaluate_fitness(temp_solution, tasks_dict, vms_dict, vm_map)
        if new_fitness < current_fitness:
            current_fitness, best_solution = new_fitness, temp_solution
            
    return best_solution


# --- Algoritma Utama CloudyGSA ---

def cloudy_gsa_scheduler(tasks: list[Task], vms: list[VM], iterations: int = 1000) -> dict:
    print(f"Memulai Cloudy-GSA (V4 - Unified Fitness, Adaptive Inertia, {iterations} iterasi)...")

    n_tasks, n_vms = len(tasks), len(vms)
    vms_dict = {vm.name: vm for vm in vms}; tasks_dict = {task.id: task for task in tasks}
    vm_map = [vm.name for vm in vms]

    pos = np.random.uniform(0, n_vms - 1, (POP_SIZE, n_tasks))
    vel = np.zeros((POP_SIZE, n_tasks))

    fitness = np.array([_evaluate_fitness(_map_to_solution(p, n_vms), tasks_dict, vms_dict, vm_map) for p in pos])

    gbest_idx = np.argmin(fitness)
    gbest_val = fitness[gbest_idx]
    gbest_pos = pos[gbest_idx].copy()
    
    print(f"Estimasi Fitness Awal (Acak): {gbest_val:.2f}")

    inertia_weight = INERTIA_MAX
    stagnation_counter = 0

    for t in range(iterations):
        G = G0 * math.exp(-ALPHA * (t / iterations))
        mass = _compute_mass(fitness)
        
        force = np.zeros((POP_SIZE, n_tasks))
        for i in range(POP_SIZE):
            for j in range(POP_SIZE):
                if i != j:
                    diff = pos[j] - pos[i]
                    dist = np.linalg.norm(diff) + EPS
                    f = G * (mass[j] / (dist**2)) * diff
                    force[i] += np.random.rand(n_tasks) * f
        
        accel = force / (mass[:, np.newaxis] + EPS)
        vel = inertia_weight * vel + accel
        pos += vel
        pos = np.clip(pos, 0, n_vms - 1)

        sorted_indices = np.argsort(fitness)
        for i, p_idx in enumerate(sorted_indices):
            if i < LOCAL_SEARCH_CANDIDATES:
                current_solution = _map_to_solution(pos[p_idx], n_vms)
                pos[p_idx] = _intelligent_local_search(current_solution, tasks_dict, vms_dict, vm_map).astype(float)
            if np.random.rand() < MUTATION_RATE:
                task_to_mutate, new_vm = np.random.randint(n_tasks), np.random.randint(n_vms)
                pos[p_idx, task_to_mutate] = new_vm
            fitness[p_idx] = _evaluate_fitness(_map_to_solution(pos[p_idx], n_vms), tasks_dict, vms_dict, vm_map)
        
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < gbest_val:
            gbest_val, gbest_pos = fitness[current_best_idx], pos[current_best_idx].copy()
            stagnation_counter = 0
            # [OPTIMASI #2] Inersia Adaptif: Sukses -> turunkan inersia (lebih presisi)
            inertia_weight = max(INERTIA_MIN, inertia_weight - 0.05)
            if t > 0 and t % 100 == 0: print(f"Iterasi {t}: Fitness Baru Terbaik: {gbest_val:.2f}")
        else:
            stagnation_counter += 1
            # [OPTIMASI #2] Inersia Adaptif: Stagnan -> naikkan inersia (eksplorasi)
            if stagnation_counter > 20: inertia_weight = min(INERTIA_MAX, inertia_weight + 0.05)
            pos[np.argmax(fitness)] = gbest_pos

    print(f"Cloudy-GSA Selesai. Fitness Terbaik: {gbest_val:.2f}")

    best_solution_discrete = _map_to_solution(gbest_pos, n_vms)
    best_assignment = {task.id: vm_map[best_solution_discrete[i]] for i, task in enumerate(tasks)}
    return best_assignment

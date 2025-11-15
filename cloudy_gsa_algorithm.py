
import numpy as np
import math
from collections import namedtuple

# Definisi Tipe
VM = namedtuple('VM', ['name', 'ip', 'cpu_cores', 'ram_gb'])
Task = namedtuple('Task', ['id', 'name', 'index', 'cpu_load'])

# --- Konstanta Algoritma ---
POP_SIZE = 50
G0 = 100.0
ALPHA = 20.0
EPS = 1e-12
INERTIA = 0.75 # Sedikit meningkatkan inertia untuk eksplorasi
VELOCITY_CLAMP = 2.0
MUTATION_RATE = 0.02
STAGNATION_LIMIT = 75 # [OPTIMASI #2] Batas iterasi sebelum G di-reset
LOCAL_SEARCH_CANDIDATES = 5 # Jumlah partikel terbaik yang dioptimasi local search

# --- [OPTIMASI #1] Bobot untuk Fitness Function 3-Objektif ---
W_MAKESPAN = 1.0   # Tetap penting untuk kecepatan
W_STD_DEV = 1.2    # Menjaga keseimbangan
W_IDLE_TIME = 0.5  # Mendorong utilisasi sumber daya

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
    [OPTIMASI #1] Fitness Function 3-Objektif: Makespan, Keseimbangan, dan Utilisasi.
    """
    vm_loads = _get_vm_loads(solution, tasks_dict, vms_dict, vm_map)
    
    # 1. Minimalkan Makespan
    makespan = np.max(vm_loads)
    
    # 2. Minimalkan ketidakseimbangan (Standar Deviasi Beban)
    load_std_dev = np.std(vm_loads)
    
    # 3. Minimalkan waktu menganggur (untuk menaikkan utilisasi)
    total_idle_time = np.sum(makespan - vm_loads)
    
    fitness_score = (W_MAKESPAN * makespan) + \
                    (W_STD_DEV * load_std_dev) + \
                    (W_IDLE_TIME * total_idle_time)
    
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

# --- [OPTIMASI #3] Mekanisme Local Search Cerdas (Move & Swap) ---

def _intelligent_local_search(solution: np.ndarray, tasks_dict: dict, vms_dict: dict, vm_map: list) -> np.ndarray:
    """
    Mencoba perbaikan lokal dengan memindahkan atau menukar tugas
    antara VM yang paling sibuk dan paling lengang.
    """
    current_fitness = _evaluate_fitness(solution, tasks_dict, vms_dict, vm_map)
    best_solution = solution.copy()

    for _ in range(2): # Lakukan beberapa kali iterasi perbaikan
        vm_loads = _get_vm_loads(best_solution, tasks_dict, vms_dict, vm_map)
        most_loaded_vm_idx = np.argmax(vm_loads)
        least_loaded_vm_idx = np.argmin(vm_loads)

        if most_loaded_vm_idx == least_loaded_vm_idx: break

        tasks_on_most = [i for i, vm in enumerate(best_solution) if vm == most_loaded_vm_idx]
        tasks_on_least = [i for i, vm in enumerate(best_solution) if vm == least_loaded_vm_idx]

        if not tasks_on_most: break

        # 1. Coba 'Move'
        for task_idx in tasks_on_most:
            temp_solution = best_solution.copy()
            temp_solution[task_idx] = least_loaded_vm_idx
            new_fitness = _evaluate_fitness(temp_solution, tasks_dict, vms_dict, vm_map)
            if new_fitness < current_fitness:
                current_fitness = new_fitness
                best_solution = temp_solution

        # 2. Coba 'Swap' (jika ada tugas di VM yg lengang)
        if tasks_on_least:
            for t_most in tasks_on_most:
                for t_least in tasks_on_least:
                    temp_solution = best_solution.copy()
                    temp_solution[t_most], temp_solution[t_least] = temp_solution[t_least], temp_solution[t_most]
                    new_fitness = _evaluate_fitness(temp_solution, tasks_dict, vms_dict, vm_map)
                    if new_fitness < current_fitness:
                        current_fitness = new_fitness
                        best_solution = temp_solution
                        
    return best_solution


# --- Algoritma Utama CloudyGSA ---

def cloudy_gsa_scheduler(tasks: list[Task], vms: list[VM], iterations: int = 1000) -> dict:
    """
    Menjalankan algoritma Cloudy-GSA (V3) dengan Fitness 3-Objektif,
    Dynamic G, dan Intelligent Local Search.
    """
    print(f"Memulai Cloudy-GSA (V3 - Utilization Focus, Dynamic G, {iterations} iterasi)...")

    n_tasks = len(tasks)
    n_vms = len(vms)
    vms_dict = {vm.name: vm for vm in vms}
    tasks_dict = {task.id: task for task in tasks}
    vm_map = [vm.name for vm in vms]

    pos = np.random.uniform(0, n_vms - 1, (POP_SIZE, n_tasks))
    vel = np.zeros((POP_SIZE, n_tasks))

    fitness = np.array([_evaluate_fitness(_map_to_solution(p, n_vms), tasks_dict, vms_dict, vm_map) for p in pos])

    gbest_idx = np.argmin(fitness)
    gbest_val = fitness[gbest_idx]
    gbest_pos = pos[gbest_idx].copy()
    
    print(f"Estimasi Fitness Awal (Acak): {gbest_val:.2f}")

    stagnation_counter = 0
    last_gbest_val = gbest_val

    for t in range(iterations):
        G = G0 * math.exp(-ALPHA * (t / iterations))

        # [OPTIMASI #2] Mekanisme Anti-Stagnasi
        if stagnation_counter >= STAGNATION_LIMIT:
            G = G0 # Reset gravitasi untuk "mengguncang" sistem
            stagnation_counter = 0
            # print(f"Iterasi {t}: Stagnasi terdeteksi. Mereset G.")

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
        vel = INERTIA * vel + accel
        vel = np.clip(vel, -VELOCITY_CLAMP, VELOCITY_CLAMP)
        pos += vel
        pos = np.clip(pos, 0, n_vms - 1)

        # Evaluasi ulang, terapkan Local Search dan Mutasi
        sorted_indices = np.argsort(fitness)
        for i, p_idx in enumerate(sorted_indices):
            # Terapkan local search cerdas pada beberapa partikel terbaik
            if i < LOCAL_SEARCH_CANDIDATES:
                current_solution = _map_to_solution(pos[p_idx], n_vms)
                improved_solution = _intelligent_local_search(current_solution, tasks_dict, vms_dict, vm_map)
                pos[p_idx] = improved_solution.astype(float)

            # Terapkan mutasi acak
            if np.random.rand() < MUTATION_RATE:
                task_to_mutate = np.random.randint(n_tasks)
                new_vm = np.random.randint(n_vms)
                pos[p_idx, task_to_mutate] = new_vm
            
            fitness[p_idx] = _evaluate_fitness(_map_to_solution(pos[p_idx], n_vms), tasks_dict, vms_dict, vm_map)
        
        # Elitisme: Simpan solusi terbaik
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < gbest_val:
            gbest_val = fitness[current_best_idx]
            gbest_pos = pos[current_best_idx].copy()
            stagnation_counter = 0
            if t > 0 and t % 100 == 0:
                print(f"Iterasi {t}: Fitness Baru Terbaik: {gbest_val:.2f}")
        else:
            stagnation_counter += 1
            pos[np.argmax(fitness)] = gbest_pos # Ganti yg terburuk dengan yg terbaik

    print(f"Cloudy-GSA Selesai. Fitness Terbaik: {gbest_val:.2f}")

    best_solution_discrete = _map_to_solution(gbest_pos, n_vms)
    best_assignment = {
        task.id: vm_map[best_solution_discrete[i]]
        for i, task in enumerate(tasks)
    }
        
    return best_assignment

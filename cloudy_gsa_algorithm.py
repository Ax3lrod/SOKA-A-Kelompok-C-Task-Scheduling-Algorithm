import numpy as np
import math
from collections import namedtuple

# Definisi Tipe
VM = namedtuple('VM', ['name', 'ip', 'cpu_cores', 'ram_gb'])
Task = namedtuple('Task', ['id', 'name', 'index', 'cpu_load'])

# --- Konstanta Algoritma (Dioptimalkan) ---
POP_SIZE = 60  # Populasi lebih besar untuk eksplorasi
G0 = 150.0  # Gravitasi awal lebih tinggi untuk konvergensi cepat
ALPHA = 25.0  # Decay lebih agresif
EPS = 1e-12
INERTIA_MAX = 0.9  # Inersia adaptif
INERTIA_MIN = 0.4
VELOCITY_CLAMP = 1.5  # Velocity lebih rendah untuk stabilitas
MUTATION_RATE = 0.03
LOCAL_SEARCH_RATE = 0.15  # Lebih banyak partikel di-improve
ELITE_SIZE = 5  # Simpan 5 solusi terbaik

# --- Bobot Fitness Function (Balanced) ---
W_MAKESPAN = 1.0
W_STD_DEV = 2.0  # Prioritas lebih tinggi pada keseimbangan beban
W_UTILIZATION = 0.5  # Bonus untuk utilisasi tinggi

# --- Fungsi Fitness dengan Utilisasi Resource ---

def _get_vm_loads(solution: np.ndarray, tasks_dict: dict, vms_dict: dict, vm_map: list) -> tuple:
    """Menghitung beban dan utilisasi VM."""
    n_vms = len(vm_map)
    vm_load_time = np.zeros(n_vms)
    vm_capacities = np.zeros(n_vms)
    
    for vm_idx, vm_name in enumerate(vm_map):
        vm_capacities[vm_idx] = vms_dict[vm_name].cpu_cores
    
    for task_idx, vm_idx in enumerate(solution):
        task = tasks_dict[task_idx]
        vm_name = vm_map[int(vm_idx)]
        vm = vms_dict[vm_name]
        exec_time = task.cpu_load / vm.cpu_cores
        vm_load_time[int(vm_idx)] += exec_time
    
    return vm_load_time, vm_capacities

def _evaluate_fitness(solution: np.ndarray, tasks_dict: dict, vms_dict: dict, vm_map: list) -> float:
    """
    Multi-objective fitness: Makespan + Load Balance + Resource Utilization
    """
    vm_loads, vm_caps = _get_vm_loads(solution, tasks_dict, vms_dict, vm_map)
    
    makespan = np.max(vm_loads)
    load_std = np.std(vm_loads)
    
    # Penalty untuk VM idle (utilisasi rendah)
    mean_load = np.mean(vm_loads)
    utilization_score = -mean_load if mean_load > 0 else 0  # Negatif = reward
    
    fitness = (W_MAKESPAN * makespan) + (W_STD_DEV * load_std) + (W_UTILIZATION * utilization_score)
    return fitness

# --- Helper Functions ---

def _compute_mass(fitness: np.ndarray):
    """Mass calculation dengan smoothing."""
    best, worst = np.min(fitness), np.max(fitness)
    if (worst - best) < EPS:
        return np.ones_like(fitness) / len(fitness)
    
    raw_mass = (worst - fitness + EPS) / (worst - best + EPS)
    total_mass = np.sum(raw_mass)
    return raw_mass / (total_mass + EPS)

def _map_to_solution(position: np.ndarray, n_vms: int) -> np.ndarray:
    """Mapping dengan rounding probabilistik untuk diversitas."""
    rounded = np.round(position)
    # Random rounding 5% untuk eksplorasi
    mask = np.random.rand(len(position)) < 0.05
    rounded[mask] = np.floor(position[mask]) if np.random.rand() > 0.5 else np.ceil(position[mask])
    return np.clip(rounded, 0, n_vms - 1).astype(int)

# --- Optimized Local Search ---

def _local_search_greedy(solution: np.ndarray, tasks_dict: dict, vms_dict: dict, vm_map: list) -> np.ndarray:
    """
    Greedy local search: Pindahkan task dari VM terbeban ke VM optimal.
    """
    best_solution = solution.copy()
    best_fitness = _evaluate_fitness(solution, tasks_dict, vms_dict, vm_map)
    improved = True
    max_iterations = 3  # Batasi iterasi untuk efisiensi
    
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        vm_loads, _ = _get_vm_loads(best_solution, tasks_dict, vms_dict, vm_map)
        overloaded_vm = np.argmax(vm_loads)
        
        tasks_on_overloaded = [i for i, vm in enumerate(best_solution) if vm == overloaded_vm]
        
        if not tasks_on_overloaded:
            break
        
        # Coba pindahkan task terberat
        task_loads = [(i, tasks_dict[i].cpu_load / vms_dict[vm_map[overloaded_vm]].cpu_cores) 
                      for i in tasks_on_overloaded]
        task_loads.sort(key=lambda x: x[1], reverse=True)
        
        for task_idx, _ in task_loads[:5]:  # Cek 5 task terberat
            for new_vm in range(len(vm_map)):
                if new_vm == overloaded_vm:
                    continue
                
                test_solution = best_solution.copy()
                test_solution[task_idx] = new_vm
                test_fitness = _evaluate_fitness(test_solution, tasks_dict, vms_dict, vm_map)
                
                if test_fitness < best_fitness:
                    best_fitness = test_fitness
                    best_solution = test_solution
                    improved = True
                    break
            
            if improved:
                break
    
    return best_solution

def _local_search_swap(solution: np.ndarray, tasks_dict: dict, vms_dict: dict, vm_map: list) -> np.ndarray:
    """Swap optimization antara VM paling sibuk dan paling idle."""
    vm_loads, _ = _get_vm_loads(solution, tasks_dict, vms_dict, vm_map)
    
    most_loaded = np.argmax(vm_loads)
    least_loaded = np.argmin(vm_loads)
    
    if most_loaded == least_loaded or abs(vm_loads[most_loaded] - vm_loads[least_loaded]) < 1.0:
        return solution
    
    tasks_most = [i for i, vm in enumerate(solution) if vm == most_loaded]
    tasks_least = [i for i, vm in enumerate(solution) if vm == least_loaded]
    
    if not tasks_most:
        return solution
    
    best_solution = solution.copy()
    best_fitness = _evaluate_fitness(solution, tasks_dict, vms_dict, vm_map)
    
    # Swap paling efektif
    for task1 in tasks_most[:3]:
        if tasks_least:
            for task2 in tasks_least[:3]:
                test = solution.copy()
                test[task1], test[task2] = test[task2], test[task1]
                fit = _evaluate_fitness(test, tasks_dict, vms_dict, vm_map)
                if fit < best_fitness:
                    best_fitness = fit
                    best_solution = test
        
        # Coba pindah tanpa swap
        for new_vm in range(len(vm_map)):
            if new_vm == most_loaded:
                continue
            test = solution.copy()
            test[task1] = new_vm
            fit = _evaluate_fitness(test, tasks_dict, vms_dict, vm_map)
            if fit < best_fitness:
                best_fitness = fit
                best_solution = test
    
    return best_solution

# --- Advanced Mutation ---

def _adaptive_mutation(pos: np.ndarray, vm_idx: int, n_tasks: int, n_vms: int, fitness: float, avg_fitness: float):
    """Mutasi adaptif berdasarkan fitness relatif."""
    if fitness > avg_fitness:  # Partikel buruk = mutasi lebih agresif
        n_mutations = np.random.randint(2, max(3, n_tasks // 10))
    else:
        n_mutations = 1
    
    for _ in range(n_mutations):
        task_idx = np.random.randint(n_tasks)
        pos[vm_idx, task_idx] = np.random.randint(n_vms)

# --- Main Algorithm ---

def cloudy_gsa_scheduler(tasks: list[Task], vms: list[VM], iterations: int = 1000) -> dict:
    """
    Optimized Cloudy-GSA dengan:
    - Adaptive inertia weight
    - Elite preservation
    - Hybrid local search (greedy + swap)
    - Adaptive mutation
    - Multi-objective fitness
    """
    print(f"🚀 Memulai Optimized Cloudy-GSA ({iterations} iterasi)...")
    
    n_tasks = len(tasks)
    n_vms = len(vms)
    
    vms_dict = {vm.name: vm for vm in vms}
    tasks_dict = {task.id: task for task in tasks}
    vm_map = [vm.name for vm in vms]
    
    # Inisialisasi populasi dengan strategi hybrid
    pos = np.zeros((POP_SIZE, n_tasks))
    for i in range(POP_SIZE):
        if i < POP_SIZE // 4:  # 25% random
            pos[i] = np.random.rand(n_tasks) * n_vms
        elif i < POP_SIZE // 2:  # 25% round-robin
            pos[i] = np.arange(n_tasks) % n_vms
        else:  # 50% task-aware (assign ke VM dengan core tertinggi)
            vm_cores = [vms_dict[vm_name].cpu_cores for vm_name in vm_map]
            best_vm = np.argmax(vm_cores)
            pos[i] = best_vm + np.random.rand(n_tasks) * 0.5
    
    vel = np.random.randn(POP_SIZE, n_tasks) * 0.1
    
    fitness = np.array([
        _evaluate_fitness(_map_to_solution(p, n_vms), tasks_dict, vms_dict, vm_map)
        for p in pos
    ])
    
    # Elite tracking
    elite_indices = np.argsort(fitness)[:ELITE_SIZE]
    elite_pos = pos[elite_indices].copy()
    elite_fitness = fitness[elite_indices].copy()
    
    gbest_idx = elite_indices[0]
    gbest_val = elite_fitness[0]
    gbest_pos = elite_pos[0].copy()
    
    print(f"📊 Fitness Awal: {gbest_val:.2f}")
    
    stagnation_counter = 0
    prev_best = gbest_val
    
    for t in range(iterations):
        # Adaptive inertia
        inertia = INERTIA_MAX - (INERTIA_MAX - INERTIA_MIN) * (t / iterations)
        
        mass = _compute_mass(fitness)
        G = G0 * math.exp(-ALPHA * (t / iterations))
        
        # Gravitational force dengan knearest
        k_best = max(1, int(POP_SIZE * (1 - t / iterations)))
        k_best_indices = np.argsort(fitness)[:k_best]
        
        force = np.zeros((POP_SIZE, n_tasks))
        for i in range(POP_SIZE):
            for j in k_best_indices:
                if i != j:
                    diff = pos[j] - pos[i]
                    dist = np.linalg.norm(diff) + EPS
                    f = G * (mass[i] * mass[j] / dist) * diff
                    force[i] += np.random.rand(n_tasks) * f
        
        accel = force / (mass[:, np.newaxis] + EPS)
        vel = inertia * vel + accel
        vel = np.clip(vel, -VELOCITY_CLAMP, VELOCITY_CLAMP)
        pos += vel
        pos = np.clip(pos, 0, n_vms - 1)
        
        # Evaluasi dan optimasi lokal
        avg_fitness = np.mean(fitness)
        sorted_idx = np.argsort(fitness)
        
        for rank, p_idx in enumerate(sorted_idx):
            solution = _map_to_solution(pos[p_idx], n_vms)
            
            # Local search untuk top 15%
            if rank < int(POP_SIZE * LOCAL_SEARCH_RATE):
                if rank % 2 == 0:
                    solution = _local_search_greedy(solution, tasks_dict, vms_dict, vm_map)
                else:
                    solution = _local_search_swap(solution, tasks_dict, vms_dict, vm_map)
                pos[p_idx] = solution.astype(float)
            
            # Adaptive mutation (skip elite)
            if rank >= ELITE_SIZE and np.random.rand() < MUTATION_RATE:
                _adaptive_mutation(pos, p_idx, n_tasks, n_vms, fitness[p_idx], avg_fitness)
            
            fitness[p_idx] = _evaluate_fitness(_map_to_solution(pos[p_idx], n_vms), 
                                               tasks_dict, vms_dict, vm_map)
        
        # Update elite
        current_elite_idx = np.argsort(fitness)[:ELITE_SIZE]
        for i, idx in enumerate(current_elite_idx):
            if fitness[idx] < elite_fitness[i]:
                elite_fitness[i] = fitness[idx]
                elite_pos[i] = pos[idx].copy()
        
        # Replace worst dengan elite
        worst_indices = np.argsort(fitness)[-ELITE_SIZE:]
        for i, w_idx in enumerate(worst_indices):
            if fitness[w_idx] > elite_fitness[i]:
                pos[w_idx] = elite_pos[i].copy()
                vel[w_idx] *= 0.5
                fitness[w_idx] = elite_fitness[i]
        
        # Update global best
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < gbest_val:
            improvement = gbest_val - fitness[current_best_idx]
            gbest_val = fitness[current_best_idx]
            gbest_pos = pos[current_best_idx].copy()
            stagnation_counter = 0
            if t % 50 == 0 or improvement > 1.0:
                print(f"✨ Iterasi {t}: Fitness = {gbest_val:.2f} (↓ {improvement:.2f})")
        else:
            stagnation_counter += 1
        
        # Restart mechanism jika stagnasi
        if stagnation_counter > 150:
            print(f"🔄 Restart di iterasi {t} (stagnasi terdeteksi)")
            worst_half = np.argsort(fitness)[POP_SIZE//2:]
            pos[worst_half] = np.random.rand(len(worst_half), n_tasks) * n_vms
            vel[worst_half] = np.random.randn(len(worst_half), n_tasks) * 0.1
            stagnation_counter = 0
    
    print(f"✅ Selesai! Fitness Terbaik: {gbest_val:.2f}")
    
    best_solution = _map_to_solution(gbest_pos, n_vms)
    
    # Analisis hasil
    vm_loads, _ = _get_vm_loads(best_solution, tasks_dict, vms_dict, vm_map)
    print(f"📈 Makespan: {np.max(vm_loads):.2f}s")
    print(f"📊 Std Dev: {np.std(vm_loads):.2f}s")
    print(f"⚖️  Load Balance: {vm_loads}")
    
    assignment = {
        task.id: vm_map[best_solution[i]]
        for i, task in enumerate(tasks)
    }
    
    return assignment
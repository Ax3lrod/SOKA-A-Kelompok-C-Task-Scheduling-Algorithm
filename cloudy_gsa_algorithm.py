import numpy as np
import math
from collections import namedtuple

# Definisi Tipe
VM = namedtuple('VM', ['name', 'ip', 'cpu_cores', 'ram_gb'])
Task = namedtuple('Task', ['id', 'name', 'index', 'cpu_load'])

# --- Konstanta Algoritma ---
POP_SIZE = 40
G0 = 100.0
ALPHA = 20.0
EPS = 1e-12
INERTIA = 0.7
VELOCITY_CLAMP = 2.0
LOCAL_TRIES = 4  # Jumlah percobaan pencarian lokal per partikel

# --- Fungsi Fitness (Fungsi Biaya) ---

def _evaluate_fitness(solution: np.ndarray, tasks_dict: dict, vms_dict: dict, vm_map: list) -> float:
    """
    Fungsi Biaya (Cost Function) Multi-Objektif.
    Menghitung skor fitness berdasarkan kombinasi dari:
    1. Makespan (waktu selesai maksimum).
    2. Rata-rata waktu tunggu tugas.
    3. Standar deviasi waktu selesai VM (untuk load balancing).
    """
    n_vms = len(vm_map)
    n_tasks = len(solution)
    
    vm_finish_times = np.zeros(n_vms)
    total_wait_time = 0.0
    
    vm_name_to_idx = {name: i for i, name in enumerate(vm_map)}
    
    for task_idx in range(n_tasks):
        vm_assigned_idx = int(solution[task_idx])
        task = tasks_dict[task_idx]
        vm_name = vm_map[vm_assigned_idx]
        vm = vms_dict[vm_name]
        vm_array_idx = vm_name_to_idx[vm_name]
        
        exec_time = task.cpu_load / vm.cpu_cores
        wait_time = vm_finish_times[vm_array_idx]
        total_wait_time += wait_time
        vm_finish_times[vm_array_idx] += exec_time
        
    makespan = np.max(vm_finish_times)
    avg_wait_time = total_wait_time / n_tasks if n_tasks > 0 else 0
    std_dev = np.std(vm_finish_times)
    
    # Skor fitness gabungan
    fitness_score = makespan + avg_wait_time + (0.5 * std_dev)
    return fitness_score

# --- Fungsi Helper ---

def _compute_mass(fitness: np.ndarray):
    """Menghitung massa setiap partikel berdasarkan nilai fitness-nya."""
    best, worst = np.min(fitness), np.max(fitness)
    if (worst - best) < EPS:
        return np.ones(POP_SIZE) / POP_SIZE
    raw_mass = (worst - fitness) / (worst - best)
    total_mass = np.sum(raw_mass)
    return raw_mass / total_mass if total_mass > 0 else np.ones(POP_SIZE) / POP_SIZE

def _map_to_solution(position: np.ndarray, n_vms: int) -> np.ndarray:
    """Membulatkan posisi kontinu ke solusi diskrit (indeks VM)."""
    return np.clip(np.round(position), 0, n_vms - 1).astype(int)

# --- Algoritma Utama CloudyGSA ---

def cloudy_gsa_scheduler(tasks: list[Task], vms: list[VM], iterations: int = 1000) -> dict:
    """
    Menjalankan algoritma Cloudy Gravitational Search Algorithm (Cloudy-GSA)
    dengan Elitisme dan Pencarian Lokal.
    """
    print(f"Memulai Cloudy-GSA (dengan Elitisme & Pencarian Lokal, {iterations} iterasi)...")

    n_tasks = len(tasks)
    n_vms = len(vms)

    vms_dict = {vm.name: vm for vm in vms}
    tasks_dict = {task.id: task for task in tasks}
    vm_map = [vm.name for vm in vms]

    # Inisialisasi posisi dan kecepatan
    pos = np.random.rand(POP_SIZE, n_tasks) * (n_vms - EPS)
    vel = (np.random.rand(POP_SIZE, n_tasks) - 0.5) * 0.1

    # Evaluasi fitness awal
    fitness = np.array([
        _evaluate_fitness(_map_to_solution(p, n_vms), tasks_dict, vms_dict, vm_map)
        for p in pos
    ])

    # Simpan solusi terbaik global (gbest)
    gbest_idx = np.argmin(fitness)
    gbest_val = fitness[gbest_idx]
    gbest_pos = pos[gbest_idx].copy()
    
    print(f"Estimasi Fitness Awal (Acak): {gbest_val:.2f}")

    # Iterasi utama
    for t in range(iterations):
        # Simpan gbest sebelum update untuk Elitisme
        gbest_pos_before_update = gbest_pos.copy()
        gbest_val_before_update = gbest_val

        # Perhitungan GSA standar
        mass = _compute_mass(fitness)
        G = G0 * math.exp(-ALPHA * (t / iterations))

        force = np.zeros((POP_SIZE, n_tasks))
        for i in range(POP_SIZE):
            for j in range(POP_SIZE):
                if i == j: continue
                diff = pos[j] - pos[i]
                dist = np.sqrt(np.sum(diff**2)) + EPS
                f = G * (mass[i] * mass[j] / dist) * diff
                force[i] += np.random.rand() * f

        accel = force / (mass[:, np.newaxis] + EPS)
        
        # Update kecepatan dan posisi
        vel = INERTIA * vel + accel
        vel = np.clip(vel, -VELOCITY_CLAMP, VELOCITY_CLAMP)
        pos += vel
        pos = np.clip(pos, 0, n_vms - 1)

        # --- TAMBAHAN: MEKANISME PENCARIAN LOKAL (LOCAL SEARCH) ---
        for p in range(POP_SIZE):
            current_solution = _map_to_solution(pos[p], n_vms)
            # Coba beberapa perbaikan kecil pada setiap partikel
            for _ in range(LOCAL_TRIES):
                # Pilih satu tugas acak untuk dipindahkan
                task_to_move = np.random.randint(n_tasks)
                original_vm_idx = current_solution[task_to_move]
                
                # Coba pindahkan ke VM acak yang baru
                new_vm_idx = np.random.randint(n_vms)
                if new_vm_idx == original_vm_idx: continue

                current_solution[task_to_move] = new_vm_idx
                
                # Evaluasi fitness dari solusi yang diubah
                new_fitness = _evaluate_fitness(current_solution, tasks_dict, vms_dict, vm_map)

                # Jika lebih baik, pertahankan. Jika tidak, kembalikan.
                if new_fitness < fitness[p]:
                    fitness[p] = new_fitness
                    # Update posisi kontinu juga untuk konsistensi
                    pos[p, task_to_move] = new_vm_idx
                else:
                    current_solution[task_to_move] = original_vm_idx
        
        # Evaluasi ulang fitness setelah GSA move + Local Search
        for p in range(POP_SIZE):
             fitness[p] = _evaluate_fitness(_map_to_solution(pos[p], n_vms), tasks_dict, vms_dict, vm_map)

        # --- TAMBAHAN: MEKANISME ELITISME ---
        # Ganti partikel terburuk dengan gbest dari sebelum iterasi ini
        worst_idx = np.argmax(fitness)
        if fitness[worst_idx] > gbest_val_before_update:
            pos[worst_idx] = gbest_pos_before_update
            vel[worst_idx] = (np.random.rand(n_tasks) - 0.5) * 0.1 # Reset kecepatan
            fitness[worst_idx] = gbest_val_before_update

        # Update gbest global
        local_best_idx = np.argmin(fitness)
        if fitness[local_best_idx] < gbest_val:
            gbest_val = fitness[local_best_idx]
            gbest_pos = pos[local_best_idx].copy()
            if t % 50 == 0:
                print(f"Iterasi {t}: Fitness Baru Terbaik: {gbest_val:.2f}")

    print(f"Cloudy-GSA Selesai. Fitness Terbaik: {gbest_val:.2f}")

    # Bentuk hasil akhir
    best_solution_discrete = _map_to_solution(gbest_pos, n_vms)
    best_assignment = {}
    for i, task in enumerate(tasks):
        vm_idx = best_solution_discrete[i]
        vm_name = vm_map[vm_idx]
        best_assignment[task.id] = vm_name
        
    return best_assignment

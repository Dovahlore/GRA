import copy
import statistics
import heapq
from collections import deque
import random
import time


def detect_unit_frequency_conflict(unit, beam):
    # 频率冲突检测
    # TODO: 任务指向相同没判断
    for on_going_beam in unit.on_goning_beams:  # 遍历该阵元正在运行的波束
        if abs(beam.F - on_going_beam.F) < 5 and beam.father != on_going_beam.father:  # 检测频率冲突
            return True

    return False


# 在指定的阵面中寻找复用值最低的阵元
def find_lowest_reuse_unit(units_on_surface, beam, target_N):
    lowest_reuse_value = float('inf')
    lowest_reuse_unit = None
    # 在给定的阵面上遍历45个阵元
    for unit in units_on_surface.values():
        reuse_value = unit.N
        if reuse_value < target_N and reuse_value < lowest_reuse_value and detect_unit_frequency_conflict(unit,
                                                                                                          beam) == False:
            lowest_reuse_value = reuse_value
            lowest_reuse_unit = unit

    return lowest_reuse_unit


# 寻找最短路径，但只在当前阵面内寻找，不考虑跨阵面的相邻阵元
def find_shortest_path(env, adjacency, start_unit, target_count, beam_F, N):
    # 此方法优点是算法简单，分配速度快，分配紧凑，为高优先级的任务空出较多复用为0的阵元（？）
    queue = deque([(start_unit, [start_unit])])
    visited = set()

    while queue:
        current_unit, path = queue.popleft()

        if len(path) == target_count:
            for unit in path:
                unit.N += 1
            return path

        for neighbor_id in env.get_unit_adj(current_unit):
            # 只考虑与当前阵元在同一个阵面的邻接阵元
            neighbor = env.get_RF_instance(neighbor_id)
            if neighbor.startswith(
                    current_unit[:4]) and neighbor not in visited and neighbor.N < N and detect_unit_frequency_conflict(
                neighbor, beam_F) == False:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None


def find_shortest_path_dijkstra(env, start_unit, target_count, beam_F):
    # 此方法优点是找到的是当前阵面中最优的路径，可以保证复用数很低，缺点是遍历慢
    queue = [(0, start_unit.id, [start_unit.id])]
    visited = {}  # 记录访问过的阵元及其路径权重

    while queue:
        current_weight, current_unit, path = heapq.heappop(queue)

        if len(path) == target_count:
            for unit in path:
                unit = env.get_RF_instance(unit)
                unit.N += 1
            return path

        if current_unit in visited and visited[current_unit] <= current_weight:
            continue

        visited[current_unit] = current_weight
        ids = env.get_unit_adj(current_unit)

        current_unit = env.get_RF_instance(current_unit)
        for neighbor_id in ids:

            neighbor = env.get_RF_instance(neighbor_id)
            if neighbor.father == current_unit.father:  # 只考虑同阵面的邻接阵元
                reuse_value = neighbor.N
                if reuse_value < 3 and detect_unit_frequency_conflict(neighbor, beam_F) == False:
                    new_weight = current_weight + reuse_value
                    heapq.heappush(queue, (new_weight, neighbor.id, path + [neighbor.id]))

    return None


def find_min_weight_path_dp(env, surfaces, Task, beam, target_N, required_units=None):
    """
    使用动态规划寻找必须走满d步的最小权重和路径
    """
    current_index = 0  # 初始化从第一个最优阵面开始
    all_units = {}
    path_surface_target = {}
    max_target_count = 0
    beam_success = False
    for surface, theta in surfaces:
        for unit in surface.units.values():
            # 排除不可用的阵元
            if unit.N < target_N and not detect_unit_frequency_conflict(unit, beam):
                all_units[unit.id] = unit
        # 如果给定了阵元数，则直接使用
        if required_units:
            target_count = required_units
        else:
            target_count = env.calc_N(beam, Task, theta, target_N)
            print(f"{surface.id}:target_count={target_count}")
        path_surface_target[surface.id] = target_count

        if target_count > max_target_count:
            max_target_count = target_count
    k = 0
    INF = float('inf')
    path_founded = False
    # dp[i][k] 表示经过k步到达节点i的最小权重和
    dp = {i: [INF for _ in range(max_target_count + 1)] for i in all_units}
    # 路径记录，用于追踪最优路径
    path = {i: [[] for _ in range(max_target_count + 1)] for i in all_units}
    path_node_target = {i: max_target_count for i in all_units}

    while current_index < len(surfaces) and not beam_success:
        # 寻找当前阵面中复用值最低的可用阵元
        current_surface, theta = surfaces[current_index]
        usable_units = [u for u in current_surface.units.values() if u.id in all_units]
        start_unit = lowest_reuse_usable_unit(usable_units)

        if not start_unit:
            # 如果当前阵面没有可用阵元，则切换到下一个阵面
            current_index += 1
            continue

        # 进行最短路径部署
        # dp初始化
        start_id = start_unit.id
        dp[start_id][1] = start_unit.N + 1
        path[start_id][1] = [start_unit.id]
        path_node_target[start_id] = path_surface_target[start_unit.father.id]
        # 动态规划过程
        for k in range(1, max_target_count + 1):  # 当前步数
            for node_id in all_units:  # 当前节点
                if dp[node_id][k] < INF:  # 如果从该节点在k长度时可以到达
                    # 如果阵元数满足条件，则出现可用路径，本次循环结束即可得出最短路径
                    if path_node_target[node_id] == k:
                        path_founded = True
                        continue
                    # 检查所有邻接节点
                    neighbors = []
                    for u in path[node_id][k]:
                        neighbors += env.get_unit_adj(u)
                    for neighbor_id in neighbors:
                        # 跳过已经分配过的，或不可使用的阵元
                        if neighbor_id not in all_units or neighbor_id in path[node_id][k]:
                            continue
                        neighbor = env.get_RF_instance(neighbor_id)
                        new_weight = dp[node_id][k] + neighbor.N + 1
                        # 如果通过node到达neighbor的权重和更小，且目标路径长度更短，更新dp
                        if path_node_target[node_id] <= path_node_target[neighbor_id] and new_weight <= \
                                dp[neighbor_id][k + 1]:
                            dp[neighbor_id][k + 1] = new_weight
                            path[neighbor_id][k + 1] = path[node_id][k] + [neighbor_id]
                            path_node_target[neighbor_id] = max(path_node_target[node_id],
                                                                path_surface_target[neighbor.father.id])
            if path_founded:
                break

        # 找到走满d长度时的最小权重和路径
        min_weight_sum = INF
        best_path = []

        print(f"K={k}")
        for node_id in all_units:
            if path_node_target[node_id] == k and dp[node_id][k] < min_weight_sum:
                min_weight_sum = dp[node_id][k]
                best_path = path[node_id][k]

        if min_weight_sum != INF:
            beam_success = True
        else:
            # 删除当前起始点所在的孤岛
            ilet = [start_unit.id]
            while ilet:
                node_id = ilet.pop(0)
                del all_units[node_id]
                for neighbor_id in env.get_unit_adj(node_id):
                    if neighbor_id in all_units and neighbor_id not in ilet:
                        ilet.append(neighbor_id)

    if beam_success:
        return best_path
    else:
        return None


# 在指定的阵元组中寻找复用值最低的阵元
def lowest_reuse_usable_unit(usable_units):
    lowest_reuse_value = float('inf')
    lowest_reuse_unit = None
    # 在给定的阵面上遍历45个阵元
    for unit in usable_units:
        reuse_value = unit.N
        if reuse_value < lowest_reuse_value:
            lowest_reuse_value = reuse_value
            lowest_reuse_unit = unit

    return lowest_reuse_unit


def find_min_weight_path_greedy(env, surfaces, Task, beam, target_N, required_units=None):
    current_index = 0  # 初始化从第一个最优阵面开始
    all_units = {}
    path_surface_target = {}
    max_target_count = 0
    beam_success = False
    for surface, theta in surfaces:
        for unit in surface.units.values():
            # 排除不可用的阵元
            if unit.N < target_N and not detect_unit_frequency_conflict(unit, beam):
                all_units[unit.id] = unit
        # 如果给定了阵元数，则直接使用
        if required_units:
            target_count = required_units
        else:
            target_count = env.calc_N(beam, Task, theta, target_N)
        path_surface_target[surface.id] = target_count

        if target_count > max_target_count:
            max_target_count = target_count

    while current_index < len(surfaces) and not beam_success:
        # 寻找当前阵面中复用值最低的可用阵元
        current_surface, theta = surfaces[current_index]
        usable_units = [u for u in current_surface.units.values() if u.id in all_units]
        start_unit = lowest_reuse_usable_unit(usable_units)

        if not start_unit:
            # 如果当前阵面没有可用阵元，则切换到下一个阵面
            current_index += 1
            continue

        # 进行最短路径部署
        path = [start_unit.id]
        neibors = [n for n in env.get_unit_adj(start_unit.id) if n in all_units]
        path_target = path_surface_target[start_unit.father.id]  # 当前路径的目标长度
        # 检查最低复用度的邻接节点
        while neibors and len(path) < path_target:
            min_N = target_N
            min_target = max_target_count
            min_neighbor = None
            for neighbor_id in neibors:
                neighbor = env.get_RF_instance(neighbor_id)
                if neighbor.N <= min_N and min_target >= path_surface_target[neighbor.father.id]:
                    min_N = neighbor.N
                    min_target = path_surface_target[neighbor.father.id]
                    min_neighbor = neighbor
            path.append(min_neighbor.id)
            # 将新相邻节点加入待检查列表
            for unit_id in env.get_unit_adj(min_neighbor.id):
                if unit_id not in neibors and unit_id not in path and unit_id in all_units:
                    neibors.append(unit_id)
            neibors.remove(min_neighbor.id)
            path_target = min_target

        # 如果找不到路径，再次切换到下一个起始阵元，并把当前路径从可用阵元中移除
        if len(path) < path_target:
            for unit_id in path:
                del all_units[unit_id]
        else:  # 成功部署
            beam_success = True

    if beam_success:
        return path
    else:
        return None


def find_min_weight_path_heap(env, surfaces, Task, beam, target_N, required_units=None):
    all_units = []
    path_surface_target = {}
    heap = []
    all_usable_paths = []
    cheched_nodes = []
    usable_surface_ids = []

    for surface, theta in surfaces:
        usable_surface_ids.append(surface.id)
        for unit in surface.units.values():
            # 排除不可用的阵元
            if unit.N < target_N and not detect_unit_frequency_conflict(unit, beam):
                all_units.append(unit)
        # 如果给定了阵元数，则直接使用
        if required_units:
            target_count = required_units
        else:
            target_count = env.calc_N(beam, Task, theta, target_N)
            print(f"{surface.id}:target_count={target_count}")
        path_surface_target[surface.id] = target_count

    all_units = sorted(all_units, key=lambda x: x.N)
        # 进行最短路径部署
    for start_unit in all_units:
        if start_unit.id in cheched_nodes:
            continue
        heapq.heappush(heap, (start_unit.N,start_unit.id))
        path = []
        target_count = path_surface_target[start_unit.father.id]
        weight_sum = 0
        cheched_nodes.append(start_unit.id)
        cheched_nodes.extend(env.get_unit_adj(start_unit.id))
        while len(path) < target_count and heap:
            weight, min_node_id = heapq.heappop(heap)
            min_node = env.get_RF_instance(min_node_id)
            path.append(min_node_id)
            weight_sum += min_node.N
            new_target_count = path_surface_target[min_node.father.id]
            # 如果跨阵面，则需要更新目标路径长度
            if target_count < new_target_count:
                target_count = new_target_count
            for neighbor_id in env.get_unit_adj(min_node_id):
                neighbor = env.get_RF_instance(neighbor_id)
                if (neighbor.N < target_N and neighbor_id not in path and neighbor.father.id in usable_surface_ids
                        and (neighbor.N + path_surface_target[neighbor.father.id] * 10, neighbor_id) not in heap):
                    heapq.heappush(heap, (neighbor.N + path_surface_target[neighbor.father.id] * 10, neighbor_id))

        if len(path) == target_count:
            all_usable_paths.append((path, weight_sum))
    min_weight_sum = float('inf')
    min_path = []
    min_path_target = float('inf')
    for path, weight_sum in all_usable_paths:
        if weight_sum < min_weight_sum and len(path) <= min_path_target:
            min_weight_sum = weight_sum
            min_path = path
            min_path_target = len(path)

    if min_path:
        return min_path
    else:
        return None  # 无法找到满足条件的路径


def select_next_surface(surface_list, current_index):
    """
    根据当前序号从可选择的阵面列表中选择下一个阵面。
    :param surface_list: 可选择的阵面列表（按优先级从高到低排序）
    :param current_index: 当前选中的阵面序号
    :return: 更新后的 surface_list 和 current_index 或返回错误信息
    """
    # 如果当前序号已经是最后一个，则返回错误信息
    if current_index >= len(surface_list) - 1:
        print("Error: No more available surfaces to choose from.")
        return None, current_index  # 返回 None 表示分配失败

    # 否则选择下一个阵面
    current_index += 1
    next_surface = surface_list[current_index]
    print(f"Switching to next surface: {next_surface}")

    return surface_list, current_index


def meta_task_chain_deploy(meta_task, meta_tasks, index, servers, individual):
    # 嵌套随机找到合理解
    servers_key_list = list(servers.keys())
    random.shuffle(servers_key_list)
    for server_key in servers_key_list:
        server = servers[server_key]
        if can_assign(meta_task, server):
            server.server_run(meta_task, test=True)
            if index == len(meta_tasks) - 1:
                individual.append({"metatask": meta_task, "server": server_key})
                return True
            else:
                success = meta_task_chain_deploy(meta_tasks[index + 1], meta_tasks, index + 1, servers, individual)
            if success:
                individual.append({"metatask": meta_task, "server": server_key})
                return True
            else:
                server.server_release(meta_task, test=True)
                continue
    if individual == []:
        return False


def create_individual(meta_tasks, servers):
    # 注意这里要使用tempenv
    temp_servers = copy.deepcopy(servers)
    individual = []
    meta_tasks = list(meta_tasks.values())
    meta_task_chain_deploy(meta_tasks[0], meta_tasks, 0, temp_servers, individual)
    individual.reverse()
    return individual


# 检查是否可以分配任务到服务器
def can_assign(metatask, server):
    return server.idle_cores >= metatask.core and server.idle_gpus >= metatask.GPU and depends_and_repels(metatask,
                                                                                                          server)


# TODO：计算服务器负载均衡度
# 适应度函数
def fitness_function(individual, servers, k=0.9):
    """
    负载均衡优化
    k分配GPU和CPU的权重,k(0-1)
    """

    temp_servers = copy.deepcopy(servers)

    cpu_loads = []
    gpu_loads = []
    individual_deploy(individual, temp_servers)
    for server in temp_servers.values():
        if server.cpu:
            cpu_loads.append(server.idle_cores)
        if server.gpu:
            gpu_loads.append(server.idle_gpus)
    cpu_var = statistics.variance(cpu_loads)
    gpu_var = statistics.variance(gpu_loads)
    fit = k * cpu_var + (1 - k) * gpu_var

    return fit


"""在使用遗传算法（Genetic Algorithm, GA）进行部署或优化之前，需要有一些初始条件或初始值。
遗传算法是一种搜索启发式算法，它模拟自然选择和遗传学的原理来寻找最优或近似最优的解。
然而，这个搜索过程是基于一个初始种群（population）开始的，而不是从完全空白的状态开始。"""


# 遗传算法函数
def genetic_algorithm(meta_tasks, servers, pop_size=40, generations=100):
    random.seed(time.time())
    # 初始化种群
    mating_size = pop_size // 2
    elder_size = pop_size - mating_size
    population = []
    for _ in range(pop_size):
        individual = create_individual(meta_tasks, servers)
        if individual:
            population.append(individual)
        else:
            print("Error: No individual can be created.")
            return None  # 无任何方案
    # 进化过程
    for gen in range(generations):
        # 计算适应度并排序
        population = sorted(population, key=lambda ind: fitness_function(ind, servers))

        mating_pool = population[:(pop_size // 2)]  # 适应度较高的前一半作为交配池
        # 交叉
        new_population = []
        # print("new_population:",new_population)
        trys = 0
        while len(new_population) < mating_size:

            parent1, parent2 = random.sample(mating_pool, 2)
            if len(parent1)==1:
                new_population.append(parent1)
                continue

            if trys > 10:
                trys = 0
                new_population.append(parent1)
                continue
            temp_servers1 = copy.deepcopy(servers)

            crossover_point = random.randint(1, len(parent1) - 1)  # 随机选择一个交叉点
            child = parent1[:crossover_point] + parent2[crossover_point:]  # 通过交叉产生子代个体
            # TODO：check
            if individual_deploy(child, temp_servers1):
                # 子代个体部署成功
                trys = 0

                new_population.append(child)
            else:

                trys += 1
                continue
            # 子代个体添加到 新的种群中
        # 新种群中保留上一代最优个体
        new_population.extend(population[:elder_size])

        # 最优个体直接进入新的种群
        for individual in new_population:
            if random.random() < 0.1:  # 以0.1的概率进行变异
                mutation_point = random.randint(0, len(individual) - 1)  # 随机选择一个变异点
                old_server_key = individual[mutation_point]['server']
                server_list = list(servers.keys())
                server_list.remove(old_server_key)
                random.shuffle(server_list)
                for server_key in server_list:
                    temp_servers2 = copy.deepcopy(servers)
                    individual[mutation_point]['server'] = server_key
                    if individual_deploy(individual, temp_servers2):
                        break
                # 变异失败，选择原先的server_key
                individual[mutation_point]['server'] = old_server_key
        population = sorted(new_population, key=lambda ind: fitness_function(ind, servers))
    # 按适应度从最优到最次排序并返回

    return population[0]


# 元任务冲突依赖检测
def depends_and_repels(meta_task, server):
    ongoing_meta_tasks = server.ongoing_meta_tasks

    if meta_task.depends[0]:
        for depend in meta_task.depends:
            if depend not in[meta_task1.meta_task_type for meta_task1 in ongoing_meta_tasks if
                                             meta_task1.father.task_id == meta_task.father.task_id]:
                return False
    if meta_task.repels[0]:
        repels = meta_task.repels
        for repel in repels:
            if repel in [meta_task.meta_task_type for meta_task in ongoing_meta_tasks]:
                return False
    return True


def individual_deploy(individual, servers):
    for item in individual:
        server_key = item['server']
        metatask = item['metatask']
        server = servers[server_key]
        if not server.server_run(metatask, test=True):
            return False
    return True

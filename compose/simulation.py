# 仿真程序

import argparse
import yaml
from logger import Logger
from env import Env
import json
from task_func import *
import bisect
import copy
from utils.ang_calc import ang_calc
from utils.calc import *
import schedule_func
import sympy as sp
import time as Time
import pickle

class Event:
    def __init__(self, simulator, timestamp, type, tasks, ):
        self.timestamp = timestamp
        self.type = type
        self.tasks = tasks  # 任务列表
        self.simulator = simulator

    def __str__(self):
        taskstr = ''
        for task in self.tasks:
            taskstr += task.__str__() + "\n"
        return "EVENT :\n{\ntimestamp: " + str(self.timestamp) + ' type: ' + self.type + ' tasks: ' + taskstr + "}\n"


class Simulator(Env):

    def __init__(self, config='config.yaml',args=None):
        self.args = args
        if args.yaml:
            config = args.yaml


        self.task_dict = {}
        with open(config, 'r', encoding='utf-8') as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)
        super().__init__(self.config)
        print('MESSAGE: Simulation start, loading confing file：%s \n' % config)

        if not args.method:
            self.args.method = 'greedy'







        # logger初始化


        self.event_list = []
        self.task_event_connect = {}

        if args.plan:
            plan_path = args.plan

        else:
            plan_path =self.config["planner"]["plan_path"]
        self.load_plan(plan_path)
        # 加载仿真plan
        self.logger = Logger(self.config["logger"], self,plan_path,config)


    def call(self, Event):  # 根据事件类型执行一个事件

        if Event.type == 'req':
            # 模拟请求到达，请求处理
            print("REQ", Event)
            temp=[]
            for task in Event.tasks:
                bisect.insort(temp, task, key=lambda x: x.priority)
            for task in temp:
                time1 = Time.perf_counter()
                res, message = self.run(task)

                if res:
                    time2=Time.perf_counter()
                    time=time2-time1
                    # if 分配成功，生成终止event:
                    self.logger.log(Event.timestamp, Event.type, "SUCCESS", task,time)
                    if task.lifetime != float('inf'):#非无限执行
                        endtime = task.start_time + task.lifetime
                        # 关联任务和release event
                        e = self.add_event(endtime, 'release', [task])
                        self.task_event_connect[task.task_id] = e
                else:
                    self.logger.log(Event.timestamp, Event.type, "FAILED %s" % message, task)
            return True


        elif Event.type == 'kill':
            # 模拟任务被终止
            # TODO
            print("KILL TASK ", Event)
            for task in Event.tasks:

                if task.status == 1:
                    self.release(task)
                    e = self.task_event_connect[task.task_id]
                    self.event_list.remove(e)
                    self.logger.log(Event.timestamp, Event.type, "SUCCESS", task)
                else:
                    self.logger.log(Event.timestamp, Event.type, "ERROR: cannot kill task , task not running", task)
                    print("ERROR: cannot kill task %s, task not running.\n" % task.task_id)
            return True
        elif Event.type == 'release':
            # 模拟任务结束释放
            print("RELEASE TASK ", Event)
            self.release(Event.tasks[0])
            self.logger.log(Event.timestamp, Event.type, "SUCCESS", Event.tasks[0])
            return True
        elif Event.type == 'rob':
            # 模拟高优先级抢占
            print("rob task", Event)
            return True
        else:
            print("unknown event type")
            return False

    def load_plan(self, plan_file):
        # 加载plan文件
        print('MESSAGE: Loading plan file：%s' % plan_file)

        plans = json.load(open(plan_file, 'r'))
        self.plan = plans
        for data in self.plan:
            if data['act_type'] == 'req':
                tasks = data['tasks']
                temp = []
                # 创建任务实例

                for task_info in tasks:
                    self.task_dict[task_info['task_id']] = Task(**task_info, TASKS_CONFIG=self.config['Tasks'])
                    temp.append(self.task_dict[task_info['task_id']])
                self.add_event(data['act_time'], 'req', temp)
            elif data['act_type'] == 'kill':
                tasks_ids = data['tasks']
                temp = []
                for task_id in tasks_ids:
                    temp.append(self.task_dict[task_id])
                self.add_event(data['act_time'], 'kill', temp)
        print('Done\n')

    def add_event(self, timestamp, type, tasks):
        e = Event(self, timestamp, type, tasks)
        bisect.insort(self.event_list, e, key=lambda x: x.timestamp)
        return e

    def simulate(self):
        print("MESSAGE: Simulation start")
        start = Time.perf_counter()

        while len(self.event_list) > 0:

            event = self.event_list[0]
            result = self.call(event)
            if result:
                self.event_list.pop(0)

        end = Time.perf_counter()
        runtime=end-start
        self.logger.conclude(runtime,self.args)
        self.logger.save()
        print("MESSAGE: Simulation end")

    def calc_N(self, beam, Task, angle, target_N):
        # TODO:计算阵元数N

        F = beam.F  # 频率
        S = beam.S  # 信号强度

        R = Task.kwargs['R'] if 'R' in Task.kwargs.keys() else None  # 接收天线距离
        task_type = Task.task_type  # 任务类型
        beam_type = beam.beam_type  # 波束类型
        N = target_N  # 设定的最大复用数
        P_max = 0  # 最大功率
        M = sp.symbols('M')
        G = 10 * sp.log(M) + 6.5 + 1.5 * 10 * math.log10(abs(math.cos(angle)))
        P = P_max - 10 * math.log10(N) - min(10 * math.log10(N), 10)
        GT = 10 * sp.log(M) + 6.5 + 1.5 * 10 * math.log10(abs(math.cos(angle))) - 22.8
        match task_type:
            case 1:
                # 卫星-测控
                if beam_type == "R":
                    EP = sp.Eq(128.54 + G + P - 20 * math.log(F, 10) - 10 * math.log(S, 10) - 13, 10)
                    EP_solution = sp.solve(EP, M)
                else:
                    EP = sp.Eq(163.54 + GT - 20 * math.log10(F) - math.log10(S) - 13, 10)
                    EP_solution = sp.solve(EP, M)
            case 2:
                # 卫星-数传
                if beam_type == "R":
                    EP = sp.Eq(128.54 + G + P - 20 * math.log10(F) - 10 * math.log10(S) - 5, 10)
                    EP_solution = sp.solve(EP, M)
                else:
                    EP = sp.Eq(163.54 + GT - 20 * math.log10(F) - math.log10(S) - 5, 10)
                    EP_solution = sp.solve(EP, M)
            case 3:
                # 无人机-测控
                if beam_type == "R":
                    EP = sp.Eq(140.1 + G + P - 20 * math.log10(F) - 10 * math.log10(S) - 13, 10)
                    EP_solution = sp.solve(EP, M)
                else:
                    EP = sp.Eq(148.1 + GT - 20 * math.log10(F) - math.log10(S) - 13, 10)
                    EP_solution = sp.solve(EP, M)
            # case 4:
            #     # 无人机-数传
            #     if beam_type == "R":
            #         EP = sp.Eq(140.1 + G + P - 20 * math.log10(F) - 10 * math.log10(S) - 5, 10)
            #         EP_solution = sp.solve(EP, M)
            #     else:
            #         EP = sp.Eq(148.1 + GT - 20 * math.log10(F) - math.log10(S) - 5, 10)
            #         EP_solution = sp.solve(EP, M)
            case 5:
                # 有源定位
                Ts = 22.4
                theta_ = 0
                lambda_ = 0.130434783
                tau = 0.1
                Ls = 2
                K = -228.6
                snr = 8
                Pt = P_max - 10 * math.log(N, 10) - min(10 * math.log(N, 10), 10)
                G = M * (10 ** 0.65) * (math.cos(angle) ** 1.5)
                EP = sp.Eq((db2value(Pt) * (tau) * (G ** 2) * (lambda_ ** 2) * db2value(theta_) / (
                        (4 * math.pi) ** 3 * db2value(K) * db2value(Ts) * db2value(Ls) * db2value(snr))) ** 0.25, R)
                EP_solution = sp.solve(EP, M)

            case 6:
                # 无源定位
                EP = sp.Eq(126.87 + GT - 20 * math.log(R) - math.log(S) - 10, 10)
                EP_solution = sp.solve(EP, M)
            case 7:
                # 协同感知
                EP = sp.Eq(126.87 + GT - 20 * math.log(R) - math.log(S) - 10, 5)
                EP_solution = sp.solve(EP, M)
        if int(abs(EP_solution[0])) + 1<5:
            return 5
        else:
            return int(abs(EP_solution[0])) + 1

        # 任务计算函数，利用if来对不同type创建计算函数

    def schedule(self, Task, target_N=10):
        # case5 需要的参数
        case5_T_used_unit_num = None
        T_surface_theta, T_surface_phi = None, None
        # 调用 usable_surface 获取可用的阵面列表
        print("MESSAGE: Scheduling task {}".format(Task.task_id))

        tempenv = copy.deepcopy(self)

        schedule_res = {"antenna": [], "server": []}

        # 角度冲突检测

        if tempenv.angle_conflict_detect(Task):
            print("任务频角度冲突")
            return False, "angle conflict"

        surface_list = tempenv.usable_surface(Task.angle, Task.delta)["surfaces"]

        # 波束遍历部署

        usable_antenna = list(tempenv.Antennas.keys())

        for beam in Task.beams.values():
            print(f"scheduling Beam: {beam.beam_id}")
            current_index = 0  # 初始化从第一个最优阵面开始
            beam_succcess = False
            #有源定位不同波束需要在不同球面上
            if Task.task_type in [5, 6, 7] and beam.beam_type == "R":
                surface_list = [(surface, theta) for surface,theta in surface_list if surface.father.id in usable_antenna]
            # 尝试部署

            # 判断有源定位收发波束角度是否满足30°要求,将不满足要求的阵面剔除
            if Task.task_type == 5 and beam.beam_type == "R":
                while current_index < len(surface_list):
                    current_surface, theta = surface_list[current_index]
                    theta1, phi1 = current_surface.angle
                    ang = ang_calc(theta1, phi1, T_surface_theta, T_surface_phi)  # 计算发射阵面与接收阵面的夹角
                    if ang >= 60/180*math.pi:
                        surface_list.pop(current_index)
                        continue
                    current_index += 1

            # 进行最短路径部署，逐个球体尝试部署
            for surface in surface_list:
                antenna_id = surface[0].father.id
                usable_surfaces = []
                for surface, theta in surface_list:
                    if surface.father.id == antenna_id:
                        usable_surfaces.append((surface, theta))
                if Task.task_type == 5 and beam.beam_type == "R":
                    # 有源定位收波束阵元数与发波束相同
                    required_units = case5_T_used_unit_num
                    path = function_map[self.args.method](tempenv, usable_surfaces, Task,
                                                                 beam, target_N, required_units)
                else:
                    if Task.task_type in [1,2] and beam.beam_type == "R" and self.args.method=='dp':
                        path =function_map["heap"](tempenv, usable_surfaces, Task,
                                                                     beam, target_N)
                    else:
                        path = function_map[self.args.method](tempenv, usable_surfaces, Task,
                                                          beam, target_N)

                # if Task.task_type == 5 and beam.beam_type == "R":
                    # 有源定位收波束阵元数与发波束相同
                    # required_units = case5_T_used_unit_num
                    # path = schedule_func.find_min_weight_path_heap(tempenv, usable_surfaces, Task,
                    #                                              beam, target_N, required_units)
                # else:
                    # path = schedule_func.find_min_weight_path_heap(tempenv, usable_surfaces, Task,
                    #                                                beam, target_N)
                if not path:
                    print(f"ERROR:no available surface in antenna {antenna_id} found for {beam.beam_id}")
                else:# 成功部署
                    beam_succcess = True
                    print(f"Successfully allocated")
                    print(f"PATH for {beam.beam_id}:", [unit_id for unit_id in path])
                    # 记录分配方案
                    schedule_res["antenna"].append({"beam": beam, "unit_ids": path,"antenna_id":self.Antennas[antenna_id].name})
                    for unit_id in path:  # 测试部署
                        unit = tempenv.get_RF_instance(unit_id)
                        unit.on_goning_beams.append(beam)
                        unit.N += 1

                    if Task.task_type in [5, 6, 7] and beam.beam_type == "R":
                            # 非合作目标波束需要部署在不同球体上
                        usable_antenna.remove(antenna_id)
                    if Task.task_type == 5 and beam.beam_type == "T":
                        case5_T_used_unit_num = len(path)
                        current_surface = tempenv.get_RF_instance(path[0]).father
                        T_surface_theta = current_surface.angle[0]
                        T_surface_phi = current_surface.angle[1]
                    break

            if not beam_succcess:
                return False, "schedule antenna failed"


        # 服务器部署
        servers = tempenv.Server
        meta_tasks = Task.meta_tasks

        print('mts:', meta_tasks)
        # TODO：根据负载均衡度进行尝试部署

        schedule_list = schedule_func.genetic_algorithm(meta_tasks,servers)
        if schedule_list is None:
            return False, "schedule meta_task failed"
        else:
            print("genetic_algorithmn_schedule_list:",schedule_list)
            for task_assign in schedule_list:
                task_try = task_assign["metatask"]
                server_try = task_assign["server"]
                schedule_res["server"].append({"id": server_try, "metatask": task_try})



        # 如果波束和计算资源都满足，返回success信号
        Task.task_resources = schedule_res
        return True, "Schedule success"




    def run(self, Task):


        # schedule = self.schedule(Task)
        success, message = self.schedule(Task)
        if not success:
            return False, message

        try:
            antenna_sche = Task.task_resources['antenna']

            for unit_sche in antenna_sche:
                beam = unit_sche['beam']
                id = unit_sche['unit_ids'][0]
                antenna = self.Antennas[id[:2]]
                antenna.ongoing_beam_num += 1
                for unit in unit_sche['unit_ids']:
                    unit = self.get_RF_instance(unit)
                    unit.on_goning_beams.append(beam)
                    unit.N += 1

            server_sche = Task.task_resources['server']

            for metatask_sche in server_sche:
                server = self.get_server(metatask_sche['id'])
                success=server.server_run(metatask_sche['metatask'])
                if not success:
                    return False, "server run failed"
            Task.status = 1
            print("TASK RUNNING: Task {} is running".format(Task.task_id) + "\n")
            return True, None

        except Exception as e:
            message = "ERROR: Task {} failed to run".format(Task.task_id) + "\n" + str(e)
            print("ERROR: Task {} failed to run".format(Task.task_id) + "\n", e)

            return False, message

    def release(self, Task):

        antenna_sche = Task.task_resources['antenna']
        server_sche = Task.task_resources['server']
        for unit_sche in antenna_sche:
            beam = unit_sche['beam']
            id = unit_sche['unit_ids'][0]
            antenna = self.Antennas[id[:2]]
            antenna.ongoing_beam_num -= 1
            for unit in unit_sche['unit_ids']:
                unit = self.get_RF_instance(unit)
                unit.on_goning_beams.remove(beam)
                unit.N -= 1

        for schedule in server_sche:
            server = self.get_server(schedule['id'])
            server.server_release(schedule['metatask'])
        Task.status = 3
        print("MESSAGE: Task {} is released".format(Task.task_id) + "\n")
        return True

    def angle_conflict_detect(self, Task):
        """
        检查当前任务角度是否与其他任务同频,返回True表示有冲突
        """

        for beam1 in Task.beams.values():
            f1 = beam1.F
            # 遍历所有正在运行的任务，检查它们波束频段和方向
            for task in self.task_dict.values():
                # 时间冲突分信息
                if task.status == 1 and task!=Task:#任务内部不会冲突
                    for beam2 in task.beams.values():
                        f2 = beam2.F
                        # 空间冲突分析
                        if ang_calc(*beam1.angle, *beam2.angle) <= math.pi / 1800 and abs(f1 - f2) < 5:
                            return True
        return False

    def unit_frequency_conflict_detect(self, Task):
        """
        检查当前任务是否与阵元频率冲突,返回True表示有冲突
        """
        # 遍历所有正在运行的任务，检查它们阵元频率
        T_antenna_sche = Task.task_resources['antenna']
        for beam_sche in T_antenna_sche:  # 遍历任务的每个波束分配方案
            F = beam_sche['beam'].F  # 得到波束频率
            for unit_id in beam_sche['unit_ids']:  # 该波束的阵元
                unit = self.get_RF_instance(unit_id)  # 得到阵元实例
                for beam in unit.on_goning_beams:  # 遍历
                    # 该阵元正在运行的波束
                    if abs(F - beam.F) < 5:  # 检测频率冲突
                        return True

        return False

    def task_conflict_detect(self, Task):
        # 在当前环境中检测任务是否与其他任务冲突，包括角度冲突和阵元频率冲突

        if self.angle_conflict_detect(Task):
            print("任务角度同频冲突")
            return True
        if self.unit_frequency_conflict_detect(Task):
            print("任务阵元频率冲突")
            return True
        else:
            return False

        # 检查阵元检查当前阵元频率是否占用

    def usable_surface(self, angle, delta):
        #  计算可用的表面
        # angle是Task方向 delta是参数，返回阵面与任务夹角小于delta的阵面列表
        delta_rad = delta * math.pi / 180
        surfaces = []
        for antenna in self.Antennas.values():  # 遍历天线
            for surface_num in range(1, antenna.surface_num + 1):  # 遍历天线上的阵面

                type = int(antenna.type)  # 阵面类型
                if type == 1:
                    id = "{:}:{:02d}".format(antenna.id, surface_num)  # 阵面id
                else:
                    id = "{:}{:02d}".format(antenna.id, surface_num)  # 阵面id
                surface = self.get_RF_instance(id, type)  # 获取阵面实例
                print(surface)

                theta1, phi1 = surface.angle
                theta2, phi2 = angle
                ang = ang_calc(theta1, phi1, theta2, phi2)  # 计算阵面与任务的夹角
                temp = (surface, ang)

                if delta_rad >= ang:
                    bisect.insort(surfaces, temp, key=lambda x: x[1])  # 夹角满足参数限制时阵面可用

        return {"delta": delta, "surfaces": surfaces}
function_map = {
            'heap': schedule_func.find_min_weight_path_heap,
            'dp': schedule_func.find_min_weight_path_dp,
            'greedy': schedule_func.find_min_weight_path_greedy,
        }
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate the environment')
    parser.add_argument('--method','-m', type=str, help='选择算法', required=False,choices=['greedy','dp','heap'])
    parser.add_argument('--load', '-l', type=str, help='选择断点', required=False)
    parser.add_argument('--yaml', '-y', type=str, help='配置文件路径' )
    parser.add_argument('--plan', '-p', type=str, help='仿真计划文件路径')
    args = parser.parse_args()
    if args.load:
        picklef = open(args.load, 'rb')
        sim = pickle.load(picklef)
        if args.method:
            sim.args.method=args.method
        if args.plan:
            path=args.plan
        else:
            path=sim.config["planner"]["plan_path"]
        sim.load_plan(args.plan)
        sim.logger = Logger(sim.config["logger"], sim, path, args.yaml if args.yaml else 'config.yaml')


        sim.simulate()
    else:
        sim = Simulator(args=args)
        sim.simulate()
    # sim.conflict_detect()

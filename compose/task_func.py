# 任务模型
class Task:
    """
    任务模型
    """

    def __init__(self, task_id, task_type, start_time, lifetime, priority, angle, F ,TASKS_CONFIG, **kwargs):
        self.R_beam_num = None
        self.T_beam_num = None
        self.delta=90
        self.task_id = task_id
        self.priority = priority
        self.task_type = task_type
        self.kwargs = kwargs
        self.angle = angle
        self.start_time = start_time
        self.lifetime = lifetime
        # 任务波束类型，支持自定义类型的任务
        self.beams = {}#beam字典
        self.task_name = TASKS_CONFIG[task_type]['task_name']
        self.init_beam( task_type,TASKS_CONFIG,F, kwargs)
        self.meta_tasks = self.init_meta_tasks(TASKS_CONFIG, task_type)
        # 运行状态
        # status={0:'unscheduled',1:'running',2:'robbed',3:'finished',4:'failed',5:'killed'}
        self.status = 0  # 0:未运行
        self.task_resources = {"antenna": {}, "server": []}
    # 根据调度方案占用环境资源
    def init_beam(self, task_type, TASKS_CONFIG, F,kwargs):
        beams_data = TASKS_CONFIG[task_type]['beams']
        if beams_data["R_beam_num"] == -1:
            self.R_beam_num = kwargs['R_beam_num']
        else:
            self.R_beam_num = beams_data["R_beam_num"]
        if beams_data["T_beam_num"] == -1:
            self.T_beam_num = kwargs['T_beam_num']
        else:
            self.T_beam_num = beams_data["T_beam_num"]
        for i in range(self.T_beam_num):
            self.beams[f"T{i + 1}"] =Beam(f"T{i + 1}","T",F["T"][i][0],F["T"][i][1],self,self.angle)
        for i in range(self.R_beam_num):
            self.beams[f"R{i + 1}"] = Beam(f"R{i + 1}", "R", F["R"][i][0],F["R"][i][1],self,self.angle)



    def init_meta_tasks(self, config, task_type):
        temp = {}
        id = 0
        task = config[task_type]
        for meta_task in task['meta_tasks']:
            x = MetaTask(**meta_task, father=self, meta_task_id=str(self.task_id) + "/" + f"{id:02d}")
            temp[f"{id:02d}"] = x
            id += 1
        return temp

    def __str__(self):  # 打印任务信息
        return "Task ID:{}, Type:{}, Start Time:{}, Lifetime:{}, Status:{}".format(self.task_id, self.task_type,
                                                                                   self.start_time, self.lifetime,
                                                                                   self.status)
class Beam:
    def __init__(self, beam_id, beam_type, F,S,father,angle):
        self.beam_id = beam_id
        self.beam_type = beam_type
        self.F = F
        self.S = S
        self.father = father# 所属任务
        self.angle=angle

class MetaTaskType:
    """
    元任务类型
    type: 元任务类型
    core: 元任务需要的CPU核心数
    GPU: 元任务需要的GPU核心数
    relation: 元任务其他元任务的依赖排斥关系
    """

    def __init__(self, meta_task_name, meta_task_type, GPU, core, depends, repels):
        self.meta_task_name = meta_task_name
        self.meta_task_type = meta_task_type
        self.core = core
        self.GPU = GPU
        self.depends = depends
        self.repels = repels



class MetaTask(MetaTaskType):
    """
    元任务实例
    """
    def __init__(self, meta_task_id, father, meta_task_name, meta_task_type, GPU, core, depends, repels):
        super().__init__(meta_task_name, meta_task_type, GPU, core, depends, repels)
        self.meta_task_id = meta_task_id
        self.father = father
        self.running_on = None
        self.status = 0  # 0:未运行 1:运行中 2:运行完成 3:失败或终止 4:被抢占

    def __str__(self):  # 打印元任务信息
        return "MetaTask ID:{}, Name:{}, Type:{}, Status:{}".format(self.meta_task_id, self.meta_task_name,
                                                                    self.meta_task_type, self.status)


if __name__ == '__main__':
    pass

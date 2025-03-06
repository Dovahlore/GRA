import json
import random


class Planner():
    """
       生成仿真计划
    """

    def __init__(self, path="plan.json"):
        self.task_current_id = 0
        self.plan = []
        self.path = path
        print("Planner initialized,path： %s" % self.path)
        self.file = open(self.path, 'w')

    def kill_task(self, act_time, tasks):
        """
        :param tasks id列表 e.g. [1,3]
        :param act_time:
        """
        self.plan.append({"tasks": tasks, "act_type": "kill", "act_time": act_time})

    def new_req(self, tasks, act_time=None):
        if not act_time:
            act_time = random.randint(0, 500)
        for task in tasks:
            task["start_time"] = act_time
        self.plan.append({"tasks": tasks, "act_type": "req", "act_time": act_time})

    def save_plan(self):
        json.dump(self.plan, self.file, indent=4)
        self.file.close()

    def create_task(self, task_type, F, task_id=None, priority=None, lifetime=None, angle=None, **kwargs):
        """
        :param task_id: 任务id
        :param priority: 优先级
        :param task_type: 任务类型
        :param start_time: 任务开始时间
        :param lifetime: 任务生命周期
        :param angle: 任务角度
        :param F: 任务波束
        :param kwargs: 其他参数
        :return: 任务字典
        """
        # utils
        self.task_current_id += 1
        task = {"task_id": task_id if task_id else self.task_current_id,
                "priority": priority if priority else random.randint(1, 3),
                "task_type": task_type,
                "start_time": None,
                "lifetime": lifetime if lifetime else float('inf'),
                **kwargs,
                "angle": angle if angle else (random.randint(-180, 180), random.randint(10, 90)), "F": F}
        return task


if __name__ == "__main__":
    planner = Planner(path="plan.json")
    for i in range(16):
        t2 = planner.create_task(1, F={'R': [(random.randint(2200, 2400),10000000) ], "T": [(random.randint(2200, 2400), 100000000)]})
        planner.new_req([t2])
    for i in range(17):
        t3 = planner.create_task(2, F={'R': [(random.randint(2200, 2400), 10000000)]})
        planner.new_req([t3])
    for i in range(17):
        t4 = planner.create_task(3, F={'R': [(random.randint(2200, 2400), 10000000)], "T": [(random.randint(2200, 2400), 10000000)]})
        planner.new_req([t4])
    for i in range(16):
        t1 = planner.create_task(5, F={'R': [(2300, None), (2300, None), (2300, None)], "T": [(2300, None)]}, R=10000)
        planner.new_req([t1])
    for i in range(17):
        t6 = planner.create_task(6, F={'R': [(random.randint(2200, 2400), 10000000), (random.randint(2100, 2400), 10000000), (random.randint(2300, 2400),10000000), (random.randint(2200, 2400),10000000)]}, R=10)
        planner.new_req([t6])
    for i in range(17):
        t7 = planner.create_task(7, F={
            'R': [(random.randint(2200, 2400), 10000000), (random.randint(2200, 2400), 10000000),
                  (random.randint(2200, 2400), 10000000), (random.randint(2200, 2400), 10000000)]}, R=10,
                                 R_beam_num=4)
        planner.new_req([t7])
    # # for i in range(40):   #测试支持波束数40波束
    #
    #     t = planner.create_task(3 ,F={'R': [(random.randint(2100,2400), 8000000)], "T": [(random.randint(2200,2400), 8000000)]})
    #     planner.new_req( [t])
    # for i in range(60):   #测试支持波束数40波束
    #
    #     t = planner.create_task(7 ,F={'R': [(random.randint(2100,2200), 80000000),(random.randint(2100,2200), 80000000),(random.randint(2100,2200), 80000000),(random.randint(2100,2200), 80000000)], }, lifetime=100,R=10,R_beam_num=4)
    #     planner.new_req( [t])
    planner.save_plan()
    print("Done")

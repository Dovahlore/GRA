import copy
class Server:
    """
    服务器类实例
    """
    def __init__(self, para,copy=False):
        self.id = para['id']
        if not copy:
            print("loading server %s"%self.id)
        self.cpu = para['cpu_core_num']
        self.gpu = para['GPU_num']
        self.ongoing_meta_tasks = []
        self.idle_cores = self.cpu
        self.idle_gpus = self.gpu
    def get_status(self):
        print("SERVER STATUS:server %s info:"%self.id)
        print("cpu cores: %d, idle cores: %d"%(self.cpu, self.idle_cores))
        print("gpu cards: %d, idle cards: %d"%(self.gpu, self.idle_gpus))
        for meta_task in self.ongoing_meta_tasks:
            print("meta task %s is onging"%meta_task.meta_task_id)
    def __deepcopy__(self, memodict={}):
        Server_copy = Server(para={'id':self.id, 'cpu_core_num':self.cpu, 'GPU_num':self.gpu},copy=True)
        Server_copy.idle_cores = self.idle_cores
        Server_copy.idle_gpus = self.idle_gpus
        Server_copy.ongoing_meta_tasks = copy.copy(self.ongoing_meta_tasks)
        return Server_copy


    def server_run(self,meta_task,test=False):

        if self.idle_cores >= meta_task.core and self.idle_gpus >= meta_task.GPU:
            inner_ongoing_meta_types=[metatask.meta_task_type for metatask in self.ongoing_meta_tasks if metatask.father.task_id==meta_task.father.task_id]
            grand_ongoing_meta_types=[metatask.meta_task_type for metatask in self.ongoing_meta_tasks]
            for depend in meta_task.depends:
                if not depend :
                    break
                elif depend not in inner_ongoing_meta_types:
                    if not test:
                        print("SERVER WARNING:server %s cannot run meta task %s because of depend condition"%(self.id, meta_task.meta_task_id))

                    return False
            for repel in meta_task.repels:
                if repel in grand_ongoing_meta_types:
                    if not test:
                        print("SERVER WARNING:server %s cannot run meta task %s because of repel condition" % (
                        self.id, meta_task.meta_task_id))

                    return False
            # all depends and repels conditions are satisfied
            self.ongoing_meta_tasks.append(meta_task)
            self.idle_cores -= meta_task.core
            self.idle_gpus -= meta_task.GPU

            meta_task.status=1
            if not test:
                meta_task.running_on = self.id
                print("SERVER RUN:server %s running meta task %s"%(self.id, meta_task.meta_task_id))
            return True



    def server_release(self,meta_task,test=False):
        if meta_task in self.ongoing_meta_tasks:
            self.ongoing_meta_tasks.remove(meta_task)
            self.idle_cores += meta_task.core
            self.idle_gpus += meta_task.GPU
            meta_task.status=3
            if not test:
                print("SERVER RELEASE:server %s released meta task %s\n"%(self.id, meta_task.meta_task_id))
    def __str__(self):
        return "ServerID:%s,server idle cores: %d, idle gpus: %d, onging meta tasks num: %d"%(self.id,self.idle_cores, self.idle_gpus, len(self.ongoing_meta_tasks))
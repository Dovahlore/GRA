import json
class Antenna:
    """
    天线类实例
    """

    def __str__(self):
        return "antenna id:%s name:%s" % (self.id, self.name)

    def __init__(self, para):

            self.type = int(para['type'])
            self.name = para['name']
            print("loading %s" % self.name)
            self.id = para['antenna_id']
            self.surface_num = int(para['surface_num'])
            self.unit_num = int(para['unit_num'])
            self.adj_file_path = para['adj_file_path']
            self.config_file_path = para['config_file_path']
            self.type = para['type']
            # adj
            self.adj_matrix = json.load(open(self.adj_file_path, 'r'))  # dict邻接表
            # surfaces
            self.initilize_antenna_surface()
            self.ongoing_beam_num = 0


    def initilize_antenna_surface(self):
        """
        初始化天线面实例
        """
        temp_load = json.load(open(self.config_file_path, 'r'))["surfaces"]
        self.surfaces = {}
        for k, v in temp_load.items():
            self.surfaces[k]=Antenna_surface(k, self, v["units"], v["direction"])
        self.surface_num = len(self.surfaces)

    def get_ongoing_unit(self):
        num = 0
        # n = 0
        # print(self.surfaces)
        for i in self.surfaces.values():
            surface = i
            for unit in surface.units.values():
                if unit.N!=0:
                    num += 1
                    # n+=unit.N
        return num
class Antenna_surface:
    """
    天线面类实例
    """
    def __str__(self):
        return "surface id:%s" % self.id
    def __init__(self, id, father, unit_ids, angle):
        self.id = id
        self.father = father  # 父亲天线球

        self.angle = angle  # 天线面倾角
        self.units = {}
        for unit_id in unit_ids:
            self.units[unit_id]=Antenna_unit(unit_id, self)
        self.unit_num = len(self.units)


class Antenna_unit:
    """
    天线阵元类实例
    """
    def __str__(self):
        return "unit id:%s" % self.id
    def __init__(self, id, father):
        self.father = father
        self.id = id

        self.N = 0
        self.on_goning_beams = []



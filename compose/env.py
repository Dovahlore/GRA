# 资源环境初始化
# 天线编号01.02,03,04……两位
# 阵面编号0101,0102,0201……四位
# 阵元010101，010102，010201……六位
from antenna import *
from server import *


class Env():
    """
    读取yaml配置初始化波束和计算环境。优先编写波束环境。
    """

    def __init__(self, CONFIG):
        print("Initializing ENV")

        self.Antennas = self.initilize_antenna(CONFIG["env"])
        self.Server = self.initilize_server(CONFIG["env"])
        print(self.Antennas)
        print("Done\n")

    def initilize_antenna(self, data):
        """
        读取yaml配置.
        初始化创建天线
        """
        print("initilizing_antenna")

        self.antenna_num = data['Antenna_num']
        print("total antenna_num:", self.antenna_num)

        temp_antenna = {}
        for i in range(self.antenna_num):
            para = data['Antennas'][i]
            x = Antenna(para)

            temp_antenna[data['Antennas'][i]['antenna_id']] = x

        return temp_antenna

    def initilize_server(self, data):
        """
        读取yaml配置.
        初始化创建天线
        """
        print("initilizing_server")
        try:
            self.server_num = data['Server_num']
            print("total server_num:", self.server_num)

            temp_server = {}
            for i in range(self.server_num):
                para = data['Servers'][i]
                x = Server(para)
                temp_server[data['Servers'][i]['id']] = x
            return temp_server
        except Exception as e:
            print("initilize_server error ",e)
            return None


    def get_server(self, id):
        return self.Server[id]

    def get_RF_instance(self, id, type=1):
        """
        返回任意id对应实例，可输入天线，阵面，阵元id
        """
        if type == 0:  # 编号规则0
            lens = len(id)
            if lens == 2:
                try:
                    return self.Antennas[id]
                except:
                    print("antenna not found")
                    return None
            elif lens == 4:
                try:
                    antenna_id = id[:2]
                    return self.Antennas[antenna_id].surfaces[id]
                except:
                    print("surface not found")
                    return None
            elif lens == 6:
                try:
                    return self.Antennas[id[0:2]].surfaces[id[:4]].units[id]
                except:
                    print("unit not found")
                    return None
        elif type == 1:  # 编号规则1 antenna_id：surface_id：unit_id
            list = id.split(":")
            if len(list) == 4:
                # print("输入格式错误")
                return None
            elif len(list) == 3:
                antenna_id = list[0]
                surface_id = list[1]
                units_id = list[2]
                return self.Antennas[antenna_id].surfaces[antenna_id + ":" + surface_id].units[
                    antenna_id + ":" + surface_id + ":" + units_id]
            elif len(list) == 2:
                antenna_id = list[0]
                surface_id = list[1]
                return self.Antennas[antenna_id].surfaces[antenna_id + ":" + surface_id]
            elif len(list) == 1:
                antenna_id = list[0]
                return self.Antennas[antenna_id]

    def get_unit_adj(self, target):
        """
        输入id或者实例，返回邻接阵元
        返回邻接阵元id，如['010103', '010105', '010101', '010104', '010203']
        """
        target_id = ""
        if type(target) == str:
            target_id = target
            target = self.get_RF_instance(target)
        elif type(target) == Antenna_unit:
            target_id = target.id
        antenna = target.father.father
        adj = antenna.adj_matrix[target_id]
        return adj




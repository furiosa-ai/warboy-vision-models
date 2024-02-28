import datetime
import multiprocessing as mp
import os
import queue
import subprocess
import threading
import time

import cv2
from furiosa.device.sync import list_devices


class ResultViewer:
    def __init__(self):
        pass


class WarboyViewer:
    def __init__(self):
        self.state = True
        self.devices = list_devices()
        self.proc = threading.Thread(target=self.view_npu_utilization)

    def start(self):
        self.proc.start()

    def join(self):
        self.proc.join()

    def view_npu_utilization(self):
        last_pc = {}

        ts = time.time()
        while self.state:
            os.system("clear")
            timer = datetime.datetime.now()
            device_states = []
            for device in self.devices:
                device_name = str(device)
                per_counters = device.performance_counters()

                state = ""
                for per_counter in per_counters:
                    pe_name = str(per_counter[0])
                    cur_pc = per_counter[1]

                    if pe_name in last_pc:
                        warboy_util, comp_ratio, io_ratio = self.calculate_result(
                            cur_pc, last_pc[pe_name]
                        )
                        device_ = "{0:<9}".format(pe_name)
                        graph, util, comp, io = self.state_to_str(warboy_util, comp_ratio, io_ratio)
                        state += "{}:{} {} {} {}\n".format(device_, graph, util, comp, io)
                    last_pc[pe_name] = cur_pc

                if len(state) == 0:
                    device_ = "{0:<9}".format(device_name)
                    graph, util, comp, io = self.state_to_str(0.0, 0.0, 0.0)
                    state += "{}:{} {} {} {}\n".format(device_, graph, util, comp, io)
                device_states.append(state)

            print(f"Datetime: {timer}, Run-Time: {time.time()-ts}s")
            print("-" * 70)
            print("{0:9}  {0:<34} Util(%) Comp(%) I/O(%)".format("Device", "Status- Comp:▓  I/O:▒"))
            for device_state in device_states:
                print(device_state, end="\r")
            print("-" * 70)
            time.sleep(0.3)

    def calculate_result(self, cur_pc, last_pc):
        result = cur_pc.calculate_utilization(last_pc)

        warboy_util = result.npu_utilization()
        comp_ratio = result.computation_ratio()
        io_ratio = result.io_ratio()

        return warboy_util, comp_ratio, io_ratio

    def state_to_str(self, warboy_util, comp_ratio, io_ratio):
        comp_graph = (int(round(comp_ratio * warboy_util * 100 / 3))) * "▓"
        io_graph = (int(round(io_ratio * warboy_util * 100 / 3))) * "▒"
        graph = "[{0: <33}]".format(comp_graph + io_graph)
        util = "% 7.2f" % (warboy_util * 100)

        comp = "{0:>7}".format("-")
        io = "{0:>6}".format("-")
        if warboy_util != 0.0:
            comp = "% 7.2f" % (comp_ratio * 100)
            io = "% 6.2f" % (io_ratio * 100)

        return graph, util, comp, io

from furiosa.device import list_devices

class WARBOYDevice:
    def __init__(self):
        pass

    @classmethod
    async def create(cls):
        self = cls()
        self.warboy_devices = await list_devices()
        self.last_pc = {}
        self.time_count = 0
        return self

    async def __call__(self):
        power_info, util_info, temper_info, devices = await self._get_warboy_device_status(
        )
        self.time_count += 1
        return power_info, util_info, temper_info, self.time_count, devices

    
    async def _get_warboy_device_status(self,):
        status = [[] for _ in range(4)]

        for device in self.warboy_devices:
            warboy_name = str(device)
            device_idx = warboy_name[3:]
            per_counters = device.performance_counters()

            if len(per_counters) != 0:
                fetcher = device.get_hwmon_fetcher()
                temper = await fetcher.read_temperatures()
                peak_device_temper = int(str(temper[0]).split(" ")[-1]) // 1000
                power_info = str((await fetcher.read_powers_average())[0])
                p = int(float(power_info.split(" ")[-1]) / 1000000.0)
                
                status[0].append(p)
                status[2].append(peak_device_temper)
                status[3].append(device_idx)
                
            t_utils = 0.0
            for pc in per_counters:
                pe_name = str(pc[0])
                cur_pc = pc[1]

                if pe_name in self.last_pc:
                    result = cur_pc.calculate_utilization(self.last_pc[pe_name])
                    util = result.npu_utilization()
                    if not ("0-1" in pe_name):
                        util /= 2.0
                    t_utils += util

                self.last_pc[pe_name] = cur_pc

            if len(per_counters) != 0:
                t_utils = int(t_utils * 100.0)
                status[1].append(t_utils)
        return status
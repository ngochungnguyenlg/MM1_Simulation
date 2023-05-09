import numpy as np
import copy
import matplotlib.pyplot as plt



MAX_FLOAT = float('inf')

class MM1:
    def __init__(self, lamd, service_rate):
        self.lamd = lamd
        self.service_rate = service_rate
        self.inter_arr = 1/lamd
        self.serive_time = 1/service_rate
        self.num_served_package = 0

    def set_inter_arrival_time(self, val):
        self.inter_arr = val

    def set_inter_service_time(self, val):
        self.inter_arrival = val

    def initial_events(self, simulating_time):
        time = 0
        self.inter_arr_time_l = []
        self.arr_time_l = []
        self.wait_time_l = []
        self.service_time_l = []
        self.starting_process_time = []
        self.finish_time = []
        while (time < simulating_time):
            current_inter_arr_time = np.random.exponential(self.inter_arr)
            self.inter_arr_time_l.append(current_inter_arr_time)
            self.arr_time_l.append(time)
            self.wait_time_l.append(0)
            curr_service_time = np.random.exponential(self.serive_time)
            self.service_time_l.append(curr_service_time)
            self.starting_process_time.append(time)
            self.finish_time.append(time + curr_service_time)
            time += current_inter_arr_time

    def adjust_time_line(self):
        self.response_time = []
        for idx in range(1, len(self.starting_process_time)):
            self.starting_process_time[idx] = max(
                self.starting_process_time[idx], self.finish_time[idx-1])
            self.finish_time[idx] = self.starting_process_time[idx] + \
                self.service_time_l[idx-1]
            self.wait_time_l[idx] = self.starting_process_time[idx] - \
                self.arr_time_l[idx]
            self.response_time.append(
                self.finish_time[idx] - self.arr_time_l[idx])

    def MM1(self, simulating_time=100):

        self.initial_events(simulating_time)
        self.adjust_time_line()
        
        arrive_idx = 0
        start_idx = 0
        finish_idx = 0

        num_packages_in_system = 0
        num_packages_in_queue = 0
        nums_package = len(self.inter_arr_time_l)

        self.packages_q_over_time_l = []  # in queue
        self.packages_s_over_time_l = []  # in system
        while finish_idx < nums_package:
            arrival_time = self.arr_time_l[arrive_idx] if  arrive_idx< nums_package else MAX_FLOAT
            start_time = self.starting_process_time[start_idx] if start_idx< nums_package else MAX_FLOAT
            complete_time = self.finish_time[finish_idx]
            if arrival_time <= start_time and arrival_time <= complete_time:
                nums_change, numsq_change = 1, 1
                arrive_idx += 1
            elif start_time <= arrival_time and start_time <= complete_time:
                nums_change, numsq_change = 0, -1
                start_idx += 1
            else:
                nums_change, numsq_change = -1, 0
                finish_idx += 1
            num_packages_in_system += nums_change
            num_packages_in_queue += numsq_change
            self.packages_q_over_time_l.append(num_packages_in_queue)
            self.packages_s_over_time_l.append(num_packages_in_system)


    def figure1(self):
        simulating_time = 6000
        inter_arrival_time_config = [1.1, 1.5, 2.0,]
        waiting_l = []
        inter_arrival_time_l = []
        service_time_l = []
        for iat in inter_arrival_time_config:
            self.set_inter_arrival_time(iat)
            self.MM1(simulating_time = simulating_time)
            waiting_l.append(copy.deepcopy(self.wait_time_l))
            inter_arrival_time_l.append(copy.deepcopy(self.inter_arr_time_l))
            service_time_l.append(copy.deepcopy(
                self.response_time))  # response time

        
        for idx, iat in enumerate(inter_arrival_time_config):
            plt.figure(figsize= (14, 8))
            ax1 = plt.subplot(311)
            ax1.plot(waiting_l[idx], label=str(iat))
            ax1.set_ylabel("Waiting time in the queue", fontsize=12)
            
            ax2 = plt.subplot(312)
            ax2.plot(inter_arrival_time_l[idx], label=str(iat))
            ax2.set_ylabel("Inter arrival time", fontsize=12)

            ax3 = plt.subplot(313)
            ax3.plot(service_time_l[idx], label=str(iat))
            ax3.set_ylabel("service time", fontsize=12)
            plt.xlabel('Inter arrival time = {:.1f}'.format(
                inter_arrival_time_config[idx]), fontsize=12)
            plt.savefig('Inter arrival time = {:.1f}.png'.format(
                inter_arrival_time_config[idx]), dpi = 300 )
            # plt.show()
            plt.close()
    def figure2(self):
        simulating_time = 60
        inter_arrival_time_config = [1.1, 1.5, 2.0,]

        plt.figure(figsize= (14, 8))
        
        for idx, iat in enumerate(inter_arrival_time_config):
            self.set_inter_arrival_time(iat)
            self.MM1(simulating_time = simulating_time)
            plt.plot(self.packages_q_over_time_l, label = 'Inter arrival time = {:.1f}'.format(iat))
        
        plt.ylabel("Number task in queue over time")
        plt.xlabel('Timeline {:.1f} seconds'.format(simulating_time))
        plt.grid()
        plt.legend()
        plt.savefig("task in queue overtime", dpi=300)
        plt.show()
            
if __name__ == "__main__":
    mm1 = MM1(1, 1)
    mm1.figure1()
    mm1.figure2()

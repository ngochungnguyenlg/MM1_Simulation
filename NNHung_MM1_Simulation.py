import numpy as np
import copy
import matplotlib.pyplot as plt
import plotly.express as px
import plotly
import pandas as pd


MAX_FLOAT = float('inf')


class MM1:
    def __init__(self, lamd, service_rate):
        self.lamd = lamd
        self.service_rate = service_rate
        self.inter_arr = 1/lamd
        self.serive_time = 1/service_rate
        text_file = open('simulation_results.txt', 'w')
        text_file.close()

    def set_inter_arrival_time(self, val):
        self.inter_arr = val

    def set_inter_service_time(self, val):
        self.serive_time = val

    def initial_events(self, simulating_time):
        time = 0
        self.inter_arr_time_l = []
        self.arr_time_l = []
        self.wait_time_l = []
        self.service_time_l = []
        self.starting_process_time = []
        self.finish_time = []
        while (time <= simulating_time):
            current_inter_arr_time = np.random.exponential(self.inter_arr)
            self.inter_arr_time_l.append(current_inter_arr_time)
            self.arr_time_l.append(time)
            self.wait_time_l.append(0)
            curr_service_time = np.random.exponential(self.serive_time)
            self.service_time_l.append(curr_service_time)
            self.starting_process_time.append(time)
            self.finish_time.append(time + curr_service_time)
            time += current_inter_arr_time
        self.nums_package = len(self.inter_arr_time_l)

    def adjust_time_line(self):
        self.response_time = [self.service_time_l[0]]
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

        time_gap = 0
        now = 0
        pre = 0

        self.time_gap_l = []
        
        self.time_actual_event = []

        while finish_idx < nums_package:
            arrival_time = self.arr_time_l[arrive_idx] if arrive_idx < nums_package else MAX_FLOAT
            start_time = self.starting_process_time[start_idx] if start_idx < nums_package else MAX_FLOAT
            complete_time = self.finish_time[finish_idx]
            if arrival_time <= start_time and arrival_time <= complete_time:
                nums_change, numsq_change = 1, 1
                arrive_idx += 1
                now = arrival_time
            elif start_time <= arrival_time and start_time <= complete_time:
                nums_change, numsq_change = 0, -1
                start_idx += 1
                now = start_time

            else:
                nums_change, numsq_change = -1, 0
                finish_idx += 1
                now = complete_time

            time_gap = now - pre
            
            self.time_gap_l.append(time_gap)
            if len(self.time_gap_l) ==1:
                self.time_actual_event.append(0)
            else: 
                self.time_actual_event.append(time_gap + self.time_actual_event[-1])
            num_packages_in_system += nums_change
            num_packages_in_queue += numsq_change
            self.packages_q_over_time_l.append(num_packages_in_queue)
            self.packages_s_over_time_l.append(num_packages_in_system)
            pre = now
        self.simulation_printing(0, simulating_time)

    def box_plot(self, waiting_l, inter_arrival_time_l, service_time_l, titles=[1.1, 1.4, 2.0]):

        labels = ['Waiting time', 'Inter arrival time', 'Service time']

        # combine the whole time into one
        com_waiting_l = []
        com_inter_arrival_time_l = []
        com_service_time_l = []
        com_arrival_rate = []
        for i in range(len(waiting_l)):
            com_arrival_rate += [str(titles[i])]*len(waiting_l[i])
            com_waiting_l += waiting_l[i]
            com_inter_arrival_time_l += inter_arrival_time_l[i]
            com_service_time_l += service_time_l[i]

        com_index = list(range(0, len(com_arrival_rate)))

        df = {
            "index": com_index,
            labels[0]: com_waiting_l,
            labels[1]:  com_inter_arrival_time_l,
            labels[2]:  com_service_time_l,
            'Inter Arrival rate':  com_arrival_rate
        }
        df = pd.DataFrame(df, index=None)
        new_df = df.melt(
            id_vars=['index', 'Inter Arrival rate'], value_vars=labels)
        new_df.rename({'variable': 'Metrics', 'value': 'Time'},
                      axis=1, inplace=True)

        fig = px.box(new_df, y="Time", facet_col="Metrics", color="Inter Arrival rate",
                     boxmode="group", points='all')
        fig.update_layout(
            font=dict(
                family="Courier New, monospace",
                size=30,
                color="RebeccaPurple"
            )
        )
        return fig

    def simulation_printing(self, plot=False, time=100):
        sim_duration = self.finish_time[self.nums_package-1]
        inter_arrival_mean = np.mean(self.inter_arr_time_l)
        arrival_mean = 1/inter_arrival_mean
        waiting_mean = np.mean(self.wait_time_l)
        service_mean = np.mean(self.service_time_l)
        service_rate = 1/service_mean
        respone_mean = np.mean(self.response_time)
        nums_q_package_mean = np.sum(
            np.array(self.packages_q_over_time_l)*np.array(self.time_gap_l))/sim_duration
        nums_s_package_mean = np.sum(
            np.array(self.packages_s_over_time_l)*np.array(self.time_gap_l))/sim_duration
        text_file = open('simulation_results.txt', 'a')

        print(
            "---------simulation result of inter arrival rate (IAR)={:.1f} and in {:d} seconds ------------".format(self.inter_arr, time))
        text_file.write(
            "---------simulation result of inter arrival rate (IAR)={:.1f} and in {:d} seconds ------------\n".format(self.inter_arr, time))
        print("simulation duration {:1f}".format(sim_duration))

        print("Inter arrival time mean {:3f}".format(inter_arrival_mean))
        print("Arrival time mean {:3f}".format(arrival_mean))
        print("Waiting time mean {:3f}".format(waiting_mean))
        print("Service time mean {:3f}".format(service_mean))
        print("Service rate mean {:3f}".format(service_rate))
        print("Response time mean {:3f}".format(respone_mean))
        print("Number of packages in queue {:3f}".format(nums_q_package_mean))
        print("Number of packages in system {:3f}".format(nums_s_package_mean))

        text_file.write("simulation duration {:1f}\n".format(sim_duration))

        text_file.write(
            "Inter arrival time mean {:3f}\n".format(inter_arrival_mean))
        text_file.write("Arrival time mean {:3f}\n".format(arrival_mean))
        text_file.write("Waiting time mean {:3f}\n".format(waiting_mean))
        text_file.write("Service time mean {:3f}\n".format(service_mean))
        text_file.write("Service rate mean {:3f}\n".format(service_rate))
        text_file.write("Response time mean {:3f}\n".format(respone_mean))
        text_file.write(
            "Number of packages in queue {:3f}\n".format(nums_q_package_mean))
        text_file.write(
            "Number of packages in system {:3f}\n".format(nums_s_package_mean))
        text_file.close()
        if plot:
            fig = self.box_plot([self.wait_time_l], [self.inter_arr_time_l], [
                self.response_time], titles=[self.serive_time])
            plotly.offline.plot(
                fig, filename='./MM1_result_for_{}.html'.format(self.inter_arr))
            fig.write_image(
                './MM1_result_for_{}.png'.format(self.inter_arr), scale=6, width=2160, height=1080)

    def figure1(self):
        simulating_time = 6000
        inter_arrival_time_config = [1.1, 1.5, 2.0,]
        waiting_l = []
        inter_arrival_time_l = []
        service_time_l = []
        for iat in inter_arrival_time_config:
            self.set_inter_arrival_time(iat)
            self.set_inter_service_time(1)
            self.MM1(simulating_time=simulating_time)
            waiting_l.append(copy.deepcopy(self.wait_time_l))
            inter_arrival_time_l.append(copy.deepcopy(self.inter_arr_time_l))
            service_time_l.append(copy.deepcopy(
                self.response_time))  # response time
        fig = self.box_plot(waiting_l, inter_arrival_time_l, service_time_l)
        plotly.offline.plot(
            fig, filename='./MM1_different_inter_arr_rate.html')
        fig.write_image('./MM1_different_inter_arr_rate.png',
                        scale=6, width=2160, height=1080)
        for idx, iat in enumerate(inter_arrival_time_config):
            plt.figure(figsize=(14, 8))
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
                inter_arrival_time_config[idx]), dpi=300)
            # plt.show()
            plt.close()

    def figure2(self):
        simulating_time = 60
        inter_arrival_time_config = [1.1, 1.5, 2.0]

        plt.figure(figsize=(14, 8))

        for idx, iat in enumerate(inter_arrival_time_config):
            self.set_inter_arrival_time(iat)
            self.set_inter_service_time(1)
            self.MM1(simulating_time=simulating_time)
            even_l = [0]*len(self.time_gap_l)
            plt.plot(self.time_actual_event,self.packages_q_over_time_l,
                     label='Inter arrival time = {:.1f}'.format(iat))

        plt.axvline(x=simulating_time,ls='--', color = 'red')
        
        plt.ylabel("Number task in queue over time")
        plt.xlabel('Timeline {:.1f} seconds'.format(simulating_time))
        plt.grid()
        plt.legend()
        plt.xlim(0)
        plt.ylim(0)
        plt.savefig("task in queue overtime", dpi=300)
        plt.show()


if __name__ == "__main__":
    mm1 = MM1(1, 1)
    mm1.figure1()
    mm1.figure2()

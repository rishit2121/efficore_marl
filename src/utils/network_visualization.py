import matplotlib.pyplot as plt
import networkx as nx
import pypsa
import numpy as np
from typing import Dict, Any
import matplotlib.animation as animation

class NetworkVisualizer:
    def __init__(self, network: pypsa.Network):
        self.network = network
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111)

    def plot_network(self, title: str = "Power Network", save_path: str = None):
        G = nx.Graph()

        # Add buses
        for bus in self.network.buses.index:
            G.add_node(bus, pos=(self.network.buses.loc[bus, 'x'], 
                                 self.network.buses.loc[bus, 'y']))

        # Add lines
        for line in self.network.lines.index:
            G.add_edge(self.network.lines.loc[line, 'bus0'],
                       self.network.lines.loc[line, 'bus1'])

        # Add generators and loads as separate markers
        pos = nx.get_node_attributes(G, 'pos')
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
                node_size=500, font_size=8, font_weight='bold')

        # Add power flow labels if available
        try:
            for line in self.network.lines.index:
                bus0 = self.network.lines.loc[line, 'bus0']
                bus1 = self.network.lines.loc[line, 'bus1']
                if hasattr(self.network, 'lines_t') and 'p0' in self.network.lines_t:
                    p0 = self.network.lines_t.p0.loc[self.network.snapshots[0], line]
                    self.ax.annotate(f"{p0:.1f} MW",
                                     xy=pos[bus0],
                                     xytext=pos[bus1],
                                     arrowprops=dict(arrowstyle='->', color='red'))
        except Exception as e:
            print(f"Warning: Could not add power flow labels: {e}")

        # Highlight houses, solar panels, and backup gens
        for load in self.network.loads.index:
            bus = self.network.loads.loc[load, 'bus']
            if "house" in load:
                plt.scatter(*pos[bus], s=700, c='green', marker='s', label='House')
            elif "solar" in load:
                plt.scatter(*pos[bus], s=700, c='orange', marker='^', label='Solar Panel')
            elif "backup" in load:
                plt.scatter(*pos[bus], s=700, c='gray', marker='x', label='Backup Gen')

        plt.title(title)
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), loc='upper left')

        if save_path:
            plt.savefig(save_path)
        plt.close()

    def plot_power_flow(self, save_path: str = None):
        plt.figure(figsize=(12, 8))

        try:
            if hasattr(self.network, 'lines_t') and 'p0' in self.network.lines_t:
                for line in self.network.lines.index:
                    p0 = self.network.lines_t.p0.loc[self.network.snapshots[0], line]
                    p1 = self.network.lines_t.p1.loc[self.network.snapshots[0], line]
                    plt.plot([0, 1], [p0, p1], label=f"Line {line}")

                plt.xlabel('Time')
                plt.ylabel('Power Flow (MW)')
                plt.title('Power Flow in Network')
                plt.legend()
                plt.grid(True)
            else:
                plt.text(0.5, 0.5, "No power flow data available",
                         horizontalalignment='center',
                         verticalalignment='center')
                plt.title('Power Flow Data Not Available')
        except Exception as e:
            plt.text(0.5, 0.5, f"Error plotting power flow: {e}",
                     horizontalalignment='center',
                     verticalalignment='center')
            plt.title('Error in Power Flow Visualization')

        if save_path:
            plt.savefig(save_path)
        plt.close()

    def plot_bus_voltages(self, save_path: str = None):
        plt.figure(figsize=(12, 8))

        try:
            if hasattr(self.network, 'buses_t') and 'v_mag_pu' in self.network.buses_t:
                for bus in self.network.buses.index:
                    v = self.network.buses_t.v_mag_pu.loc[self.network.snapshots[0], bus]
                    plt.bar(bus, v, label=f"Bus {bus}")

                plt.xlabel('Bus')
                plt.ylabel('Voltage (p.u.)')
                plt.title('Bus Voltages')
                plt.legend()
                plt.grid(True)
            else:
                plt.text(0.5, 0.5, "No voltage data available",
                         horizontalalignment='center',
                         verticalalignment='center')
                plt.title('Voltage Data Not Available')
        except Exception as e:
            plt.text(0.5, 0.5, f"Error plotting voltages: {e}",
                     horizontalalignment='center',
                     verticalalignment='center')
            plt.title('Error in Voltage Visualization')

        if save_path:
            plt.savefig(save_path)
        plt.close()

    def create_animation(self, steps: int, save_path: str):
        fig, ax = plt.subplots(figsize=(12, 8))

        def update(frame):
            ax.clear()
            self.network.lpf()

            G = nx.Graph()
            for bus in self.network.buses.index:
                G.add_node(bus, pos=(self.network.buses.loc[bus, 'x'],
                                     self.network.buses.loc[bus, 'y']))

            for line in self.network.lines.index:
                G.add_edge(self.network.lines.loc[line, 'bus0'],
                           self.network.lines.loc[line, 'bus1'])

            pos = nx.get_node_attributes(G, 'pos')
            nx.draw(G, pos, with_labels=True, node_color='lightblue',
                    node_size=500, font_size=8, font_weight='bold')

            for line in self.network.lines.index:
                bus0 = self.network.lines.loc[line, 'bus0']
                bus1 = self.network.lines.loc[line, 'bus1']
                p0 = self.network.lines_t.p0.loc[self.network.snapshots[0], line]
                ax.annotate(f"{p0:.1f} MW",
                            xy=pos[bus0],
                            xytext=pos[bus1],
                            arrowprops=dict(arrowstyle='->', color='red'))

            plt.title(f"Power Network State - Step {frame}")

        anim = animation.FuncAnimation(fig, update, frames=steps, interval=200)
        anim.save(save_path, writer='pillow', fps=5)
        plt.close()

    def plot_agent_actions(self, actions: Dict[str, np.ndarray], save_path: str = None):
        plt.figure(figsize=(12, 8))

        for agent_id, agent_actions in actions.items():
            plt.plot(agent_actions, label=f"Agent {agent_id}")

        plt.xlabel('Time Step')
        plt.ylabel('Action Value')
        plt.title('Agent Actions Over Time')
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
        plt.close()

"""Hardcoded Mask-D analysis annotations (corridor indices, biclique groups, key
transitions) plus small layout/color helpers.

Mask-D-specific analysis configuration: producer/plot scripts instantiate
``MaskDSpec()`` and pass it into the generic package functions as
``maskd_special_params``. Distinct from :class:`manhattan_maze.mask.MaskDSpecial`
(the per-session Mask-D geometry subclass): ``MaskDSpecial`` carries the loaded
maze geometry/graph for one session, whereas ``MaskDSpec`` holds the fixed display
topology (biclique groups, bottleneck/shortest-path corridor indices, plotting
order, and colors) used by the Mask-D transition/algorithm figures.

Lives in the package (rather than ``scripts/``) because it is importable analysis
logic reused across several gen/plot scripts, per ``docs/data_contracts.md`` §6
(Mask-D-only logic belongs in a package module, not scattered in scripts).
"""
import numpy as np


class MaskDSpec:
    """
    Hardcoded special parameters for Mask D analysis (corridor topology + layout).
    """
    def __init__(self, bottleneck_corridor_indices = None, biclique_1_corridor_indices = None,
                 biclique_1_groups = None, biclique_2_corridor_indices = None, biclique_2_groups = None, shortest_path_corridor_indices = None,
                 unvisited_corridors = None, out_corridor=None, home_corridor=None, out_key_transitions = None, home_key_transitions=None):
        """
        Initialize the Mask-D special parameters; defaults encode the standard layout.
        """
        if bottleneck_corridor_indices is None:
            bottleneck_corridor_indices = [1]
        if biclique_1_corridor_indices is None:
            biclique_1_corridor_indices = [5, 13, 3, 15, 7, 17, 9, 19, ]
        if biclique_1_groups is None:
            biclique_1_groups = [[5, 3, 7, 9], [ 13, 15, 17, 19,]]
        if biclique_2_corridor_indices is None:
            biclique_2_corridor_indices = [ 12, 4, 18, 6, 20,8, 14, 2] # interleaving corridor for visualization
        if biclique_2_groups is None:
            biclique_2_groups = [[14, 18, 20, 12,], [4, 6, 8, 2],]
        if shortest_path_corridor_indices is None:
            shortest_path_corridor_indices = [5, 19, 1, 12, 2, 16]
        if unvisited_corridors is None:
            unvisited_corridors = [0, 10, 11, 21]
        if out_corridor is None:
            out_corridor = [16]
        if home_corridor is None:
            home_corridor = [5]
        if out_key_transitions is None:
            out_key_transitions = [[19, 1], [1, 12], [2, 16]]
        if home_key_transitions is None:
            home_key_transitions = [[12, 1], [1, 19]]


        self.bottleneck_corridor_indices = bottleneck_corridor_indices
        self.biclique_1_corridor_indices = biclique_1_corridor_indices
        self.biclique_1_groups = biclique_1_groups
        self.biclique_2_corridor_indices = biclique_2_corridor_indices
        self.biclique_2_groups = biclique_2_groups
        self.shortest_path_corridor_indices = shortest_path_corridor_indices
        self.unvisited_corridors = unvisited_corridors
        self.plot_corridor_order = biclique_1_corridor_indices + bottleneck_corridor_indices + biclique_2_corridor_indices + out_corridor # leave out the unvisited corridors
        self.corridor_index_order = self.plot_corridor_order + unvisited_corridors

        self.shortest_path_corridor_order = [self.corridor_index_order.index(i) for i in shortest_path_corridor_indices]
        self.corridor_order_operator_array = np.argsort(self.corridor_index_order)
        self.out_corridor = out_corridor
        self.home_corridor = home_corridor
        self.out_key_transitions = out_key_transitions
        self.home_key_transitions = home_key_transitions

        self.corridor_color_indices_dict = {"Biclique 1": ("tab:blue", np.ravel(self.biclique_1_corridor_indices)),
                                            "Biclique 2": ("tab:orange", np.ravel(self.biclique_2_corridor_indices)),
                                            "Bottleneck": ("black", np.ravel(self.bottleneck_corridor_indices)),
                                            "Unvisited": ("tab:grey", np.ravel(self.unvisited_corridors)),
                                            "Shortest Path": ("black", np.ravel(self.shortest_path_corridor_indices)),
                                            "Out": ("black", np.ravel(out_corridor)),
                                            "Home": ("black", np.ravel(home_corridor))}
        self.out_node_set = (19, 1, [3, 7, 9]) # start node, goal node,  control nodes
        self.home_node_set = (12, 1, [4, 6, 8])


    def biclique_column_layout(self, biclique_group=None, add_all_nodes=True):
        '''
        Create a biclique layout for the graph G based on the partite dict.
        :param G:
        :param partite_dict: a dictionary with node as key and partite set as value {nodes: 0 or 1}
        :return:
        '''
        pos = {}
        if biclique_group is None:
            # merge the two bicliques
            for i, node in enumerate(self.biclique_1_groups[0]):
                pos[node] = (0, -i)
            for i, node in enumerate(self.biclique_1_groups[1]):
                pos[node] = (0.8, -i)
            # add 1 to increase the gap between the two bicliques
            for i, node in enumerate(self.biclique_2_groups[0]):
                pos[node] = (2.2, -i) # horizontal put on the right side
            for i, node in enumerate(self.biclique_2_groups[1]):
                pos[node] = (1.2, -i)
        else: # every node in the same column
            for i, node in enumerate(biclique_group[0]):
                pos[node] = (0, -i)
            for i, node in enumerate(biclique_group[1]):
                pos[node] = (1, -i)
        if add_all_nodes:
            # add also the shortest path corridors
            pos[1] = (1, 1) # botteleneck to be put in the middle
            pos[16] = (2.4, 1) # out corridor to be put on the right side
        return pos

    def biclique_column_colors(self, horizontal_color="tab:blue", vertical_color="tab:orange", highlight_list=None, highlight_color="red"):
        """
        Create a color map for the biclique layout.
        :param horizontal_color: color for the horizontal corridors
        :param vertical_color: color for the vertical corridors
        :return:
        """
        color_map = {}
        for i, node in enumerate(self.biclique_1_groups[0]):
            color_map[node] = horizontal_color
        for i, node in enumerate(self.biclique_1_groups[1]):
            color_map[node] = vertical_color

        for i, node in enumerate(self.biclique_2_groups[0]):
            color_map[node] = horizontal_color
        for i, node in enumerate(self.biclique_2_groups[1]):
            color_map[node] = vertical_color

        if highlight_list is None:
            highlight_list = self.shortest_path_corridor_indices
            for node in highlight_list:
                color_map[node] = highlight_color
        return color_map

from gym_idsgame.envs.rendering.network.node import Node
from gym_idsgame.envs.rendering.util.render_util import batch_line
from gym_idsgame.envs.rendering.network.resource_node import ResourceNode
from gym_idsgame.envs.dao.render_config import RenderConfig
from gym_idsgame.envs.rendering.network.data_resource import Data
from gym_idsgame.envs.rendering.network.server_resource import Server
class Network:
    """
    Class representing the resource network in the rendering
    """
    def __init__(self, render_config: RenderConfig):
        self.render_config = render_config
        self.grid = [[self.create_node(i,j) for j in range(self.render_config.game_config.num_cols)] for i in
                     range(self.render_config.game_config.num_rows)]

    def create_node(self, i,j):
        # Data node
        if i == 0 and j == self.render_config.game_config.num_cols//2:
            return Data(self.render_config, i, j)
        # Start node
        if i == self.render_config.game_config.num_rows-1 and j == self.render_config.game_config.num_cols//2:
            return None
        if i is not 0 and i is not self.render_config.game_config.num_rows-1:
            return Server(self.render_config, i, j)
        return None
        # if i == 0:
        #     return ResourceNode()


    def get_cell(self, row, col):
        """
        Gets a specific cell from the grid

        :param row: the row of the cell in the grid
        :param col: the column of the cell in the grid
        :return: the cell at the given row and column
        """
        return self.grid[row][col]

    def root_edge(self, n1, n2, color, batch, group, line_width):
        x1, y1, col1, row1 = n1.get_link_coords(lower=True, upper=False)
        x2, y2, col2, row2 = n2.get_link_coords(upper=True, lower=False)
        return batch_line(x1, y1 + self.cell_size / 6, x2, y2, color, batch, group, line_width)

    def connect_start_and_server_nodes(self, n1, n2, color, batch, group, line_width):
        x1, y1, col1, row1 = n1.get_link_coords(lower=True, upper=False)
        x2, y2, col2, row2 = n2.get_link_coords(upper=True, lower=False)
        edges = []
        e1 = batch_line(x1, y1, x2, y1, color, batch, group, line_width)
        e2 = batch_line(x2, y1, x2, y2, color, batch, group, line_width)
        edges.append(e1)
        edges.append(e2)
        return edges

    def connect_server_and_server_nodes(self, n1, n2, color, batch, group, line_width):
        x1, y1, col1, row1 = n2.get_link_coords(lower=True, upper=False)
        x2, y2, col2, row2 = n1.get_link_coords(upper=True, lower=False)
        e1 = batch_line(x2, y1, x2, y2, color, batch, group, line_width)
        return [e1]

    def connect_server_and_data_nodes(self, n1, n2, color, batch, group, line_width):
        x1, y1, col1, row1 = n2.get_link_coords()
        x2, y2, col2, row2 = n1.get_link_coords(upper=False, lower=True)
        edges = []
        e1 = batch_line(x1, y1, x2, y1, color, batch, group, line_width)
        e2 = batch_line(x2, y1, x2, y2, color, batch, group, line_width)
        edges.append(e1)
        edges.append(e2)
        if row1 == 0 and col1 == col2:
            e3 = batch_line(x2, y2, x2, y2-self.cell_size/3, color, batch, group, line_width)
            edges.append(e3)
        return edges
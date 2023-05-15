from abc import ABC
from TF_Gridworld import TFGridWorld


class FourRooms(TFGridWorld, ABC):
    def __init__(self, rows=13, cols=13, terminal_idx=-1, agent_start_idx=14):
        bottom_border = list(range(0, 13))
        top_border = [x + (13 * 12) for x in bottom_border]
        left_border = [13 * x for x in range(0, 13)]
        right_border = [x + 12 for x in left_border]
        vertical_wall = [(13 * x) + 6 for x in [1, 3, 4, 5, 6, 7, 8, 10, 11]]
        left_hoz_wall = [x + (6 * 13) for x in [1, 3, 4, 5]]
        right_hoz_wall = [x + (5 * 13) for x in [7, 8, 10, 11]]
        walls = bottom_border + top_border + left_border + right_border + vertical_wall + left_hoz_wall + right_hoz_wall

        super().__init__(rows=rows, cols=cols, walls=walls, terminal_idx=terminal_idx, agent_start_idx=agent_start_idx)


class NineRooms(TFGridWorld, ABC):
    def __init__(self, rows=19, cols=19, terminal_idx=-1, agent_start_idx=20):
        walls = [342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 323,
                 329, 335, 341, 304, 316, 322, 285, 291, 297, 303, 266, 272, 284, 247, 253, 259, 265, 228, 229, 230,
                 232, 233, 234, 235, 236, 237, 239, 240, 241, 242, 244, 245, 246, 209, 216, 222, 227, 190, 197, 203,
                 208, 171, 189, 152, 159, 165, 170, 133, 140, 146, 151, 114, 115, 116, 118, 119, 120, 121, 122, 123,
                 125, 126, 127, 128, 130, 131, 132, 95, 101, 107, 113, 76, 82, 94, 57, 69, 75, 38, 44, 50, 56, 19, 25,
                 31, 37, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        super().__init__(rows, cols, terminal_idx, walls, agent_start_idx)

class JointSeparator:
    def __init__(self):
        self.left_indices = [4, 5, 6, 11, 12, 13]
        self.right_indices = [1, 2, 3, 14, 15, 16]
        self.mid_indices = [0, 7, 8, 9, 10]

    def separate(self, x):
        left_joints = x[:, :, [i * 2 for i in self.left_indices] + [i * 2 + 1 for i in self.left_indices]]
        right_joints = x[:, :, [i * 2 for i in self.right_indices] + [i * 2 + 1 for i in self.right_indices]]
        mid_joints = x[:, :, [i * 2 for i in self.mid_indices] + [i * 2 + 1 for i in self.mid_indices]]
        return left_joints, right_joints, mid_joints
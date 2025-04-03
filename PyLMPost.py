import numpy as np
import os
import matplotlib.pyplot as plt

# 解析 LAMMPS dump 文件并提取原子坐标及 StructureType
def extract_positions_dump_type(lammps_file, output_file):
    with open(lammps_file, 'r') as f:
        lines = f.readlines()
    
    start_idx = lines.index("ITEM: ATOMS id type x y z StructureType\n") + 1
    positions = []
    for line in lines[start_idx:]:
        tokens = line.split()
        if len(tokens) < 6:
            continue
        x, y, z, struct_type = map(float, tokens[2:6])
        positions.append([x, y, z, int(struct_type)])   
    np.savetxt(output_file, positions, fmt='%f %f %f %d', delimiter=' ')

# 定义计算原子到点的面，距离的函数
def distance_to_plane(atom, vector=(1, 0, 0), point=[0, 0, 0]):
    atom_position = np.array(atom[:3], dtype=float)  # 确保是 NumPy 数组
    vector = np.array(vector, dtype=float)
    point = np.array(point, dtype=float)
    
    vector /= np.linalg.norm(vector)  # 归一化法向量
    return np.abs(np.dot(atom_position - point, vector))

# 定义原子对过原子体系中心点的面的距离
def distance_center_plane(crystal, vector=(1, 0, 0)):
    center = np.mean(crystal[:, :3], axis=0)
    return  np.array([distance_to_plane(atom, vector, center) for atom in crystal])

# 筛选过中心平面周围的原子
def filter_atoms_near_center_plane(crystal, vector=(1, 0, 0), threshold = 2):
    distances = distance_center_plane(crystal, vector)
    near_plane_atoms = crystal[distances < threshold]
    return  near_plane_atoms, distances

# 绘制三维原子图，支持不同Type
def scatter_atoms_3d(positions, atom_types=None):
    """
    绘制三维原子散点图，支持不同类型的原子使用不同颜色。
    
    参数：
    positions: ndarray, shape (N, 3)
        每行表示一个原子的 (x, y, z) 坐标。
    atom_types: list or ndarray, shape (N,)
        每个原子的类型（可选）。如果提供，将不同类型的原子用不同颜色表示。
    """
    positions = np.array(positions)
    
    if atom_types is None:
        atom_types = np.zeros(len(positions))  # 如果未提供类型，则假设所有原子类型相同
    else:
        atom_types = np.array(atom_types)
    
    unique_types = np.unique(atom_types)
    cmap = plt.get_cmap("tab10")  # 选择颜色映射
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for i, atom_type in enumerate(unique_types):
        mask = atom_types == atom_type
        ax.scatter(positions[mask, 0], positions[mask, 1], positions[mask, 2],
                   label=f'Type {atom_type}', color=cmap(i / len(unique_types)), s=30)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

def get_orthogonal_basis(n):
    """
    自动生成平面上的两个正交基向量 u 和 v。
    :param n: 平面的法向量 (n_x, n_y, n_z)
    :return: 两个基向量 u 和 v
    """
    # 选择一个与 n 不共线的任意向量 a
    if np.allclose(n, [1, 0, 0]) or np.allclose(n, [-1, 0, 0]):
        a = np.array([0, 1, 0])  # 避免平行
    else:
        a = np.array([1, 0, 0])
    
    # 计算第一个基向量 u
    u = a - (np.dot(a, n) / np.dot(n, n)) * n
    u = u / np.linalg.norm(u)  # 归一化
    
    # 计算第二个基向量 v
    v = np.cross(n, u)
    v = v / np.linalg.norm(v)  # 归一化
      
    return u, v

def construct_projection_matrix(n):
    """
    构造投影矩阵 M，将三维坐标转换为二维坐标。
    :param n: 平面的法向量 (n_x, n_y, n_z)
    :return: 2x3 的投影矩阵 M
    """
    # 自动生成基向量 u 和 v
    u, v = get_orthogonal_basis(n)
    
    # 构造转换矩阵 M
    M = np.vstack([u, v])
    return M

def point3d_to_2d(point, n):
    """
    将三维点投影到以法向量 n 定义的平面，并转换为二维坐标。
    :param point: 三维点 (x, y, z)
    :param n: 平面的法向量 (n_x, n_y, n_z)
    :return: 二维坐标 (s, t)
    """
    # 构造投影矩阵
    M = construct_projection_matrix(n)
    
    # 将三维点转换为二维坐标
    point_2d = M @ np.array(point)
    return point_2d

# 将所有三维平行投影在二维面上
def crystal3d_to_2d(crystal, n):
    return np.array([point3d_to_2d(atom, n) for atom in crystal[:, :3]])

def scatter_atoms_2d(positions, atom_types=None):
    """
    绘制二维原子散点图，支持不同类型的原子使用不同颜色。
    
    参数：
    positions: ndarray, shape (N, 2)
        每行表示一个原子的 (x, y) 坐标。
    atom_types: list or ndarray, shape (N,)
        每个原子的类型（可选）。如果提供，将不同类型的原子用不同颜色表示。
    """
    positions = np.array(positions)
    
    if atom_types is None:
        atom_types = np.zeros(len(positions))  # 如果未提供类型，则假设所有原子类型相同
    else:
        atom_types = np.array(atom_types)
    
    unique_types = np.unique(atom_types)
    cmap = plt.get_cmap("tab10")  # 选择颜色映射
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for i, atom_type in enumerate(unique_types):
        mask = atom_types == atom_type
        ax.scatter(positions[mask, 0], positions[mask, 1],
                   label=f'Type {atom_type}', color=cmap(i / len(unique_types)), s=30)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    plt.show()
    
def plot_vector_3Dfield(filtered_positions, filtered_eigvecs):
    """
    绘制矢量场以及原子点。
    
    参数：
    positions: ndarray, shape (N, 3)
        每行表示一个原子的 (x, y, z) 坐标。
    eigvecs: ndarray, shape (N, 3)
        每个原子的矢量方向。
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制矢量场
    ax.quiver(filtered_positions[:, 0], filtered_positions[:, 1], filtered_positions[:, 2],
              filtered_eigvecs[:, 0], filtered_eigvecs[:, 1], filtered_eigvecs[:, 2],
              color='black', length=0.5, normalize=True)
    
    # 绘制原子点（空心球）
    ax.scatter(filtered_positions[:, 0], filtered_positions[:, 1], filtered_positions[:, 2],
               facecolors='none', edgecolors='black', s=50)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(f'Vibrational Mode at Frequency')
    plt.show()
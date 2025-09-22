import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import random

# 定义环境参数
BOUNDARY = [-9, 9]
RESOLUTION = 0.05

# 定义障碍物
# obstacles_array: [x, y, radius]
obstacles_array = np.array([
    [-6, 6, 1], [-3, 6, 1], [2, 6, 1], [6, 6, 1],
    [-6, 3, 1], [-3, 3, 1], [2, 3, 1], [6, 3, 1],
    [-6, 0, 1], [-3, 0, 1.5], [2, 0, 1], [6, 0, 1],
    [-6, -3, 1.7], [-3, -3, 2], [2, -3, 1.7], [6, -3, 1.7],
    [-6, -6, 0.5], [-3, -6, 0.5], [2, -6, 0.5], [6, -6, 0.5],
])

# RRT-Star参数
STEP_SIZE = 0.5
NEIGHBOR_RADIUS = 2.0
MAX_ITERATIONS_PER_PATH = 5000

# 网格覆盖参数
GRID_SIZE = RESOLUTION
GRID_X_MIN = int((BOUNDARY[0] + 9) / GRID_SIZE)
GRID_X_MAX = int((BOUNDARY[1] + 9) / GRID_SIZE)
GRID_Y_MIN = int((BOUNDARY[0] + 9) / GRID_SIZE)
GRID_Y_MAX = int((BOUNDARY[1] + 9) / GRID_SIZE)

# 碰撞检测函数
def is_collision(point, obstacles):
    for obs in obstacles:
        obs_x, obs_y, obs_r = obs
        if np.linalg.norm(point - np.array([obs_x, obs_y])) <= obs_r:
            return True
    return False

def is_path_collision(start, end, obstacles):
    num_steps = 100
    for i in range(num_steps + 1):
        t = i / num_steps
        point = start + t * (end - start)
        if is_collision(point, obstacles):
            return True
    return False

# RRT-Star节点类
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0

# 查找最近邻节点
def find_nearest_node(node_list, point):
    distances = [np.linalg.norm([node.x - point[0], node.y - point[1]]) for node in node_list]
    return node_list[np.argmin(distances)]

# RRT-Star路径生成函数
def rrt_star_path(start_point, obstacles):
    node_list = [Node(start_point[0], start_point[1])]
    
    for _ in range(MAX_ITERATIONS_PER_PATH):
        # 1. 随机采样
        rand_point = np.array([
            random.uniform(BOUNDARY[0], BOUNDARY[1]),
            random.uniform(BOUNDARY[0], BOUNDARY[1])
        ])

        # 如果采样点在障碍物内，重新采样
        if is_collision(rand_point, obstacles):
            continue

        # 2. 寻找最近邻
        x_nearest = find_nearest_node(node_list, rand_point)

        # 3. 生成新节点
        theta = np.arctan2(rand_point[1] - x_nearest.y, rand_point[0] - x_nearest.x)
        x_new_coords = np.array([
            x_nearest.x + STEP_SIZE * np.cos(theta),
            x_nearest.y + STEP_SIZE * np.sin(theta)
        ])

        # 边界检查
        if not (BOUNDARY[0] <= x_new_coords[0] <= BOUNDARY[1] and BOUNDARY[0] <= x_new_coords[1] <= BOUNDARY[1]):
            continue

        # 4. 碰撞检测
        if is_path_collision(np.array([x_nearest.x, x_nearest.y]), x_new_coords, obstacles):
            continue
        
        x_new = Node(x_new_coords[0], x_new_coords[1])

        # 5. RRT-Star优化
        # 找到邻近节点
        x_near = [node for node in node_list if np.linalg.norm([node.x - x_new.x, node.y - x_new.y]) <= NEIGHBOR_RADIUS]

        # 选择最佳父节点
        min_cost = x_nearest.cost + np.linalg.norm([x_new.x - x_nearest.x, x_new.y - x_nearest.y])
        best_parent = x_nearest
        
        for neighbor in x_near:
            dist = np.linalg.norm([x_new.x - neighbor.x, x_new.y - neighbor.y])
            if neighbor.cost + dist < min_cost:
                if not is_path_collision(np.array([neighbor.x, neighbor.y]), x_new_coords, obstacles):
                    min_cost = neighbor.cost + dist
                    best_parent = neighbor
        
        x_new.parent = best_parent
        x_new.cost = min_cost
        node_list.append(x_new)

        # 重新连接
        for neighbor in x_near:
            dist = np.linalg.norm([x_new.x - neighbor.x, x_new.y - neighbor.y])
            if x_new.cost + dist < neighbor.cost:
                if not is_path_collision(x_new_coords, np.array([neighbor.x, neighbor.y]), obstacles):
                    neighbor.parent = x_new
                    neighbor.cost = x_new.cost + dist

    return node_list

# 将 RRT-Star 树转换为轨迹点列表
def get_path_from_tree(node_list):
    path_points = []
    
    # 从树中随机选择一个末端节点
    end_node = random.choice(node_list)

    current_node = end_node
    while current_node is not None:
        path_points.append(np.array([current_node.x, current_node.y]))
        current_node = current_node.parent
    
    return path_points[::-1] # 反转列表，从起点到终点

# 将轨迹点列表转换为三次样条曲线
def get_spline_path(path_points, num_points=100):
    if len(path_points) < 2:
        return np.array([])
    
    path_points = np.array(path_points)
    t = np.linspace(0, 1, len(path_points))
    cs_x = CubicSpline(t, path_points[:, 0])
    cs_y = CubicSpline(t, path_points[:, 1])

    t_new = np.linspace(0, 1, num_points)
    x_new = cs_x(t_new)
    y_new = cs_y(t_new)
    
    return np.vstack((x_new, y_new)).T

# 更新覆盖地图
def update_coverage_map(coverage_map, path_points):
    updated = False
    for point in path_points:
        x, y = point
        grid_x = int((x + 9) / GRID_SIZE)
        grid_y = int((y + 9) / GRID_SIZE)
        
        if 0 <= grid_x < GRID_X_MAX and 0 <= grid_y < GRID_Y_MAX:
            if coverage_map[grid_y, grid_x] == 0:
                coverage_map[grid_y, grid_x] = 1
                updated = True
    return updated

# 寻找未覆盖的起始点
def find_uncovered_start_point(coverage_map):
    uncovered_indices = np.argwhere(coverage_map == 0)
    if len(uncovered_indices) == 0:
        return None
    
    # 随机选择一个未覆盖的网格作为新的起点
    idx = random.choice(uncovered_indices)
    grid_y, grid_x = idx
    
    x = grid_x * GRID_SIZE - 9 + GRID_SIZE / 2
    y = grid_y * GRID_SIZE - 9 + GRID_SIZE / 2
    
    return np.array([x, y])

# 绘制结果
def plot_results(all_paths, obstacles):
    fig, ax = plt.subplots(figsize=(10, 10))

    # 绘制障碍物
    for obs in obstacles:
        circle = plt.Circle((obs[0], obs[1]), obs[2], color='red', alpha=0.6)
        ax.add_artist(circle)
    
    # 绘制所有轨迹
    for i, path in enumerate(all_paths):
        if len(path) > 0:
            ax.plot(path[:, 0], path[:, 1], linestyle='-', linewidth=2, label=f'Trajectory {i+1}')

    ax.set_xlim(BOUNDARY[0], BOUNDARY[1])
    ax.set_ylim(BOUNDARY[0], BOUNDARY[1])
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Robot Exploration Path with Cubic Spline')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.grid(True)
    plt.show()

# 主函数
def main():
    coverage_map = np.zeros((GRID_Y_MAX, GRID_X_MAX))
    all_spline_paths = []
    
    # 初始化第一个起点
    start_point = np.array([0, 0])
    
    for i in range(100): # 尝试生成最多100条路径
        print(f"Generating path {i+1}...")
        
        # 如果起点在障碍物内，重新找一个
        if is_collision(start_point, obstacles_array):
            start_point = find_uncovered_start_point(coverage_map)
            if start_point is None:
                break # 所有区域都被覆盖

        # 生成RRT-Star路径
        node_list = rrt_star_path(start_point, obstacles_array)
        rrt_path = get_path_from_tree(node_list)

        # 将RRT路径转换为样条曲线
        spline_path = get_spline_path(rrt_path)
        
        if len(spline_path) > 0:
            all_spline_paths.append(spline_path)
            
            # 更新覆盖地图
            if update_coverage_map(coverage_map, spline_path):
                print(f"Coverage updated. Total coverage: {np.sum(coverage_map) / coverage_map.size * 100:.2f}%")
            
        # 寻找下一个未覆盖的起点
        start_point = find_uncovered_start_point(coverage_map)
        if start_point is None:
            print("All reachable areas have been covered.")
            break
            
    plot_results(all_spline_paths, obstacles_array)

if __name__ == "__main__":
    main()

import numpy as np
from scipy.interpolate import CubicHermiteSpline
current_pos=[0,0]
current_quat=[1,0,0,0]
next_pos=[0.1,0]
next_quat=[0.707,0,0.707,0] 
steps=10


A = np.array(current_pos).squeeze()
B = np.array(next_pos).squeeze()
C = 0 
D = 0.95   
distance = np.square(B - A).sum()**0.5

# calculate tangent vectors
tangent_A = np.array([np.cos(C), np.sin(C)])
tangent_B = np.array([np.cos(D), np.sin(D)])



t = np.array([0, distance])
points = np.vstack((A, B))
tangents = np.vstack((tangent_A, tangent_B))

# create cubic Hermite splines for x and y coordinates
path_x = CubicHermiteSpline(t, points[:, 0], tangents[:, 0])
path_y = CubicHermiteSpline(t, points[:, 1], tangents[:, 1])

t_values = np.linspace(0, distance, steps*10)

x_values = path_x(t_values)
y_values = path_y(t_values)
points = np.column_stack((x_values, y_values))


vel_x = path_x.derivative()
vel_y = path_y.derivative()
dx_values = vel_x(t_values)
dy_values = vel_y(t_values)
direction_values = np.arctan2(dy_values, dx_values)



actions = points[1:] - points[:-1]
dir_acts = direction_values[1:] - direction_values[:-1]
distances = np.sum(np.square(actions), axis=1)**0.5
new_point_list = []
new_direction_values_list = []
new_point_list.append(points[0])
new_direction_values_list.append(direction_values[0])
sum1 = 0
sum2 = 0
for i in range(steps*10-1):
    sum1 += distances[i]
    sum2 += abs(dir_acts[i])
    if sum1 > 0.05 or sum2 > 0.05:
        new_point_list.append(points[i-1])
        new_direction_values_list.append(direction_values[i-1])
        sum1 = distances[i]
        sum2 = abs(dir_acts[i])
new_point_list.append(points[-1])
new_direction_values_list.append(direction_values[-1])



    
print('end!')
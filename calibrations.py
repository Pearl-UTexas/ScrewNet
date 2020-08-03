import math
import numpy as np

## Real Data:
# %% Kinect Color Camera
color_cam_matrix = np.array(
    [1.0526303338534365e+03, 0., 9.3528526085572480e+02, 0., 1.0534191001014469e+03, 5.2225718970556716e+02, 0., 0.,
     1.]).reshape(3, 3)
color_distortion_coeffs = np.array(
    [4.5467150011699140e-02, -7.4470107942918126e-02, -6.1697129558609537e-03, -2.5667037404509380e-03,
     -1.4503959457133547e-02]).reshape(1, 5)
color_rotation = np.eye(3)
color_projection = np.array(
    [1.0526303338534365e+03, 0., 9.3528526085572480e+02, 0., 0., 1.0534191001014469e+03, 5.2225718970556716e+02, 0., 0.,
     0., 1., 0., 0., 0., 0., 1.]).reshape(4, 4)

# %% Kinect IR Camera
ir_cam_matrix = np.array(
    [3.5706872738709285e+02, 0., 2.5037220752105404e+02, 0., 3.5700920458183873e+02, 2.0803230739018434e+02, 0., 0.,
     1.]).reshape(3, 3)
ir_distortion_coeffs = np.array(
    [5.5998048975189132e-02, -2.5691440815038830e-01, -5.3889184410447575e-03, -1.6922667364749613e-03,
     1.9674519800098919e-01]).reshape(1, 5)
ir_rotation = np.eye(3)
ir_projection = np.array(
    [3.5706872738709285e+02, 0., 2.5037220752105404e+02, 0., 0., 3.5700920458183873e+02, 2.0803230739018434e+02, 0., 0.,
     0., 1., 0., 0., 0., 0., 1.]).reshape(4, 4)
depthShift = -2.7989551644219979e+01

# %% Pose Calibration between depth and color
rotation = np.array([9.9997222955499243e-01, -7.4399336788120839e-03, 4.3301925190808763e-04, 7.4347723554060875e-03,
                     9.9991294780487039e-01, 1.0900503300210780e-02, -5.1408057825089366e-04, -1.0896981188819882e-02,
                     9.9994049399058227e-01]).reshape(3, 3)
translation = np.array([-5.2291985456630448e-02, -1.9227292627499695e-04, 1.7173350151375650e-03]).reshape(3, 1)
essential = np.array([-1.2669151118394222e-05, -1.7150903228939863e-03, -2.1098130088050980e-04, 1.6904050298585356e-03,
                      -5.8260164046387006e-04, 5.2289617408374921e-02, -1.9651142111198186e-04, -5.2288863822328481e-02,
                      -5.6992570216587654e-04]).reshape(3, 3)
fundamental = np.array(
    [-8.8142664830290771e-09, -1.1934330447023842e-06, 1.9806702972926870e-04, 1.1751792885051283e-06,
     -4.0509553642475600e-07, 1.2770218257581496e-02, -7.4941574482561516e-04, -3.6972004067303506e-02, 1.]).reshape(3,
                                                                                                                     3)

# %% Color Params
color_height = 1080
color_width = 1920
color_fov_x = 360 / math.pi * math.atan2(color_width, 2 * color_cam_matrix[0, 0])
color_fov_y = 360 / math.pi * math.atan2(color_height, 2 * color_cam_matrix[1, 1])
color_fx = color_cam_matrix[0, 0]
color_fy = color_cam_matrix[1, 1]
color_cx = color_cam_matrix[0, 2]
color_cy = color_cam_matrix[1, 2]

color_fx
color_fy
color_fov_x
color_fov_y

# %% IR Field of View, Width, Height computation
ir_width = 512
ir_height = 424
ir_aspect = ir_width / ir_height
depth_fov_x = 360 / math.pi * math.atan2(ir_width, 2 * color_cam_matrix[0, 0])
depth_fov_y = 360 / math.pi * math.atan2(ir_height, 2 * color_cam_matrix[1, 1])
ir_fx = ir_cam_matrix[0, 0]
ir_fy = ir_cam_matrix[1, 1]
ir_cx = ir_cam_matrix[0, 2]
ir_cy = ir_cam_matrix[1, 2]

## transform into camera frame. useful for reconstruction!
T_magic_to_cam = np.array([[0., -1., 0., 0.],
                           [0., 0., -1., 0.],
                           [1., 0., 0., 0.],
                           [0., 0., 0., 1.0]])

## Simulation Camera Params
# %%
znear = 0.1
zfar = 12
sim_width = 192
sim_height = 108
# sim_width = 720 * 4
# sim_height = 405 * 4

old_sim_fovy = 60 * math.pi / 180
old_sim_fovx = 2 * math.atan(math.tan(old_sim_fovy / 2) * sim_width / sim_height)

old_sim_fovy * 180 / math.pi
old_sim_fovx * 180 / math.pi

old_sim_focal_y = (sim_height / 2) / math.tan(old_sim_fovy / 2)
old_sim_focal_x = (sim_width / 2) / math.tan(old_sim_fovx / 2)
old_sim_proj_matrix = np.array([[old_sim_focal_x, 0, sim_width / 2],
                                [0, old_sim_focal_y, sim_height / 2],
                                [0, 0, 1]])

# new sim cam Params, using color fov_y
sim_focal_y = (sim_height / 2) / math.tan(color_fov_y * 3.14 / 180.0 / 2)
sim_focal_x = sim_focal_y
sim_proj_matrix = np.array([[sim_focal_x, 0, sim_width / 2],
                            [0, sim_focal_y, sim_height / 2],
                            [0, 0, 1]])

# checking that these are reasonable
color_fov_x = 360 / math.pi * math.atan2(color_width, 2 * color_cam_matrix[0, 0])
color_fov_y = 360 / math.pi * math.atan2(color_height, 2 * color_cam_matrix[1, 1])

color_fov_x
color_fov_y

test_sim_fov_y = 360 / math.pi * math.atan2(sim_height, 2 * sim_proj_matrix[1, 1])
test_sim_fov_x = 360 / math.pi * math.atan2(sim_width, 2 * sim_proj_matrix[0, 0])

# fake real sim cam Params (ie, size is the full 1920 x 1080)
fake_focal_y = (color_height / 2) / math.tan(color_fov_y * 3.14 / 180.0 / 2)
fake_focal_x = (color_width / 2) / math.tan(color_fov_x * 3.14 / 180.0 / 2)
fake_proj_matrix = np.array([[fake_focal_x, 0, color_width / 2],
                             [0, fake_focal_y, color_height / 2],
                             [0, 0, 1]])

if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    print(' \n simulated cam matrix: \n\t', str(np.round(fake_proj_matrix, 0)).replace('\n', '\n\t'))
    print(' \n real cam matrix: \n\t', str(np.round(color_cam_matrix, 0)).replace('\n', '\n\t'))
    print(' \n ')

    print(color_fov_y)

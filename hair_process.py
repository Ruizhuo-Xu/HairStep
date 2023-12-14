import os

import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation

# from scripts.get_lmk import _get_lmk
# from scripts.recon3D import _recon3D
# from scripts.img2hairstep import _img2hairstep
# from scripts.opt_cam import _opt_cam
import subprocess
from glob import glob


# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

        
def matrix_to_quaternion(matrix):
    """
    将3x3旋转矩阵转换为四元数
    """
    # 矩阵的迹
    trace = np.trace(matrix)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (matrix[2, 1] - matrix[1, 2]) * s
        y = (matrix[0, 2] - matrix[2, 0]) * s
        z = (matrix[1, 0] - matrix[0, 1]) * s
    else:
        if matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            w = (matrix[2, 1] - matrix[1, 2]) / s
            x = 0.25 * s
            y = (matrix[0, 1] + matrix[1, 0]) / s
            z = (matrix[0, 2] + matrix[2, 0]) / s
        elif matrix[1, 1] > matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            w = (matrix[0, 2] - matrix[2, 0]) / s
            x = (matrix[0, 1] + matrix[1, 0]) / s
            y = 0.25 * s
            z = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            w = (matrix[1, 0] - matrix[0, 1]) / s
            x = (matrix[0, 2] + matrix[2, 0]) / s
            y = (matrix[1, 2] + matrix[2, 1]) / s
            z = 0.25 * s
    
    return np.array([w, x, y, z])

# def find_similarity_transform_matrix(A, B):
#     '''
#     Calculates the least-squares best-fit transform between corresponding 3D points A->B
#     Input:
#       A: Nx3 numpy array of corresponding 3D points
#       B: Nx3 numpy array of corresponding 3D points
#     Returns:
#       T: 4x4 homogeneous transformation matrix
#       R: 3x3 rotation matrix
#       t: 3x1 translation vector
#     '''
#     A = np.array(A)
#     B = np.array(B)

#     assert len(A) == len(B)

#     # get number of dimensions
#     dim = len(B[0])

#     # translate points to their centroids
#     centroid_A = np.mean(A, axis=0)
#     centroid_B = np.mean(B, axis=0)
    
#     AA = A - centroid_A
#     BB = B - centroid_B

#     # rotation matrix
#     H = np.dot(AA.T, BB)
#     U, S, Vt = np.linalg.svd(H)
#     R = np.dot(Vt.T, U.T)

#     # special reflection case
#     if np.linalg.det(R) < 0:
#         Vt[dim-1,:] *= -1
#         R = np.dot(Vt.T, U.T)

#     # translation
#     t = centroid_B.T - np.dot(R,centroid_A.T)

#     # homogeneous transformation
#     T = np.eye(dim+1)
#     T[:dim, :dim] = R
#     T[:dim, dim] = t

#     return T

def find_similarity_transform_matrix(points1, points2, is_scale=True):
    """
    寻找相似变换矩阵
    
    参数：
    points1, points2: 两个形状为 (4, 3) 的numpy数组,每一行代表一个3维点的坐标。
    
    返回：
    T: 3x4的相似变换矩阵
    """
    # 计算质心
    centroid1 = np.mean(points1, axis=0)
    centroid2 = np.mean(points2, axis=0)
    
    # 将点云中心化
    centered_points1 = points1 - centroid1
    centered_points2 = points2 - centroid2
    
    # 计算协方差矩阵
    H = centered_points1.T @ centered_points2
    
    # 使用奇异值分解求解旋转矩阵 R
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # 计算尺度因子
    if is_scale:
        scale = np.trace(H @ R.T) / np.trace(centered_points1.T @ centered_points1)
    else:
        scale = 1
    
    # 构建相似变换矩阵
    T = np.eye(4)
    # quaternion = matrix_to_quaternion(R)
    T[:3, :3] = scale * R
    T[:3, 3] = centroid2 - scale * R @ centroid1
    
    # return quaternion, scale, T[:3, 3]
    return T

def detect_and_crop_faces(image_path, output_path, scale=1.5):
    # 加载分类器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 读取图像
    img = cv2.imread(image_path)
    print(image_path, img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 人脸检测
    faces = face_cascade.detectMultiScale(gray, minNeighbors=5)

    # 遍历检测到的人脸
    for (x, y, w, h) in faces:
        # 计算扩大后的坐标
        new_x = max(0, int(x - 0.5 * (scale - 1) * w))
        new_y = max(0, int(y - 0.5 * (scale - 1) * h))
        new_w = int(w * scale)
        new_h = int(h * scale)

        # 在原始图像上绘制扩大后的矩形
        # cv2.rectangle(img, (new_x, new_y), (new_x + new_w, new_y + new_h), (255, 0, 0), 2)

        # 裁剪人脸区域
        face_roi = img[new_y:new_y+new_h, new_x:new_x+new_w]

        # 保存裁剪的人脸图像
        print(cv2.imwrite(output_path, face_roi))

def filter_lineset(line_set, threshold_y):
    """
    Filter a given Open3D LineSet based on a y-coordinate threshold.

    Parameters:
        line_set (o3d.geometry.LineSet): The input LineSet to be filtered.
        threshold_y (float): The y-coordinate threshold.

    Returns:
        o3d.geometry.LineSet: The filtered LineSet.
    """
    # Get point coordinates and line indices
    points = np.asarray(line_set.points)
    lines = np.asarray(line_set.lines)

    # Find indices of points that satisfy the y-coordinate threshold
    selected_point_indices = np.where(points[:, 1] >= threshold_y)[0]

    # Find indices of lines that satisfy the y-coordinate threshold
    selected_line_indices = np.where(np.all(points[lines] >= threshold_y, axis=1))[0]

    # Create a mapping from old point indices to new point indices
    point_mapping = {old_index: new_index for new_index, old_index in enumerate(selected_point_indices)}

    # Update line indices by mapping old point indices to new point indices
    new_lines = np.vectorize(point_mapping.get)(lines[selected_line_indices])

    # Update LineSet with filtered points and lines
    line_set.points = o3d.utility.Vector3dVector(points[selected_point_indices])
    line_set.lines = o3d.utility.Vector2iVector(new_lines)

    return line_set

def _hair_modeling(script_path):
    result = subprocess.run(['sh', script_path])

    if result.returncode == 0:
        print("Success!")
    else:
        print("Failed!")
    # _img2hairstep()
    # _get_lmk()
    # _opt_cam()
    # _recon3D()


# def _hari_model_alignment(hair_lineset_path, head_mesh_path, smpl_model):
#     hair_lineset = o3d.io.read_line_set(hair_lineset_path)
#     # smpl_mesh = o3d.io.read_triangle_mesh(smpl_mesh_path)
#     head_mesh = o3d.io.read_triangle_mesh(head_mesh_path)
#     source_f = np.array(head_mesh.triangles)
#     # target_f = np.array(smpl_mesh.triangles)
#     source_v = np.array(head_mesh.vertices)
#     # target_v = np.array(smpl_mesh.vertices)
#     target_v = smpl_model
#     target_idxs = [
#         [1311, 1100, 2263],
#         [411, 446, 302],
#         [9011, 9003, 2132],
#         [8754, 8751, 8755]
#     ]
#     source_idxs = [6984, 5330, 11444, 6374]
#     source_keypoints = []
#     for source_face_idx in source_idxs:
#         points = [source_v[i] for i in source_f[source_face_idx]]
#         points = sum(points) / len(points)
#         source_keypoints.append(points)
#     target_keypoints = []
#     for target_idx in target_idxs:
#         points = [target_v[i] for i in target_idx]
#         points = sum(points) / len(points)
#         target_keypoints.append(points)
#     transform_matrix = find_similarity_transform_matrix(source_keypoints, target_keypoints)
#     # aligned_hair_mesh = hair_lineset.transform(transform_matrix)
#     # return np.asarray(aligned_hair_mesh.points), np.asarray(aligned_hair_mesh.lines)
#     return transform_matrix
def _hari_model_alignment(hair_lineset_path, head_mesh_path, smpl_model):
    hair_lineset = o3d.io.read_line_set(hair_lineset_path)
    # smpl_mesh = o3d.io.read_triangle_mesh(smpl_mesh_path)
    head_mesh = o3d.io.read_triangle_mesh(head_mesh_path)
    source_f = np.array(head_mesh.triangles)
    # target_f = np.array(smpl_mesh.triangles)
    source_v = np.array(head_mesh.vertices)
    # target_v = np.array(smpl_mesh.vertices)
    target_v = smpl_model
    target_idxs = [
        [1311, 1100, 2263],
        [411, 446, 302],
        [9003, 9003, 9003],
        [8952, 8952, 8952]
    ]
    source_f_idxs = [6984, 5330, 11444, 1093]
    source_idxs = [
        [source_f[source_f_idxs[0]][0], source_f[source_f_idxs[0]][1], source_f[source_f_idxs[0]][2]],
        [source_f[source_f_idxs[1]][0], source_f[source_f_idxs[1]][1], source_f[source_f_idxs[1]][2]],
        [source_f[source_f_idxs[2]][0], source_f[source_f_idxs[2]][0], source_f[source_f_idxs[2]][0]],
        [source_f[source_f_idxs[3]][1], source_f[source_f_idxs[3]][1], source_f[source_f_idxs[3]][1]]
    ]
    source_keypoints = []
    for k, source_idx in enumerate(source_idxs):
        points = [source_v[i] for i in source_idx]
        points = sum(points) / len(points)
        source_keypoints.append(points)
    target_keypoints = []
    for k, target_idx in enumerate(target_idxs):
        points = [target_v[i] for i in target_idx]
        points = sum(points) / len(points)
        target_keypoints.append(points)
    transform_matrix = find_similarity_transform_matrix(source_keypoints, target_keypoints)
    # aligned_hair_mesh = hair_lineset.transform(transform_matrix)
    # return np.asarray(aligned_hair_mesh.points), np.asarray(aligned_hair_mesh.lines)
    return transform_matrix

def hair_modeling(script_path):
    # exists_files = glob('results/real_imgs/*/*.*')
    # [subprocess.run(['rm', file]) for file in exists_files]
    # output_path = './results/real_imgs/img/output.png'
    # detect_and_crop_faces(image_path, output_path, scale=crop_scale)
    _hair_modeling(script_path)
    hair_path = './results/real_imgs/hair3D/output.ply'
    hair_model = o3d.io.read_line_set(hair_path)
    return np.asarray(hair_model.points), np.asarray(hair_model.lines)
    
def hair_model_alignment(smpl_model):
    hair_path = './results/real_imgs/hair3D/output.ply'
    head_path = './data/head_model.obj'
    transform_matrix = _hari_model_alignment(hair_path, head_path, smpl_model)
    return transform_matrix

def smpl_head_alignment(pre_smpl_verts, cur_smpl_verts, head_index):
    pre_smpl_head = pre_smpl_verts[head_index]
    cur_smpl_head = cur_smpl_verts[head_index]
    transform_matrix = find_similarity_transform_matrix(pre_smpl_head, cur_smpl_head, is_scale=False)
    return transform_matrix

    
if __name__ == '__main__':
    image_path = './temp_images/uploaded_image.png'
    script_path = './test.sh'
    hair_modeling(image_path, script_path, crop_scale=1.8)
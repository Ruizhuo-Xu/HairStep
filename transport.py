from flask import Flask, request, jsonify
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import os
from tqdm import tqdm
import open3d as o3d

from hair_process import hair_modeling, hair_model_alignment, smpl_head_alignment


app = Flask(__name__)
body_image = None

    
def base64_to_numpy(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return np.array(image)


@app.route('/upload', methods=['POST'])
def upload_image():
    print("request received!")

    data = request.get_json()
    if not data:
        return jsonify({'error': 'no input data provided'})

    if 'smplx_verts' not in data:
        return jsonify({"error": "smplx verts data not provided"}), 400

    batch_smplx_verts = np.array(data['smplx_verts'])
    # np.save('smpl_verts.npy', batch_smplx_verts)

    print(batch_smplx_verts.shape)

    # if 'image' not in data:
    #     return jsonify({"error": "Image data not provided"}), 400

    # base64_string = data['image']
    # image_array = base64_to_numpy(base64_string)
    # save_path = './temp_images'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)

    # image_filename = 'uploaded_image.png'  # 替换为你想使用的图像文件名
    # image_path = os.path.join(save_path, image_filename)

    # Image.fromarray(image_array).save(image_path)
    # print("image saved:" + image_path)
    script_path = './test.sh'
    points, lines = hair_modeling(script_path)
    print("hair model done!")

    matrices = []

    head_path = './data/head_model.obj'
    head_mesh = o3d.io.read_triangle_mesh(head_path)
    offset = batch_smplx_verts.shape[1] - 10475
    matrix_0 = hair_model_alignment(head_mesh, batch_smplx_verts[0], offset=offset)
    matrices.append(matrix_0.tolist())
    ref_smpl_verts = batch_smplx_verts[0]
    smplx_head_index = np.load('./data/SMPL-X__FLAME_vertex_ids.npy')
    smplx_head_index = smplx_head_index + offset
    # smplx_head_index = [30838, 30839, 30837, 29877, ]
    for smplx_verts in tqdm(batch_smplx_verts[1:]):
        transform_matrix = smpl_head_alignment(ref_smpl_verts, smplx_verts, smplx_head_index)
        # transform_matrix = hair_model_alignment(head_mesh, smplx_verts)
        transform_matrix = transform_matrix @ matrix_0
        matrices.append(transform_matrix.tolist())
    # for smplx_verts in tqdm(batch_smplx_verts[:]):
    #     transform_matrix = hair_model_alignment(smplx_verts)
    #     matrices.append(transform_matrix.tolist())

    return jsonify({"points": points.tolist(),
                    "lines": lines.tolist(),
                    "matrices": matrices})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5101)
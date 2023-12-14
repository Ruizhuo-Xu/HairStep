from flask import Flask, request, jsonify
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import os
from tqdm import tqdm
import subprocess
from glob import glob
from utils import detect_and_crop_faces

app = Flask(__name__)
body_image = None
    
def base64_to_numpy(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return np.array(image)


@app.route('/test', methods=['POST'])
def detect_and_crop():
    print("request received!")

    data = request.get_json()
    if not data:
        return jsonify({'error': 'no input data provided'})

    # if 'smplx_verts' not in data:
    #     return jsonify({"error": "smplx verts data not provided"}), 400

    # batch_smplx_verts = np.array(data['smplx_verts'])
    # np.save('smpl_verts.npy', batch_smplx_verts)

    # print(batch_smplx_verts.shape)

    if 'image' not in data:
        return jsonify({"error": "Image data not provided"}), 400

    base64_string = data['image']
    image_array = base64_to_numpy(base64_string)
    save_path = './temp_images'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_filename = 'uploaded_image.png'  # 替换为你想使用的图像文件名
    image_path = os.path.join(save_path, image_filename)

    Image.fromarray(image_array).save(image_path)
    print("image saved:" + image_path)
    save_path = '../results/real_imgs/img'
    cropped_image_filename = 'output.png'
    cropped_image_path = os.path.join(save_path, cropped_image_filename)

    exists_files = glob('../results/real_imgs/*/*.*')
    [subprocess.run(['rm', file]) for file in exists_files]
    script_path = './run.sh'
    result = subprocess.run(['sh', script_path])
    if result.returncode == 0:
        return jsonify({"msg": 0})
    else:
        return jsonify({"msg": -1})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5100)
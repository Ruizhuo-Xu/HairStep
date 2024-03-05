docker run -itd \
    --gpus all \
    --net host \
    -w /workspace \
    -v /home/rz/HairStep:/workspace \
    hairstep:1.1 \
    /bin/bash -c "source ~/.bashrc && python transport.py"

docker run -itd \
    --gpus all \
    --net host \
    -w /workspace \
    -v /home/rz/HairStep:/workspace \
    face_detect \
    python face_detect/detect_and_crop_face.py
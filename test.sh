CUDA_VISIBLE_DEVICES=3 python -m scripts.img2hairstep
CUDA_VISIBLE_DEVICES=3 python scripts/get_lmk.py
CUDA_VISIBLE_DEVICES=3 python -m scripts.opt_cam
CUDA_VISIBLE_DEVICES=3 python -m scripts.recon3D

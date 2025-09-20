python convert.py -s D:\gaussian-splatting\data\20imgonlybuild\ --mask_path D:\gaussian-splatting\data\20imgonlybuild\masks\ --colmap_executable "D:\colmap-x64-windows-cuda\bin\colmap.exe"




python train.py -s <data dir>

python render.py -s D:\gaussian-splatting\data\20imgonlybuild\ -m D:\gaussian-splatting\output\6652a1d4-6\ --iteration 30000 --eval


python metrics.py -m D:\gaussian-splatting\output\20tree\
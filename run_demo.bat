@echo off
setlocal
python train.py --data data/sample_turkish.txt --out-dir checkpoints/demo --max-iters 300 --eval-interval 50 --eval-iters 10 --batch-size 16 --block-size 64 --n-layer 2 --n-head 2 --n-embd 64
if errorlevel 1 exit /b %errorlevel%
python generate.py --checkpoint checkpoints/demo/best.pt --prompt "Bilim" --max-new-tokens 300 --temperature 0.8 --top-k 20
endlocal

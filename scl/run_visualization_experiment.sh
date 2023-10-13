# CIFAR10 ViT
python main_scl_visualization.py \
--dataset cifar10 \
--imgsize 32 \
--batchsize 256 \
--numworkers 0 \
--valsplit 0.01 \
--checkpoint_path tb_logs/scl/resnet18_cifar10_batch_512_Ncoeff_0.78_embeddim_256/version_3/checkpoints/last.ckpt \
--plot_name test
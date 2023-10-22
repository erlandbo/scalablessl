# CIFAR10 ViT
python main_supervised.py \
--dataset stl10 \
--modelarch resnet18 \
--maxpool1 \
--first_conv \
--embed_dim 128 \
--imgsize 96 \
--batchsize 256 \
--numworkers 20 \
--lr 3e-4 \
--scheduler None \
--optimizer adam \
--valsplit 0.05 \
--maxepochs 50 \
# --checkpoint_path tb_logs/scl/resnet18_stl10_batch_256_Ncoeff_0.78_embeddim_128/version_4/checkpoints/last.ckpt \


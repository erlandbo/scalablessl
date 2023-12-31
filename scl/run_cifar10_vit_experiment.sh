# CIFAR10 ViT
python main_scl.py \
--dataset cifar10 \
--modelarch vittorch \
--in_channels 3 \
--embed_dim 128 \
--transformer_patchdim 4 \
--transformer_numlayers 8 \
--transformer_dmodel 512 \
--transformer_nhead 8 \
--transformer_dff_ration 4 \
--transformer_dropout 0.1 \
--transformer_activation relu \
--jitterstrength 0.5 \
--gausblur \
--imgsize 32 \
--batchsize 128 \
--numworkers 20 \
--lr 3e-4 \
--scheduler linwarmup_cosanneal \
--optimizer adam \
--alpha 0.5 \
--titer 1_000_000 \
--ncoeff 0.7 \
--simmetric gaussian \
--clamp 50

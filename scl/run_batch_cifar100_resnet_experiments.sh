# CIFAR100 resnet18
# python main_scl.py \
# --dataset cifar100 \
# --modelarch resnet18 \
# --embed_dim 128 \
# --jitterstrength 0.5 \
# --imgsize 32 \
# --batchsize 512 \
# --numworkers 20 \
# --lr 3e-4 \
# --scheduler None \
# --optimizer adam \
# --valsplit 0.01 \
# --alpha 0.5 \
# --titer 1_000_000 \
# --ncoeff 0.78 \
# --sinv_init_coeff 2.0 \
# --simmetric gaussian \
# --var 2.0 \
# --maxepochs 1000 \
# --finetune_lr 3e-4 \
# --finetune_batchsize 512 \
# --finetune_knn \
# --finetune_linear \
# --finetune_interval 10 \
# --finetune_n_neighbours 20 \

python main_scl.py \
--dataset cifar100 \
--modelarch resnet18 \
--embed_dim 128 \
--jitterstrength 0.5 \
--imgsize 32 \
--batchsize 256 \
--numworkers 20 \
--lr 3e-4 \
--scheduler None \
--optimizer adam \
--valsplit 0.01 \
--alpha 0.5 \
--titer 1_000_000 \
--ncoeff 0.7 \
--sinv_init_coeff 2.0 \
--simmetric gaussian \
--var 2.0 \
--maxepochs 1000 \
--finetune_lr 3e-4 \
--finetune_batchsize 512 \
--finetune_knn \
--finetune_linear \
--finetune_interval 10 \
--finetune_n_neighbours 20 \

python main_scl.py \
--dataset cifar100 \
--modelarch resnet18 \
--embed_dim 128 \
--jitterstrength 0.5 \
--imgsize 32 \
--batchsize 128 \
--numworkers 20 \
--lr 3e-4 \
--scheduler None \
--optimizer adam \
--valsplit 0.01 \
--alpha 0.5 \
--titer 1_000_000 \
--ncoeff 0.65 \
--sinv_init_coeff 2.0 \
--simmetric gaussian \
--var 2.0 \
--maxepochs 1000 \
--finetune_lr 3e-4 \
--finetune_batchsize 512 \
--finetune_knn \
--finetune_linear \
--finetune_interval 10 \
--finetune_n_neighbours 20 \

python main_scl.py \
--dataset cifar100 \
--modelarch resnet18 \
--embed_dim 128 \
--jitterstrength 0.5 \
--imgsize 32 \
--batchsize 64 \
--numworkers 20 \
--lr 3e-4 \
--scheduler None \
--optimizer adam \
--valsplit 0.01 \
--alpha 0.5 \
--titer 1_000_000 \
--ncoeff 0.6 \
--sinv_init_coeff 2.0 \
--simmetric gaussian \
--var 2.0 \
--maxepochs 1000 \
--finetune_lr 3e-4 \
--finetune_batchsize 512 \
--finetune_knn \
--finetune_linear \
--finetune_interval 10 \
--finetune_n_neighbours 20 \

python main_scl.py \
--dataset cifar100 \
--modelarch resnet18 \
--embed_dim 128 \
--jitterstrength 0.5 \
--imgsize 32 \
--batchsize 32 \
--numworkers 20 \
--lr 3e-4 \
--scheduler None \
--optimizer adam \
--valsplit 0.01 \
--alpha 0.5 \
--titer 1_000_000 \
--ncoeff 0.525 \
--sinv_init_coeff 2.0 \
--simmetric gaussian \
--var 2.0 \
--maxepochs 1000 \
--finetune_lr 3e-4 \
--finetune_batchsize 512 \
--finetune_knn \
--finetune_linear \
--finetune_interval 10 \
--finetune_n_neighbours 20 \

python main_scl.py \
--dataset cifar100 \
--modelarch resnet18 \
--embed_dim 128 \
--jitterstrength 0.5 \
--imgsize 32 \
--batchsize 8 \
--numworkers 20 \
--lr 3e-4 \
--scheduler None \
--optimizer adam \
--valsplit 0.01 \
--alpha 0.5 \
--titer 1_000_000 \
--ncoeff 0.4 \
--sinv_init_coeff 2.0 \
--simmetric gaussian \
--var 2.0 \
--maxepochs 1000 \
--finetune_lr 3e-4 \
--finetune_batchsize 512 \
--finetune_knn \
--finetune_linear \
--finetune_interval 10 \
--finetune_n_neighbours 20 \

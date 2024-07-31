m='ml-100k'
y='yelp'
b='beer'
c='Amazon_Beauty'
a='ml-1m'
t='taobao'
i='ijcai'
s='steam'

# nohup python run_recbole_layer.py --dataset=${s} --integra=add --pooling_mode=mean  --model=ours4_no  --weight_norm=0.5 --num_layers=4 --gpu_id=0>>layerout/8s01.out&
# nohup python run_recbole_layer.py --dataset=${s} --integra=add --pooling_mode=mean  --model=ours4_realmore0  --weight_norm=0.5 --num_layers=4 --gpu_id=1>>layerout/8s01.out&
# nohup python run_recbole_layer.py --dataset=${s} --integra=add --pooling_mode=mean  --model=ours4_realmore01  --weight_norm=0.5 --num_layers=4 --gpu_id=2>>layerout/8s02.out&
# nohup python run_recbole_layer.py --dataset=${s} --integra=add --pooling_mode=mean  --model=ours4_realmore1  --weight_norm=0.5 --num_layers=4 --gpu_id=2>>layerout/8s02.out&
# nohup python run_recbole_layer.py --dataset=${s} --integra=add --pooling_mode=mean  --model=sasrec_  --weight_norm=0.5 --num_layers=4 --gpu_id=2>>layerout/8s02.out&


# nohup python run_recbole_layer.py --dataset=${s} --integra=add --pooling_mode=mean  --model=ours4_no  --weight_norm=0.5 --num_layers=3 --gpu_id=0>>layerout/8s01.out&
# # nohup python run_recbole_layer.py --dataset=${s} --integra=add --pooling_mode=mean  --model=ours4_realmore0  --weight_norm=0.5 --num_layers=3 --gpu_id=3>>layerout/8s03.out&
# # nohup python run_recbole_layer.py --dataset=${s} --integra=add --pooling_mode=mean  --model=ours4_realmore01 --weight_norm=0.5 --num_layers=3 --gpu_id=4>>layerout/8s04.out&
# nohup python run_recbole_layer.py --dataset=${s} --integra=add --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=0.5 --num_layers=3 --gpu_id=1>>layerout/8s04.out&
# nohup python run_recbole_layer.py --dataset=${s} --integra=add --pooling_mode=mean  --model=ours4_realmore3 --weight_norm=0.5 --num_layers=3 --gpu_id=2>>layerout/8s0000.out&
# nohup python run_recbole_layer.py --dataset=${s} --integra=add --pooling_mode=mean  --model=sasrec_  --weight_norm=0.5 --num_layers=3 --gpu_id=3>>layerout/8s02.out&

# nohup python run_recbole_layer.py --dataset=${s} --integra=add --pooling_mode=mean  --model=ours4_no  --weight_norm=0.5 --num_layers=2 --gpu_id=4>>layerout/8s01.out&
# nohup python run_recbole_layer.py --dataset=${s} --integra=add --pooling_mode=mean  --model=ours4_realmore0  --weight_norm=0.5 --num_layers=2 --gpu_id=5>>layerout/8s05.out&
# nohup python run_recbole_layer.py --dataset=${s} --integra=add --pooling_mode=mean  --model=ours4_realmore01  --weight_norm=0.5 --num_layers=2 --gpu_id=6>>layerout/8s06.out&
# nohup python run_recbole_layer.py --dataset=${s} --integra=add --pooling_mode=mean  --model=ours4_sas  --weight_norm=0.5 --num_layers=2 --gpu_id=5>>layerout/8s01.out&
# nohup python run_recbole_layer.py --dataset=${s} --integra=add --pooling_mode=mean  --model=ours4_dif  --weight_norm=0.5 --num_layers=2 --gpu_id=6>>layerout/8s01.out&
# nohup python run_recbole_layer.py --dataset=${s} --integra=add --pooling_mode=mean  --model=ours4_realmore2  --weight_norm=0.5 --num_layers=2 --gpu_id=5>>layerout/8s06.out&
# nohup python run_recbole_layer.py --dataset=${s} --integra=add --pooling_mode=mean  --model=ours4_realmore3  --weight_norm=0.5 --num_layers=2 --gpu_id=6>>layerout/8s06.out&
# nohup python run_recbole_layer.py --dataset=${s} --integra=add --pooling_mode=mean  --model=sasrec_  --weight_norm=0.5 --num_layers=2 --gpu_id=7>>layerout/8s06.out&


# nohup python run_recbole_layer.py --dataset=${s} --integra=add --pooling_mode=mean  --model=ours4_no  --weight_norm=0.5 --num_layers=1 --gpu_id=9>>layerout/8s01.out&
# # nohup python run_recbole_layer.py --dataset=${s} --integra=add --pooling_mode=mean  --model=ours4_realmore0  --weight_norm=0.5 --num_layers=1 --gpu_id=7>>layerout/8s07.out&
# # nohup python run_recbole_layer.py --dataset=${s} --integra=add --pooling_mode=mean  --model=ours4_realmore01  --weight_norm=0.5 --num_layers=1 --gpu_id=8>>layerout/8s08.out&
# nohup python run_recbole_layer.py --dataset=${s} --integra=add --pooling_mode=mean  --model=ours4_realmore2  --weight_norm=0.5 --num_layers=1 --gpu_id=10>>layerout/8s08.out&
# nohup python run_recbole_layer.py --dataset=${s} --integra=add --pooling_mode=mean  --model=ours4_realmore3  --weight_norm=0.5 --num_layers=1 --gpu_id=11>>layerout/8s08.out&
# nohup python run_recbole_layer.py --dataset=${s} --integra=add --pooling_mode=mean  --model=sasrec_  --weight_norm=0.5 --num_layers=1 --gpu_id=8>>layerout/8s08.out&

# nohup python run_recbole_layer.py --dataset=${y} --integra=add --pooling_mode=mean  --model=sasrec_  --weight_norm=0.1 --num_layers=1 --gpu_id=11>>layerout/8y11.out&
# nohup python run_recbole_layer.py --dataset=${y} --integra=add --pooling_mode=mean  --model=ours4_realmore2  --weight_norm=0.1 --num_layers=1 --gpu_id=10>>layerout/8y10.out&

# nohup python run_recbole_layer.py --dataset=${y} --integra=add --pooling_mode=mean  --model=sasrec_  --weight_norm=0.1 --num_layers=2 --gpu_id=9>>layerout/8y9.out&
# nohup python run_recbole_layer.py --dataset=${y} --integra=add --pooling_mode=mean  --model=ours4_realmore2  --weight_norm=0.1 --num_layers=2 --gpu_id=8>>layerout/8y8.out&

# nohup python run_recbole_layer.py --dataset=${y} --integra=add --pooling_mode=mean  --model=sasrec_  --weight_norm=0.1 --num_layers=3 --gpu_id=7>>layerout/8y7.out&
# nohup python run_recbole_layer.py --dataset=${y} --integra=add --pooling_mode=mean  --model=ours4_realmore2  --weight_norm=0.1 --num_layers=3 --gpu_id=6>>layerout/8y6.out&

# nohup python run_recbole_layer.py --dataset=${y} --integra=add --pooling_mode=mean  --model=sasrec_  --weight_norm=0.1 --num_layers=4 --gpu_id=5>>layerout/8y5.out&


# nohup python run_recbole_layer.py --dataset=${c} --integra=add --pooling_mode=mean  --model=sasrec_  --weight_norm=0.5 --num_layers=1 --gpu_id=0>>layerout/8s08.out&
# nohup python run_recbole_layer.py --dataset=${c} --integra=add --pooling_mode=mean  --model=ours4_realmore2  --weight_norm=0.5 --num_layers=1 --gpu_id=0>>layerout/8s08.out&

# nohup python run_recbole_layer.py --dataset=${c} --integra=add --pooling_mode=mean  --model=sasrec_  --weight_norm=0.5 --num_layers=2 --gpu_id=1>>layerout/8s08.out&
# nohup python run_recbole_layer.py --dataset=${c} --integra=add --pooling_mode=mean  --model=ours4_realmore2  --weight_norm=0.5 --num_layers=2 --gpu_id=1>>layerout/8s08.out&


# nohup python run_recbole_layer.py --dataset=${c} --integra=add --pooling_mode=mean  --model=sasrec_  --weight_norm=0.5 --num_layers=3 --gpu_id=2>>layerout/8s08.out&
# nohup python run_recbole_layer.py --dataset=${c} --integra=add --pooling_mode=mean  --model=ours4_realmore2  --weight_norm=0.5 --num_layers=3 --gpu_id=4>>layerout/8s08.out&

# nohup python run_recbole_layer.py --dataset=${c} --integra=add --pooling_mode=mean  --model=sasrec_  --weight_norm=0.5 --num_layers=4 --gpu_id=3>>layerout/8s08.out&

# nohup python run_recbole_layer.py --dataset=${c} --integra=add --pooling_mode=mean  --model=ours4_realmore2  --weight_norm=0.5 --num_layers=1 --gpu_id=0>>layerout/8s08.out&

# nohup python run_recbole_layer.py --dataset=${c} --integra=add --pooling_mode=mean  --model=sasrec_  --weight_norm=0.5 --num_layers=2 --gpu_id=1>>layerout/8s08.out&
# nohup python run_recbole_layer.py --dataset=${c} --integra=add --pooling_mode=mean  --model=ours4_realmore2  --weight_norm=0.5 --num_layers=2 --gpu_id=1>>layerout/8s08.out&


nohup python run_recbole_layer.py --dataset=${a} --integra=add --pooling_mode=sum  --model=ours4_realmore2  --weight_norm=0.5 --num_layers=2 --gpu_id=3>>layerout/8a3.out&
nohup python run_recbole_layer.py --dataset=${a} --integra=add --pooling_mode=sum  --model=ours4_realmore2  --weight_norm=0.5 --num_layers=1 --gpu_id=10>>layerout/8a10.out&

nohup python run_recbole_layer.py --dataset=${a} --integra=add --pooling_mode=sum  --model=ours4_realmore2 --weight_norm=0.5 --num_layers=3 --gpu_id=11>>layerout/8a11.out&
m='ml-100k'
y='yelp'
b='beer'
c='Amazon_Beauty'
a='ml-1m'
t='taobao'
i='ijcai'
s='steam'

n='base'


nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=sasrec_item --weight_norm=0.5  --gpu_id=0>>baseout1/item00.out&
nohup python run_recbole_${n}.py --dataset=${c} --integra=add --pooling_mode=mean  --model=sasrec_item --weight_norm=0.5  --gpu_id=1>>baseout1/item01.out&
nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=mean  --model=sasrec_item --weight_norm=0.5  --gpu_id=2>>baseout1/item02.out&

nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=sasrec_posf --weight_norm=0.5  --gpu_id=3>>baseout1/pf00.out&
nohup python run_recbole_${n}.py --dataset=${c} --integra=add --pooling_mode=mean  --model=sasrec_posf --weight_norm=0.5  --gpu_id=4>>baseout1/pf01.out&
nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=mean  --model=sasrec_posf --weight_norm=0.5  --gpu_id=5>>baseout1/pf02.out&

nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=sasrec_posa --weight_norm=0.5  --gpu_id=6>>baseout1/pa00.out&
nohup python run_recbole_${n}.py --dataset=${c} --integra=add --pooling_mode=mean  --model=sasrec_posa --weight_norm=0.5  --gpu_id=7>>baseout1/pa01.out&
nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=mean  --model=sasrec_posa --weight_norm=0.5  --gpu_id=8>>baseout1/pa02.out&

nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=sasrec_pos --weight_norm=0.5  --gpu_id=9>>baseout1/pa00.out&
nohup python run_recbole_${n}.py --dataset=${c} --integra=add --pooling_mode=mean  --model=sasrec_pos --weight_norm=0.5  --gpu_id=10>>baseout1/pa01.out&
nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=mean  --model=sasrec_pos --weight_norm=0.5  --gpu_id=11>>baseout1/pa02.out&

nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=sasrec_posv --weight_norm=0.5  --gpu_id=0>>baseout1/pv00.out&
nohup python run_recbole_${n}.py --dataset=${c} --integra=add --pooling_mode=mean  --model=sasrec_posv --weight_norm=0.5  --gpu_id=1>>baseout1/pv01.out&
nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=mean  --model=sasrec_posv --weight_norm=0.5  --gpu_id=2>>baseout1/pv02.out&


nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=sasrec_itemf --weight_norm=0.5  --gpu_id=3>>baseout1/if00.out&
nohup python run_recbole_${n}.py --dataset=${c} --integra=add --pooling_mode=mean  --model=sasrec_itemf --weight_norm=0.5  --gpu_id=4>>baseout1/if01.out&
nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=mean  --model=sasrec_itemf --weight_norm=0.5  --gpu_id=5>>baseout1/if02.out&


nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=sasrec_catea --weight_norm=0.5  --gpu_id=6>>baseout1/ca00.out&
nohup python run_recbole_${n}.py --dataset=${c} --integra=add --pooling_mode=mean  --model=sasrec_catea --weight_norm=0.5  --gpu_id=7>>baseout1/ca01.out&
nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=mean  --model=sasrec_catea --weight_norm=0.5  --gpu_id=8>>baseout1/ca02.out&


nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=sasrec_cate --weight_norm=0.5  --gpu_id=9>>baseout1/cate01.out&
nohup python run_recbole_${n}.py --dataset=${c} --integra=add --pooling_mode=mean  --model=sasrec_cate --weight_norm=0.5  --gpu_id=10>>baseout1/cate02.out&
nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=mean  --model=sasrec_cate --weight_norm=0.5  --gpu_id=11>>baseout1/cate03.out&


nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=sasrec_catev --weight_norm=0.5  --gpu_id=3>>baseout1/cv00.out&
nohup python run_recbole_${n}.py --dataset=${c} --integra=add --pooling_mode=mean  --model=sasrec_catev --weight_norm=0.5  --gpu_id=4>>baseout1/cv01.out&
nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=mean  --model=sasrec_catev --weight_norm=0.5  --gpu_id=5>>baseout1/cv02.out&



nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=sasrec_catey --weight_norm=0.5  --gpu_id=6>>baseout1/cy00.out&
nohup python run_recbole_${n}.py --dataset=${c} --integra=add --pooling_mode=mean  --model=sasrec_catey --weight_norm=0.5  --gpu_id=7>>baseout1/cy01.out&
nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=mean  --model=sasrec_catey --weight_norm=0.5  --gpu_id=8>>baseout1/cy02.out&













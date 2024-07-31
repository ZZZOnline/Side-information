m='ml-100k'
y='yelp'
b='beer'
c='Amazon_Beauty'
a='ml-1m'
t='taobao'
i='ijcai'
s='steam'


nohup python run_recbole_head.py --dataset=${s} --integra=add --pooling_mode=mean  --model=ours4_no --p_heads=4 --weight_norm=0.5 --gpu_id=7>>disenout/8s07.out&
nohup python run_recbole_head.py --dataset=${s} --integra=add --pooling_mode=mean  --model=ours4_sas1 --p_heads=4 --weight_norm=0.5 --gpu_id=5>>disenout/8s05.out&
nohup python run_recbole_head.py --dataset=${s} --integra=add --pooling_mode=mean  --model=ours4_dif1 --p_heads=4 --weight_norm=0.5 --gpu_id=6>>disenout/8s06.out&
m='ml-100k'
y='yelp'
b='beer'
c='Amazon_Beauty'
a='ml-1m'
t='taobao'
i='ijcai'
s='steam'

n='head0'

# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=0.1 --p_heads=1 --gpu_id=1>>headout/y1.out&
# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=0.1  --p_heads=2 --gpu_id=5>>headout/y5.out&
# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=0.1 --p_heads=4 --gpu_id=6>>headout/y6.out&
# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=0.1 --p_heads=8 --gpu_id=7>>headout/y7.out&
# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=0.1 --p_heads=16 --gpu_id=10>>headout/y10.out&
# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=0.1 --p_heads=32 --gpu_id=11>>headout/y11.out&
# nohup python run_recbole_${n}.py --dataset=${y} --integra=item --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=0.1 --p_heads=64 --gpu_id=10>>headout/s3.out&

# nohup python run_recbole_${n}.py --dataset=${c} --integra=add --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=0.5 --p_heads=1 --gpu_id=4>>headout/c1.out&
# nohup python run_recbole_${n}.py --dataset=${c} --integra=add --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=0.5  --p_heads=2 --gpu_id=5>>headout/c5.out&
# nohup python run_recbole_${n}.py --dataset=${c} --integra=add --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=0.5 --p_heads=4 --gpu_id=6>>headout/c6.out&
# nohup python run_recbole_${n}.py --dataset=${c} --integra=add --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=0.5 --p_heads=8 --gpu_id=7>>headout/c7.out&
# nohup python run_recbole_${n}.py --dataset=${c} --integra=add --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=0.5 --p_heads=16 --gpu_id=10>>headout/c10.out&
# nohup python run_recbole_${n}.py --dataset=${c} --integra=add --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=0.5 --p_heads=32 --gpu_id=11>>headout/c11.out&


# nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=0.5 --p_heads=1 --gpu_id=4>>headout/s1.out&
# nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=0.5  --p_heads=2 --gpu_id=5>>headout/s5.out&
# nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=0.5 --p_heads=4 --gpu_id=6>>headout/s6.out&
# nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=0.5 --p_heads=8 --gpu_id=7>>headout/s7.out&
# nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=0.5 --p_heads=16 --gpu_id=10>>headout/s10.out&
# nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=0.5 --p_heads=32 --gpu_id=11>>headout/s11.out&

# nohup python run_recbole_${n}.py --dataset=${a} --integra=add --pooling_mode=sum  --model=ours4_realmore2 --p_heads=1 --weight_norm=0.5 --gpu_id=0>>8a0.out&
# nohup python run_recbole_${n}.py --dataset=${a} --integra=add --pooling_mode=sum  --model=ours4_realmore2 --p_heads=2 --weight_norm=0.5 --gpu_id=1>>8a1.out&
# nohup python run_recbole_${n}.py --dataset=${a} --integra=add --pooling_mode=sum  --model=ours4_realmore2 --p_heads=4 --weight_norm=0.5 --gpu_id=2>>8a2.out&
# nohup python run_recbole_${n}.py --dataset=${a} --integra=add --pooling_mode=sum  --model=ours4_realmore2 --p_heads=8 --weight_norm=0.5 --gpu_id=4>>8a4.out&
# nohup python run_recbole_${n}.py --dataset=${a} --integra=add --pooling_mode=sum  --model=ours4_realmore2 --p_heads=16 --weight_norm=0.5 --gpu_id=5>>8a5.out&
# nohup python run_recbole_${n}.py --dataset=${a} --integra=add --pooling_mode=sum  --model=ours4_realmore2 --p_heads=32 --weight_norm=0.5 --gpu_id=6>>8a6.out&

nohup python run_recbole_${n}.py --dataset=taobao-20000 --integra=add --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=1 --p_heads=1 --gpu_id=1>>headout/t1.out&
nohup python run_recbole_${n}.py --dataset=taobao-20000 --integra=add --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=1 --p_heads=2 --gpu_id=2>>headout/t2.out&
nohup python run_recbole_${n}.py --dataset=taobao-20000 --integra=add --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=1 --p_heads=4 --gpu_id=4>>headout/t4.out&
nohup python run_recbole_${n}.py --dataset=taobao-20000 --integra=add --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=1 --p_heads=8 --gpu_id=8>>headout/t8.out&
nohup python run_recbole_${n}.py --dataset=taobao-20000 --integra=add --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=1 --p_heads=16 --gpu_id=10>>headout/t10.out&
nohup python run_recbole_${n}.py --dataset=taobao-20000 --integra=add --pooling_mode=mean  --model=ours4_realmore2 --weight_norm=1 --p_heads=32 --gpu_id=9>>headout/t9.out&

m='ml-100k'
y='yelp'
b='beer'
c='Amazon_Beauty'
a='ml-1m'
t='taobao'
i='ijcai'
s='steam'

n='base'

# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=adamct_decode1 --weight_norm=0.1 --gpu_id=3>>baseout/y3.out&
# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=gru4rec_decode --weight_norm=0.1  --gpu_id=4>>baseout/y4.out&
# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=nova_decode --weight_norm=0.1  --gpu_id=5>>baseout/y5.out&
# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=difsr_decode1 --weight_norm=0.1  --gpu_id=6>>baseout/y6.out&
# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=mlp4rec_decode1 --weight_norm=0.1 --gpu_id=7>>baseout/y07.out&
# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=fdsa_decode --weight_norm=0.1  --gpu_id=10>>baseout/y10.out&
# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=mlp4rec --weight_norm=0.1 --gpu_id=5>>baseout/y05.out&
# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=fdsa --weight_norm=0.1  --gpu_id=6>>baseout/y6.out&
# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=bert4rec_decode --weight_norm=0.1  --gpu_id=11>>baseout/y11.out&

# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=ours4_realmore22 --weight_norm=0.1  --gpu_id=1>>baseout/y0001.out&
# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=ours4_realmore22both0 --weight_norm=0.1  --gpu_id=2>>baseout/y0002.out&
# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=ours4_realmore2aw --weight_norm=0.1  --gpu_id=0>>baseout/y0000.out&

# nohup python run_recbole_${n}.py --dataset=${c} --integra=add --pooling_mode=mean  --model=adamct_decode1 --weight_norm=0.5 --gpu_id=7>>baseout/y3.out&
# nohup python run_recbole_${n}.py --dataset=${c} --integra=add --pooling_mode=mean  --model=gru4rec_decode --weight_norm=0.5  --gpu_id=9>>baseout/y4.out&
# nohup python run_recbole_${n}.py --dataset=${c} --integra=add --pooling_mode=mean  --model=nova_decode --weight_norm=0.5  --gpu_id=10>>baseout/y5.out&
# nohup python run_recbole_${n}.py --dataset=${c} --integra=add --pooling_mode=mean  --model=difsr_decode1 --weight_norm=0.5  --gpu_id=11>>baseout/y6.out&
# # nohup python run_recbole_${n}.py --dataset=${c} --integra=add --pooling_mode=mean  --model=mlp4rec_decode1 --weight_norm=0.5 --gpu_id=7>>baseout/y7.out&
# nohup python run_recbole_${n}.py --dataset=${c} --integra=add --pooling_mode=mean  --model=fdsa_decode --weight_norm=0.5  --gpu_id=3>>baseout/c3.out&
# nohup python run_recbole_${n}.py --dataset=${c} --integra=add --pooling_mode=mean  --model=bert4rec_decode --weight_norm=0.5  --gpu_id=9>>baseout/c9.out&

# nohup python run_recbole_${n}.py --dataset=${c} --integra=add --pooling_mode=mean  --model=fdsa --weight_norm=0.5  --gpu_id=11>>baseout/c11.out&

# nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=mean  --model=adamct_decode1 --weight_norm=0.5 --gpu_id=10>>baseout/y3.out&
# nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=mean  --model=gru4rec_decode --weight_norm=0.5  --gpu_id=11>>baseout/y4.out&
# nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=mean  --model=nova_decode --weight_norm=0.5  --gpu_id=4>>baseout/y5.out&
# nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=mean  --model=difsr_decode1 --weight_norm=0.5  --gpu_id=9>>baseout/y6.out&
# nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=mean  --model=mlp4rec_decode1 --weight_norm=0.5 --gpu_id=7>>baseout/y7.out&
# nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=mean  --model=fdsa_decode --weight_norm=0.5  --gpu_id=0>>baseout/s0.out&
# nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=mean  --model=bert4rec_decode --weight_norm=0.5  --gpu_id=8>>baseout/s8.out&
# nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=mean  --model=sasrec_ --weight_norm=0.5  --gpu_id=8>>baseout/s008.out&

# nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=mean  --model=gru4rec_decode --weight_norm=0.5 --p_heads=8 --gpu_id=5>>baseout/s0005.out&
# nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=mean  --model=fdsa --weight_norm=0.5  --gpu_id=1>>baseout/s100.out&
# nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=mean  --model=bert4rec --weight_norm=0.5  --gpu_id=8>>baseout/s800.out&

# nohup python run_recbole_${n}.py --dataset=${a} --integra=add --pooling_mode=sum  --model=adamct_decode1 --weight_norm=0.5 --gpu_id=6>>baseout/a3.out&
# nohup python run_recbole_${n}.py --dataset=${a} --integra=add --pooling_mode=sum  --model=gru4rec_decode --weight_norm=0.5  --gpu_id=7>>baseout/a7.out&
# nohup python run_recbole_${n}.py --dataset=${a} --integra=add --pooling_mode=sum  --model=nova_decode --weight_norm=0.5  --gpu_id=8>>baseout/a8.out&
# nohup python run_recbole_${n}.py --dataset=${a} --integra=add --pooling_mode=sum  --model=difsr_decode1 --weight_norm=0.5  --gpu_id=9>>baseout/a9.out&
# nohup python run_recbole_${n}.py --dataset=${a} --integra=add --pooling_mode=sum  --model=mlp4rec_decode1 --weight_norm=0.5 --gpu_id=10>>baseout/a10.out&
# nohup python run_recbole_${n}.py --dataset=${a} --integra=add --pooling_mode=sum  --model=fdsa_decode --weight_norm=0.5  --gpu_id=11>>baseout/a11.out&
# nohup python run_recbole_${n}.py --dataset=${a} --integra=add --pooling_mode=sum  --model=bert4rec_decode --weight_norm=0.5  --gpu_id=3>>baseout/a3.out&


# nohup python run_recbole_${n}.py --dataset=${a} --integra=add --pooling_mode=sum  --model=mlp4rec --weight_norm=0.5 --gpu_id=7>>baseout/a07.out&
# nohup python run_recbole_${n}.py --dataset=${a} --integra=add --pooling_mode=sum  --model=fdsa --weight_norm=0.5  --gpu_id=6>>baseout/a06.out&
# nohup python run_recbole_${n}.py --dataset=${a} --integra=add --pooling_mode=sum  --model=bert4rec --weight_norm=0.5  --gpu_id=8>>baseout/a08.out&
# nohup python run_recbole_${n}.py --dataset=${a} --integra=add --pooling_mode=sum  --model=adamct --weight_norm=0.5 --gpu_id=9>>baseout/a09.out&
# nohup python run_recbole_${n}.py --dataset=${a} --integra=add --pooling_mode=sum  --model=gru4rec --weight_norm=0.5  --gpu_id=10>>baseout/a010.out&
# nohup python run_recbole_${n}.py --dataset=${a} --integra=add --pooling_mode=sum  --model=nova_ --weight_norm=0.5  --gpu_id=3>>baseout/a03.out&
# nohup python run_recbole_${n}.py --dataset=${a} --integra=add --pooling_mode=sum  --model=difsr --weight_norm=0.5  --gpu_id=11>>baseout/a011.out&
# nohup python run_recbole_${n}.py --dataset=${a} --integra=add --pooling_mode=sum  --model=sasrec_ --weight_norm=0.5  --gpu_id=4>>baseout/a04.out&

# # nohup python run_recbole_${n}.py --dataset=taobao-20000 --integra=add --pooling_mode=mean  --model=bert4rec --weight_norm=1  --gpu_id=1>>baseout/t001.out&
# nohup python run_recbole_${n}.py --dataset=taobao-20000 --integra=add --pooling_mode=mean  --model=gru4rec --weight_norm=1  --gpu_id=3>>baseout/t003.out&


# nohup python run_recbole_${n}.py --dataset=taobao-20000 --integra=add --pooling_mode=mean  --model=sasrec_ --weight_norm=1  --gpu_id=7>>baseout/t000.out&




# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=sasrec_cate --weight_norm=0.5  --gpu_id=1>>baseout/cate01.out&
# nohup python run_recbole_${n}.py --dataset=${c} --integra=add --pooling_mode=sum  --model=sasrec_cate --weight_norm=0.5  --gpu_id=2>>baseout/cate02.out&
# nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=sum  --model=sasrec_cate --weight_norm=0.5  --gpu_id=3>>baseout/cate03.out&
# nohup python run_recbole_${n}.py --dataset=${a} --integra=add --pooling_mode=sum  --model=sasrec_cate --weight_norm=0.5  --gpu_id=4>>baseout/cate04.out&


# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=sasrec_item --weight_norm=0.5  --gpu_id=0>>baseout/item00.out&
# nohup python run_recbole_${n}.py --dataset=${c} --integra=add --pooling_mode=sum  --model=sasrec_item --weight_norm=0.5  --gpu_id=1>>baseout/item01.out&
# nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=sum  --model=sasrec_item --weight_norm=0.5  --gpu_id=2>>baseout/item02.out&
# nohup python run_recbole_${n}.py --dataset=${a} --integra=add --pooling_mode=sum  --model=sasrec_item --weight_norm=0.5  --gpu_id=3>>baseout/item03.out&

# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=sasrec_catey --weight_norm=0.5  --gpu_id=4>>baseout/cy00.out&
nohup python run_recbole_${n}.py --dataset=${c} --integra=add --pooling_mode=sum  --model=sasrec_catey --weight_norm=0.5  --gpu_id=5>>baseout/cy01.out&
# nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=sum  --model=sasrec_catey --weight_norm=0.5  --gpu_id=6>>baseout/cy02.out&
# nohup python run_recbole_${n}.py --dataset=${a} --integra=add --pooling_mode=sum  --model=sasrec_catey --weight_norm=0.5  --gpu_id=7>>baseout/cy03.out&

# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=sasrec_catev --weight_norm=0.5  --gpu_id=4>>baseout/cv00.out&
# nohup python run_recbole_${n}.py --dataset=${c} --integra=add --pooling_mode=sum  --model=sasrec_catev --weight_norm=0.5  --gpu_id=5>>baseout/cv01.out&
# nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=sum  --model=sasrec_catev --weight_norm=0.5  --gpu_id=6>>baseout/cv02.out&
# nohup python run_recbole_${n}.py --dataset=${a} --integra=add --pooling_mode=sum  --model=sasrec_catev --weight_norm=0.5  --gpu_id=7>>baseout/cv03.out&

# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=sasrec_itemf --weight_norm=0.5  --gpu_id=0>>baseout/if00.out&
# nohup python run_recbole_${n}.py --dataset=${c} --integra=add --pooling_mode=sum  --model=sasrec_itemf --weight_norm=0.5  --gpu_id=1>>baseout/if01.out&
# nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=sum  --model=sasrec_itemf --weight_norm=0.5  --gpu_id=2>>baseout/if02.out&
# nohup python run_recbole_${n}.py --dataset=${a} --integra=add --pooling_mode=sum  --model=sasrec_itemf --weight_norm=0.5  --gpu_id=3>>baseout/if03.out&

# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=sasrec_posf --weight_norm=0.5  --gpu_id=0>>baseout/pf00.out&
# nohup python run_recbole_${n}.py --dataset=${c} --integra=add --pooling_mode=sum  --model=sasrec_posf --weight_norm=0.5  --gpu_id=1>>baseout/pf01.out&
# nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=sum  --model=sasrec_posf --weight_norm=0.5  --gpu_id=2>>baseout/pf02.out&
# nohup python run_recbole_${n}.py --dataset=${a} --integra=add --pooling_mode=sum  --model=sasrec_posf --weight_norm=0.5  --gpu_id=3>>baseout/pf03.out&

# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=sasrec_posa --weight_norm=0.5  --gpu_id=4>>baseout/pa00.out&
# nohup python run_recbole_${n}.py --dataset=${c} --integra=add --pooling_mode=sum  --model=sasrec_posa --weight_norm=0.5  --gpu_id=5>>baseout/pa01.out&
# nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=sum  --model=sasrec_posa --weight_norm=0.5  --gpu_id=6>>baseout/pa02.out&
# nohup python run_recbole_${n}.py --dataset=${a} --integra=add --pooling_mode=sum  --model=sasrec_posa --weight_norm=0.5  --gpu_id=7>>baseout/pa03.out&

# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=sasrec_catea --weight_norm=0.5  --gpu_id=8>>baseout/ca00.out&
# nohup python run_recbole_${n}.py --dataset=${c} --integra=add --pooling_mode=mean  --model=sasrec_catea --weight_norm=0.5  --gpu_id=9>>baseout/ca04.out&
# nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=mean  --model=sasrec_catea --weight_norm=0.5  --gpu_id=10>>baseout/ca05.out&
# nohup python run_recbole_${n}.py --dataset=${a} --integra=add --pooling_mode=mean  --model=sasrec_catea --weight_norm=0.5  --gpu_id=11>>baseout/ca06.out&

# nohup python run_recbole_${n}.py --dataset=${y} --integra=add --pooling_mode=mean  --model=sasrec_posv --weight_norm=0.5  --gpu_id=8>>baseout/pv00.out&
# nohup python run_recbole_${n}.py --dataset=${c} --integra=add --pooling_mode=sum  --model=sasrec_posv --weight_norm=0.5  --gpu_id=9>>baseout/pv01.out&
# nohup python run_recbole_${n}.py --dataset=${s} --integra=add --pooling_mode=sum  --model=sasrec_posv --weight_norm=0.5  --gpu_id=10>>baseout/pv02.out&
# nohup python run_recbole_${n}.py --dataset=${a} --integra=add --pooling_mode=sum  --model=sasrec_posv --weight_norm=0.5  --gpu_id=11>>baseout/pv03.out&


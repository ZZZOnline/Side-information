
import torch
import pytorch_lightning as pl
from network import *
from new_utils import *
from new_metric import *

class casrmodel(pl.LightningModule):
    def __init__(self, itemmodel, catemodel,  args
        ):
        super().__init__()
        self.teachingflag = True
        self.m_logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.loss = torch.nn.CrossEntropyLoss()
        self.args = args
        self.itemmodel = itemmodel
        self.catemodel = catemodel
        self.starmodel = starcateseq(args.hidden_units, args.hidden_units, args.dropout_rate, args)
         

        self.train_epoch = 0
        self.dropout = torch.nn.Dropout(p=args.dropout_rate)

    def training_step(self, batch, batch_idx):
        args = self.args
        self.train_epoch +=1
        if self.train_epoch>8:
            self.teachingflag = False
        user_seq_batch, item_y_batch, cate_y_batch, beha_y_batch, item_seq_batch, cate_seq_batch, beha_seq_batch, time_seq_batch, pad_incateitem_seq_batch, pad_incatetime_seq_batch, pad_cateset_seq_batch, itemy_seq_batch, catey_seq_batch = batch


        item_seq_batch = item_seq_batch.squeeze(0)
        item_y_batch = item_y_batch.squeeze(0)
        cate_seq_batch = cate_seq_batch.squeeze(0)
        cate_y_batch = cate_y_batch.squeeze(0)
        catey_seq_batch = catey_seq_batch.squeeze(0)
        itemy_seq_batch = itemy_seq_batch.squeeze(0)
        time_seq_batch = time_seq_batch.squeeze(0)
        item_idx = []

        cate_logits = self.catemodel.log2feats(cate_seq_batch, time_seq_batch)
        cate_outputs = self.catemodel.head(cate_logits[:,-1,:], None).squeeze()



        cateloss = self.loss(cate_outputs, cate_y_batch)

        for param in self.catemodel.item_emb.parameters(): cateloss += args.l2_emb * torch.norm(param)

        item_test = np.repeat(np.array(item_idx).reshape(1,-1), args.batch_size, 0)

        mask = (item_seq_batch>0)
        if True:

            cate_input = self.starmodel(self.itemmodel.item_emb(item_seq_batch), self.catemodel.item_emb(cate_seq_batch)+ self.dropout(self.catemodel.time_emb(time_seq_batch)), self.catemodel.item_emb(catey_seq_batch), mask)


            starinput = cate_input





        outputs = self.itemmodel.trainq(starinput, itemy_seq_batch, None)

        valid_numpy = np.array(itemy_seq_batch.to('cpu'))>0
        label_numpy = valid_numpy.nonzero()


        valid_logits = outputs[label_numpy]
        valid_labels = itemy_seq_batch.squeeze()[label_numpy]

        itemloss = self.loss(valid_logits, valid_labels)



        for param in self.itemmodel.item_emb.parameters(): itemloss += args.l2_emb * torch.norm(param)
        loss = 0.8*itemloss + 0.2*cateloss

        return {'loss':loss}

    def training_epoch_end(self, training_step_outputs):
        loss = torch.stack([o['loss'] for o in training_step_outputs], 0).mean()
        self.log('train_loss', loss)


    def validation_step(self, batch, batch_idx):
        user_seq_batch, item_y_batch, cate_y_batch, beha_y_batch, item_seq_batch, cate_seq_batch, beha_seq_batch, time_seq_batch, pad_incateitem_seq_batch, pad_incatetime_seq_batch, pad_cateset_seq_batch, itemy_seq_batch, catey_seq_batch = batch
        args = self.args
        itemnum = self.itemmodel.item_num

        item_seq_batch = item_seq_batch.squeeze(0)
        item_y_batch = item_y_batch.squeeze(0)
        cate_seq_batch = cate_seq_batch.squeeze(0)
        catey_seq_batch = catey_seq_batch.squeeze(0)
        time_seq_batch = time_seq_batch.squeeze(0)


        #obtain predicted preferred categories   
        cate_logits = self.catemodel.log2feats(cate_seq_batch, time_seq_batch)
        cate_outputs = self.catemodel.head(cate_logits[:,-1,:], None).squeeze()
        pred_item_prob1 = None
        mask = (item_seq_batch>0)
        log_prob_cate_short = self.m_logsoftmax(cate_outputs)
        log_prob_cate_short, pred_cate_index = torch.topk(log_prob_cate_short, args.beam_size, dim=-1)
        pred_cate_index = pred_cate_index.detach()

        #generate item preference based on category preferences
        for beam_index in range(args.beam_size):

            prob_cate_beam = log_prob_cate_short[:, beam_index]

            cate_input = self.starmodel(self.itemmodel.item_emb(item_seq_batch), self.catemodel.item_emb(cate_seq_batch) + self.dropout(self.catemodel.time_emb(time_seq_batch)), self.catemodel.item_emb(pred_cate_index[:,beam_index].reshape(-1,1)), mask)

            starinput = cate_input[:,-1,:]


            logit_2 = self.itemmodel.head(starinput, None)
            outputs = logit_2 

            test_prob_batch = self.m_logsoftmax(outputs)

            if pred_item_prob1 is None:
                pred_item_prob1 = test_prob_batch+prob_cate_beam.reshape(-1, 1)
                pred_item_prob1 = pred_item_prob1.unsqueeze(-1)
            else:
                pred_item_prob_beam1 = test_prob_batch+prob_cate_beam.reshape(-1, 1)
                pred_item_prob_beam1 = pred_item_prob_beam1.unsqueeze(-1)
                pred_item_prob1 = torch.cat((pred_item_prob1, pred_item_prob_beam1), dim=-1)

    
        pred_item_prob1 = torch.logsumexp(pred_item_prob1, dim=-1)

        valid_labels = torch.zeros((item_y_batch.shape[0])).long().to('cuda')

        cate_recall_5, cate_mrr, cate_ndcg = evaluate_part(cate_outputs, cate_y_batch, torch.tensor([True]).expand_as(valid_labels).to('cuda'), k=5)
    

        metric = defaultdict(list)


        item_idx = np.arange(0, itemnum+1)
        cate_idx = [args.imap[i] for i in item_idx]
        cate_all = torch.tensor(cate_idx).to('cuda')
        
        
        #generate recommendation based on category preferences
        for k in args.recall:
            _, indices = torch.topk(pred_item_prob1, k-args.topkc, -1)


            a = pred_item_prob1.clone().detach()

            output_pool = a.scatter(1, indices, -torch.inf)
            qwe = output_pool.clone().detach()

            for i in range(0, args.topkc):
                output_beam = torch.where(cate_all == pred_cate_index[:,i].reshape(-1,1), output_pool, (torch.ones((itemnum+1))*-torch.inf).to('cuda'))
                _, index = torch.topk(output_beam, 1, -1)
                indices = torch.cat([indices,index],-1)
                qwe = torch.where(cate_all == pred_cate_index[:,i].reshape(-1,1),  (torch.ones((itemnum+1))*-torch.inf).to('cuda'), qwe)

            metric['Recall@%d' % k], metric['mrr@%d' % k], metric['ndcg@%d' % k]= newevaluate(indices, item_y_batch, torch.tensor([True]).expand_as(valid_labels).to('cuda'))

        return {'metrics':metric,  'validcate_recall_5':cate_recall_5}

    


    def validation_epoch_end(self, validation_step_outputs):
        newkeys = validation_step_outputs[0]['metrics'].keys()
        metric =  defaultdict(list)
        for o in validation_step_outputs:
            for k in newkeys:
                metric[k].append(o['metrics'][k])


        for k in newkeys:
            self.log(f'Val:{k}', torch.Tensor(metric[k]).mean(), on_epoch=True)

            

        validcate_recall_5 = torch.stack([o['validcate_recall_5'] for o in validation_step_outputs], 0).mean()
        self.log('validcate_recall_5', validcate_recall_5, on_epoch=True)

        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, betas=(0.9, 0.98))
        return optimizer





class sasrecmodel(pl.LightningModule):
    def __init__(self, itemmodel, catemodel, args
        ):
        super().__init__()
        self.m_logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.loss = torch.nn.CrossEntropyLoss()
        self.beloss = torch.nn.BCEWithLogitsLoss()
        self.args = args
        self.itemmodel = itemmodel
        self.catemodel = catemodel


        self.train_epoch = 0
        self.dropout = torch.nn.Dropout(p=args.dropout_rate)

        

    def training_step(self, batch, batch_idx):
        args = self.args
        self.train_epoch +=1
        if self.train_epoch>8:
            self.teachingflag = False
        user_seq_batch, item_y_batch, cate_y_batch, beha_y_batch, item_seq_batch, cate_seq_batch, beha_seq_batch, time_seq_batch, pad_incateitem_seq_batch, pad_incatetime_seq_batch, pad_cateset_seq_batch, itemy_seq_batch, catey_seq_batch = batch


        item_seq_batch = item_seq_batch.squeeze(0)
        item_y_batch = item_y_batch.squeeze(0)
        itemy_seq_batch = itemy_seq_batch.squeeze(0)
        time_seq_batch = time_seq_batch.squeeze(0)


        item_logits = self.itemmodel.log2feats(item_seq_batch)
        outputs = self.itemmodel.losshead(item_logits, None)

        indices = np.where((item_seq_batch != 0).to('cpu'))
        loss = self.loss(outputs[indices], itemy_seq_batch[indices])


        for param in self.itemmodel.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
        return {'loss':loss}

    def training_epoch_end(self, training_step_outputs):
        loss = torch.stack([o['loss'] for o in training_step_outputs], 0).mean()
        self.log('train_loss', loss)



    def validation_step(self, batch, batch_idx):
        args = self.args
        itemnum = self.itemmodel.item_num
        user_seq_batch, item_y_batch, cate_y_batch, beha_y_batch, item_seq_batch, cate_seq_batch, beha_seq_batch, time_seq_batch, pad_incateitem_seq_batch, pad_incatetime_seq_batch, pad_cateset_seq_batch, itemy_seq_batch, catey_seq_batch = batch
        
        item_seq_batch = item_seq_batch.squeeze(0)
        item_y_batch = item_y_batch.squeeze(0)
        itemy_seq_batch = itemy_seq_batch.squeeze(0)
        cate_seq_batch = cate_seq_batch.squeeze(0)
        time_seq_batch = time_seq_batch.squeeze(0)

        item_logits = self.itemmodel.log2feats(item_seq_batch)
        pred_item_prob1 = None

        if True:
            logit_1 = self.itemmodel.head(item_logits[:,-1,:], None)
            outputs =  logit_1
            pred_item_prob1 = self.m_logsoftmax(outputs)

        valid_labels = torch.zeros((pred_item_prob1.shape[0])).long().to(args.device)
        metrics = evaluate(pred_item_prob1, item_y_batch, torch.tensor([True]).expand_as(valid_labels).to('cuda'), recalls=args.recall)

        return {'metrics':metrics}


    def validation_epoch_end(self, validation_step_outputs):
        keys = validation_step_outputs[0]['metrics'].keys()
        metric =  defaultdict(list)
        for o in validation_step_outputs:
            for k in keys:
                metric[k].append(o['metrics'][k])

        for k in keys:
            self.log(f'Val:{k}', torch.Tensor(metric[k]).mean(), on_epoch=True)


        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, betas=(0.9, 0.98))
        return optimizer


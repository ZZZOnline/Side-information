import argparse
from logging import getLogger
from trainer import *

from recbole.config import Config
from recbole.data import create_dataset
from recbole.utils import init_logger, get_model, get_trainer, init_seed, set_color
import importlib

def get_model_by_string(model_name):
    try:
        model_module = importlib.import_module('newmodel5.' + model_name)
        model = getattr(model_module, model_name)
        return model
    except ImportError:
        print(f"模块 {model_name} 不存在")
    except AttributeError:
        print(f"模块 {model_name} 中没有 {model_name} 对象")

# # 示例调用
# model_name = 'difsr'  # 根据输入的字符串获取模块名
# model = get_model_by_string(model_name)



import numpy as np
import pandas as pd
import torch
import torch.nn.utils.rnn as rnn_utils
def _convert_to_tensor(data):
    """This function can convert common data types (list, pandas.Series, numpy.ndarray, torch.Tensor) into torch.Tensor.

    Args:
        data (list, pandas.Series, numpy.ndarray, torch.Tensor): Origin data.

    Returns:
        torch.Tensor: Converted tensor from `data`.
    """
    elem = data[0]
    if isinstance(elem, (float, int, np.float, np.int64)):
        new_data = torch.as_tensor(data)
    elif isinstance(elem, (list, tuple, pd.Series, np.ndarray, torch.Tensor)):
        seq_data = [torch.as_tensor(d) for d in data]
        new_data = rnn_utils.pad_sequence(seq_data, batch_first=True)
    else:
        raise ValueError(f'[{type(elem)}] is not supported!')
    if new_data.dtype == torch.float64:
        new_data = new_data.float()
    return new_data

class newDataset(object):

    def __init__(self, interaction):
        self.interaction = dict()
        if isinstance(interaction, dict):
            for key, value in interaction.items():
                if isinstance(value, (list, np.ndarray)):
                    self.interaction[key] = _convert_to_tensor(value)
                elif isinstance(value, torch.Tensor):
                    self.interaction[key] = value
                else:
                    raise ValueError(f'The type of {key}[{type(value)}] is not supported!')
        elif isinstance(interaction, pd.DataFrame):
            for key in interaction:
                value = interaction[key].values
                self.interaction[key] = _convert_to_tensor(value)
        else:
            raise ValueError(f'[{type(interaction)}] is not supported for initialize `Interaction`!')
        self.length = -1
        for k in self.interaction:
            if self.interaction[k].shape:
                self.length = max(self.length, self.interaction[k].shape[0])
            else:
                self.length = max(self.length, 1)
            # self.length = max(self.length, 1)

    def __iter__(self):
        return self.interaction.__iter__()

    def __getattr__(self, item):
        if 'interaction' not in self.__dict__:
            raise AttributeError(f"'Interaction' object has no attribute 'interaction'")
        if item in self.interaction:
            return self.interaction[item]
        raise AttributeError(f"'Interaction' object has no attribute '{item}'")

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.interaction[index]
        else:
            ret = {}
            for k in self.interaction:
                ret[k] = self.interaction[k][index]
            return ret

    def __contains__(self, item):
        return item in self.interaction

    def __len__(self):
        return self.length

    def __str__(self):
        info = [f'The batch_size of interaction: {self.length}']
        for k in self.interaction:
            inter = self.interaction[k]
            temp_str = f"    {k}, {inter.shape}, {inter.device.type}, {inter.dtype}"
            info.append(temp_str)
        info.append('\n')
        return '\n'.join(info)

from torch.utils.data import *
class DataModule(pl.LightningDataModule):
    def __init__(self, traindataset, evaldataset, testdataset, args, config):
        super().__init__()
        self.traindataset = traindataset
        self.evaldataset = evaldataset
        self.testdataset = testdataset
        self.args = args
        self.config = config

    def train_dataloader(self):
        return DataLoader(self.traindataset, self.config['train_batch_size'],  persistent_workers=False)

    def val_dataloader(self):
        return DataLoader(self.evaldataset, self.config['eval_batch_size'],  persistent_workers=False)
    
    def test_dataloader(self):
        return DataLoader(self.testdataset, self.config['eval_batch_size'],  persistent_workers=False)




if __name__ == '__main__':

    parameter_dict = {
        'neg_sampling': None,
        'gpu_id':2
        # 'attribute_predictor':'not',
        # 'attribute_hidden_size':"[256]",
        # 'fusion_type':'gate',
        # 'seed':212,
        # 'n_layers':4,
        # 'n_heads':1
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id',  type=int, default=2, help='name of gpu')
    parser.add_argument('--boost_size', default=2, type=int)

    parser.add_argument('--defaultmodel', '-m', type=str, default='SASRec', help='name of models')

    parser.add_argument('--dataset', '-d', type=str, default='beer', help='name of datasets')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--dropout_rate', default=0.3, type=float)

    parser.add_argument('--config_files', type=str, default='None', help='config files')

    parser.add_argument('--final', type=str, default='add')
    parser.add_argument('--gamma', default=2, type=float)
    parser.add_argument('--hidden_units', default=128, type=int)
    parser.add_argument('--integra', type=str, default='add')


    parser.add_argument('--lamda', default=5, type=float)
    parser.add_argument('--lastk', type=int, default=5)
    parser.add_argument('--lr', default=.0005, type=float)
    parser.add_argument('--l2_emb', default=0.005, type=float)
    parser.add_argument('--maxlen', type=int, default=50)
    parser.add_argument('--model', type=str, default='difsr_decode')


    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--num_heads', default=2, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)


    parser.add_argument('--optimizer_type', default='Adagrad', type=str)
    parser.add_argument('--patience', default=20, type=int)
    parser.add_argument('--pooling_mode', default='mean', type=str)
    parser.add_argument('--p_heads', default=4, type=int)


    parser.add_argument('--recall', default=[5, 10, 20, 50], type=list)
#ours
    parser.add_argument('--shared', default='shared', type=str)
    parser.add_argument('--weight_norm', default=0.5, type=float)

    args, _ = parser.parse_known_args()



    args.config_files = './configs/'+args.dataset +'.yaml'
    print(args.config_files)
    parameter_dict['gpu_id'] = args.gpu_id  
    config_file_list = args.config_files.strip().split(' ') if args.config_files else None

    config = Config(model=args.defaultmodel, dataset=args.dataset, config_file_list=config_file_list, config_dict=parameter_dict)

    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    if config['save_dataset']:
        dataset.save()
    logger.info(dataset)

    # save pickle
    # path_item = '{}_item.pickle'.format(dataset.dataset_name)
    # path_user = '{}_user.pickle'.format(dataset.dataset_name)
    # with open(path_item,'wb') as f:
    #     pickle.dump(dataset.field2token_id['item_id'],f)
    # with open(path_user, 'wb') as f:
    #     pickle.dump(dataset.field2token_id['session_id'],f)
    
    #modification
    dataset.item_feat = dataset.item_feat.drop_duplicates(subset='item_id')
    # dataset splitting
    # train_data, valid_data, test_data = data_preparation(config, dataset)
    built_datasets = dataset.build()
    logger = getLogger()

    train_dataset, valid_dataset, test_dataset = built_datasets

    # if config['save_dataloaders']:
    #     save_split_dataloaders(config, dataloaders=(train_data, valid_data, test_data))

    print(args.model)
    # config['n_layers'] = args.num_layers
    args.lr = config['learning_rate']

    model = get_model_by_string(args.model)(config, dataset, args)

    # model = difsr_decode(config, train_data.dataset, args)

    recmodel = casr(model,  args)
    train, valid, test = newDataset(train_dataset.inter_feat.interaction), newDataset(valid_dataset.inter_feat.interaction), newDataset(test_dataset.inter_feat.interaction)
    mydata = DataModule(train, valid, test, args, config)


    from pytorch_lightning.callbacks.early_stopping import EarlyStopping

    from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


    early_stop_callback = EarlyStopping(monitor="Val:Recall@10", min_delta=0.00, patience=args.patience, verbose=False, mode="max")
    modelcheckpoint = ModelCheckpoint(monitor='Val:Recall@10', mode='max')

    feature_num = str(len(config['selected_features']))
    if args.dataset=='beer':
        log_steps = 5
    else:
        log_steps = 50

    # print('weight_norm',args.weight_norm)
    trainer = pl.Trainer(devices=[0], callbacks=[early_stop_callback, modelcheckpoint], accelerator="gpu", accumulate_grad_batches=1, max_epochs=200,  default_root_dir='./script1/head0/{}/hidden{}/lr{}/layer{}/head{}/{}/{}/norm{}/{}/head{}'.format(args.dataset+feature_num, config['hidden_size'],config['learning_rate'],  config['n_layers'], config['n_heads'],  args.model, args.integra, args.weight_norm, args.pooling_mode, args.p_heads), log_every_n_steps=log_steps)
# original sequence augmentation with 0 padding at last
# test1
    trainer.fit(recmodel, mydata)
    trainer.test(recmodel, mydata, ckpt_path='best')



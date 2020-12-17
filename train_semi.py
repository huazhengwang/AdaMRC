import re
import os
import sys
import random
import string
import logging
import argparse
import json
import torch
import msgpack
import pdb
# import pandas as pd
import numpy as np
from shutil import copyfile
from datetime import datetime
from collections import Counter, defaultdict
from src.model import DocReaderModel
from src.batcher import load_meta, BatchGen
from config import set_args
from my_utils.utils import set_environment
from my_utils.log_wrapper import create_logger
from my_utils.squad_eval import evaluate_file, load_gold
# from my_utils.marco_eval import load_rank_score, generate_submit
# from my_utils.ms_marco_eval_pretoken import MAX_BLEU_ORDER, compute_metrics_from_files
import configparser
# from my_utils.utils import repeat_save

args = set_args()



# input('check')

if args.valid_batch_size is None:
    args.valid_batch_size=args.batch_size
if args.logDir is not None:
    args.log_file=os.path.join(args.logDir,'san.log')
if args.modelDir is not None:
    args.model_dir = args.modelDir
if args.dev_datasets is None:
    args.dev_datasets=args.train_datasets
# if args.elmo_path is None:
#   args.elmo_path=args.data_dir
# if args.elmo_model=='big':
#   args.elmo_options_path=os.path.join(args.elmo_path,'elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json')
#   args.elmo_weight_path=os.path.join(args.elmo_path,'elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5')
# elif args.elmo_model=='small':
#   args.elmo_options_path=os.path.join(args.elmo_path,'elmo_2x1024_128_2048cnn_1xhighway_options.json')
#   args.elmo_weight_path=os.path.join(args.elmo_path,'elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5')
# if args.elmo_gamma == -1:
#   args.elmo_gamma = None
    # print('gamma set to None.')

args.num_gpu = torch.cuda.device_count() if args.multi_gpu else 1
    
config_path = os.path.join(args.model_dir, 'config.json')
for i in range(10):
    try:
        json.dump(vars(args),open(config_path,'w'), indent=4)
        break
    except:
        print('output json failed.')


# args.dataset_config_path='dataset_configs/dataset_config_%d.ini' % args.dataset_config_id
# print('dataset config path:', args.dataset_config_path)
# config = configparser.ConfigParser()
# config.read(args.dataset_config_path)
# args.dataset_configs=config
# args.dataset_configs['marco_test']=args.dataset_configs['marco']

# args.elmo_config_path='elmo_configs/elmo_config_%d.ini' % args.elmo_config_id
# print('elmo config path:', args.elmo_config_path)
# config = configparser.ConfigParser()
# config.read(args.elmo_config_path)
# args.elmo_configs={}
# for key in config['elmo']:
#     args.elmo_configs[key]=int(config['elmo'][key])
# print('elmo config:', args.elmo_configs[key])

print(vars(args))
print('cuda:',args.cuda)
os.system('nvidia-smi')
# print('directory:',os.listdir("./"))
print('output directory:',args.model_dir)
print('log directory:',args.log_file)

# parse multitask dataset names
args.train_datasets=args.train_datasets.split(',')
args.dev_datasets=args.dev_datasets.split(',')  

# set model dir
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)
model_dir = os.path.abspath(model_dir)

# set environment
set_environment(args.seed, args.cuda)
# setup logger
logger =  create_logger(__name__, to_disk=True, log_file=args.log_file)

def check(model, data, gold_data):
    data.reset()
    predictions = {}
    for batch_list, name_map in data:
        phrase, _ = model.predict(batch_list, name_map, dataset_name=data.dataset_name)
        uids = batch_list[name_map['uids']]
        for uid, pred in zip(uids, phrase):
            predictions[uid] = pred

    results = evaluate_file(gold_data, predictions)
    return results['exact_match'], results['f1'], predictions

# def eval_model_marco(model, data):
#     data.reset()
#     dev_predictions = []
#     dev_best_scores = []
#     dev_ids_list = []
#     for batch_list, name_map in data:
#         phrase, phrase_score = model.predict(batch_list, name_map, dataset_name=data.dataset_name)
#         dev_predictions.extend(phrase)
#         dev_best_scores.extend(phrase_score)
#         dev_ids_list.extend(batch_list[name_map['uids']])
#     return dev_predictions, dev_best_scores, dev_ids_list
#     generate_submit(output, span_output, output_yn, dev_ids_list, dev_best_scores, dev_predictions, match_score, yn_dict)



def main():
    logger.info('Launching the SAN')
    start_test=False
  

    opt = vars(args)
    logger.info('Loading data')
    embedding, opt = load_meta(opt, os.path.join(args.multitask_data_path, args.meta))
    gold_data = load_gold(args.dev_datasets, args.data_dir)
    best_em_score, best_f1_score = 0.0, 0.0


    if args.batch_mix_domain:
        args.valid_batch_size = int(args.valid_batch_size/2)
    
    if not args.test_mode:
        all_train_batchgen=[]
        all_dev_batchgen=[]
        print(args.train_datasets)
        # input('ha')
        for dataset_name in args.train_datasets:
            path=os.path.join(args.multitask_data_path, dataset_name+'_train.json')
            dataset_ratio = 1.0
            if args.semi_dataset:
                if args.semi_dataset == dataset_name:
                    dataset_ratio = args.semi_ratio
            all_train_batchgen.append(BatchGen(path,
                                  batch_size=args.batch_size,
                                  gpu=args.cuda, dataset_name = dataset_name, add_domain_tag = args.add_domain_tag,
                                  doc_maxlen=args.doc_maxlen, drop_less = args.drop_less,
                                  num_gpu = args.num_gpu, mask_words=args.test_fill_embedding,
                                  dropout_w=args.dropout_w, dw_type=args.dw_type, dataset_ratio = dataset_ratio))
        all_train_iters=[iter(item) for item in all_train_batchgen]
        for dataset_name in args.dev_datasets:
            path=os.path.join(args.multitask_data_path, dataset_name+'_dev.json')
            all_dev_batchgen.append(BatchGen(path,
                                  batch_size=args.valid_batch_size,
                                  gpu=args.cuda, is_train=False, 
                                  dataset_name = dataset_name, add_domain_tag = args.add_domain_tag,
                                  doc_maxlen=args.doc_maxlen, num_gpu = args.num_gpu, mask_words=args.test_fill_embedding))
            # if 'marco' in dataset_name:
            #     rank_path = os.path.join(args.data_dir,dataset_name)
            #     dev_rank_path=os.path.join(rank_path, 'dev_oracle_scores.json') if args.marco_oracle \
            #         else os.path.join(rank_path, 'dev_rank_scores.json')
            #     dev_rank_scores = load_rank_score(dev_rank_path)
            #     dev_yn=json.load(open(os.path.join(rank_path,'dev_yn_dict.json')))
                # test_rank_path=os.path.join(rank_path, 'test_rank_scores.json')
                # test_rank_scores = load_rank_score(test_rank_path)
    else:
        all_dev_batchgen=[]
        for dataset_name in args.dev_datasets:
            path=os.path.join(args.multitask_data_path, dataset_name+'_test.json')
            all_dev_batchgen.append(BatchGen(path,
                                  batch_size=args.valid_batch_size,
                                  gpu=args.cuda, is_train=False, dataset_name = dataset_name, doc_maxlen=args.doc_maxlen))

            all_dev_batchgen.append(BatchGen(path,
                                  batch_size=args.valid_batch_size,
                                  gpu=args.cuda, is_train=False, dataset_name = dataset_name, doc_maxlen=args.doc_maxlen))
    if args.resume_last_epoch:
        # find_checkpoint = False
        latest_time=0
        for o in os.listdir(model_dir):
            if o.startswith('checkpoint_epoch_'):
                edit_time = os.path.getmtime(os.path.join(model_dir, o))
                # print(edit_time)
                if edit_time>latest_time:
                    latest_time=edit_time
                    args.resume_dir = model_dir
                    args.resume = o


    if args.resume_dir is not None:
        print('resuming model in ', os.path.join(args.resume_dir, args.resume))
        checkpoint = torch.load(os.path.join(args.resume_dir, args.resume))
        
        model_opt = checkpoint['config'] if args.resume_options else opt
        if 'outside_embed' not in model_opt:
            model_opt['outside_embed']=False
            model_opt['add_elmo']=False
            # model_opt['elmo_configs']=opt['elmo_configs']
            model_opt['multi_gpu']=False
        model_opt['multitask_data_path']=opt['multitask_data_path']
        model_opt['covec_path']=opt['covec_path']
        model_opt['data_dir']=opt['data_dir']

        if args.resume_options:
            logger.info('resume old options')
        else:
            logger.info('use new options.')
        # model_opt['train_datasets']=checkpoint['config']['train_datasets']

        state_dict = checkpoint['state_dict']
        model = DocReaderModel(model_opt, embedding, state_dict)

        # logger.info('use old random state.')
        # random.setstate(checkpoint['random_state'])
        # torch.random.set_rng_state(checkpoint['torch_state'])
        # if args.cuda:
        #     torch.cuda.set_rng_state(checkpoint['torch_cuda_state'])

        if model.scheduler:
            if 'scheduler_state' in checkpoint:
                model.scheduler.load_state_dict(checkpoint['scheduler_state'])
            else:
                print('warning: not loading scheduler state because didn\'t save.')
        # start_epoch = checkpoint['epoch']+1
        start_epoch = 0
    else:
        model = DocReaderModel(opt, embedding)
        start_epoch=0
        
    # model meta str
    logger.info('using {} GPUs'.format(torch.cuda.device_count()))
    headline = '############# Model Arch of SAN #############'
    # print network
    logger.info('\n{}\n{}\n'.format(headline, model.network))
    if not args.outside_embed:
        model.setup_eval_embed(embedding)

    logger.info("Total number of params: {}".format(model.total_param))
    if args.cuda:
        model.cuda()



    all_lens=[len(bg) for bg in all_train_batchgen]
    num_all_batches=args.epoches * sum(all_lens)

    for epoch in range(start_epoch, args.epoches):
        logger.warning('At epoch {}'.format(epoch))

        for train_data in all_train_batchgen:
            if args.data_reset:
                train_data.reset()

        # batch indices
        all_call_indices=[]
        for i in range(len(all_train_batchgen)):
            if 'squad' in all_train_batchgen[i].dataset_name:
                all_call_indices+=[i]*int(args.squad_ratio*len(all_train_batchgen[i]))
            else:
                all_call_indices+=[i]*len(all_train_batchgen[i])
        all_call_indices=np.random.permutation(all_call_indices)
        # print (len(all_call_indices), all_call_indices)
        # pdb.set_trace()
        # all_call_indices=all_call_indices[:1]
        
        start = datetime.now()
        # for iiii in range(10):
        train_steps = len(all_call_indices)
        if args.batch_mix_domain:
            train_steps = int(train_steps/2)
        for i in range(train_steps):
            # if epoch == 2:       
            #     # pdb.set_trace()
            #     model.network.test_mode=True
            if args.batch_mix_domain:
                batch_list_1, name_map_1=next(all_train_iters[all_call_indices[i]])
                batch_list_2, name_map_2=next(all_train_iters[1-all_call_indices[i]])
                batch_list = batch_list_1+batch_list_2
                # print (name_map_1, name_map_2)
                name_map = name_map_1
            else:
                batch_list, name_map=next(all_train_iters[all_call_indices[i]])
            # try:
            #     batch_list, name_map=next(all_train_iters[all_call_indices[i]])
            # except StopIteration:
            #     all_train_batchgen[all_call_indices[i]].reset()
            #     print (all_train_batchgen[all_call_indices[i]].offset)
            #     pdb.set_trace()
            #     continue
            # print (i) 
                # batch_list, name_map=next(all_train_iters[all_call_indices[i]])
            # pdb.set_trace()
            dataset_name = args.train_datasets[all_call_indices[i]]
            # if model.updates==231 or torch.isnan(model.network.mem_highway.doc_mem_highway_marco_test.comp[0].gate.weight).sum()>0:
            #   print(model.updates)
            #   pdb.set_trace()
            if not args.no_adv:
                p = float(i) / train_steps
                l = 2. / (1. + np.exp(-10. * p)) - 1
                adv_grad_scale = 0.01 / (1. + 10 * p)**0.75  
                # print ('adv_grad_scale: ', adv_grad_scale)
            else:
                adv_grad_scale = 1.0       

            model.update(batch_list, name_map, dataset_name, adv_grad_scale*2)
            
            if (model.updates) % args.log_per_updates == 0 or i == 0:
                logger.info('o(*^~^*) Task [{0:2}] #updates[{1:6}] train loss[{2:.5f}] remaining[{3}]'.format(
                    all_call_indices[i], model.updates, model.train_loss.avg,
                    str((datetime.now() - start) / (i + 1) * (len(all_call_indices) - i - 1)).split('.')[0]))
                if not model.opt['no_adv']:
                    logger.info('Domain predictor acc [{0:.4f}]'.format(model.domain_acc.avg))
            if (model.updates) % args.progress_per_updates ==0 or i==0:
                print("PROGRESS: {0:.4f}%".format(100.0 * (model.updates) / num_all_batches))         

        # print('temp saving!')
        # model.save('temp_test.pt', 0)
        em_sum=0
        f1_sum=0
        # dev eval
        # if epoch % 10 ==0:
            # if epoch == 50:
            #   start_test=True
        for i in range(len(all_dev_batchgen)):
            # dev_gold_path = os.path.join(args.data_dir, args.dev_datasets[i],'dev.json')
            dataset_name = args.dev_datasets[i]
            if dataset_name in ['squad','newsqa', 'newsqa_unsupervised', 'newsqa_unsupervised_old_filtered','SelfRC','ParaphraseRC',]:
                em, f1, results = check(model, all_dev_batchgen[i], gold_data[dataset_name])
                output_path = os.path.join(model_dir, 'dev_output_{}_{}.json'.format(dataset_name,epoch))
                try:
                    with open(output_path, 'w') as f:
                        json.dump(results, f)
                except OSError:
                    print('save predict failed.')
                em_sum+=em
                f1_sum+=f1
                logger.warning("Epoch {0} - Task {1:6} dev EM: {2:.3f} F1: {3:.3f}".format(epoch, dataset_name, em, f1))
            # elif 'marco' in dataset_name:
            #     # dev eval
            #     output = os.path.join(model_dir, 'dev_pred_{}.json'.format(epoch))
            #     output_yn = os.path.join(model_dir, 'dev_pred_yn_{}.json'.format(epoch))
            #     span_output = os.path.join(model_dir, 'dev_pred_span_{}.json'.format(epoch))
            #     # if start_test:
            #     #   pdb.set_trace()
            #     dev_predictions, dev_best_scores, dev_ids_list=eval_model_marco(model, all_dev_batchgen[i])

            #     dev_gold_path = os.path.join(args.data_dir, dataset_name, 'dev_original.json')
            #     metrics = compute_metrics_from_files(dev_gold_path, \
            #                                             output, \
            #                                             MAX_BLEU_ORDER)
            #     metrics_ys = compute_metrics_from_files(dev_gold_path, \
            #                                             output_yn, \
            #                                             MAX_BLEU_ORDER)
            #     rouge_score = metrics['rouge_l']
            #     blue_score = metrics['bleu_1']
            #     logger.warning("Epoch {0} - dev ROUGE-L: {1:.4f} BLEU-1: {2:.4f}".format(epoch, rouge_score, blue_score))

            #     for metric in sorted(metrics):
            #         logger.info('%s: %s' % (metric, metrics[metric]))
            #     logger.info('############################')
            #     logger.info('-----------YES/NO------')
            #     logger.info(metrics_ys)

        # setting up scheduler
        if model.scheduler is not None:
            logger.info('scheduler_type {}'.format(opt['scheduler_type']))
            if opt['scheduler_type'] == 'rop':
                model.scheduler.step(f1, epoch=epoch)
            else:
                model.scheduler.step()
        # save
        for try_id in range(10):
            try:
                model_file = os.path.join(model_dir, 'checkpoint_epoch_{}.pt'.format(epoch))
                model.save(model_file, epoch, best_em_score, best_f1_score)
                if em_sum + f1_sum > best_em_score + best_f1_score:
                    copyfile(os.path.join(model_dir, model_file), os.path.join(model_dir, 'best_checkpoint.pt'))
                    best_em_score, best_f1_score = em_sum, f1_sum
                    logger.info('Saved the new best model and prediction')
                break
            except OSError:
                print('save model failed: outer step. Probably fail when saving the best model.')

if __name__ == '__main__':
    main()

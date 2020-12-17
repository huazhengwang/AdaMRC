import os
import sys
import json
import torch
import random
import string
import logging
import numpy as np
import pickle as pkl
from shutil import copyfile
from allennlp.modules.elmo import batch_to_ids

PAD_ID = 0
UNK_ID = 1
STA_ID = 2
END_ID = 3

def load_meta(opt, meta_path):
    with open(meta_path, 'rb') as f:
        meta = pkl.load(f)
    embedding = torch.Tensor(meta['embedding'])
    opt['pos_vocab_size'] = len(meta['vocab_tag'])
    opt['ner_vocab_size'] = len(meta['vocab_ner'])
    opt['vocab_size'] = len(meta['vocab'])
    return embedding, opt

class BatchGen:
    def __init__(self, data_path, batch_size, gpu, is_train=True, doc_maxlen=1300, dataset_name = 'squad', add_domain_tag = False, drop_less=False,
                 num_gpu=1, mask_words=False, dropout_w=0.0, dw_type=0, dataset_ratio = 1.0):
        self.batch_size = batch_size
        self.doc_maxlen = doc_maxlen
        self.is_train = is_train
        self.gpu = gpu
        self.data_path = data_path
        self.add_domain_tag = add_domain_tag
        self.data = self.load(self.data_path, is_train, doc_maxlen, drop_less, dataset_name)
        if is_train:
            indices = list(range(len(self.data)))
            random.shuffle(indices)
            indices = indices[:int(len(indices)*dataset_ratio)]
            print('Use {} samples out of {} in semi-supervised setting in {}'.format(len(indices), len(self.data), dataset_name)) 
            data = [self.data[i] for i in indices]
            self.data = data
        data = [self.data[i:i + batch_size] for i in range(0, len(self.data), batch_size)]
        self.data = data
        self.ordered_data = data
        self.offset = 0
        self.dataset_name = dataset_name
        self.doc_maxlen = 1300 if is_train else None
        self.num_gpu = num_gpu
        self.mask_words=mask_words
        self.dropout_w = dropout_w
        self.dw_type=dw_type
        # self.last_indices=None

    def load(self, path, is_train=True, doc_maxlen=1300, drop_less =False, dataset_name='squad'):
        with open(path, 'r', encoding='utf-8') as reader:
            # filter
            data = []
            cnt = 0
            for line in reader:
                sample = json.loads(line)
                cnt += 1
                # if cnt > 200: #test code
                #     break                
                if len(sample['span'])==0:
                    continue
                if len(sample['doc_tok']) > doc_maxlen:
                    continue
                if is_train:
                    if sample['start'] is None or sample['end'] is None:
                        continue
                    if sample['start'] == -1 or sample['end'] == -1: #duorc
                        continue
                    if len(sample['doc_tok']) > doc_maxlen:
                        if not drop_less:
                            continue
                        if sample['start']>doc_maxlen or sample['end']>doc_maxlen:
                            continue
                        sample['doc_tok']=sample['doc_tok'][:doc_maxlen]
                        sample['doc_pos']=sample['doc_pos'][:doc_maxlen]
                        sample['doc_ner']=sample['doc_ner'][:doc_maxlen]
                        sample['doc_text']=sample['doc_text'][:doc_maxlen]

                        sample['span']=sample['span'][:doc_maxlen]
                    if self.add_domain_tag:
                        if 'squad' in dataset_name:
                            sample['doc_tok'].append(STA_ID)
                            sample['doc_pos'].append(0)
                            sample['doc_ner'].append(0)
                            sample['query_tok'].append(STA_ID)
                            sample['query_pos'].append(0)
                            sample['query_ner'].append(0)
                        else:
                            sample['doc_tok'].append(END_ID)
                            sample['doc_pos'].append(0)
                            sample['doc_ner'].append(0)
                            sample['query_tok'].append(END_ID)
                            sample['query_pos'].append(0)
                            sample['query_ner'].append(0)
                    # import pdb; pdb.set_trace()
                    # print(sample['uid'], 'is invalid. with length ', len(sample['doc_tok']), 'start=',sample['start'], 'end=',sample['end'] )
                    # continue
                data.append(sample)
                # if cnt%1000 == 0:
                #     print('Loading {} samples'.format(len(data)))

            print('Loaded {} samples out of {}'.format(len(data), cnt))
            return data


    def reset(self):
        if self.is_train:
            indices = list(range(len(self.data)))
            random.shuffle(indices)
            self.data = [self.ordered_data[i] for i in indices]
            # self.last_indices=indices
        self.offset = 0

    def __len__(self):
        return len(self.data)

    def patch(self, v):
        v = v.cuda(async=True)
        # else:
        #     v = Variable(v)
        return v        

    def __random_select__(self, arr):
        if self.dropout_w > 0:
            if self.dw_type > 0:
                ids = list(set(arr))
                ids_size = len(ids)
                random.shuffle(ids)
                ids = set(ids[:int(ids_size * self.dropout_w)])
                return [UNK_ID if e in ids else e for e in arr]
            else:
                return [UNK_ID if random.uniform(0, 1) < self.dropout_w else e for e in arr]
        else: return arr

    def __iter__(self):
        while self.offset < len(self):
            batch = self.data[self.offset]
            batch_size = len(batch)
            if batch_size<self.num_gpu:
                valid_size = batch_size
                real_batch = batch
                while len(batch)<=self.num_gpu:
                    batch+=real_batch
                batch_size = len(batch)
                real_valid_size = valid_size
                print('small batch size: gave special treatment.')
            else:
                valid_size=-1
                real_valid_size = len(batch)

            batch_dict = {}

            doc_len = max(len(x['doc_tok']) for x in batch)
            # feature vector
            feature_len = len(eval(batch[0]['doc_fea'])[0]) if len(batch[0].get('doc_fea', [])) > 0 else 0
            doc_id = torch.LongTensor(batch_size, doc_len).fill_(0)
            doc_tag = torch.LongTensor(batch_size, doc_len).fill_(0)
            doc_ent = torch.LongTensor(batch_size, doc_len).fill_(0)
            doc_feature = torch.Tensor(batch_size, doc_len, feature_len).fill_(0)
            doc_text = []

            query_len = max(len(x['query_tok']) for x in batch)
            query_id = torch.LongTensor(batch_size, query_len).fill_(0)
            query_tag = torch.LongTensor(batch_size, query_len).fill_(0)
            query_ent = torch.LongTensor(batch_size, query_len).fill_(0)
            query_feature = torch.Tensor(batch_size, query_len, feature_len).fill_(0) 
            query_text = []

            for i, sample in enumerate(batch):
                select_len = min(len(sample['doc_tok']), doc_len)
                try:
                    doc_id[i, :select_len] = torch.LongTensor(self.__random_select__(sample['doc_tok'][:select_len]))
                    doc_tag[i, :select_len] = torch.LongTensor(sample['doc_pos'][:select_len])
                    doc_ent[i, :select_len] = torch.LongTensor(sample['doc_ner'][:select_len])
                except:
                    import pdb;pdb.set_trace()
                doc_text.append(sample['doc_text'])
                for j, feature in enumerate(eval(sample['doc_fea'])):
                    if self.doc_maxlen and j>=self.doc_maxlen:
                        break
                    doc_feature[i, j, :] = torch.Tensor(feature)
                # parse query
                select_len = min(len(sample['query_tok']), query_len)
                query_id[i, :len(sample['query_tok'])] = torch.LongTensor(self.__random_select__(sample['query_tok'][:select_len]))
                query_tag[i, :select_len] = torch.LongTensor(sample['query_pos'][:select_len])
                query_ent[i, :select_len] = torch.LongTensor(sample['query_ner'][:select_len])
                query_text.append(sample['query_text'])
                # for j, feature in enumerate(eval(sample['query_fea'])):
                #     if self.query_maxlen and j>=self.query_maxlen:
                #         break
                #     query_feature[i, j, :] = torch.Tensor(feature)

            doc_mask = torch.eq(doc_id, 0)
            query_mask = torch.eq(query_id, 0)

            class_dict = {
                'squad':0,
                'newsqa':1,
                'newsqa_unsupervised':1,
                'newsqa_unsupervised_old_filtered':1,
                'marco':1,
                'SelfRC':0,
                'ParaphraseRC':1,
            }
            data_source = torch.LongTensor(batch_size).fill_(class_dict[self.dataset_name])
            
            b_doc_tok = doc_id
            b_doc_pos = doc_tag
            b_doc_ner = doc_ent
            b_doc_fea = doc_feature
            b_doc_text = batch_to_ids(doc_text)
            b_query_tok = query_id
            b_query_pos = query_tag
            b_query_ner = query_ent
            b_query_fea = query_feature
            b_doc_mask = doc_mask
            b_query_mask = query_mask
            b_query_text = batch_to_ids(query_text)

            b_data_source =data_source
        

            if self.mask_words:
                b_doc_tok = b_doc_tok.fill_(0)
                b_doc_pos = b_doc_pos.fill_(0)
                b_doc_ner = b_doc_ner.fill_(0)
                b_doc_fea = b_doc_fea.fill_(0)
                b_query_tok = b_query_tok.fill_(0)
                b_query_pos = b_query_pos.fill_(0)
                b_query_ner = b_query_ner.fill_(0)
                b_query_fea = b_query_fea.fill_(0)
                # b_doc_text = b_doc_text.fill_(0)
                # b_query_text = b_query_text.fill_(0)

            batch_list = [b_doc_tok, b_doc_pos, b_doc_ner, b_doc_fea, 
                   b_doc_text, b_query_tok, b_query_pos, b_query_ner,
                   b_query_fea, b_doc_mask, b_query_mask,
                   b_query_text]

            batch_name_ids = {
                'valid_size': valid_size,
                'doc_tok': 0,
                'doc_pos': 1,
                'doc_ner': 2,
                'doc_fea': 3,
                'doc_text': 4,
                'query_tok': 5,
                'query_pos': 6,
                'query_ner': 7,
                'query_fea': 8,
                'doc_mask': 9,
                'query_mask': 10,
                'query_text': 11,
                # 'input_len': 12 # length of input to core reader
                'data_source':12,
            }

            batch_list += [b_data_source]

            if self.is_train:
                start = [sample['start'] for sample in batch]
                end = [sample['end'] for sample in batch]
                score = [-1 for sample in batch]
                # score = [sample['score'] for sample in batch]
                b_start = torch.LongTensor(start)
                b_end = torch.LongTensor(end)
                b_score = torch.FloatTensor(score)
                batch_list += [b_start, b_end, b_score]
                batch_name_ids['start'] = 13
                batch_name_ids['end'] = 14
                batch_name_ids['score'] = 15


            if self.gpu:
                for i, item in enumerate(batch_list):
                    batch_list[i] = self.patch(item.pin_memory())


            b_text = [sample['context'] for sample in batch]
            b_span = [sample['span'] for sample in batch]
            b_uids = [sample['uid'] for sample in batch]

            b_text = b_text[:real_valid_size]
            b_span = b_span[:real_valid_size]
            b_uids = b_uids[:real_valid_size]
            self.offset += 1
            if self.is_train and self.offset == len(self):
                self.offset = 0

            if self.is_train:
                batch_name_ids['text'] = 16
                batch_name_ids['span'] = 17
                batch_name_ids['uids'] = 18
            else:
                batch_name_ids['text'] = 13
                batch_name_ids['span'] = 14
                batch_name_ids['uids'] = 15



            yield batch_list + [b_text, b_span, b_uids], batch_name_ids
            # yield batch_dict

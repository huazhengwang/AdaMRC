'''
High level of model for training and prediction
Created October, 2017
Author: xiaodl@microsoft.com
'''


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
import math
from collections import defaultdict

from torch.optim.lr_scheduler import *
from torch.autograd import Variable
from my_utils.utils import AverageMeter
from .dreader import DNetwork

logger = logging.getLogger(__name__)

class DocReaderModel(object):
    def __init__(self, opt, embedding=None, state_dict=None):
        self.opt = opt
        self.updates = state_dict['updates'] if state_dict and 'updates' in state_dict else 0
        self.eval_embed_transfer = True
        self.train_loss = AverageMeter()

        self.domain_acc = AverageMeter()

        self.network = DNetwork(opt, embedding)
        if state_dict:
            new_state = set(self.network.state_dict().keys())
            for k in list(state_dict['network'].keys()):
                if k not in new_state:
                    del state_dict['network'][k]
            for k, v in list(self.network.state_dict().items()):
                if k not in state_dict['network']:
                    state_dict['network'][k] = v
            self.network.load_state_dict(state_dict['network'])

        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if opt['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(parameters, opt['learning_rate'],
                                       momentum=opt['momentum'],
                                       weight_decay=opt['weight_decay'])
        elif opt['optimizer'] == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          opt['learning_rate'],
                                          weight_decay=opt['weight_decay'])
        elif opt['optimizer'] == 'adam':
            self.optimizer = optim.Adam(parameters,
                                        opt['learning_rate'],
                                        weight_decay=opt['weight_decay'])
        elif opt['optimizer'] == 'adadelta':
            self.optimizer = optim.Adadelta(parameters,
                                            opt['learning_rate'],
                                            rho=0.95)
        else:
            raise RuntimeError('Unsupported optimizer: %s' % opt['optimizer'])
        if state_dict and 'optimizer' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])

        if opt['fix_embeddings']:
            wvec_size = 0
        else:
            wvec_size = (opt['vocab_size'] - opt['tune_partial']) * opt['embedding_dim']
        if opt.get('have_lr_scheduler', False):
            if opt.get('scheduler_type', 'rop') == 'rop':
                self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=opt['lr_gamma'], patience=3)
            elif opt.get('scheduler_type', 'rop') == 'exp':
                self.scheduler = ExponentioalLR(self.optimizer, gamma=opt.get('lr_gamma', 0.5))
            else:
                milestones = [int(step) for step in opt.get('multi_step_lr', '10,20,30').split(',')]
                self.scheduler = MultiStepLR(self.optimizer, milestones=milestones, gamma=opt.get('lr_gamma'))
        else:
            self.scheduler = None
        self.total_param = sum([p.nelement() for p in parameters]) - wvec_size

    # def update(self, batch, dataset_name='squad'):
    #     self.network.train()
    #     if self.opt['cuda']:
    #         y = Variable(batch['start'].cuda(async=True)), Variable(batch['end'].cuda(async=True))
    #     else:
    #         y = Variable(batch['start']), Variable(batch['end'])
    #     start, end = self.network(batch)
    #     loss = F.cross_entropy(start, y[0]) + F.cross_entropy(end, y[1])
    #     self.train_loss.update(loss.item(), len(start))
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     torch.nn.utils.clip_grad_norm_(self.network.parameters(),self.opt['grad_clipping'])
    #     self.optimizer.step()
    #     self.updates += 1
    #     self.reset_embeddings()
    #     self.eval_embed_transfer = True

    def update(self, batch_list, name_map, dataset_name='squad', adv_grad_scale = 2.0):
        self.network.train()
        if self.opt['cuda']:
            y = Variable(batch_list[name_map['start']].cuda(async=True)), Variable(batch_list[name_map['end']].cuda(async=True))
        else:
            y = Variable(batch_list[name_map['start']]), Variable(batch_list[name_map['end']])
        start, end, disc_out, adv_norm = self.network(batch_list, name_map, adv_grad_scale)
        # loss = F.cross_entropy(start, y[0]) + F.cross_entropy(end, y[1])
        # import pdb;pdb.set_trace()
        loss = 0
        if self.opt['not_update_decoder'] is not None:
            if not self.opt['not_update_decoder'] in dataset_name:
                loss = F.cross_entropy(start, y[0]) + F.cross_entropy(end, y[1])
        else:
            loss = F.cross_entropy(start, y[0]) + F.cross_entropy(end, y[1])
        # import pdb; pdb.set_trace()
        if not self.opt['no_adv']:
            labels = batch_list[name_map['data_source']]
            disc_loss = F.nll_loss(disc_out, labels)
            loss += disc_loss
            # print ('disc loss ', disc_loss)
            _, predicted = torch.max(disc_out, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()            
            self.domain_acc.update(1.0*correct/total, len(start))

        self.train_loss.update(loss.item(), len(start))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(),self.opt['grad_clipping'])
        self.optimizer.step()
        self.updates += 1
        self.reset_embeddings()
        self.eval_embed_transfer = True

    # def predict(self, batch, top_k=1):
    #     self.network.eval()
    #     self.network.drop_emb = False
    #     start, end = self.network(batch)
    #     start = F.softmax(start)
    #     end = F.softmax(end)
    #     start = start.data.cpu()
    #     end = end.data.cpu()
    #     text = batch['text']
    #     spans = batch['span']
    #     predictions = []
    #     best_scores = []

    #     max_len = self.opt['max_len'] or start.size(1)
    #     doc_len = start.size(1)
    #     pos_enc = self.position_encoding(doc_len, max_len)
    #     for i in range(start.size(0)):
    #         scores = torch.ger(start[i], end[i])
    #         scores = scores * pos_enc
    #         scores.triu_()
    #         scores = scores.numpy()
    #         best_idx = np.argpartition(scores, -top_k, axis=None)[-top_k]
    #         best_score = np.partition(scores, -top_k, axis=None)[-top_k]
    #         s_idx, e_idx = np.unravel_index(best_idx, scores.shape)
    #         s_offset, e_offset = spans[i][s_idx][0], spans[i][e_idx][1]
    #         predictions.append(text[i][s_offset:e_offset])
    #         best_scores.append(best_score)

    #     return (predictions, best_scores)
    def predict(self, batch_list, name_map, dataset_name='squad', top_k=1):
        self.network.eval()
        self.network.drop_emb = False
        start, end,disc_out, adv_norm = self.network(batch_list, name_map)
        start = F.softmax(start)
        end = F.softmax(end)
        start = start.data.cpu()
        end = end.data.cpu()
        text = batch_list[name_map['text']]
        spans = batch_list[name_map['span']]
        predictions = []
        best_scores = []

        max_len = self.opt['max_len'] or start.size(1)
        doc_len = start.size(1)
        pos_enc = self.position_encoding(doc_len, max_len)
        for i in range(start.size(0)):
            scores = torch.ger(start[i], end[i])
            scores = scores * pos_enc
            scores.triu_()
            scores = scores.numpy()
            best_idx = np.argpartition(scores, -top_k, axis=None)[-top_k]
            best_score = np.partition(scores, -top_k, axis=None)[-top_k]
            s_idx, e_idx = np.unravel_index(best_idx, scores.shape)
            s_offset, e_offset = spans[i][s_idx][0], spans[i][e_idx][1]
            predictions.append(text[i][s_offset:e_offset])
            best_scores.append(best_score)

        return (predictions, best_scores)
    def setup_eval_embed(self, eval_embed, padding_idx = 0):
        self.network.lexicon_encoder.eval_embed = nn.Embedding(eval_embed.size(0),
                                               eval_embed.size(1),
                                               padding_idx = padding_idx)
        self.network.lexicon_encoder.eval_embed.weight.data = eval_embed
        for p in self.network.lexicon_encoder.eval_embed.parameters():
            p.requires_grad = False
        self.eval_embed_transfer = True

        if self.opt['covec_on']:
            self.network.lexicon_encoder.ContextualEmbed.setup_eval_embed(eval_embed)

    def update_eval_embed(self):
        if self.opt['tune_partial'] > 0:
            offset = self.opt['tune_partial']
            self.network.lexicon_encoder.eval_embed.weight.data[0:offset] \
                = self.network.lexicon_encoder.embedding.weight.data[0:offset]

    def reset_embeddings(self):
        if self.opt['tune_partial'] > 0:
            offset = self.opt['tune_partial']
            if offset < self.network.lexicon_encoder.embedding.weight.data.size(0):
                self.network.lexicon_encoder.embedding.weight.data[offset:] \
                    = self.network.lexicon_encoder.fixed_embedding

    def save(self, filename, epoch, best_em_score, best_f1_score):
        # strip cove
        network_state = dict([(k, v) for k, v in self.network.state_dict().items() if k[0:4] != 'CoVe'])
        if 'eval_embed.weight' in network_state:
            del network_state['eval_embed.weight']
        if 'fixed_embedding' in network_state:
            del network_state['fixed_embedding']
        params = {
            'state_dict': {'network': network_state},
            'config': self.opt,
            'epoch': epoch,
        }
        torch.save(params, filename)
        logger.info('model saved to {}'.format(filename))

    def cuda(self):
        self.network.cuda()

    def position_encoding(self, m, threshold=4):
        encoding = np.ones((m, m), dtype=np.float32)
        for i in range(m):
            for j in range(i, m):
                if j - i > threshold:
                    encoding[i][j] = float(1.0 / math.log(j - i + 1))
        return torch.from_numpy(encoding)

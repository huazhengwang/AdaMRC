
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.utils.rnn as rnn_utils
import math

_INF = float('inf')

def check_decreasing(lengths):
    lens, order = torch.sort(lengths, 0, True) 
    if torch.ne(lens, lengths).sum() == 0:
        return None
    else:
        _, rev_order = torch.sort(order)

        return lens, Variable(order), Variable(rev_order)

class SelfAttention(nn.Module):
    def __init__(self, dim):
        super(SelfAttention, self).__init__()
        self.linear_attn = nn.Linear(dim, 1)
        self.sm = nn.Softmax()
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, context):
        context2 = context.view(-1, context.size(2))
        attn = self.linear_attn(context2).view(context.size(0), context.size(1))  # batch x sourceL
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -_INF)
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL
        weightedContext = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        return weightedContext

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.adv_att = None
        self.opt = opt
        print (opt)
        self.disc_type = opt['disc_type']
        if opt['no_adv']:
            self.discriminator = None
        else:
            if opt['disc_type'] == "DNN":
                init_in_size = opt['rnn_size']
            elif opt['disc_type'] == "RNN":
                init_in_size = opt['disc_size']
                self.num_directions = 2 if opt['disc_bi_dir'] else 1
                self.rnn = nn.LSTM(opt['rnn_size'], opt['disc_size'] // self.num_directions,
                                        num_layers=1,
                                        dropout=opt['dropout'],
                                        bidirectional=opt['disc_bi_dir'])
            elif opt['disc_type'] == "CNN":
                assert False
            else:
                assert False
            if opt['adv_att']:
                self.adv_att = SelfAttention(init_in_size)
            modules = []
            for i in range(opt['disc_layer']):
                if i == 0:
                    in_size = init_in_size
                else:
                    in_size = opt['disc_size']
                modules += [nn.Linear(in_size, opt['disc_size'])]
                if opt['batch_norm']:
                    modules += [nn.BatchNorm1d(opt['disc_size'])]
                if opt['non_linear'] == "tanh":
                    modules += [nn.Tanh()]
                elif opt['non_linear'] == "relu":
                    modules += [nn.ReLU()]
                else:
                    assert False
                modules += [nn.Dropout(opt['adv_dropout_prob'])]
            if opt['label_smooth']:
                modules += [nn.Linear(opt['disc_size'], 1)]
                modules += [nn.Sigmoid()]
            else:
                modules += [nn.Linear(opt['disc_size'], 2)]#opt.num_rb_bin)]
                if opt['disc_obj_reverse']:
                    modules += [nn.Softmax()]
                else:
                    modules += [nn.LogSoftmax()]
            self.dnn = nn.Sequential(*modules)

    def forward(self, input, context, padMask, grad_scale):
        adv_norm = []
        if self.opt['no_adv']:
            disc_out = None
            adv_norm.append(0)
        else:
            adv_context_variable = torch.mul(context, 1)
            if not self.opt['separate_update']:
                adv_context_variable.register_hook(adv_wrapper(adv_norm, grad_scale))
            else:
                adv_norm.append(0)

            # import pdb; pdb.set_trace()

            if self.disc_type == "DNN":
                # adv_context_variable = adv_context_variable.t().contiguous()
                # padMask = 
                # padMask = input.eq(0)#onmt.Constants.PAD)
                if self.opt['disc_input_type'] == 1:
                    query_mask = Variable(torch.zeros(padMask.size(0), 1).byte())
                    if self.opt['cuda']:
                        query_mask = query_mask.cuda()
                    padMask = torch.cat([query_mask, padMask], 1)
                if self.adv_att:
                    self.adv_att.applyMask(padMask.data) #let it figure out itself. Backprop may have problem if not
                    averaged_context = self.adv_att(adv_context_variable)
                else:
                    padMask = 1. - padMask.float() #batch * sourceL
                    masked_context = adv_context_variable * padMask.unsqueeze(2).expand(padMask.size(0), padMask.size(1), context.size(2))
                    sent_len = torch.sum(padMask, 1)
                    averaged_context = torch.div(torch.sum(masked_context, 1).squeeze(1), sent_len.unsqueeze(1).expand(sent_len.size(0), context.size(2)))
                disc_out = self.dnn(averaged_context)
                # disc_out = self.dnn(context)
            elif self.disc_type == "RNN":
                lengths = input.data.ne(0)#onmt.Constants.PAD).sum(1).squeeze(1)
                check_res = check_decreasing(lengths)
                if check_res is None:
                    packed_emb = rnn_utils.pack_padded_sequence(adv_context_variable, lengths.tolist())
                    packed_out, hidden_t = self.rnn(packed_emb)
                    if self.adv_att:
                        assert False
                        outputs, srclens = rnn_utils.pad_packed_sequence(packed_out)
                    else:
                        hidden_t = (_fix_enc_hidden(hidden_t[0], self.num_directions)[-1],
                                    _fix_enc_hidden(hidden_t[1], self.num_directions)[-1]) #The first one is h, the other one is c
                        #print hidden_t[0].size(), hidden_t[1].size()
                        #hidden_t = torch.cat(hidden_t, 1)
                        #print hidden_t.size()
                        disc_out = self.dnn(hidden_t[0])
                else:
                    assert False
            else:
                assert False
        return disc_out, adv_norm


def adv_wrapper(norm, grad_scale):
    def hook_func(grad):
        new_grad = -grad * grad_scale
        #print new_grad
        norm.append(math.pow(new_grad.norm().data[0], 2))
        # print (grad, new_grad)
        return new_grad
        pass
    return hook_func
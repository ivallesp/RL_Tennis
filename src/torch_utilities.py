import torch
import torch.nn.functional as F
from torch.autograd import Variable


def ohe_torch(idx, nb_digits):
    idx = idx.squeeze().view(-1, 1).long()
    if idx.device.type=="cuda":
        idx_ohe = torch.cuda.FloatTensor(idx.shape[0], nb_digits).zero_().scatter_(1, idx, 1)
    else:
        idx_ohe = torch.FloatTensor(idx.shape[0], nb_digits).zero_().scatter_(1, idx, 1)
    return idx_ohe

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    s_gumbel = sample_gumbel(logits.size())
    if logits.device.type == "cuda":
        s_gumbel = s_gumbel.cuda()
    y = logits + s_gumbel
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y
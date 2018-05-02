# coding=utf-8
from prototypical_batch_sampler import PrototypicalBatchSampler
from omniglot_dataset import OmniglotDataset
from mini_imagenet_dataset import MiniImagenetDataset
from protonet import ProtoNet
import torch
from prototypical_loss import prototypical_loss as loss, obtain_mean
from torch.autograd import Variable
import numpy as np
from parser import get_parser
from tqdm import tqdm
import os


def init_seed(opt):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)


def init_dataset(opt):
    '''
    Initialize the datasets, samplers and dataloaders
    '''
    if opt.dataset == 'omniglot':
        test_dataset = OmniglotDataset(mode='test')
    elif opt.dataset == 'mini_imagenet':
        test_dataset = MiniImagenetDataset(mode='val')
    else:
        print('Dataset is not valid')
    test_sampler = PrototypicalBatchSampler(labels=test_dataset.y,
                                            classes_per_it=opt.classes_per_it_val,
                                            num_samples=opt.num_support_val + opt.num_query_val,
                                            iterations=opt.iterations)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_sampler=test_sampler)
    return test_dataloader


def init_protonet(opt):
    '''
    Initialize the ProtoNet
    '''
    if opt.dataset == 'omniglot':
        model = ProtoNet()
    elif opt.dataset == 'mini_imagenet':
        model = ProtoNet(x_dim=3)
    model = model.cuda() if opt.cuda else model
    return model

def test(opt, test_dataloader, model):
    '''
    Test the model trained with the prototypical learning algorithm
    '''
    rand_vec = 0.01 * np.random.randn(100, 1600)
    accs = []
    test_iter = iter(test_dataloader)
    #batches = [test_iter.__next__() for i in range(100)]
    batch = test_iter.__next__()
    for idx in range(101):
        counter = 0
        #for batch in batches:
        x, y = batch
        x, y = Variable(x), Variable(y)
        if opt.cuda:
            x, y = x.cuda(), y.cuda()
        model_output = model(x)
        means = obtain_mean(model_output, target=y, n_support=opt.num_support_tr)
        if idx < 100:
            means = means.data.numpy()
            means[4] = means[4] + rand_vec[idx]
            means = Variable(torch.FloatTensor(means))

        _, acc = loss(model_output, means, target=y, n_support=opt.num_support_tr)
        #avg_acc.append(acc.data[0])

        #avg = np.mean(avg_acc)
        print('Test Acc: {}'.format(acc.data[0]))
        accs.append(acc.data[0])

    for idx in range(100):
        if accs[idx] > accs[-1]:
            print('Higher index: {}'.format(idx))
    import pdb; pdb.set_trace()

    return accs


def eval(opt):
    '''
    Initialize everything and train
    '''
    options = get_parser().parse_args()

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)
    test_dataloader = init_dataset(options)
    model = init_protonet(options)
    model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    model.load_state_dict(torch.load(model_path))

    test(opt=options,
         test_dataloader=test_dataloader,
         model=model)

def main():
    options = get_parser().parse_args()
    eval(opt=options)


if __name__ == '__main__':
    main()

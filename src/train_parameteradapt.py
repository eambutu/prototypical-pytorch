# coding=utf-8
from collections import OrderedDict
from prototypical_batch_sampler import PrototypicalBatchSampler
from omniglot_dataset import OmniglotDataset
from mini_imagenet_dataset import MiniImagenetDataset
from protonet import ProtoNet
import torch
from prototypical_loss import full_loss as loss
from torch.autograd import Variable
import numpy as np
from parser import get_parser
from tqdm import tqdm
import os
import copy


inner_lr = 0.001

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
        train_dataset = OmniglotDataset(mode='train')
        val_dataset = OmniglotDataset(mode='val')
        trainval_dataset = OmniglotDataset(mode='trainval')
        test_dataset = OmniglotDataset(mode='test')
    elif opt.dataset == 'mini_imagenet':
        train_dataset = MiniImagenetDataset(mode='train')
        val_dataset = MiniImagenetDataset(mode='val')
        trainval_dataset = MiniImagenetDataset(mode='val')
        test_dataset = MiniImagenetDataset(mode='test')

    tr_sampler = PrototypicalBatchSampler(labels=train_dataset.y,
                                          classes_per_it=opt.classes_per_it_tr,
                                          num_samples=opt.num_support_tr + opt.num_query_tr,
                                          iterations=opt.iterations)

    val_sampler = PrototypicalBatchSampler(labels=val_dataset.y,
                                           classes_per_it=opt.classes_per_it_val,
                                           num_samples=opt.num_support_val + opt.num_query_val,
                                           iterations=opt.iterations)

    trainval_sampler = PrototypicalBatchSampler(labels=trainval_dataset.y,
                                                classes_per_it=opt.classes_per_it_tr,
                                                num_samples=opt.num_support_tr + opt.num_query_tr,
                                                iterations=opt.iterations)

    test_sampler = PrototypicalBatchSampler(labels=test_dataset.y,
                                            classes_per_it=opt.classes_per_it_val,
                                            num_samples=opt.num_support_val + opt.num_query_val,
                                            iterations=opt.iterations)

    tr_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                batch_sampler=tr_sampler)

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_sampler=val_sampler)

    trainval_dataloader = torch.utils.data.DataLoader(trainval_dataset,
                                                      batch_sampler=trainval_sampler)

    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_sampler=test_sampler)
    return tr_dataloader, val_dataloader, trainval_dataloader, test_dataloader


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


def init_optim(opt, model):
    '''
    Initialize optimizer
    '''
    return torch.optim.Adam(params=model.parameters(),
                            lr=opt.learning_rate)


def init_lr_scheduler(opt, optim):
    '''
    Initialize the learning rate scheduler
    '''
    return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=opt.lr_scheduler_gamma,
                                           step_size=opt.lr_scheduler_step)


def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)


def train(opt, tr_dataloader, model, optim, lr_scheduler, val_dataloader=None):
    '''
    Train the model with the prototypical learning algorithm
    '''
    if val_dataloader is None:
        best_state = None
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    pre_metatrain_loss = []
    pre_metatrain_acc = []
    post_metatrain_loss = []
    post_metatrain_acc = []
    best_acc = 0

    best_model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    last_model_path = os.path.join(opt.experiment_root, 'last_model.pth')

    copy_net = init_protonet(opt)
    copy_opt = torch.optim.Adam(params=copy_net.parameters(),
                                lr=opt.learning_rate)

    for epoch in range(opt.epochs):
        print('=== Epoch: {} ==='.format(epoch))
        tr_iter = iter(tr_dataloader)
        model.train()
        for batch in tqdm(tr_iter):
            optim.zero_grad()
            x, y = batch
            x, y = Variable(x), Variable(y)
            if opt.cuda:
                x, y = x.cuda(), y.cuda()

            model_output = model(x)
            l, acc = loss(model_output, target=y, n_support=opt.num_support_tr, inner_loop=True)
            grads = torch.autograd.grad(l, model.parameters(), create_graph=True)

            adapted_weights = OrderedDict((name, param - inner_lr * grad) for ((name, param), grad) in zip(model.named_parameters(), grads))
            model_output_adapt = model(x, adapted_weights)
            l, acc = loss(model_output_adapt, target=y, n_support=opt.num_support_tr)

            l.backward()
            optim.step()
            train_loss.append(l.data[0])
            train_acc.append(acc.data[0])
        avg_loss = np.mean(train_loss[-opt.iterations:])
        avg_acc = np.mean(train_acc[-opt.iterations:])
        print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
        lr_scheduler.step()
        if val_dataloader is None:
            continue
        val_iter = iter(val_dataloader)
        model.eval()
        for batch in val_iter:
            copy_net.load_state_dict(model.state_dict())
            cur_opt = optim.state_dict()
            copy_opt.__setstate__(cur_opt['state'])
            copy_opt.zero_grad()
            x, y = batch
            x, y = Variable(x), Variable(y)
            if opt.cuda:
                x, y = x.cuda(), y.cuda()
            model_output = copy_net(x)
            l, acc = loss(model_output, target=y, n_support=opt.num_support_val, inner_loop=True)
            pre_metatrain_loss.append(l.data[0])
            pre_metatrain_acc.append(acc.data[0])
            l.backward()
            copy_opt.step()

            model_output = copy_net(x)
            l, acc = loss(model_output, target=y, n_support=opt.num_support_val)
            val_loss.append(l.data[0])
            val_acc.append(acc.data[0])
            l, acc = loss(model_output, target=y, n_support=opt.num_support_val, inner_loop=True)
            post_metatrain_loss.append(l.data[0])
            post_metatrain_acc.append(acc.data[0])
        avg_loss = np.mean(val_loss[-opt.iterations:])
        avg_acc = np.mean(val_acc[-opt.iterations:])
        avg_pre_metatrain_loss = np.mean(pre_metatrain_loss[-opt.iterations:])
        avg_pre_metatrain_acc = np.mean(pre_metatrain_acc[-opt.iterations:])
        avg_post_metatrain_loss = np.mean(post_metatrain_loss[-opt.iterations:])
        avg_post_metatrain_acc = np.mean(post_metatrain_acc[-opt.iterations:])
        postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(
            best_acc)
        print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(
            avg_loss, avg_acc, postfix))
        print('Avg Pre Metatrain Loss: {}, Avg Pre Metatrain Acc: {}'.format(avg_pre_metatrain_loss, avg_pre_metatrain_acc))
        print('Avg Post Metatrain Loss: {}, Avg Post Metatrain Acc: {}'.format(avg_post_metatrain_loss, avg_post_metatrain_acc))
        if avg_acc >= best_acc:
            torch.save(model.state_dict(), best_model_path)
            best_acc = avg_acc
            best_state = model.state_dict()

    torch.save(model.state_dict(), last_model_path)

    for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
        save_list_to_file(os.path.join(opt.experiment_root, name + '.txt'), locals()[name])

    return best_state, best_acc, train_loss, train_acc, val_loss, val_acc


def test(opt, test_dataloader, model, optim):
    '''
    Test the model trained with the prototypical learning algorithm
    '''
    avg_acc = list()

    copy_net = init_protonet(opt)
    copy_opt = torch.optim.Adam(params=copy_net.parameters(),
                                lr=opt.learning_rate)

    for epoch in range(10):
        test_iter = iter(test_dataloader)
        for batch in test_iter:
            copy_net.load_state_dict(model.state_dict())
            cur_opt = optim.state_dict()
            copy_opt.__setstate__(cur_opt['state'])
            copy_opt.zero_grad()

            x, y = batch
            x, y = Variable(x), Variable(y)
            if opt.cuda:
                x, y = x.cuda(), y.cuda()
            model_output = copy_net(x)
            l, acc = loss(model_output, target=y, n_support=opt.num_support_tr, inner_loop=True)
            l.backward()
            copy_opt.step()

            model_output = copy_net(x)
            _, acc = loss(model_output, target=y, n_support=opt.num_support_tr)
            avg_acc.append(acc.data[0])
    avg_acc = np.mean(avg_acc)
    print('Test Acc: {}'.format(avg_acc))

    return avg_acc


def eval(opt):
    '''
    Initialize everything and train
    '''
    options = get_parser().parse_args()

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)
    test_dataloader = init_dataset(options)[-1]
    model = init_protonet(options)
    model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    model.load_state_dict(torch.load(model_path))

    test(opt=options,
         test_dataloader=test_dataloader,
         model=model)


def main():
    '''
    Initialize everything and train
    '''
    options = get_parser().parse_args()
    if not os.path.exists(options.experiment_root):
        os.makedirs(options.experiment_root)

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)
    tr_dataloader, val_dataloader, trainval_dataloader, test_dataloader = init_dataset(
        options)
    model = init_protonet(options)
    optim = init_optim(options, model)
    lr_scheduler = init_lr_scheduler(options, optim)
    res = train(opt=options,
                tr_dataloader=tr_dataloader,
                val_dataloader=val_dataloader,
                model=model,
                optim=optim,
                lr_scheduler=lr_scheduler)
    best_state, best_acc, train_loss, train_acc, val_loss, val_acc = res
    print('Testing with last model..')
    test(opt=options,
         test_dataloader=test_dataloader,
         model=model,
         optim=optim)

    model.load_state_dict(best_state)
    print('Testing with best model..')
    test(opt=options,
         test_dataloader=test_dataloader,
         model=model,
         optim=optim)

    # optim = init_optim(options, model)
    # lr_scheduler = init_lr_scheduler(options, optim)

    # print('Training on train+val set..')
    # train(opt=options,
    #       tr_dataloader=trainval_dataloader,
    #       val_dataloader=None,
    #       model=model,
    #       optim=optim,
    #       lr_scheduler=lr_scheduler)

    # print('Testing final model..')
    # test(opt=options,
    #      test_dataloader=test_dataloader,
    #      model=model)


if __name__ == '__main__':
    main()

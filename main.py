import time
import torch
import logging
from utils.parser import parse_args
from utils.dataloader import Dataloader
from utils.utils import config, construct_negative_graph, choose_model
from utils.tester import Tester
import torch.nn.functional as F

if __name__ == '__main__':
    args = parse_args()
    early_stop = config(args)

    if args.gpu >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'
    device = torch.device(device)
    args.device = device

    data = args.dataset
    dataloader = Dataloader(args, data, device)
    sample_weight = dataloader.sample_weight.to(device)

    model = choose_model(args, dataloader)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    early_stop(99999.99, model)

    tester = Tester(args, model, dataloader)


    def info_nce_loss(anchor, positive, negatives, temperature=0.1):
        pos_pair = torch.cat([anchor.unsqueeze(1), positive.unsqueeze(1)], dim=1)
        pos_sim = F.cosine_similarity(pos_pair[:, 0], pos_pair[:, 1], dim=-1)

        neg_sim = F.cosine_similarity(anchor.unsqueeze(1), negatives, dim=-1)

        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1) / temperature
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(anchor.device)

        return F.cross_entropy(logits, labels)

    for epoch in range(args.epoch):
        start_time = time.time()
        logging.info('Epoch {}/{}'.format(epoch + 1, args.epoch))
        model.train()

        loss_train = torch.zeros(1).to(device)

        graph_pos = dataloader.train_graph
        for i in range(args.neg_number):
            graph_neg = construct_negative_graph(graph_pos, ('user', 'rate', 'item'))

            score_pos, score_neg, score_pos_lt, score_neg_lt, h, h_lt = model(graph_pos, graph_neg)

            positive_user = h['user']
            anchor_user = h_lt['user']
            negative_users = h_lt['user'][torch.randint(0, h_lt['user'].size(0), (args.neg_number,))]  # 随机负样本

            contrastive_loss = info_nce_loss(anchor_user, positive_user, negative_users)

            if not args.category_balance:
                loss_train += -(score_pos_lt - score_neg_lt).sigmoid().log().mean() + (-(score_pos - score_neg).sigmoid().log().mean()) + contrastive_loss
            else:
                loss = -(score_pos_lt - score_neg_lt).sigmoid().log() + (-(score_pos - score_neg).sigmoid().log())
                items = graph_pos.edges(etype='rate')[1]
                weight = sample_weight[items]
                loss_train += (weight * loss.squeeze(1)).mean() + contrastive_loss

        loss_train = loss_train / args.neg_number
        logging.info('train loss = {}'.format(loss_train.item()))
        opt.zero_grad()
        loss_train.backward()
        opt.step()

        model.eval()
        graph_val_pos = dataloader.val_graph
        graph_val_neg = construct_negative_graph(graph_val_pos, ('user', 'rate', 'item'))

        score_pos, score_neg, score_pos_lt, score_neg_lt , h, h_lt = model(graph_val_pos, graph_val_neg)

        if not args.category_balance:
            loss_val = -(score_pos_lt - score_neg_lt).sigmoid().log().mean() + (-(score_pos - score_neg).sigmoid().log().mean())
        else:
            loss = -(score_pos_lt - score_neg_lt).sigmoid().log() + (-(score_pos - score_neg).sigmoid().log())
            items = graph_val_pos.edges(etype = 'rate')[1]
            weight = sample_weight[items]
            loss_val = (weight * loss.squeeze(1)).mean()


        early_stop(loss_val, model)

        logging.info('Testing at epoch {}/{}'.format(epoch + 1, args.epoch))
        logging.info('begin testing')
        # tester = Tester(args, model, dataloader)
        res = tester.test(epoch + 1)

        end_time = time.time()
        epoch_time = end_time - start_time
        logging.info('Epoch {}/{} completed in {:.2f} minutes'.format(epoch + 1, args.epoch, epoch_time / 60))

        if torch.isnan(loss_val) == True:
            break

        if early_stop.early_stop:
            break

    logging.info('loading best model for test')
    model.load_state_dict(torch.load(early_stop.save_path))
    tester = Tester(args, model, dataloader)
    logging.info('begin testing')
    res = tester.test(999)

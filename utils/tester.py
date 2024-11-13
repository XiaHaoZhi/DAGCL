import pdb
import logging
import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import entropy

class Tester(object):
    def __init__(self, args, model, dataloader):
        self.args = args
        self.model = model
        self.history_dic = dataloader.historical_dict
        self.history_csr = dataloader.train_csr
        self.dataloader = dataloader.dataloader_test
        self.test_dic = dataloader.test_dic
        self.cate = np.array(list(dataloader.category_dic.values()))
        self.metrics = args.metrics

    def judge(self, users, items):

        results = {metric: 0.0 for metric in self.metrics}

        stat = self.stat(items)
        for metric in self.metrics:
            f = Metrics.get_metrics(metric)
            for i in range(len(items)):
                results[metric] += f(items[i], test_pos = self.test_dic[users[i]], num_test_pos = len(self.test_dic[users[i]]), count = stat[i], model = self.model)
        return results

    def ground_truth_filter(self, users, items):
        batch_size, k = items.shape
        res = []
        for i in range(len(users)):
            gt_number = len(self.test_dic[users[i]])
            if gt_number >= k:
                res.append(items[i])
            else:
                res.append(items[i][:gt_number])
        return res

    def test(self , epoch):
        results = {}
        h = self.model.get_long_tail_embedding()
        original_h = self.model.get_embedding()

        weight = 0.5
        count = 0

        for k in self.args.k_list:
            results[k] = {metric: 0.0 for metric in self.metrics}

        for batch in tqdm(self.dataloader):
            users = batch[0]
            count += users.shape[0]

            combined_h = {
                'user': weight * h['user'] + (1 - weight) * original_h['user'],
                'item': weight * h['item'] + (1 - weight) * original_h['item']
            }

            scores = self.model.get_score(combined_h, users)

            users = users.tolist()

            mask = torch.tensor(self.history_csr[users].todense(), device=scores.device).bool()
            scores[mask] = -float('inf')

            _, recommended_items_lt = torch.topk(scores, k=max(self.args.k_list))
            recommended_items_lt = recommended_items_lt.cpu()

            for k in self.args.k_list:
                results_batch = self.judge(users, recommended_items_lt[:, :k])

                for metric in self.metrics:
                    results[k][metric] += results_batch[metric]

        for k in self.args.k_list:
            for metric in self.metrics:
                results[k][metric] = results[k][metric] / count
        self.show_results(results , epoch)

    def show_results(self, results, epoch):
        formatted_results = {'Epoch': epoch}

        formatted_results['recall@100'] = '{:.4f}'.format(results[100]['recall'])
        formatted_results['recall@300'] = '{:.4f}'.format(results[300]['recall'])
        formatted_results['hit_ratio@100'] = '{:.4f}'.format(results[100]['hit_ratio'])
        formatted_results['hit_ratio@300'] = '{:.4f}'.format(results[300]['hit_ratio'])
        formatted_results['coverage@100'] = '{:.4f}'.format(results[100]['coverage'])
        formatted_results['coverage@300'] = '{:.4f}'.format(results[300]['coverage'])

        logging.info(formatted_results)

    def stat(self, items):
        stat = [np.unique(self.cate[item], return_counts=True)[1] for item in items]
        return stat


class Metrics(object):

    def __init__(self):
        pass

    @staticmethod
    def get_metrics(metric):

        metrics_map = {
            'recall': Metrics.recall,
            'hit_ratio': Metrics.hr,
            'coverage': Metrics.coverage
        }

        return metrics_map[metric]

    @staticmethod
    def recall(items, **kwargs):

        test_pos = kwargs['test_pos']
        num_test_pos = kwargs['num_test_pos']
        hit_count = np.isin(items, test_pos).sum()

        return hit_count/num_test_pos

    @staticmethod
    def hr(items, **kwargs):

        test_pos = kwargs['test_pos']
        hit_count = np.isin(items, test_pos).sum()

        if hit_count > 0:
            return 1.0
        else:
            return 0.0

    @staticmethod
    def coverage(items, **kwargs):

        count = kwargs['count']

        return count.size


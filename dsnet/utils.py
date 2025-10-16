import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
class AvgrageMeter(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res, target, pred.squeeze()

def output_metric(tar, pre):

    print(f"akk: {np.sum(pre==0)}, gt: {np.sum(pre==1)}")
    all_labels = [0, 1] # 0: background, 1: target
    matrix = confusion_matrix(tar, pre, labels = all_labels)
    OA, AA_mean, Kappa, AA = cal_results(matrix)

    # Precision, Recall, F1-Score 계산
    precision = precision_score(tar, pre, pos_label=0, labels=all_labels, average='binary', zero_division=1)
    recall = recall_score(tar, pre, pos_label=0, labels=all_labels, average='binary', zero_division=1)
    f1 = f1_score(tar, pre, pos_label=0, labels=all_labels, average='binary', zero_division=1)

    return OA, AA_mean, Kappa, AA, precision, recall, f1

def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float32)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA

def print_args(args):
    for k, v in zip(args.keys(), args.values()):
        print("{0}: {1}".format(k,v))

class NonZeroClipper(object):
    def __call__(self, module):
        if hasattr(module, 'weight'):
           w = module.weight.data
           w.clamp_(1e-6, 1)

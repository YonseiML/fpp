import numpy as np
import torch


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        num_classes = output.size(1)
        valid_topk = [k for k in topk if k <= num_classes]
        if not valid_topk:
            valid_topk = [1]

        maxk = max(valid_topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            if k <= num_classes:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            else:
                correct_k = correct[:1].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
        return res


def SCE_Loss(probs, labels, norm='l1', num_bins=20):
    assert probs.dim() == 2, "probs should be (N, K)"
    N, K = probs.shape
    assert labels.shape[0] == N, "labels length must match probs N"

    bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=probs.device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    bin_accuracy = [[0 for _ in range(K)] for _ in range(num_bins)]
    bin_confidence = [[0.0 for _ in range(K)] for _ in range(num_bins)]
    bin_num_sample = [[0 for _ in range(K)] for _ in range(num_bins)]

    for i in range(N):
        y = int(labels[i].item()) if torch.is_tensor(labels[i]) else int(labels[i])
        for k in range(K):
            conf = float(probs[i, k].item())
            bin_idx = -1
            for bl, bu in zip(bin_lowers, bin_uppers):
                bin_idx += 1
                bl = float(bl.item())
                bu = float(bu.item())

                if bl < conf and conf <= bu:
                    bin_num_sample[bin_idx][k] += 1
                    bin_accuracy[bin_idx][k] += 1 if y == k else 0
                    bin_confidence[bin_idx][k] += conf
                    break

    for b in range(num_bins):
        for k in range(K):
            n = bin_num_sample[b][k]
            if n != 0:
                bin_accuracy[b][k] = bin_accuracy[b][k] / n
                bin_confidence[b][k] = bin_confidence[b][k] / n

    sce_loss = 0.0
    for b in range(num_bins):
        for k in range(K):
            n = bin_num_sample[b][k]
            if n == 0:
                continue
            gap = bin_accuracy[b][k] - bin_confidence[b][k]
            if norm == 'l2':
                gap = gap * gap
            else:
                gap = abs(gap)
            sce_loss += (gap * n) / (N * K)

    return sce_loss, bin_accuracy, bin_confidence, bin_num_sample


def ECE_Loss(num_bins, predictions, confidences, correct, path=None):
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    bin_accuracy = [0]*num_bins
    bin_confidence = [0]*num_bins
    bin_num_sample = [0]*num_bins

    for idx in range(len(predictions)):
        confidence = confidences[idx]
        bin_idx = -1
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            bin_idx += 1
            bin_lower = bin_lower.item()
            bin_upper = bin_upper.item()
            if bin_lower < confidence and confidence <= bin_upper:
                bin_num_sample[bin_idx] += 1
                bin_accuracy[bin_idx] += correct[idx]
                bin_confidence[bin_idx] += confidences[idx]

    for idx in range(num_bins):
        if bin_num_sample[idx] != 0:
            bin_accuracy[idx] = bin_accuracy[idx]/bin_num_sample[idx]
            bin_confidence[idx] = bin_confidence[idx]/bin_num_sample[idx]

    ece_loss = 0.0
    for idx in range(num_bins):
        temp_abs = abs(bin_accuracy[idx]-bin_confidence[idx])
        ece_loss += (temp_abs*bin_num_sample[idx])/len(predictions)

    return ece_loss, bin_accuracy, bin_confidence, bin_num_sample


def AECE_Loss(num_bins, predictions, confidences, correct):
    confidences = torch.as_tensor(confidences).clamp(0.0, 1.0)
    correct = torch.as_tensor(correct).float()
    N = len(confidences)
    if N == 0:
        return 0.0, [0]*num_bins, [0]*num_bins, [0]*num_bins

    sorted_conf, _ = torch.sort(confidences)
    pos = (torch.linspace(0, 1, num_bins + 1, device=confidences.device) * (N - 1)).round().long()
    boundaries = torch.unique_consecutive(sorted_conf[pos])
    if boundaries.numel() < 2:
        return 0.0, [0]*num_bins, [0]*num_bins, [0]*num_bins

    bin_lowers = boundaries[:-1]
    bin_uppers = boundaries[1:]
    num_bins_eff = bin_lowers.numel()

    bin_accuracy = [0.0]*num_bins_eff
    bin_confidence = [0.0]*num_bins_eff
    bin_num_sample = [0]*num_bins_eff

    for i in range(N):
        c = confidences[i].item()
        y_ok = correct[i].item()
        for j, (lo, up) in enumerate(zip(bin_lowers, bin_uppers)):
            lo = float(lo); up = float(up)
            in_lower = (c >= lo) if j == 0 else (c > lo)
            if in_lower and (c <= up):
                bin_num_sample[j] += 1
                bin_accuracy[j] += y_ok
                bin_confidence[j] += c
                break

    for j in range(num_bins_eff):
        if bin_num_sample[j] > 0:
            bin_accuracy[j] /= bin_num_sample[j]
            bin_confidence[j] /= bin_num_sample[j]

    ace_loss = 0.0
    for j in range(num_bins_eff):
        ace_loss += abs(bin_accuracy[j] - bin_confidence[j]) * (bin_num_sample[j] / N)

    if num_bins_eff < num_bins:
        pad = num_bins - num_bins_eff
        bin_accuracy += [0.0]*pad
        bin_confidence += [0.0]*pad
        bin_num_sample += [0]*pad

    return ace_loss, bin_accuracy, bin_confidence, bin_num_sample


def MCE_Loss(num_bins, predictions, confidences, correct):
    confidences = torch.as_tensor(confidences).clamp(0.0, 1.0)
    correct = torch.as_tensor(correct).float()

    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    bin_accuracy = [0.0]*num_bins
    bin_confidence = [0.0]*num_bins
    bin_num_sample = [0]*num_bins

    N = len(confidences)
    if N == 0:
        return 0.0, bin_accuracy, bin_confidence, bin_num_sample

    for i in range(N):
        c = float(confidences[i])
        y_ok = float(correct[i])
        for j, (lo, up) in enumerate(zip(bin_lowers, bin_uppers)):
            lo = float(lo); up = float(up)
            if (c > lo) and (c <= up):
                bin_num_sample[j] += 1
                bin_accuracy[j] += y_ok
                bin_confidence[j] += c
                break

    for j in range(num_bins):
        if bin_num_sample[j] > 0:
            bin_accuracy[j] /= bin_num_sample[j]
            bin_confidence[j] /= bin_num_sample[j]

    mce_loss = 0.0
    for j in range(num_bins):
        if bin_num_sample[j] > 0:
            d = abs(bin_accuracy[j] - bin_confidence[j])
            if d > mce_loss:
                mce_loss = d

    return mce_loss, bin_accuracy, bin_confidence, bin_num_sample


def calc_aurc(confidences, labels):
    confidences = np.array(confidences)
    labels = np.array(labels)
    predictions = np.argmax(confidences, axis=1)
    max_confs = np.max(confidences, axis=1)

    n = len(labels)
    indices = np.argsort(max_confs)
    labels = labels[indices][::-1]
    predictions = predictions[indices][::-1]

    risk_cov = np.cumsum(labels != predictions) / (np.arange(1, n + 1))
    aurc = np.mean(risk_cov)

    return aurc


def Calculator(result_dict, args):
    list_max_confidence = result_dict['max_confidence']
    list_prediction = result_dict['prediction']
    list_label = result_dict['label']
    list_prob = result_dict['prob']

    torch_list_prediction = torch.tensor(list_prediction).int()
    torch_list_label = torch.tensor(list_label).int()

    torch_correct = (torch_list_prediction == torch_list_label)
    list_correct = torch_correct.tolist()

    ece_data = ECE_Loss(20, list_prediction, list_max_confidence, list_correct)

    probs = torch.cat(list_prob, dim=0)
    labels = torch.cat(list_label, dim=0)

    sce_data = SCE_Loss(probs, labels)

    aece_data = AECE_Loss(20, list_prediction, list_max_confidence, list_correct)

    mce_data = MCE_Loss(20, list_prediction, list_max_confidence, list_correct)

    probs_np = probs.cpu().numpy()
    labels_np = labels.cpu().numpy()
    aurc = calc_aurc(probs_np, labels_np)

    acc = sum(list_correct)/len(list_correct)

    print('acc: ', acc*100)
    print('ece: ', ece_data[0]*100)
    print('sce: ', sce_data[0]*100)
    print('aece: ', aece_data[0]*100)
    print('mce: ', mce_data[0]*100)
    print('aurc: ', aurc)

    return acc*100, ece_data[0]*100, sce_data[0]*100, aece_data[0]*100, mce_data[0]*100, aurc

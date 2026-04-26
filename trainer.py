import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from time import time

from gnn import GNN_module
from cnn import EmbeddingCNN, Linear_model, myModel


def tensor2cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def np2cuda(array):
    tensor = torch.from_numpy(array)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


# ---------------------------------------------------------------------------
# Physics-informed regularisation
# ---------------------------------------------------------------------------

def physics_loss(support_features, support_labels, nway):
    """Intra-class embedding variance penalty.

    For each seen class, compute per-dimension variance across its support
    embeddings, then average over classes and dimensions.

    Args:
        support_features: (B, N_support, D)  — stacked support embeddings
        support_labels:   (B, N_support)     — integer class labels 0..nway-1
        nway: int

    Returns:
        Scalar tensor.
    """
    B, N, D = support_features.shape
    var_sum = support_features.new_zeros(1)
    count = 0

    for c in range(nway):
        mask = (support_labels == c)  # (B, N) bool
        # Average across batch items that have this class
        for b in range(B):
            class_feats = support_features[b][mask[b]]  # (K_c, D)
            if class_feats.size(0) > 1:
                var_sum = var_sum + class_feats.var(dim=0).mean()
                count += 1

    if count == 0:
        return support_features.new_zeros(1).squeeze()
    return (var_sum / count).squeeze()


# ---------------------------------------------------------------------------
# GNN wrapper
# ---------------------------------------------------------------------------

class GNN(myModel):
    def __init__(self, cnn_feature_size, gnn_hidden_dim, nway):
        super(GNN, self).__init__()
        num_inputs = cnn_feature_size + nway + 1
        self.gnn_obj = GNN_module(
            nway=nway,
            input_dim=num_inputs,
            hidden_dim=gnn_hidden_dim,
        )

    def forward(self, inputs):
        return self.gnn_obj(inputs)


class gnnModel(myModel):
    def __init__(self, nway, batchsize, shots, use_gradient_checkpointing=False):
        super(myModel, self).__init__()
        self.batchsize = batchsize
        self.nway = nway
        self.shots = shots

        image_size = 100
        cnn_feature_size = 64
        cnn_hidden_dim = 32
        cnn_num_layers = 4
        gnn_hidden_dim = 16   # hidden dim of UpdateModule

        self.cnn_feature = EmbeddingCNN(
            image_size, cnn_feature_size, cnn_hidden_dim, cnn_num_layers,
            use_gradient_checkpointing=use_gradient_checkpointing
        )
        self.gnn = GNN(cnn_feature_size, gnn_hidden_dim, nway)

    def forward(self, data, return_embeddings=False):
        [x, _, _, _, xi, label_yi, one_hot_yi, _] = data

        z = self.cnn_feature(x)                                      # (B, D)
        zi = torch.stack(
            [self.cnn_feature(xi[:, i]) for i in range(xi.size(1))],
            dim=1
        )                                                             # (B, N_sup, D)

        # Uniform label for query node
        uniform_pad = x.new_full(
            (one_hot_yi.size(0), 1, one_hot_yi.size(2)),
            1.0 / one_hot_yi.size(2)
        )

        labels   = torch.cat([uniform_pad, one_hot_yi], dim=1)       # (B, N_sup+1, C+1)
        features = torch.cat([z.unsqueeze(1), zi], dim=1)            # (B, N_sup+1, D)
        nodes    = torch.cat([features, labels], dim=2)               # (B, N_sup+1, D+C+1)

        logits = self.gnn(nodes)                                      # (B, nway+1)
        log_probs = F.log_softmax(logits, dim=1)

        if return_embeddings:
            return log_probs, zi, label_yi
        return log_probs


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, trainer_dict):
        self.args = trainer_dict['args']
        self.logger = trainer_dict['logger']

        if self.args.todo == 'train':
            self.tr_dataloader = trainer_dict['tr_dataloader']

        self.model = gnnModel(
            nway=self.args.nway,
            batchsize=self.args.batch_size,
            shots=self.args.shots,
            use_gradient_checkpointing=self.args.gradient_checkpointing
        )
        self.logger.info(self.model)
        self.total_iter = 0

    def load_model(self, model_dir):
        self.model.load(model_dir)
        print('load model successfully...')

    def model_cuda(self):
        if torch.cuda.is_available():
            self.model.cuda()

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------

    def eval(self, dataloader, test_sample=None):
        self.model.eval()
        args = self.args
        if test_sample is None:
            test_sample = args.eval_sample_8gb
        batch_size = args.eval_batch_size
        iterations = max(1, int(test_sample / batch_size))

        total_correct = total_sample = 0
        seen_correct = seen_total = 0
        unseen_correct = unseen_total = 0

        y_true_all, y_pred_all = [], []

        unseen_label = args.nway

        with torch.no_grad():
            for _ in range(iterations):
                data = dataloader.load_te_batch(
                    batch_size=batch_size,
                    nway=args.nway,
                    num_shots=args.shots
                )
                data_cuda = [tensor2cuda(d) for d in data]
                log_probs = self.model(data_cuda)

                label = data_cuda[1]
                loss_val = F.nll_loss(log_probs, label).item()

                pred = torch.argmax(log_probs, dim=1)
                correct_mask = torch.eq(pred, label)

                total_correct += correct_mask.float().sum().item()
                total_sample  += pred.shape[0]

                is_unseen = (label == unseen_label)
                unseen_correct += (correct_mask & is_unseen).float().sum().item()
                unseen_total   += is_unseen.float().sum().item()
                seen_correct   += (correct_mask & ~is_unseen).float().sum().item()
                seen_total     += (~is_unseen).float().sum().item()

                y_true_all.extend(label.cpu().tolist())
                y_pred_all.extend(pred.cpu().tolist())

        overall_acc = 100.0 * total_correct / max(total_sample, 1)
        seen_acc    = 100.0 * seen_correct  / max(seen_total, 1)
        unseen_acc  = 100.0 * unseen_correct / max(unseen_total, 1)

        self.logger.info(
            f'seen={seen_acc:.2f}% unseen={unseen_acc:.2f}% overall={overall_acc:.2f}%'
        )

        # Full per-class metrics via evaluate.py
        try:
            from evaluate import compute_metrics, save_report
            nway = self.args.nway
            class_names = [str(i) for i in range(nway)] + ['unseen']
            metrics = compute_metrics(y_true_all, y_pred_all, class_names,
                                      unseen_label=nway)
            f1_summary = '  '.join(
                f'{name}:{v["f1"]:.1f}' for name, v in metrics['per_class'].items()
            )
            self.logger.info('per-class F1: ' + f1_summary)
            if self.args.save:
                config = '%dway_%dshot_%s_%s' % (
                    self.args.nway, self.args.shots,
                    self.args.model_type, self.args.affix
                )
                save_report(metrics, config, self.args.eval_output, self.total_iter)
        except Exception as e:
            self.logger.warning(f'evaluate.py integration error: {e}')

        avg_loss = loss_val  # last batch loss as proxy
        return avg_loss, overall_acc, seen_acc, unseen_acc, y_true_all, y_pred_all

    def eval_augmented(self, dataloader, test_sample=1000):
        """Second eval pass with RandomRotation360 + SpeckleNoise on query images."""
        from augment import RandomRotation360, SpeckleNoise
        rot = RandomRotation360()
        spk = SpeckleNoise(self.args.speckle_sigma)

        self.model.eval()
        args = self.args
        iterations = max(1, int(test_sample / args.batch_size))

        total_correct = total_sample = 0

        with torch.no_grad():
            for _ in range(iterations):
                data = dataloader.load_te_batch(
                    batch_size=args.batch_size,
                    nway=args.nway,
                    num_shots=args.shots
                )
                # Augment query images (data[0])
                augmented_x = torch.stack(
                    [spk(rot(img)) for img in data[0]], dim=0
                )
                data[0] = augmented_x
                data_cuda = [tensor2cuda(d) for d in data]
                log_probs = self.model(data_cuda)
                label = data_cuda[1]
                pred = torch.argmax(log_probs, dim=1)
                total_correct += torch.eq(pred, label).float().sum().item()
                total_sample  += pred.shape[0]

        aug_acc = 100.0 * total_correct / max(total_sample, 1)
        self.logger.info(f'augmented-test acc={aug_acc:.2f}%')
        return aug_acc

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def train_batch(self):
        self.model.train()
        args = self.args

        data = self.tr_dataloader.load_tr_batch(
            batch_size=args.batch_size,
            nway=args.nway,
            num_shots=args.shots
        )
        data_cuda = [tensor2cuda(d) for d in data]

        self.opt.zero_grad()

        with torch.amp.autocast('cuda', enabled=args.amp and torch.cuda.is_available()):
            if args.physics_lambda > 0:
                log_probs, support_zi, support_labels = self.model(
                    data_cuda, return_embeddings=True
                )
            else:
                log_probs = self.model(data_cuda)

            label = data_cuda[1]
            cls_loss = F.nll_loss(log_probs, label)

            if args.physics_lambda > 0:
                phys = physics_loss(support_zi, support_labels, args.nway)
                total_loss = cls_loss + args.physics_lambda * phys
            else:
                total_loss = cls_loss
                phys = None

        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.opt)
        self.scaler.update()

        return cls_loss.item(), (phys.item() if phys is not None else 0.0)

    def train(self):
        if self.args.freeze_cnn:
            self.model.cnn_feature.freeze_weight()
            print('freeze cnn weight...')

        best_acc  = 0.0
        stop      = 0
        eval_sample = self.args.eval_sample_8gb

        self.model_cuda()
        self.model_dir = os.path.join(self.args.model_folder, 'model.pth')

        self.opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.args.lr,
            weight_decay=1e-6
        )
        self.scaler = torch.amp.GradScaler('cuda',
            enabled=self.args.amp and torch.cuda.is_available()
        )

        start = time()
        tr_cls_losses, tr_phys_losses = [], []

        # Store the intended unseen probability; warmup starts with seen-only
        target_unseen_prob = self.tr_dataloader.unseen_prob
        if self.args.warmup_iters > 0:
            self.tr_dataloader.unseen_prob = 0.0
            self.logger.info(
                'Warmup: seen-only queries for %d iters, then unseen_prob=%.2f' % (
                    self.args.warmup_iters, target_unseen_prob
                )
            )

        for i in range(self.args.max_iteration):
            if self.args.warmup_iters > 0 and i == self.args.warmup_iters:
                self.tr_dataloader.unseen_prob = target_unseen_prob
                self.logger.info(
                    'iter %d: warmup done, unseen_prob restored to %.2f' % (
                        i, target_unseen_prob
                    )
                )

            cls_l, phys_l = self.train_batch()
            tr_cls_losses.append(cls_l)
            tr_phys_losses.append(phys_l)

            if i % self.args.log_interval == 0:
                self.logger.info(
                    'iter: %d  spent: %.1fs  cls_loss: %.5f  phys_loss: %.5f' % (
                        i, time() - start,
                        np.mean(tr_cls_losses),
                        np.mean(tr_phys_losses)
                    )
                )
                tr_cls_losses.clear()
                tr_phys_losses.clear()
                start = time()

            if i % self.args.eval_interval == 0:
                va_loss, va_acc, va_seen, va_unseen, _, _ = \
                    self.eval(self.tr_dataloader, eval_sample)

                self.logger.info('================== eval ==================')
                self.logger.info('iter: %d  va_loss: %.5f' % (i, va_loss))
                self.logger.info(
                    'seen=%.2f%%  unseen=%.2f%%  overall=%.2f%%' % (
                        va_seen, va_unseen, va_acc
                    )
                )
                self.logger.info('==========================================')

                # Checkpoint on overall accuracy (not loss)
                if va_acc > best_acc:
                    stop = 0
                    best_acc = va_acc
                    if self.args.save:
                        self.model.save(self.model_dir)
                else:
                    stop += 1

                start = time()

                if stop > self.args.early_stop:
                    self.logger.info('Early stop triggered.')
                    break

            self.total_iter += 1

        self.logger.info('============= best result ===============')
        self.logger.info('best overall acc: %.4f %%' % best_acc)


# ---------------------------------------------------------------------------
# Baseline trainer (closed-set CNN)
# ---------------------------------------------------------------------------

class TrainerBaseline:
    """Standard cross-entropy CNN classifier for few-shot baseline comparison."""

    def __init__(self, trainer_dict):
        self.args = trainer_dict['args']
        self.logger = trainer_dict['logger']
        self.tr_dataloader = trainer_dict['tr_dataloader']

        nway = self.args.nway
        self.cnn = EmbeddingCNN(100, 64, 32, 4)
        self.classifier = Linear_model(nway)
        self.model_cuda()

    def model_cuda(self):
        if torch.cuda.is_available():
            self.cnn.cuda()
            self.classifier.cuda()

    def train(self):
        args = self.args
        opt = torch.optim.Adam(
            list(self.cnn.parameters()) + list(self.classifier.parameters()),
            lr=args.lr, weight_decay=1e-6
        )
        scaler = torch.amp.GradScaler('cuda',
            enabled=args.amp and torch.cuda.is_available()
        )

        best_acc = 0.0
        start = time()

        for it in range(args.max_iteration):
            self.cnn.train(); self.classifier.train()
            data_list, label_list = self.tr_dataloader.get_data_list(
                self.tr_dataloader.full_train_dict
            )

            if args.baseline_kshot:
                class_samples = {}
                for d, l in zip(data_list, label_list):
                    class_samples.setdefault(l, []).append(d)
                data_list, label_list = [], []
                for cls, samples in class_samples.items():
                    chosen = random.sample(samples, min(args.shots, len(samples)))
                    data_list.extend(chosen)
                    label_list.extend([cls] * len(chosen))

            unique_cls = sorted(set(label_list))
            cls_map = {c: i for i, c in enumerate(unique_cls)}
            labels_idx = [cls_map[l] for l in label_list]

            # Mini-batch loop — processes args.batch_size images at a time
            opt.zero_grad()
            total_loss = 0.0
            total_correct = 0
            total_n = 0
            bs = args.batch_size
            for s in range(0, len(data_list), bs):
                x_mini = tensor2cuda(torch.stack(data_list[s:s + bs], 0))
                y_mini = tensor2cuda(torch.LongTensor(labels_idx[s:s + bs]))
                with torch.amp.autocast('cuda',
                        enabled=args.amp and torch.cuda.is_available()):
                    feats = self.cnn(x_mini)
                    logits = self.classifier(feats)
                    loss = F.cross_entropy(logits, y_mini)
                scaler.scale(loss).backward()
                total_loss += loss.item()
                total_correct += (logits.argmax(1) == y_mini).float().sum().item()
                total_n += y_mini.size(0)
            scaler.step(opt)
            scaler.update()

            if it % args.log_interval == 0:
                acc = 100.0 * total_correct / max(total_n, 1)
                self.logger.info(
                    'iter: %d  spent: %.1fs  loss: %.5f  train_acc: %.2f%%' % (
                        it, time() - start, total_loss, acc
                    )
                )
                start = time()

            if it % args.eval_interval == 0:
                eval_acc = self._eval()
                self.logger.info('eval acc: %.4f%%' % eval_acc)
                if eval_acc > best_acc:
                    best_acc = eval_acc

        self.logger.info('best baseline acc: %.4f%%' % best_acc)

    def _eval(self):
        self.cnn.eval(); self.classifier.eval()
        data_list, label_list = self.tr_dataloader.get_data_list(
            self.tr_dataloader.full_test_dict
        )
        unique_cls = sorted(set(label_list))
        cls_map = {c: i for i, c in enumerate(unique_cls)}
        labels_idx = [cls_map[l] for l in label_list]

        total_correct = 0
        total_n = 0
        bs = self.args.eval_batch_size
        with torch.no_grad():
            for s in range(0, len(data_list), bs):
                x = tensor2cuda(torch.stack(data_list[s:s + bs], 0))
                y = tensor2cuda(torch.LongTensor(labels_idx[s:s + bs]))
                with torch.amp.autocast('cuda',
                        enabled=self.args.amp and torch.cuda.is_available()):
                    feats = self.cnn(x)
                    logits = self.classifier(feats)
                total_correct += (logits.argmax(1) == y).float().sum().item()
                total_n += y.size(0)
        return 100.0 * total_correct / max(total_n, 1)


import random  # noqa: E402  (placed after class definitions to avoid circular import issues)

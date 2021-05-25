import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_loader, valid_loader, lamb=.1, mname=None, snr_ranges=None):
        # snr_ranges: list of tuples indicating snr ranges to be evaluated separately
        self.scorenames = ["tp","fp","tn","fn"]
        self.scorenames += [s+"_seg" for s in self.scorenames]
        self.metricnames = ["acc", "prec", "rec", "tpr", "tnr", "fnr", "fpr", "f1"]
        self.metricnames += [m+"_seg" for m in self.metricnames]
        self.lnames = ["loss", "bce", "conf", "gen", "repr"]  # loss names

        mnames = ["mlp", "cnn", "rnn_naive", "rnn", "rnn_conf", "rnn_gen", "rnn_repr"]  # model names
        if mname is None or mname not in mnames:
            print(f"WARNING: specify mname ({', '.join(mnames)})")
        self.single_target = mname in ["mlp", "cnn", "rnn_naive"]
        self.mname = mname

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model
        
        self.epochs_trained = 0
        self.splitnames = ["train", "valid"]
        self.train_loader, self.valid_loader = train_loader, valid_loader
        self.n_train, self.n_valid = train_loader.dataset.n_samples, valid_loader.dataset.n_samples
        
        self.lamb = lamb
        self.grad_norms = []
        self.lambdas = [lamb] if mname=="rnn_conf" else None
        
        snr_default_ranges = [(), (0.2, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 10.0)]
        self.snr_ranges = snr_default_ranges if snr_ranges is None else [()]+snr_ranges
        
        self.losses, self.metrics = {s:{} for s in self.splitnames}, {s:{} for s in self.splitnames}
        for s in self.splitnames:
            for rng in self.snr_ranges:
                self.losses[s] = {l:[] for l in self.lnames}
#                 self.losses[s][rng] = {l:[] for l in self.lnames}
                self.metrics[s][rng] = {m:[] for m in self.scorenames+self.metricnames}

    def get_scores(self, out, mask, transit, rdepth):
        score_dicts = {}  # score separated for snr ranges
        with torch.no_grad():

            depths = rdepth.max(dim=1).values if self.single_target else rdepth
            targets = transit > .5 if self.single_target else mask > .5
            preds = out[0] > .5
            
            if not self.single_target:
                depths_seg = rdepth.max(dim=1).values
                targets_seg = transit > .5
                preds_seg = preds.any(dim=1)
                depths_seg, targets_seg, preds_seg = depths_seg.view(-1), targets_seg.view(-1), preds_seg.view(-1)
                correct_seg_all = preds_seg==targets_seg

            depths, targets, preds = depths.view(-1), targets.view(-1), preds.view(-1)
            correct_all = preds==targets
                
            for rng in self.snr_ranges:
                if rng is ():
                    t, correct = targets, correct_all
                else:
                    in_range = (depths>=rng[0]) & (depths<rng[1])
                    t, correct = targets[in_range], correct_all[in_range]
                    
                tp = correct[t].sum().item()
                fp = (~correct[~t]).sum().item()
                tn = correct[~t].sum().item()
                fn = (~correct[t]).sum().item()

                if self.single_target:
                    tp_seg, fp_seg, tn_seg, fn_seg = tp, fp, tn, fn
                else:
                    if rng is ():
                        t_seg, correct_seg = targets_seg, correct_seg_all
                    else:
                        in_range = (depths_seg>=rng[0]) & (depths_seg<rng[1])
                        t_seg, correct_seg = targets_seg[in_range], correct_seg_all[in_range]

                    tp_seg = correct_seg[t_seg].sum().item()
                    fp_seg = (~correct_seg[~t_seg]).sum().item()
                    tn_seg = correct_seg[~t_seg].sum().item()
                    fn_seg = (~correct_seg[t_seg]).sum().item()
                
                score_dicts[rng] = dict(zip(self.scorenames, [tp,fp,tn,fn,tp_seg,fp_seg,tn_seg,fn_seg]))
        return score_dicts
        
        
    def get_loss(self, out, flux, mask, transit, rdepth):
        loss_conf = loss_gen = loss_repr = torch.tensor(np.nan)
        if self.mname in ["mlp", "cnn", "rnn_naive"]:
            loss = loss_bce = self.bce_loss(out[1], transit)
        else:
            if self.snr_weight is None:
                bce_loss = self.bce_loss
            else:
                weight = torch.ones(mask.shape).to(self.device)
                msk = mask > .5
                weight[msk] = rdepth[msk]  # if self.snr_weight == "snr"
                if self.snr_weight == "sqrt":
                    weight = torch.sqrt(weight)
                elif self.snr_weight == "power":
                    weight = weight**2

                bce_loss = torch.nn.BCEWithLogitsLoss(weight=weight.view(-1), pos_weight=self._tr_weight)

            if self.mname == "rnn":        
                loss = loss_bce = bce_loss(out[1].view(-1), mask.view(-1))
            else: # p: sigmoid transit prediction, a: additional output
                p, logits, a = out
                if self.mname == "rnn_conf":
                    half = round(p.size(0)/2)
                    p_ = torch.vstack((a[:half]*p[:half] + (1-a[:half])*mask[:half], p[half:]))
                    logits_ = -torch.log(1/p_ - 1)  # inverse sigmoid
                    loss_bce = bce_loss(logits_.view(-1), mask.view(-1))
                    loss_conf = -torch.log(a).mean()
                    loss = loss_bce + self.lamb*loss_conf
                else:
                    loss_bce = bce_loss(logits.view(-1), mask.view(-1))
                    if self.mname == "rnn_gen":
                        flux = torch.vstack((flux, flux)) if a.size(0) > flux.size(0) else flux
                        nanmsk = torch.isnan(flux)
                        loss_gen = self.mse_loss(a[~nanmsk], flux[~nanmsk])
                        loss = loss_bce + self.lamb*loss_gen
                    elif self.mname == "rnn_repr":
                        raise NotImplentedError("rnn_repr")    
        loss_list = [l.item() for l in [loss, loss_bce, loss_conf, loss_gen, loss_repr]]
        return loss, {lname:lvalue for (lname,lvalue) in zip(self.lnames, loss_list)}
    
    
    def adjust_lamb(self, loss_conf):
        if loss_conf > self.conf_budget:
            self.lamb += self.lamb_update
        elif self.lamb > self.lamb_update:
            self.lamb -= self.lamb_update
        self.lambdas.append(self.lamb)
            
    
    def evaluate(self, loader=None):
        self.model.eval()
        loader = self.valid_loader if loader is None else loader
        
        losses = {l:[] for l in self.lnames}
        scores = {rng:{s:0 for s in self.scorenames} for rng in self.snr_ranges}

        with torch.no_grad():
            for flux, mask, transit, rdepth in loader:
                flux, mask, transit = flux.to(self.device), mask.to(self.device), transit.to(self.device)
                rdepth = rdepth.to(self.device)
                rdepth.requires_grad = False
                
                out = self.model(flux)
                loss, lvalues = self.get_loss(out, flux, mask, transit, rdepth)
                
                batch_size = len(transit)
                for l, lval in lvalues.items():
                    if ~np.isnan(lval):
                        losses[l].append(lval*batch_size)
                        
                scores_dic = self.get_scores(out, mask, transit, rdepth)
                for rng in self.snr_ranges:
                    for s, sval in scores_dic[rng].items():
                        scores[rng][s] += sval
        return losses, scores
    
                        
    def train_epoch(self):
        self.model.train()

        epoch_losses = {l:[] for l in self.lnames}
        epoch_scores = {rng:{s:0 for s in self.scorenames} for rng in self.snr_ranges}
        
        for flux, mask, transit, rdepth in self.train_loader:
            flux, mask, transit = flux.to(self.device), mask.to(self.device), transit.to(self.device)
            rdepth = rdepth.to(self.device)
            rdepth.requires_grad = False

            self.optimizer.zero_grad()
            out = self.model(flux)
            loss, lvalues = self.get_loss(out, flux, mask, transit, rdepth)
            loss.backward()

            # gradient norm
            if self.clip_value:
                torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_value)
            if self.clip_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)

            total_norm = 0  # track norm per epoch
            for p in self.model.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            if np.isnan(total_norm):
                raise Exception("ERROR: grad norm is nan")
            self.grad_norms.append( total_norm ** (1./2) )

            self.optimizer.step()
    
            batch_size = len(transit)
            for l, lval in lvalues.items():
                if ~np.isnan(lval):
                    epoch_losses[l].append(lval*batch_size)
                
            scores_dic = self.get_scores(out, mask, transit, rdepth)
            for rng in self.snr_ranges:
                for s, sval in scores_dic[rng].items():
                    epoch_scores[rng][s] += sval
            
            if self.mname == "rnn_conf":
                self.adjust_lamb(lvalues["conf"])
                    
        self.epochs_trained += 1
        return epoch_losses, epoch_scores
    
    
    def compute_metric(self, metric, tp, fp, tn, fn):        
        if metric == "acc":
            return (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else np.nan
        elif metric == "prec":
            return tp / (tp + fp) if (tp + fp) > 0 else np.nan
        elif metric == "rec" or metric == "tpr":
            return tp / (tp + fn) if (tp + fn) > 0 else np.nan
        elif metric == "tnr":
            return tn / (tn + fp) if (tn + fp) > 0 else np.nan
        elif metric == "fnr":
            return fn / (fn + tp) if (fn + tp) > 0 else np.nan
        elif metric == "fpr":
            return fp / (fp + tn) if (fp + tn) > 0 else np.nan
        elif metric == "f1":
            return 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else np.nan
        else:
            print(f"WARNING: metric {metric} not recognized")
    
    
    def update_metrics(self, dset, scores):
        for rng in self.snr_ranges:
            tp, fp, tn, fn = scores[rng]["tp"], scores[rng]["fp"], scores[rng]["tn"],  scores[rng]["fn"]
            tp_seg, fp_seg = scores[rng]["tp_seg"], scores[rng]["fp_seg"]
            tn_seg, fn_seg = scores[rng]["tn_seg"], scores[rng]["fn_seg"]

            for metricname in self.metricnames:
                if metricname.endswith("_seg"):
                    mvalue = self.compute_metric(metricname[:-4], tp_seg, fp_seg, tn_seg, fn_seg)
                else:
                    mvalue = self.compute_metric(metricname, tp, fp, tn, fn)
                self.metrics[dset][rng][metricname].append(mvalue)

            for sname in self.scorenames:
                self.metrics[dset][rng][sname].append(eval(sname))
                
                
    def get_test_results(self, test_loader):
        self.model.to(self.device)
        test_result = {rng:{} for rng in self.snr_ranges}
        _, scores = self.evaluate(loader=test_loader)
        for rng in self.snr_ranges:
            tp, fp, tn, fn = scores[rng]["tp"], scores[rng]["fp"], scores[rng]["tn"],  scores[rng]["fn"]
            tp_seg, fp_seg = scores[rng]["tp_seg"], scores[rng]["fp_seg"]
            tn_seg, fn_seg = scores[rng]["tn_seg"], scores[rng]["fn_seg"]

            for metricname in self.metricnames:
                if metricname.endswith("_seg"):
                    mvalue = self.compute_metric(metricname[:-4], tp_seg, fp_seg, tn_seg, fn_seg)
                else:
                    mvalue = self.compute_metric(metricname, tp, fp, tn, fn)
                test_result[rng][metricname] = mvalue
            for sname in self.scorenames:
                test_result[rng][sname] = eval(sname)
        self.metrics["test"] = test_result
        self.model.to("cpu")
        
    def _set_transit_weight(self):
        # in case both snr weighting and transit weight is applied
        # a small adjustment needs to made for consistent results
        if self.snr_weight is None:
            self._tr_weight = None if self.transit_weight is None else torch.tensor([self.transit_weight]).to(self.device)
        else:
            tr_weight = 1 if self.transit_weight is None else self.transit_weight
            mask = self.train_loader.dataset.mask.numpy().astype(bool)
            tr = np.sum(mask)
            rdepth = self.train_loader.dataset.rdepth[mask].numpy()
            if self.snr_weight == "sqrt":
                tr_ = np.sum(np.sqrt(rdepth))
            elif self.snr_weight == "snr":
                tr_ = np.sum(rdepth)
            elif self.snr_weight == "power":
                tr_ = np.sum(rdepth**2)
            _tr_weight = tr_weight * tr / tr_
            self._tr_weight = None if _tr_weight==1 else torch.tensor([_tr_weight]).to(self.device)
        
    
    def train(self, epochs, lr, weight_decay=0, conf_budget=.1, lamb_update=5e-3,  
              clip_norm=0, clip_value=0, transit_weight=None, snr_weight=None):
        self.model.to(self.device)

        # initialize optimizer for first time training, or when a param has changed 
        init = self.epochs_trained == 0
        if self.epochs_trained > 0:
            init = lr != self.lr or weight_decay != self.weight_decay
            action = "Resetting" if init else "Reusing"
            print(f"{action} optimizer, continuing from epoch {self.epochs_trained}")

        if init:  # not sure if NOT resetting optimizer makes a difference for training at all
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            self.lr, self.weight_decay = lr, weight_decay
            
        self.conf_budget, self.lamb_update = conf_budget, lamb_update
        self.clip_norm, self.clip_value = clip_norm, clip_value
        self.transit_weight = transit_weight
        self.snr_weight = snr_weight  # None, "snr", "sqrt", "power"
        self._set_transit_weight()
        self.bce_loss, self.mse_loss = nn.BCEWithLogitsLoss(pos_weight=self._tr_weight), nn.MSELoss()
 
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            try:
                train_losses, train_scores = self.train_epoch()
                valid_losses, valid_scores = self.evaluate()
                
                for l in self.lnames:
                    if len(train_losses[l]):
                        self.losses["train"][l].append(np.sum(train_losses[l])/self.n_train)
                        self.losses["valid"][l].append(np.sum(valid_losses[l])/self.n_valid)
                
                for dset, scores in zip(["train", "valid"], [train_scores, valid_scores]):
                    self.update_metrics(dset, scores)     

            except KeyboardInterrupt:
                pbar.close()
                print("Stopping training")
                break
            except:
                pbar.close()
                self.model.to("cpu")
                raise

        self.model.to("cpu")
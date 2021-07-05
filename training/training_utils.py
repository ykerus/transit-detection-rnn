
import numpy as np
import matplotlib.pyplot as plt

def plot_stats(dic, keys, lbls=None, lbl_prepend="", valid_only=False, train_only=False, snr_range=None, 
               lstyle="-", return_c=False):
    # for plotting training statistics for single run
    keys = keys if isinstance(keys, list) else [keys]
    labels = keys if lbls is None else lbls
    for key, lbl in zip(keys, labels):
        train = dic["train"][key] if snr_range is None else dic["train"][snr_range][key]
        valid = dic["valid"][key] if snr_range is None else dic["valid"][snr_range][key]
        p = plt.plot(train, label=lbl_prepend+lbl, linestyle=lstyle) if not valid_only else plt.plot([], label=lbl_prepend+lbl)
        plt.plot(valid, color=p[0].get_color(), linestyle="--" if not valid_only else lstyle) if not train_only else 0
    plt.xticks(fontsize=14), plt.yticks(fontsize=14)
    return p[0].get_color() if return_c else None

def save_trainer_stats(trainer):
    # copies trainer tracked values (loss, accuracies, etc.)
    stats = {s:{"train":{}, "valid":{}} for s in ["losses", "metrics"]}
    splitnames = ["train", "valid"]
    for s in splitnames:
        stats["metrics"][s] = {rng:{} for rng in trainer.snr_ranges}
        for l in trainer.losses[s]:
            stats["losses"][s][l] = [i for i in trainer.losses[s][l]]
        for rng in stats["metrics"][s]:
            for m in trainer.metrics[s][rng]:
                stats["metrics"][s][rng][m] = [i for i in trainer.metrics[s][rng][m]]
    if "test" in trainer.metrics:
        stats["metrics"]["test"] = {rng:{} for rng in trainer.snr_ranges}
        for rng in stats["metrics"]["test"]:
            for m in trainer.metrics["test"][rng]:
                stats["metrics"]["test"][rng][m] = trainer.metrics["test"][rng][m]
    if trainer.lambdas is not None:
        stats["lambdas"] = [i for i in trainer.lambdas]
    stats["grad_norms"] = [i for i in trainer.grad_norms]
    return stats

def save_results(trainer, fname):
    utils.make_dir("results")
    with open("results/"+fname, "wb") as f:
        pickle.dump(save_trainer_stats(trainer), f)
        
def average_stats(stats):
    # in the case for multiple runs
    runs = len(stats)
    stats_mean, stats_std = {}, {}

    grad_norms = [stats[i]["grad_norms"] for i in range(runs)]
    stats_mean["grad_norms"] = np.mean(grad_norms, 0)
    stats_std["grad_norms"] = np.std(grad_norms, 0)

    #metrics
    stats_mean["metrics"], stats_std["metrics"] = {}, {}
    for split in stats[0]["metrics"].keys():
        stats_mean["metrics"][split], stats_std["metrics"][split] = {}, {}
        for rng in stats[0]["metrics"]["train"].keys():
            stats_mean["metrics"][split][rng], stats_std["metrics"][split][rng] = {}, {}
            for m in stats[0]["metrics"]["train"][rng].keys():
                mvals = [stats[i]["metrics"][split][rng][m] for i in range(runs)]
                stats_mean["metrics"][split][rng][m] = np.mean(mvals, 0)
                stats_std["metrics"][split][rng][m] = np.std(mvals, 0)
    #losses
    stats_mean["losses"], stats_std["losses"] = {}, {}
    for split in ["train", "valid"]:
        stats_mean["losses"][split], stats_std["losses"][split] = {}, {}
        for l in stats[0]["losses"]["train"].keys():
            lvals = [stats[i]["losses"][split][l] for i in range(runs)]
            stats_mean["losses"][split][l] = np.mean(lvals, 0)
            stats_std["losses"][split][l] = np.std(lvals, 0)
    return stats_mean, stats_std
    
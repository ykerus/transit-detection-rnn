import matplotlib.pyplot as plt

def plot(t, f, scatter=True, xticks=True, c="black", s=1, a=1, size=None, z=None):
    plt.figure(facecolor="w", figsize=size) if size is not None else 0
    if z is None:
        plt.scatter(t, f, s=s, color=c, alpha=a) if scatter else plt.plot(t, f, color=c, linewidth=s, alpha=a)
    else:
        plt.scatter(t, f, s=s, color=c, alpha=a, zorder=z) if scatter else \
        plt.plot(t, f, color=c, linewidth=s, alpha=a, zorder=z)
    plt.xticks(fontsize=13) if xticks else plt.xticks([])
    plt.yticks(fontsize=13)
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from db_read import *
from db_plot import COLOR_DICT
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt





def plot_example(color_dict:dict=COLOR_DICT):
    """
    Figure 1: example of deriving cost quantiles for 9 units and 3 cost bins.
    """

    np.random.seed(21)
    noise = np.random.randn(9)
    cost = np.array([15,15,15,50,50,50,100,100,100])
    volume = [50] * 3 + [100] * 3 + [25] * 3
    cumvol = np.cumsum(volume)
    fuel = 3 * ["nuclear"] + 3 * ["lignite"] + 3 * ["combined cycle gas turbine"]
    cost=  cost + noise * 2.5
    cost = np.sort(cost)

    fig, ax = plt.subplots(tight_layout=True)
    ax.grid(True)
    ax.step(cumvol, cost, lw=2, c="k", where="pre")
    ax.plot([0, cumvol[0]], [cost[0]]*2, lw=2, c="k")
    for i, x in enumerate(cumvol):
        if i == 0: 
            ax.fill_betweenx([0,cost[i]], 
                             0, x, 
                             color=color_dict[fuel[i]],
                             alpha=0.3)
        else:
            ax.fill_betweenx([0,cost[i]], 
                             cumvol[i-1], x,
                             color=color_dict[fuel[i]],
                             alpha=0.5)
    
    ax.set_xlabel('Cumulative volume')
    ax.set_ylabel('Price')

    cost_bins = np.quantile(cost, q=[1/3,2/3,1])
    index = np.searchsorted(cost_bins, cost, side="right")
    index = index.clip(max=2)
    _, first = np.unique(index, return_index=True)

    for i, bin in enumerate(cost_bins, start=1):     
        if i!= len(cost_bins):
            xs = cumvol[first[i]-1]
        else:
            xs = cumvol[-1]

        
        ax.plot([0,xs], [bin,bin], 
                ls='--', lw=2,
                c="blue")
        ax.plot(xs, bin, "o", markersize=5, c="blue")
        ax.plot([xs,xs], [0,bin],
                ls='--', lw=2, c="blue")
        ax.annotate(f"Bin {i}\nCost: {bin:.1f} €/MWh", 
                    [5, bin - 10], c="blue")

    ax.set_xlim(left=0)
    fig.suptitle("Example of cost bins")
    
    return fig, ax


def load_tensorboard_logs(log_dir: str) -> dict:
    """From a tensorboard logging directory, returns a dictionary of 
    all its scalar observations."""
    ea = EventAccumulator(log_dir)
    ea.Reload()

    data = {}

    # Scalars
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        data[tag] = {
            "steps": [e.step for e in events],
            "values": [e.value for e in events],
            "wall_times": [e.wall_time for e in events],
        }
    
    return data




def plot_losses(tensorboard_dir, keys, names=None):
    "Appendix Figure C1: convergence results for the trained algorithm."
    if names is None:
        names = keys

    logs = load_tensorboard_logs(tensorboard_dir)
    fig, axes = plt.subplots(nrows=len(keys), 
                             figsize=(12, 10), 
                             tight_layout=True)
    
    for ax, key, name in zip(axes, keys, names):
        log = logs[key]
        steps, vals = log["steps"], log["values"]
        ax.plot(steps, vals, linewidth=2)
        ax.set_title(name)
        ax.set_ylabel("Loss")
        ax.grid(True)

    ax.set_xlabel("Step")
        
   
    return fig, axes


def plot_unit_profits():
    """Plots Figure 3, which describes profit for unit 2 and portfolio
     as a function of the price markup."""
    mc_1, mc_2 = 18.0, 24.0
    price = lambda b: 54.0 if b < 54.0 else b
    quant = lambda b: 1000 if b < 54.0 else 500
    
    markups = np.linspace(1.5, 3, 1000)
    profits_2 = np.zeros_like(markups)
    profits_tot = np.zeros_like(markups)

    for i, M in enumerate(markups):
        bid = mc_2 * M
        p, q = price(bid),quant(bid)
        pf_2 = (p - mc_2) * q
        pf_tot = pf_2 + (p - mc_1) * 1000
        profits_2[i] = pf_2 / 1000
        profits_tot[i] = pf_tot / 1000


    fig, ax = plt.subplots(tight_layout=True)
    # Vertical line at economic kink: bid = 54
    k = 54.0 / mc_2

    ax.plot(markups, profits_tot,
            label="Profit (portfolio)",
            linewidth=3, color="olive")
    ax.fill_between(markups[markups > k], 
                profits_tot[0], profits_tot[markups > k], 
                color="olive", alpha=0.3, 
                label="Δ to lower bound (portfolio)")   
    
    ax.plot(markups, profits_2,
            label="Profit (unit 2)",
            linewidth=3, color="brown")
    ax.fill_between(markups[markups > k], 
                    profits_2[0], profits_2[markups > k], 
                    color="brown", alpha=0.3, 
                    label="Δ to lower bound (unit 2)")

    ax.axvline(k,
            color="black",
            linestyle="--",
            linewidth=3,
            label="Price-setting threshold")
    

    ax.set_xlim(1.5,3)
    ax.grid(True)

    # Labels
    ax.set_xlabel("Markup (Bid/MC)")
    ax.set_ylabel("Profit (k€)")
    ax.set_title("Profit in high load hours")
    # Clean legend
    ax.legend()

    return fig, ax



if __name__ == "__main__":

    try:
        with open("market_power/util_funcs/plot_config.yaml", "r") as f:
            config = yaml.safe_load(f)

        mpl.rcParams.update(config)

    except Exception as e: 
        print("Loading plot config failed: ",e)

    # Figure 1: example of a cost bin
    # fig1, ax1 = plot_example()
    # fig1.savefig("example.svg")
    # fig1.savefig("example.png")
    
    # Figure 3: profit as a function of markup
    fig3, ax3 = plot_unit_profits()
    fig3.savefig("profit.png")
    fig3.savefig("profit.svg")

    # Appendix Figure C1: tensorboard output 
    # keys = '03_grad/07_actor_loss', '02_train/02_reward', '01_eval/02_reward'
    # names = "Actor loss", "Train reward", "Eval reward"
    # figc1, axesc1 = plot_losses("tensorboard/base_op_base", keys, names)
    # figc1.savefig("tensorboard.png")
    # figc1.savefig("tensorboard.svg")

    
    



    
    

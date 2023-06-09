from pathlib import Path
import numpy as np
from mit_yaw_induction_wake_model import Windfarm

import matplotlib.pyplot as plt
import seaborn as sns


FIGDIR = Path("fig")
FIGDIR.mkdir(parents=True, exist_ok=True)


X = [0, 4, 8]
Y = [0, -0.5, 0.5]
Cts = [2.11, 1, 2]
yaws = np.deg2rad([24, 10, 0])

TITLES = ["$U$", "$dU/dC_t$", "$dU/d\gamma$"]
cmaps = ["YlGnBu_r", "RdYlGn", "RdYlGn"]
vmins = [None, -0.4, -2]
vmaxs = [None, 0.4, 2]


def plot_field(
    X, Y, field, turb_xs, turb_ys, yaws, ax, cmap=None, vmin=None, vmax=None, title=None
):
    ax.imshow(
        field,
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    for turb_x, turb_y, yaw in zip(turb_xs, turb_ys, yaws):
        # Draw turbine
        R = 0.5
        p = np.array([[0, 0], [+R, -R]])
        rotmat = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])

        p = rotmat @ p + np.array([[turb_x], [turb_y]])

        ax.plot(p[0, :], p[1, :], "k", lw=5)

    ax.text(
        0.5,
        0.98,
        title,
        horizontalalignment="center",
        verticalalignment="top",
        transform=ax.transAxes,
    )


def plot_gradients(grad_list, title_pre, title_post, save):
    fig, axes = plt.subplots(3, 1, sharex=True)

    cmap = "RdYlGn"
    vmax = np.abs(grad_list).max()
    vmin = -vmax

    for i, (ax, field) in enumerate(zip(axes, grad_list)):
        title = f"{title_pre}{i+1}{title_post}"
        plot_field(
            xmesh,
            ymesh,
            field,
            X,
            Y,
            yaws,
            ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            title=title,
        )
    plt.savefig(save, dpi=300, bbox_inches="tight")
    plt.close()


def jacobian_heatmap(jac, title, save):
    plt.figure()
    ax = sns.heatmap(
        jac,
        center=0,
        xticklabels=[f"T{i+1}" for i in range(N)],
        yticklabels=[f"T{i+1}" for i in range(N)],
        cmap="RdYlGn",
    )
    ax.set(
        title=title,
        xlabel="how this turbine...",
        ylabel="influences this turbine",
    )
    plt.savefig(save, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    windfarm = Windfarm.GradWindfarm(X, Y, Cts, yaws)
    N = len(windfarm.turbines)

    xs = np.linspace(-1, 12, 300)
    ys = np.linspace(-1.5, 1.5, 300)

    xmesh, ymesh = np.meshgrid(xs, ys)

    U, dUdCt, dUdyaw = windfarm.wsp(xmesh, ymesh)

    REWS, dREWSdCt, dREWSdyaw = windfarm.REWS_at_rotors()

    Cp, dCpdCt, dCpdyaw = windfarm.turbine_Cp()

    print("REWS:", REWS)
    print("Cp:", Cp)

    plt.figure()
    plot_field(xmesh, ymesh, U, X, Y, yaws, plt.gca(), "YlGnBu_r", title="U")
    plt.savefig(FIGDIR / "example_03_simple_windfarm_U", dpi=300, bbox_inches="tight")
    plt.close()

    plot_gradients(
        dUdCt,
        title_pre="$dU/dC_{t, ",
        title_post="}$",
        save=FIGDIR / "example_03_simple_windfarm_dUdCt.png",
    )

    plot_gradients(
        dUdyaw,
        title_pre="$dU/dyaw_{",
        title_post="}$",
        save=FIGDIR / "example_03_simple_windfarm_dUdyaw.png",
    )

    jacobian_heatmap(
        dREWSdCt,
        "REWS sensitivity to Ct",
        FIGDIR / "example_03_simple_windfarm_dREWSdCt.png",
    )

    jacobian_heatmap(
        dREWSdyaw,
        "REWS sensitivity to yaw",
        FIGDIR / "example_03_simple_windfarm_dREWSdyaw.png",
    )

    jacobian_heatmap(
        dCpdCt,
        "Cp sensitivity to Ct",
        FIGDIR / "example_03_simple_windfarm_dCpdCt.png",
    )

    jacobian_heatmap(
        dCpdyaw,
        "Cp sensitivity to yaw",
        FIGDIR / "example_03_simple_windfarm_dCpdyaw.png",
    )

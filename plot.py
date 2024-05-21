import os
import itertools

import numpy as np
import xarray as xr
import matplotlib as mpl
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex, Normalize
import arviz as az

from pymob.simulation import SimulationBase
from toopy.plot import letterer, draw_axis_letter

def xarray_indexer(ds, indices: dict, original_index="id"):
    for new_index, value in indices.items():
        ds_swapped = ds.swap_dims({original_index: new_index})
        idx = ds_swapped.sel({new_index: value}).id
        ds = ds.sel({original_index: np.array(idx.values, ndmin=1)})

    return ds

def cice_relation(sim: SimulationBase):
    ci = sim.observations.cint_diuron.dropna("id", "all").max("time")
    ce = sim.observations.cext_diuron.dropna("id", "all").isel(time=0)

    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.scatter(ci.cext_nom_diuron, ci.values)
    ax2.scatter(ce.cext_nom_diuron, ce.values)
    ax2.set_xlabel(r"$C_e$ nominal")
    ax2.set_ylabel(r"$C_e$ at $t_0$")
    ax1.set_ylabel(r"$C_i$ max")
    fig.savefig(f"{sim.output_path}/ci_ce_relationship.png")
    

def pretty_posterior_plot_multisubstance(sim):
    print("PRETTY PLOT: starting...")

    substances = sim.observations.attrs["substance"]
    # sim.inferer.load_results()
    old_time = sim.coordinates["time"].copy()
    sim.coordinates["time"] = np.linspace(24,120,200)
    x_dim="time"
    obs_raw = sim.observations
    obs_raw.survival.values = (obs_raw.survival / obs_raw.nzfe).values

    for s in substances:
        susbtance_mask = sim.observations.substance == s
        obs = obs_raw.where(susbtance_mask, drop=True)
        cext_0 = sim.model_parameters["y0"][f"cext"].where(susbtance_mask, drop=True)
        if s == "naproxen":
            bins = [100, 200, 300, 400, 1000, 1500]
            midpoint = 400
        elif s == "diclofenac":
            bins = [6,8,20,100]
            midpoint = 13
        elif s == "diuron":
            bins = [5, 12, 24, 48, 100]
            midpoint = 40
        digitized = np.digitize(cext_0, bins)
        
        obs = obs.assign_coords(cext_group=("id", digitized)).swap_dims(id="cext_group")
        data_variables = [v for v in  sim.data_variables if "cext" not in v]

        fig, axes = plt.subplots(nrows=len(data_variables), ncols=len(bins), 
                                sharex=True, sharey="row", figsize=(11,7), squeeze=False)

        panelletter = letterer("abcdefghijklmnopqrstuvwxyz")

        # colormaps = ["Greens", "GnBu", "Blues", "Oranges"]

        pretty_labels = {
            f"cext": f"$C_{{ext}}$ {s.capitalize()}",
            f"cint": f"$C_{{int}}$ {s.capitalize()}",
            f"nrf2": "Fold-change NRF2",
            f"survival": "Survival (ratio)"
        }

        limits = {
            "nrf2": {"diuron": 3, "diclofenac": 5, "naproxen": 5},
        }

        post_pred = sim.inferer.posterior_predictions(
            n=sim.inferer.n_predictions, 
            seed=sim.seed
        )

        cext0 = post_pred[f"cext"].where(post_pred[f"cext"] < bins[-1], drop=True).mean(("draw", "chain"))
        cmap = mpl.colormaps["cool"]
        cmax = float(cext0.max())
        cmin = float(cext0.min())
        # minmax_scaler = lambda x: (x - cmin) / (cmax - cmin)
        minmax_scaler = mpl.colors.TwoSlopeNorm(vmin=cmin, vcenter=midpoint, vmax=cmax)
        
        for bi, bin in enumerate(bins):
            print(f"PRETTY PLOT: make predictions for {s.capitalize()} in bin ({bi+1}/{len(bins)})")
            obs_bin = obs.sel(cext_group=bi)
            sample_ids = obs_bin.id.values
            cext0_bin = post_pred.sel(id=sample_ids)[f"cext"].isel(time=0).mean(("draw", "chain"))
            # cmap = cm.get_cmap("plasma")
            # cmap = mpl.colormaps["cool"]
            # cmax = float(cext0_bin.max())
            # cmin = float(cext0_bin.min())
            # minmax_scaler = lambda x: (x - cmin) / (cmax - cmin)
            for si, sample in enumerate(sample_ids):
                pp_s = post_pred.sel(id=sample)
                ob_s = obs_bin.swap_dims(cext_group="id").sel(id=sample)
                cid = float(pp_s[f"cext"].isel(time=0).mean(("draw", "chain")))
                color = to_hex(cmap(minmax_scaler(cid)) )
                hdi = az.hdi(pp_s, .95)

                for vi, data_variable in enumerate(data_variables):
                    y_mean = pp_s[data_variable].mean(dim=("chain", "draw"))
                    ax: plt.Axes = axes[vi, bi]

                    has_obs = not bool(ob_s[data_variable].isnull().all())
                    ls = "-" if has_obs else "dotted"
                    lw = 1.5 if has_obs else 1.0
                    alpha = .3 if has_obs else 0.15
                    alpha = alpha / len(sample_ids)
                    # c = "black" if bool(ob_s[data_variable].isnull().all()) else color
                    ax.fill_between(
                        post_pred[x_dim].values, *hdi[data_variable].values.T, 
                        alpha=alpha, color=color, zorder=2.001
                    )

                    ax.plot(
                        post_pred[x_dim].values, y_mean.values, lw=lw, 
                        color=color, ls=ls, zorder=2.003
                    )

                    ax.plot(
                        ob_s[x_dim].values, ob_s[data_variable].values, 
                        color=color, label=cid,
                        marker="o", ls="", ms=5, alpha=.75, zorder=2.003
                    )
                    # ax.legend()
                    if si == 0:
                        draw_axis_letter(ax, next(panelletter), loc=(0.9,0.97))
                        # ax.spines[["top", "right"]].set_visible(False)
                        ax.set_xlim(post_pred[x_dim].min()*0.95, post_pred[x_dim].max())
                        if data_variable == "nrf2":
                            ax.hlines(1, post_pred[x_dim].min()*0.95, post_pred[x_dim].max(), 
                                      color="black", lw=.25)
                        if data_variable == "nrf2" or data_variable == "survival":
                            iax = ax.inset_axes(bounds=(0.05, 0.05, 0.08, 0.2), transform=ax.transAxes)
                        else:
                            iax = ax.inset_axes(bounds=(0.05, 0.75, 0.08, 0.2), transform=ax.transAxes)
                        
                        cb = plt.colorbar(
                            mappable=cm.ScalarMappable(
                                norm=Normalize(vmin=cmin, vmax=cmax), 
                                cmap=cmap
                            ), 
                            cax=iax,
                            ticks=[np.ceil(cmin), np.floor(cmax)],
                        )
                        cb.ax.tick_params(labelsize=8)
                        cb.set_ticklabels([f"{int(t)} µmol/L" for t in cb.get_ticks()])
                    if bi == 0: ax.set_ylabel(pretty_labels[data_variable]) 
                    cext_unit = obs.attrs[f"cext_diuron"]
                    if vi == 0: ax.set_title(f"$C_{{ext}} \leq {bin}$ {cext_unit}", fontsize=10) 

                    if data_variable == "nrf2":
                        ax.set_ylim(0,limits["nrf2"][s])
                    if data_variable == "survival":
                        ax.set_ylim(-0.05, 1.05)
                    if data_variable == "cint":
                        ax.set_ylim(0, obs.cint.max()*1.25)

        fig.text(x=0.5, y=0.02, s="Time [hpf]")
        fig.subplots_adjust(left=0.08, right=0.99, bottom=0.08, top=0.95, wspace=0.1, hspace=0.1)

        fig.savefig(f"{sim.output_path}/combined_pps_figure_{s}.png")
        plt.close()

    sim.coordinates["time"] = old_time


def plot_experiments(self, plot_individual=True):
    obs = self.observations.swap_dims(id="experiment_id")
    experiments = self.dat.experiment_table("data/tox.db", self.observations)
    cmap = mpl.colormaps["cool"]

    # first grouping is by endpoint, because in endpoint experiments,
    # multiple variables may be recorded
    for endpoint, group in experiments.groupby("name"):
        experiment_ids = group.id.values
        
        # gather all observations in one endpoint
        endpoint_ids = np.concatenate(
            [np.array(obs.sel(experiment_id=eid).id.values, ndmin=1) 
                for eid in experiment_ids]
        )
        endpoint_obs = self.observations.sel(id=endpoint_ids)
        
        # get all substances in the endpoint data
        esubs = np.unique(endpoint_obs.substance)
        # n_per_substance = np.concatenate([np.unique(xarray_indexer(endpoint_obs, {"experiment_id": ei}).substance_index) for ei in experiment_ids])
        # np.bincount(n_per_substance)

        # get all data_variables with observations in the endpoint data
        evars = [k for k, ov in endpoint_obs.data_vars.items() if (~ov.isnull()).sum() > 0]

        # set up the figure with all panels
        rows = list(itertools.product(esubs, evars))
        ncols = len(experiment_ids)
        nrows = len(rows)
        
        if not plot_individual:
            figsize = (5 * ncols, 5 * nrows)
            fig, axes = plt.subplots(
                nrows, ncols, figsize=figsize, 
                sharex=True, sharey="row", squeeze=False
            )
        else:
            axes = np.full((nrows, ncols), None)

        # iterate over substances 
        for i, (s, v) in enumerate(rows):

            # extract data from the endpoint-substance combi
            data = xarray_indexer(endpoint_obs, {"substance": s})
            data = self.observations[v].sel(id=data.id.values)
            
            # create a concentration normalizer
            C = np.round(data.cext_nom.values, 1)
            norm = mpl.colors.Normalize(vmin=C.min(), vmax=C.max())
            
            # iterate over experiments
            for j, eid in enumerate(experiment_ids):
                
                # get the correct axis (or create a new figure)
                ax = axes[i, j]
                
                ax, meta, _, _ = self.mplot.plot_experiment(
                    self,
                    experiment_id=eid, 
                    substance=s, 
                    data_var=v, 
                    ax=ax, 
                    norm=None if plot_individual else norm,
                    cmap=cmap,
                )

                if not ax.has_data() and plot_individual:
                    plt.close()
                    continue

                folder = f"case_studies/tktd-rna-pulse/results/data/{endpoint}".lower()
                os.makedirs(folder, exist_ok=True)
                
                if plot_individual:
                    ep = str(meta['experimentator'])
                    d = str(meta["date"]).split(" ")[0].replace('-', '')
                    file = f"{v}_{eid}_{s}_{ep}_{d}".lower()
                    fig = ax.figure
                    fig.savefig(f"{folder}/{file}.png")
                    plt.close()

        if not plot_individual:
            fig.subplots_adjust(0.02, 0.02, 0.95, 0.95, 0.1, 0.1)
            fig.savefig(f"{folder}_experiments.png")
            plt.close()


def plot_variable_substance_combi(self: SimulationBase, ax, data_variable, substance, prediction="prior_predictions"):
    exposed = self.observations\
        .swap_dims(id="substance")\
        .sel(substance=substance)
    ids_subset = exposed.id.values

    if prediction == "posterior_predictions":
        preds = self.inferer.posterior_predictions(
            n=self.inferer.n_predictions, 
            seed=self.seed
        )
        mode = "mean+hdi"

    elif prediction == "prior_predictions":
        preds = self.inferer.prior_predictions(
            n=50, # this will plot 50 x all experiments. This is sufficient to get an idea
            seed=self.seed
        ).prior_predictive
        mode = "draws"
    
    if data_variable == "survival" or data_variable == "lethality":
        obs = self.scale_survival_data(data_variable)
    else:
        obs = self.observations

    ax = self.inferer.plot_predictions(
        observations=obs,
        predictions=preds,
        data_variable=data_variable, 
        x_dim="time",
        ax=ax,
        subset={"id": ids_subset},
        mode=mode
    )
    

    return ax


def plot_experiment(
    self, 
    experiment_id, 
    substance, 
    data_var, 
    ax=None, 
    norm=None,
    cmap=mpl.colormaps["cool"],
):
    # get metadata
    experiments = self.dat.experiment_table("data/tox.db", self.observations)
    meta = experiments.query(f"id=={experiment_id}").iloc[0]
    
    if ax is None:
        fig, ax = plt.subplots(1,1)
    
    indices = {
        "substance": substance, 
        "experiment_id": experiment_id,
    }

    try:
        data = xarray_indexer(
            ds=self.observations, 
            indices=indices, 
            original_index="id"
        )[data_var]
    except KeyError:
        ax.text(0.5, 0.5, "No\ndata", transform=ax.transAxes, 
                ha="center", va="center")
        return ax, meta, np.array([], ndmin=0), norm


    # get all observations from the same experiment
    obs_ids = np.array(data.id.values, ndmin=1)
    
    # get a list of symbols for different concentrations                    
    marker = iter(list(mpl.lines.Line2D.markers.keys())[slice(2,-1)])

    if norm is None:
        C = np.round(data.cext_nom.values, 1)
        norm = mpl.colors.Normalize(vmin=C.min(), vmax=C.max())

    for oid in obs_ids:
        o = data.sel(id=oid)
        o = o.dropna(dim="time")

        d = str(meta["date"]).split(" ")[0]
        ep = str(meta["experimentator"])
        n = meta["name"].capitalize()

        # s = str(o.substance.values).capitalize()
        hpf = float(o.hpf.values)
        c = round(float(o.cext_nom.values), 1)

        x = o.time
    
        col = cmap(norm(c))
        lab = f"$C_{{e,0}} = {c}$ ($ID_t = {int(o.treatment_id)}$)"
        if data_var == "lethality" or data_var == "survival":
            y = o.values / o.nzfe
            ax.set_ylim(-0.05,1.05)
            ax.set_ylabel(data_var.capitalize())

        elif data_var == "cext":
            y = o.values
            ax.set_ylabel(f"$C_{{e}}$ [µmol/l]")

        elif data_var == "cint":
            y = o.values
            ax.set_ylabel(f"$C_{{i}}$ [µmol/l]")

        elif data_var == "nrf2":
            y = o.values
            ax.set_ylabel(f"NRF2 [fc]")

        ax.plot(x, y, lw=.3, marker=next(marker), color=col, label=lab)
        ax.spines[["right", "top"]].set_visible(False)
        ax.set_title(f"{n}: ({ep}, {d}), {hpf} hpf, id={experiment_id}")
        ax.set_xlim(-5,120)
        ax.set_xlabel("time [h]")
        ax.legend(title=f"{substance.capitalize()} [µmol/L]")

    return ax, meta, obs_ids, norm

def plot_cext_cint_relation(self, observations):
    for s in ["diuron", "diclofenac", "naproxen"]:
        cext_nom = observations.cext.where(~observations.cext.isnull(), drop=True).cext_nom
        cext_meas = observations.cext.where(~observations.cext.isnull(), drop=True).max("time")
        cint_meas = observations.cint.where(~observations.cext.isnull(), drop=True).max("time")

        fig, ax = plt.subplots(1,1)
        ax.set_title(s.capitalize())
        ax.scatter(
            cext_nom.where(cext_nom.substance==s, drop=True), 
            cint_meas.where(cint_meas.substance==s, drop=True), 
            alpha=.75, color="tab:blue", label=f"$C_{{e,nom}}$ vs $C_{{i,max}}$")
        ax.scatter(
            cext_meas.where(cext_meas.substance==s, drop=True), 
            cint_meas.where(cint_meas.substance==s, drop=True), 
            alpha=.75, color="tab:orange",
            label=f"$C_{{e,max}}$ vs $C_{{i,max}}$")
        
        eids = cint_meas.where(cint_meas.substance==s, drop=True).experiment_id
        for i, eid in enumerate(eids):
            ax.text(
                cext_meas.where(cext_meas.substance==s, drop=True)[i],
                cint_meas.where(cint_meas.substance==s, drop=True)[i],
                int(eid)
            )
        ax.set_xlabel(f"$C_e$")
        ax.set_ylabel(f"$C_i$")
        ax.legend()

        fig.savefig(f"case_studies/reversible_damage/results/data/rel_ceci_{s}.png")
        plt.close()


def plot_simulation_results(results: xr.Dataset, cmap=None, data_variables=None, substances=None):
    if data_variables is None:
        data_variables = list(results.data_vars.keys())

    if cmap is None:
        cmap = mpl.colormaps["cool"]
    
    if substances is None:
        substances = results.attrs["substance"]
    
    nv = len(data_variables)
    ns = len(substances) 
    fig, axes = plt.subplots(nv, ns, figsize=(2 + ns*3, nv*1 + 2), 
                                sharex=True, squeeze=False)
    
    for i, v in enumerate(data_variables):
        for j, s in enumerate(substances):
            ax = axes[i, j]
            substance_res = results.where(results.substance==s, drop=True)
            C = np.round(substance_res.cext_nom.values, 1)
            norm = mpl.colors.Normalize(vmin=C.min(), vmax=C.max())
            if i == len(data_variables) - 1: ax.set_xlabel("Time [h]")
            if i == 0: ax.set_title(s.capitalize())
            for ii in substance_res.id:
                data = substance_res.sel(id=ii)
                ax.plot(data.time, data[v], color=cmap(norm(float(data.cext_nom))))
                if j == 0: ax.set_ylabel(v.capitalize()) 

    return fig
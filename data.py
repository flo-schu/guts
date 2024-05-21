from typing import List

import numpy as np
import xarray as xr
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sqlalchemy import select, or_, and_, intersect, bindparam, not_
from expyDB.database_model import Treatment, Observation, Experiment
from pymob.utils.misc import get_grouped_unique_val
# from expyDB import query


def load_tktd():
    data = xr.load_dataset("data/processed_data/ds2_tktd.nc")
    return data

def load_sublethal_effects():
    data = xr.load_dataset("data/processed_data/ds2_sublethal_effects.nc")
    return data

def load_data(substance: str, apical_effect: str, hpf: str, observations: list, dim_order: list):
    """
    observations are the variables that should be fitted (determined by
    ODE)
    dim_order is the correct order of dimension (this should match the
    output of the simulator)
    returns data that are required by the loss function
    """
    # find dataset
    ds = load_tktd()
    ds = ds.where(~ds.mix, drop=True) # exclude mix datasets
    ds = ds.sel(substance=substance)
    ds = ds.where(ds.cext > 0, drop=True)
    ds = ds.where(ds.hpf == 0, drop=True)
    if substance == "Naproxen":
        ds = ds.sel(id=["0hpf_Naproxen_rep1"])

    # sublethal effects from Bernhards dataset
    # TODO: ATTENTION! Currently I search the closest lethality 
    #       dataset fitting to the concentration I am using
    #       this could be difficult. A better approach may be interpolation

    dss = load_sublethal_effects()
    da = query(dss, q=dict(hpf=int(hpf), substance=substance))[apical_effect]
    nearest_conc = float(da.cext[np.argmin(abs(da.cext.values - ds.cext.values[0]))])
    lethality = da.swap_dims(id="cext" ).sel(cext=nearest_conc).mean("cext")

    ds["lethality"] = lethality.expand_dims("id").assign_coords(id=ds.id)
    # select only correct obs and put dims in correct order
    obs_arr = ds[observations].to_array().transpose(*dim_order)
    
    # transform log fc to fc by exponentiation
    # obs_arr.loc[:, :, "log_fc"] = np.exp(obs_arr.loc[:, :, "log_fc"])
    
    # remove selected data points
    # obs_arr.loc[f"0hpf_{substance}_rep1", [1.5, 3.0], "log_fc"] = np.nan
    
    # get starting point for simulation model
    y0_arr = obs_arr.sel(time=0)
    
    # fits a scaler for each id
    # TODO: define a scaling strategy (overall it is probably better to 
    #       define a scaler for each feature and be done with it)
    obs_normalized = obs_arr.copy()

    scalers = [StandardScaler().fit(obs_arr.sel(id=i)) for i in obs_arr.id]

    # iterates over each id and scales the dataset and returns it to the array
    for i in range(len(obs_arr.id)):
        selector = dict(id=i)
        obs_normalized[selector] = scalers[i].transform(obs_arr[selector])

 
    # find simulation times
    t = np.unique(np.concatenate([
        np.linspace(0, 120, 1000),
        obs_arr.time.values
    ]))
    t.sort()
    obs_idx = np.searchsorted(t, ds.time.values)

    return dict(
        y0=y0_arr, t=t, scalers=scalers, obs_idx=obs_idx, 
        obs_normalized=obs_normalized, obs=obs_arr)



def query_xarray(ds: xr.Dataset, q:dict={}, dim:str="id"):
    """
    This form helps to filter a dimension with mutliple indexes. This is 
    probably a workaround for a multiindex coordinate and is useful, where
    the dataset is not fully crossed but has a sparse structure. In such 
    cases, usage of every dim as a full coordinate would blow up the
    size of the dataset

    query a dataset which has additional coordinates that all relate to id
    TODO: assert that all queries in dict have same dimensionality as 'id'
    """
    # set first key
    old_key = dim
    for key, value in q.items():
        # swap the first key in the query with the old key to allow selecting
        # from the new dimension
        ds = ds.swap_dims({old_key: key}).sel({key: value})

        # save the new key as the new old key
        old_key = key

    # change the dimension again back to id
    ds = ds.swap_dims({old_key: dim})
    return ds


def query_database_single_substance(
        database: str,  
        hpf: float,
        substance: str, 
        substance_range=[0, np.inf],
    ):
    substances = ["diuron", "naproxen", "diclofenac"]
    zero_substances = [s for s in substances if s != substance]

    stmt = (
        select(Observation, Treatment)
        .join(Treatment.observations)
        .where(or_(
            not_(Treatment.name.contains("recovery")),
            Treatment.name.is_(None)
        ))
        .where(getattr(Treatment, f"cext_nom_{substance}") > substance_range[0])
        .where(getattr(Treatment, f"cext_nom_{substance}") <= substance_range[1])
        .where(getattr(Treatment, f"cext_nom_{zero_substances[0]}") == 0)
        .where(getattr(Treatment, f"cext_nom_{zero_substances[1]}") == 0)
        .where(Treatment.hpf == hpf)
        .where(or_(
            Observation.measurement == f"cint_{substance}",
            Observation.measurement == f"cext_{substance}",
            Observation.measurement == "nrf2",
            Observation.measurement == "lethality",
        ))
        .order_by(Observation.id)
    )

    return query(database=database, statement=stmt)



def query(database, statement):
    """ask a query to the database.

    Parameters
    ----------

    database[str]: path to a database
    statement[Select]: SQLAlchemy select statement
    """
    # ask query to database
    data = pd.read_sql(statement, con=f"sqlite:///{database}")
    
    data = data.drop(columns=["id_1", "experiment_id_1"])
    data["id"] = data["treatment_id"].astype(str) + "_" + data["replicate_id"].astype(str)

    data = data.set_index(["time", "id"])
    hpf_id = get_grouped_unique_val(data, "hpf", "id")
    cext_diu_id = get_grouped_unique_val(data, "cext_nom_diuron", "id")
    cext_dic_id = get_grouped_unique_val(data, "cext_nom_diclofenac", "id")
    cext_nap_id = get_grouped_unique_val(data, "cext_nom_naproxen", "id")
    nzfe_id = get_grouped_unique_val(data, "nzfe", "id")
    treat_id = get_grouped_unique_val(data, "treatment_id", "id")
    experi_id = get_grouped_unique_val(data, "experiment_id", "id")

    meta = (data["measurement"] + "___" + data["unit"]).unique()
    meta = {measure:unit for measure, unit in [m.split("___") for m in meta]}

    data = data.drop(columns=[
        "name", "experiment_id", "replicate_id", "unit", "nzfe", 
        "hpf", "treatment_id"
    ])

    data = data.pivot(columns="measurement", values="value")

    ds = xr.Dataset.from_dataframe(data)
    ds = ds.assign_coords({
        "cext_nom_naproxen": ("id", cext_nap_id),
        "cext_nom_diclofenac": ("id", cext_dic_id),
        "cext_nom_diuron": ("id", cext_diu_id),
        "hpf": ("id", hpf_id),
        "nzfe": ("id", nzfe_id),
        "treatment_id": ("id", treat_id),
        "experiment_id": ("id", experi_id),

    })

    ds.attrs.update(meta)

    return ds


def experiment_table(database, observations):
    stmt = select(Experiment).where(
        Experiment.id.in_(bindparam("id", expanding=True))
    )
    
    ids_b = np.unique(observations.experiment_id.values)

    experiments = pd.read_sql(
        stmt, con=f"sqlite:///{database}", 
        params={"id":[int(i) for i in ids_b]}
    )

    return experiments


def treatment_table(database, observations):
    stmt = select(Treatment).where(
        Treatment.id.in_(bindparam("id", expanding=True))
    )
    
    ids_b = np.unique(observations.treatment_id.values)

    treatments = pd.read_sql(
        stmt, con=f"sqlite:///{database}", 
        params={"id":[int(i) for i in ids_b]}
    )

    return treatments
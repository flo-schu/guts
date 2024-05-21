import numpyro
from numpyro import distributions as dist
from pymob.inference.numpyro_backend import LogNormalTrans
import jax
import jax.numpy as jnp


def preprocessing(obs, masks):

    # indexes all observations that are not NAN
    obs_idx = {k:jnp.where(~jnp.isnan(v)) for k, v in obs.items()}

    # substance index of all non-NAN observations
    si_cext = jnp.broadcast_to(jnp.array([0, 1, 2]), obs["cext"].shape)[obs_idx["cext"]]
    si_cint = jnp.broadcast_to(jnp.array([0, 1, 2]), obs["cint"].shape)[obs_idx["cint"]]

    return {
        "obs": obs,
        "masks": masks,
        "obs_idx": obs_idx,
        "si_cext": si_cext,
        "si_cint": si_cint,
    }

def indexer(sim, obs, data_var, idx):
    sim_indexed = sim[data_var][*idx[data_var]]
    obs_indexed = obs[data_var][*idx[data_var]]
    return sim_indexed, obs_indexed


def lognormal_prior(name, loc, scale, normal_base=False):
    loc = jnp.array(loc)
    scale = jnp.array(scale)
    if normal_base:
        prior_norm = numpyro.sample(
            name=f"{name}_norm",
            fn=dist.Normal(loc=jnp.zeros_like(loc), scale=jnp.ones_like(scale))
        )

        prior = numpyro.deterministic(
            name=name,
            value=jnp.exp(prior_norm * scale + jnp.log(loc))
        )
    
    else:
        prior = numpyro.sample(
            name=name,
            fn=dist.LogNormal(loc=jnp.log(loc), scale=scale)
        )

    return prior

def halfnormal_prior(name, loc, scale, normal_base=False):
    loc = jnp.array(loc)
    scale = jnp.array(scale)
    if normal_base:
        prior_norm = numpyro.sample(
            name=f"{name}_norm",
            fn=dist.Normal(loc=jnp.zeros_like(loc), scale=jnp.ones_like(scale))
        )

        prior = numpyro.deterministic(
            name=name,
            value=loc + scale * jnp.abs(prior_norm)
        )
    
    else:
        prior = numpyro.sample(
            name=name,
            fn=dist.HalfNormal(scale=scale)
        )

    return prior


def model_guts_reduced(solver, obs, masks, only_prior=False):
    """Probability model with substance specific parameters and a conditional 
    binomial probability model for survival. 

    The probability model fits the ODE system 'tktd_guts_reduced'
    
    Description of the probability model
    ------------------------------------
    The model is written so that MCMC samples are drawn from standard normal
    distributions, which are then deterministically mapped onto the log-normal
    space. This is helpful for using stochastic variational inference (SVI) with
    a multivariate normal target distribution. This target distribution can then
    in theory be used as a sampling distribution to generate proposals for true
    MCMC with NUTS.

    The model than inputs the parameters into the ODE solver, which maps the 
    parameters onto the different experiments with the different substances. And
    calculates ODE solutions for all experiments entered into the Simulation.
    Finally the log-probabilities of the observations (masked to exclude missing
    observations) are compared given the solutions of the ODE and the sigmas
    of the error model.

    Model parameters
    ----------------
    k_d: dominant rate constant
    h_b: background hazard
    z: hazard threshold according to GUTS model
    kk: killing rate of the survival model

    Dependency structure
    --------------------
    The model assumes complete parameter independence for all substances,
    including sigmas on the error models.

    """
    EPS = 1e-10

    # deterministic parameters
    k_d     = lognormal_prior(name="k_d",    loc=[0.01 , 0.01 , 0.01 ], scale=2, normal_base=True)
    z       = lognormal_prior(name="z",      loc=[10.0 , 10.0 , 10.0 ], scale=2, normal_base=True)
    h_b     = lognormal_prior(name="h_b",    loc=[1e-8, 1e-8, 1e-8], scale=2, normal_base=True)
    kk      = lognormal_prior(name="kk",     loc=[0.02 , 0.02 , 0.02 ], scale=2, normal_base=True)
    
    if only_prior:
        return

    # deterministic computations
    theta = {
        "k_d": k_d,
        "z": z,
        "h_b": h_b,
        "kk": kk,
    }
    sim = solver(theta=theta)
    cext = numpyro.deterministic("cext", sim["cext"])
    D = numpyro.deterministic("D", sim["D"])
    leth = numpyro.deterministic("lethality", sim["lethality"])
    surv = numpyro.deterministic("survival", sim["survival"])

    # indexing to long form
    substance_idx = obs["substance_index"]

    # Error models
    S = jnp.clip(surv, EPS, 1 - EPS) 
    S_cond = S[:, 1:] / S[:, :-1]
    S_cond_ = jnp.column_stack([jnp.ones_like(substance_idx), S_cond])

    n_surv = obs["survivors_before_t"]
    S_mask = masks["survival"]
    obs_survival = obs["survival"]
    
    numpyro.sample(
        "survival_obs", 
        dist.Binomial(probs=S_cond_, total_count=n_surv).mask(S_mask), 
        obs=obs_survival
    )


def model_guts_scaled_damage(solver, obs, masks, only_prior=False):
    """Probability model with substance specific parameters and a conditional 
    binomial probability model for survival. 

    The probability model fits the ODE system 'tktd_guts_scaled_damage'
    
    Description of the probability model
    ------------------------------------
    The model is written so that MCMC samples are drawn from standard normal
    distributions, which are then deterministically mapped onto the log-normal
    space. This is helpful for using stochastic variational inference (SVI) with
    a multivariate normal target distribution. This target distribution can then
    in theory be used as a sampling distribution to generate proposals for true
    MCMC with NUTS.

    The model than inputs the parameters into the ODE solver, which maps the 
    parameters onto the different experiments with the different substances. And
    calculates ODE solutions for all experiments entered into the Simulation.
    Finally the log-probabilities of the observations (masked to exclude missing
    observations) are compared given the solutions of the ODE and the sigmas
    of the error model.

    Model parameters
    ----------------
    k_i: uptake rate
    k_e: elimination rate
    k_d: dominant rate constant
    h_b: background hazard
    z: hazard threshold according to GUTS model
    kk: killing rate of the survival model
    sigma_cint: scale parameter of the error-distributions.

    Dependency structure
    --------------------
    The model assumes complete parameter independence for all substances,
    including sigmas on the error models.

    """
    EPS = 1e-10

    # deterministic parameters
    k_i     = lognormal_prior(name="k_i",    loc=[1.0  , 1.0  , 1.0  ], scale=2, normal_base=True)
    k_e     = lognormal_prior(name="k_e",    loc=[0.5  , 0.5  , 0.5  ], scale=2, normal_base=True)
    k_d     = lognormal_prior(name="k_d",    loc=[0.01 , 0.01 , 0.01 ], scale=2, normal_base=True)
    z       = lognormal_prior(name="z",      loc=[10.0 , 10.0 , 10.0 ], scale=2, normal_base=True)
    h_b     = lognormal_prior(name="h_b",    loc=[1e-8, 1e-8, 1e-8], scale=2, normal_base=True)
    kk      = lognormal_prior(name="kk",     loc=[0.02 , 0.02 , 0.02  ], scale=2, normal_base=True)
    
    # error model sigmas
    sigma_cint = halfnormal_prior(name="sigma_cint", loc=0, scale=[5, 5, 5], normal_base=True)

    if only_prior:
        return

    # deterministic computations
    theta = {
        "k_i": k_i,
        "k_e": k_e,
        "k_d": k_d,
        "z": z,
        "h_b": h_b,
        "kk": kk,
    }
    sim = solver(theta=theta)
    cext = numpyro.deterministic("cext", sim["cext"])
    cint = numpyro.deterministic("cint", sim["cint"])
    D = numpyro.deterministic("D", sim["D"])
    leth = numpyro.deterministic("lethality", sim["lethality"])
    surv = numpyro.deterministic("survival", sim["survival"])

    # indexing to long form
    substance_idx = obs["substance_index"]
    sigma_cint_indexed = sigma_cint[substance_idx]
    sigma_cint_ix_bc = jnp.broadcast_to(sigma_cint_indexed.reshape((-1, 1)), obs["cint"].shape)

    # Error models
    S = jnp.clip(surv, EPS, 1 - EPS) 
    S_cond = S[:, 1:] / S[:, :-1]
    S_cond_ = jnp.column_stack([jnp.ones_like(substance_idx), S_cond])

    n_surv = obs["survivors_before_t"]
    S_mask = masks["survival"]
    obs_survival = obs["survival"]
    
    numpyro.sample("cint_obs", dist.LogNormal(loc=jnp.log(cint + EPS), scale=sigma_cint_ix_bc).mask(masks["cint"]), obs=obs["cint"])
    numpyro.sample(
        "survival_obs", 
        dist.Binomial(probs=S_cond_, total_count=n_surv).mask(S_mask), 
        obs=obs_survival
    )


def model_guts_full(solver, obs, masks, only_prior=False):
    """Probability model with substance specific parameters and a conditional 
    binomial probability model for survival. 

    The probability model fits the ODE system 'tktd_guts_full'
    
    Description of the probability model
    ------------------------------------
    The model is written so that MCMC samples are drawn from standard normal
    distributions, which are then deterministically mapped onto the log-normal
    space. This is helpful for using stochastic variational inference (SVI) with
    a multivariate normal target distribution. This target distribution can then
    in theory be used as a sampling distribution to generate proposals for true
    MCMC with NUTS.

    The model than inputs the parameters into the ODE solver, which maps the 
    parameters onto the different experiments with the different substances. And
    calculates ODE solutions for all experiments entered into the Simulation.
    Finally the log-probabilities of the observations (masked to exclude missing
    observations) are compared given the solutions of the ODE and the sigmas
    of the error model.

    Model parameters
    ----------------
    k_i: uptake rate constant
    k_e: elimination rate constant
    k_a: damage accrual rate constant (damage-accrual)
    k_r: damage repair rate constant (damage-repair)
    h_b: background hazard
    z: hazard threshold according to GUTS model
    kk: killing rate of the survival model
    sigma_*: scale parameter of the error-distributions.

    Dependency structure
    --------------------
    The model assumes complete parameter independence for all substances,
    including sigmas on the error models.

    """
    EPS = 1e-10

    # deterministic parameters
    k_i     = lognormal_prior(name="k_i",    loc=[1.0  , 1.0  , 1.0  ], scale=2, normal_base=True)
    k_e     = lognormal_prior(name="k_e",    loc=[1.0  , 0.05  , 0.001], scale=2, normal_base=True)
    k_a     = lognormal_prior(name="k_a",    loc=[0.001, 0.001, 0.001], scale=2, normal_base=True)
    k_r     = lognormal_prior(name="k_r",    loc=[1.0  , 1.0  , 1.0  ], scale=2, normal_base=True)
    z       = lognormal_prior(name="z",      loc=[1.0  , 1.0  , 1.0  ], scale=1, normal_base=True)
    h_b     = lognormal_prior(name="h_b",    loc=[1e-8 , 1e-8 , 1e-8 ], scale=2, normal_base=True)
    kk      = lognormal_prior(name="kk",     loc=[0.02 , 0.02 , 0.02 ], scale=2, normal_base=True)
    
    # error model sigmas
    sigma_cint = halfnormal_prior(name="sigma_cint", loc=0, scale=[5, 5, 5], normal_base=True)
    sigma_nrf2 = halfnormal_prior(name="sigma_nrf2", loc=0, scale=[5, 5, 5], normal_base=True)

    if only_prior:
        return

    # deterministic computations
    theta = {
        "k_i": k_i,
        "k_e": k_e,
        "k_a": k_a,
        "k_r": k_r,
        "z": z,
        "h_b": h_b,
        "kk": kk,
    }
    sim = solver(theta=theta)
    cext = numpyro.deterministic("cext", sim["cext"])
    cint = numpyro.deterministic("cint", sim["cint"])
    nrf2 = numpyro.deterministic("nrf2", sim["nrf2"])
    leth = numpyro.deterministic("lethality", sim["lethality"])
    surv = numpyro.deterministic("survival", sim["survival"])

    # indexing to long form
    substance_idx = obs["substance_index"]
    sigma_cint_indexed = sigma_cint[substance_idx]
    sigma_nrf2_indexed = sigma_nrf2[substance_idx]
    sigma_cint_ix_bc = jnp.broadcast_to(sigma_cint_indexed.reshape((-1, 1)), obs["cint"].shape)
    sigma_nrf2_ix_bc = jnp.broadcast_to(sigma_nrf2_indexed.reshape((-1, 1)), obs["nrf2"].shape)

    # Error models
    S = jnp.clip(surv, EPS, 1 - EPS) 
    S_cond = S[:, 1:] / S[:, :-1]
    S_cond_ = jnp.column_stack([jnp.ones_like(substance_idx), S_cond])

    n_surv = obs["survivors_before_t"]
    S_mask = masks["survival"]
    obs_survival = obs["survival"]
    
    numpyro.sample("cint_obs", dist.LogNormal(loc=jnp.log(cint + EPS), scale=sigma_cint_ix_bc).mask(masks["cint"]), obs=obs["cint"])
    numpyro.sample("nrf2_obs", dist.LogNormal(loc=jnp.log(nrf2), scale=sigma_nrf2_ix_bc).mask(masks["nrf2"]), obs=obs["nrf2"])    
    numpyro.sample(
        "survival_obs", 
        dist.Binomial(probs=S_cond_, total_count=n_surv).mask(S_mask), 
        obs=obs_survival
    )
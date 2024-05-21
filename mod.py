from functools import partial

import jax.numpy as jnp
import jax
from diffrax import (
    diffeqsolve, 
    Dopri5, 
    ODETerm, 
    SaveAt, 
    PIDController, 
    RecursiveCheckpointAdjoint
)

from pymob.sim.solvetools import mappar

def simplified_ode_solver(model, post_processing, parameters, coordinates, indices, data_variables, n_ode_states, seed=None):
    coords = coordinates
    y0_arr = parameters["y0"].sel(id=coords["id"])
    params = parameters["parameters"]
    time = tuple(coords["time"])

    # collect parameters
    ode_args = mappar(model, params, exclude=["t", "X"])
    pp_args = mappar(post_processing, params, exclude=["t", "time", "results"])

    # index parameter according to substance
    s_idx = indices["substance"].values
    ode_args = [jnp.array(a, ndmin=1)[s_idx] for a in ode_args]
    pp_args = [jnp.array(a, ndmin=1)[s_idx] for a in pp_args]

    # transform Y0 to jnp.arrays
    Y0 = [jnp.array(v.values) for _, v in y0_arr.data_vars.items()]

    initialized_eval_func = partial(
        odesolve_splitargs,
        model=model,
        post_processing=post_processing,
        time=time,
        odestates = tuple(y0_arr.keys()),
        n_odeargs=len(ode_args),
        n_ppargs=len(pp_args),

    )
        
    loop_eval = jax.vmap(
        initialized_eval_func, 
        in_axes=(
            *[0 for _ in range(n_ode_states)], 
            *[0 for _ in range(len(ode_args))],
            *[0 for _ in range(len(pp_args))],
        )
    )
    result = loop_eval(*Y0, *ode_args, *pp_args)
    return result


@partial(jax.jit, static_argnames=["model", "post_processing", "time", "odestates", "n_odeargs", "n_ppargs"])
def odesolve_splitargs(*args, model, post_processing, time, odestates, n_odeargs, n_ppargs):
    n_odestates = len(odestates)
    y0 = args[:n_odestates]
    odeargs = args[n_odestates:n_odeargs+n_odestates]
    ppargs = args[n_odeargs+n_odestates:n_odeargs+n_odestates+n_ppargs]
    sol = odesolve(model=model, y0=y0, time=time, args=odeargs)
    
    res_dict = {v:val for v, val in zip(odestates, sol)}

    return post_processing(res_dict, jnp.array(time), *ppargs)


@partial(jax.jit, static_argnames=["model"])
def odesolve(model, y0, time, args):
    f = lambda t, y, args: model(t, y, *args)
    
    term = ODETerm(f)
    solver = Dopri5()
    saveat = SaveAt(ts=time)
    stepsize_controller = PIDController(rtol=1e-5, atol=1e-7)
    t_min = jnp.array(time).min()
    t_max = jnp.array(time).max()

    sol = diffeqsolve(
        terms=term, 
        solver=solver, 
        t0=t_min, 
        t1=t_max, 
        dt0=0.1, 
        y0=tuple(y0), 
        args=args, 
        saveat=saveat, 
        stepsize_controller=stepsize_controller,
        adjoint=RecursiveCheckpointAdjoint(),
        max_steps=10**5,
        # throw=False returns inf for all t > t_b, where t_b is the time 
        # at which the solver broke due to reaching max_steps. This behavior
        # happens instead of throwing an exception.
        throw=False,   
    )
    
    return list(sol.ys)


def calculate_psurv(results, t, z, kk, h_b):
    # calculate survival 
    p_surv = survival_jax(t, results["D"], z, kk, h_b)
    results["survival"] = p_surv
    results["lethality"] = 1 - p_surv
    return results


def calculate_psurv_nrf2(results, t, z, kk, h_b):
    # calculate survival 
    p_surv = survival_jax(t, results["nrf2"], z, kk, h_b)
    results["survival"] = p_surv
    results["lethality"] = 1 - p_surv
    return results


@jax.jit
def survival_jax(t, damage, z, kk, h_b):
    """
    survival probability derived from hazard 
    first calculate cumulative Hazard by integrating hazard cumulatively over t
    then calculate the resulting survival probability
    It was checked that `survival_jax` behaves exactly the same as `survival`
    """
    hazard = kk * jnp.where(damage - z < 0, 0, damage - z) + h_b
    # H = jnp.array([jax.scipy.integrate.trapezoid(hazard[:i+1], t[:i+1]) for i in range(len(t))])
    H = jnp.array([jnp.trapz(hazard[:i+1], t[:i+1], axis=0) for i in range(len(t))])
    S = jnp.exp(-H)

    return S


def tktd_guts_reduced(t, X, k_d):
    """
    ODE system modeling of th reduced guts model with scaled damage.

    This function models the dynamics of external concentration (Ce)
    The model can only handle single-substances

    Parameters
    ----------
    X : tuple
        A tuple containing 2 elements: 
        - Cext : float
            External concentration of the substance.
        - D : float
            Scaled damage.

    t : float
        Time at which the model is evaluated.

    k_d : float
        Dominant rate constant of scaled damage.

    Returns
    -------
    dCe_dt : float
        Assumed no change in environmental concentration (dCe_dt = 0)

    dD_dt : float
        The scaled damage
        
    """
    Ce, D = X

    dCe_dt = 0.0
    dD_dt = k_d * (Ce - D)

    return dCe_dt, dD_dt



def tktd_guts_scaled_damage(t, X, k_i, k_e, k_d):
    """
    ODE system modeling substance uptake, elimination, and scaled damage.

    This function models the dynamics of external concentration (Cext), internal
    concentration (Cint), and scaled Damage within an organism. The model can 
    only handle single-substances

    Parameters
    ----------
    X : tuple
        A tuple containing three elements: - Cext : float
            External concentration of the substance.
        - Cint : float
            Internal concentration of the substance within the organism.
        - D : float
            Scaled damage.

    t : float
        Time at which the model is evaluated.

    k_i : float
        Internal uptake rate constant.

    k_e : float
        Internal elimination rate constant.

    k_d : float
        Dominant rate constant of scaled damage.

    Returns
    -------
    dCe_dt : float
        Assumed no change in environmental concentration (dCe_dt = 0)

    dCi_dt : float
        The rate of change of internal concentration.

    dD_dt : float
        The scaled damage
        
    """
    Ce, Ci, D = X

    dCe_dt = 0.0
    dCi_dt = k_i * Ce - k_e * Ci
    dD_dt = k_d * (Ci - D)

    return dCe_dt, dCi_dt, dD_dt


def tktd_guts_full(t, X, d_0, k_i, k_e, k_a, k_r):
    """
    ODE system modeling of a full guts model.

    This model describes the dynamics of external concentration (Cext), internal
    concentration (Cint), and Damage within an organism. 
    The model can only handle single-substances

    Parameters
    ----------
    X : tuple
        A tuple containing three elements: 
        - Cext : float
            External concentration of the substance.
        - Cint : float
            Internal concentration of the substance within the organism.
        - D : float
            Damage in the organism

    t : float
        Time at which the model is evaluated.

    d_0 : float
        Initial damage inside the organism can also express the differential
        damage with respect to a control organism (Then d_0 should equal 1)

    k_i : float
        Internal uptake rate constant.

    k_e : float
        Internal elimination rate constant.

    k_a : float
        Damage accrual rate constant.

    k_r : float
        Damage repair rate constant.

    Returns
    -------
    dCe_dt : float
        Assumed no change in environmental concentration (dCe_dt = 0)

    dCi_dt : float
        The rate of change of internal concentration.

    dD_dt : float
        The rate of change of internal damage.
        
    """
    Ce, Ci, D = X

    dCe_dt = 0.0
    dCi_dt = k_i * Ce - k_e * Ci
    dD_dt = Ci * k_a - (D - d_0) * k_r

    return dCe_dt, dCi_dt, dD_dt

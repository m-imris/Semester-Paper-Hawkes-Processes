import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint, OptimizeResult
from scipy.stats import genpareto
from scipy.special import logsumexp
import sys
import math

class SEPOT():
    # arrival_times should be a (n, ) dimensional array containing the arrival times of the events
    # marks should be a (n, ) dimensional array containing the marks of the point process
    # threshold specifies the threshold value
    # SE_fun specifies what self-exciting function should be used
    # impact_fun specifies what impact function should be used
    # mark_dist specifies the what distribution should be used for the marks
    # random_state specifies the random seed
    def __init__(self,
                 SE_fun: str = "Exp",
                 impact_fun: str | None = None,
                 mark_dist: str = "GPD",
                 arrival_times: np.ndarray | None = None,
                 marks: np.ndarray | None = None,
                 T: float | None = None,
                 threshold: float | None = None,
                 seed: int | None = None):
        
        self.SE_fun = SE_fun
        self.impact_fun = impact_fun    
        self.mark_dist = mark_dist
        self.arrival_times = arrival_times
        self.marks = marks
        self.T = T
        self.threshold = threshold
        self.seed = seed
        self.required_params_num = 0

        if self.arrival_times.ndim == 2:
            self.arrival_times = self.arrival_times.flatten()
        if self.marks.ndim == 2:
            self.marks = self.marks.flatten()

        self.__check_model_specification()
        self.__check_for_missing_data()
        self.__check_dimensions()
        self.__check_temporal_order()
        self.__set_required_params_num()


    # Checks whether the specified model is valid
    def __check_model_specification(self):
        if self.SE_fun not in {"Exp", "Pow"}:
            raise ValueError(f"Invalid self-exciting function {self.SE_fun}. Please choose either {"Exp"} or {"Pow"}")
        if self.impact_fun not in {"Exp", "Pow", None}:
            raise ValueError(f"Invalid impact function {self.impact_fun}. Please choose either {"Exp"}, {"Pow"} or {None}")
        if self.mark_dist not in {None, "GPD", "SE GPD"}:
            raise ValueError(f"Invalid mark distribution {self.impact_fun}. Please choose either {None}, {"GPD"} or {"SE GPD"}")


    # Checks if there is any missing data
    def __check_for_missing_data(self):

        required_data = {"Arrival Times": self.arrival_times,
                         "Marks": self.marks,
                         "T": self.T,
                         "Threshold": self.threshold}
        missing_data = [name for name, val in required_data.items() if val is None]

        if missing_data:
            raise Exception(f"The following values were not specified: {", ".join(missing_data)}")
        

    # Checks whether the dimensions of the arrival times and marks are compatible
    def __check_dimensions(self):

        if self.arrival_times.shape != self.marks.shape:
            raise Exception("Mismatched dimensions of arrival times and marks")
        
        if self.arrival_times.ndim > 2 or self.marks.ndim > 2:
            raise Exception("The dimensions of arrival_times or marks are invalid")
    

    # Makes sure that the passed arrival times are in temporal ascending order
    def __check_temporal_order(self):

        if not np.array_equal(self.arrival_times, np.sort(self.arrival_times)):
            raise Exception("Arrival times are not in ascending order")
    

    # Specifies the number of required parameters for the specified model
    def __set_required_params_num(self):
        self.ground_process_params_num = 3

        if self.SE_fun == "Pow":
            self.ground_process_params_num += 1
        if self.impact_fun is not None:
            self.ground_process_params_num += 1
        if self.mark_dist == "GPD":
            self.mark_params_num = 2
        else:
            self.mark_params_num = 3
        self.required_params_num = self.ground_process_params_num + self.mark_params_num

    def __LL_MIN(self):
        return -np.log(sys.float_info.max) * len(self.arrival_times) * 100
    
    def __NLL_MAX(self):
        return -self.__LL_MIN()



    # Computes the values of the impact function c(m_i)
    def c(self,
          m_i: float,
          alpha: np.ndarray):
        if self.impact_fun == "Exp":
            return np.exp(alpha * (m_i - self.threshold))
        elif self.impact_fun == "Pow":
            return (m_i/self.threshold) ** alpha


    # Computes the values of the triggering function g(s, m_i)
    def g(self,
          s: float,
          m_i: float,
          params: np.ndarray):
        if self.SE_fun == "Exp" and self.impact_fun is None:
            K_0, beta = params
            return K_0 * np.exp(-beta * s)
        elif self.SE_fun == "Exp" and self.impact_fun is not None:
            K_0, beta, alpha = params
            return K_0 * np.exp(-beta * s) * self.c(m_i=m_i,
                                                    alpha=alpha)
        elif self.SE_fun == "Pow" and self.impact_fun is None:
            K_0, gamma, omega = params
            return K_0/(gamma * s + 1) ** (1 + omega)
        elif self.SE_fun == "Pow" and self.impact_fun is not None:
            K_0, gamma, omega, alpha = params
            return K_0 / (gamma * s + 1) ** (1 + omega) * self.c(m_i=m_i,
                                                                 alpha=alpha)

    #********************************************************************************************************************************************************************************************************
    #********************************************************************************************************************************************************************************************************

    # Works, provides the same values as self.cond_intensity up to some rounding errors
    def log_cond_intensity(self,
                           params: np.ndarray,
                           times: np.ndarray | float,
                           with_log_mu: bool = True,
                           reparam: bool = False) -> np.ndarray:

        if isinstance(times, float):
            times = np.array([times])
        res = np.zeros_like(times).astype(dtype=float)
        # Replace .astype(dtype=float)

        indicies = np.searchsorted(self.arrival_times, times)
        arrival_times_before_times = [self.arrival_times[:idx] for idx in indicies]
        marks_before_times = [self.marks[:idx] for idx in indicies]

        if reparam:
            log_mu, log_K_0 = params[:2]
        else:
            mu, K_0 = params[:2]
            if mu == 0.0 or K_0 == 0:
                print(f"mu = {mu}")
                print(f"K_0 = {K_0}")
                raise Exception("Error")
            
            log_mu, log_K_0 = np.log(mu), np.log(K_0)

        for i, t in enumerate(times):
            if self.SE_fun == "Exp":
                if reparam:
                    beta = np.exp(params[2])
                else:
                    beta = params[2]
                if self.impact_fun is None:
                    w_i = log_K_0 - beta * (t - arrival_times_before_times[i])
                else:
                    if reparam:
                        alpha = np.exp(params[3])
                    else:
                        alpha = params[3]
                    if self.impact_fun == "Exp":
                        w_i = log_K_0 - beta * (t - arrival_times_before_times[i]) + alpha * (marks_before_times[i] - self.threshold)
                    else:
                        w_i = log_K_0 - beta * (t - arrival_times_before_times[i]) + alpha * (np.log(marks_before_times[i]) - np.log(self.threshold))
            else:
                if reparam:
                    gamma, omega = np.exp(params[2:4])
                else:
                    gamma, omega = params[2:4]
                if self.impact_fun is None:
                    w_i = log_K_0 - (1 + omega) * np.log(gamma * (t - arrival_times_before_times[i]) + 1)
                else:
                    if reparam:
                        alpha = np.exp(params[4])
                    else:
                        alpha = params[4]
                    if self.impact_fun == "Exp":
                        w_i = log_K_0 - (1 + omega) * np.log(gamma * (t - arrival_times_before_times[i]) + 1) + alpha * (marks_before_times[i] - self.threshold)
                    else:
                        w_i = log_K_0 - (1 + omega) * np.log(gamma * (t - arrival_times_before_times[i]) + 1) + alpha * (np.log(marks_before_times[i]) - np.log(self.threshold))
            if with_log_mu:
                res[i] = logsumexp(np.concatenate([[log_mu],
                                                   w_i]))
            else:
                res[i] = logsumexp(w_i)
        return res


    def cond_intensity(self,
                       params: np.ndarray,
                       times: np.ndarray,
                       reparam: bool = False) -> np.ndarray:
        return np.exp(self.log_cond_intensity(params=params,
                                              times=times,
                                              reparam=reparam))
    

    def ll_gp(self,
              params: np.ndarray,
              reparam: bool = False) -> float:
            return self.log_cond_intensity(params=params,
                                           times=self.arrival_times,
                                           reparam=reparam).sum() - self.compensator(starts=0.0,
                                                                                     ends=self.T,
                                                                                     params=params,
                                                                                     reparam=reparam)


    def ll_marks(self,
                 mark_params: np.ndarray,
                 cond_intensity_params: np.ndarray | None = None,
                 reparam: bool = False,
                 xi_pos: bool = False) -> float:
        
        ll_marks = 0.0

        if self.mark_dist == "GPD":
            if reparam and xi_pos == False:
                xi, log_sigma = mark_params
                sigma = np.exp(log_sigma)
            elif reparam and xi_pos:
                log_xi, log_sigma = mark_params
                xi = np.exp(log_xi)
            else:
                xi, sigma = mark_params
                log_sigma = np.log(sigma)
            
            ll_marks += -log_sigma * len(self.arrival_times)
            
            if not (reparam and xi_pos) and xi < 0:
                valid_mask = np.log(-xi) + np.log(self.marks - self.threshold) < log_sigma
                invalid_mask = ~valid_mask
                num_invalid_marks = invalid_mask.sum()
                ll_marks -= num_invalid_marks * np.log(sys.float_info.max) * 100
                valid_marks = self.marks[valid_mask]
                
                ll_marks += (-(xi + 1)/xi) * np.log1p(xi * (valid_marks - self.threshold)/sigma).sum()
            else:
                if not (reparam and xi_pos):
                    log_xi = np.log(xi)
                ll_marks += (-(xi + 1)/xi) * np.logaddexp([0],
                                                          log_xi + np.log(self.marks - self.threshold) - log_sigma).sum()
        
        else:
            if cond_intensity_params is None:
                raise ValueError(f"For the specified model a numpy array must be passed for cond_intensity_params.")
            if reparam and xi_pos == False:
                xi, log_phi, log_eta = mark_params
            elif reparam and xi_pos:
                log_xi, log_phi, log_eta = mark_params
                xi = np.exp(log_xi)
            else:
                xi, phi, eta = mark_params
                log_phi, log_eta = np.log(phi), np.log(eta)

            log_cond_int_minus_mu = self.log_cond_intensity(params=cond_intensity_params,
                                                            times=self.arrival_times,
                                                            with_log_mu=False,
                                                            reparam=reparam)

            log_sigma_ts = np.logaddexp(log_phi,
                                        log_eta + log_cond_int_minus_mu)
            ll_marks += -log_sigma_ts.sum()
            
            if not(reparam and xi_pos) and xi < 0:
                valid_mask = np.log(-xi) + np.log(self.marks - self.threshold) < log_sigma_ts
                invalid_mask = ~valid_mask
                num_invalid_marks = invalid_mask.sum()
                ll_marks -= num_invalid_marks * np.log(sys.float_info.max) * 100
                valid_marks = self.marks[valid_mask]
                valid_log_sigma_ts = log_sigma_ts[valid_mask]


                ll_marks += (-(xi + 1)/xi) * np.log1p(xi * (valid_marks - self.threshold)/np.exp(valid_log_sigma_ts)).sum()
            else:
                if not (reparam and xi_pos):
                    log_xi = np.log(xi)
                ll_marks += (-(xi + 1)/xi) * np.logaddexp([0],
                                                          log_xi + np.log(self.marks - self.threshold) - log_sigma_ts).sum()
        return ll_marks
    

    def nll(self,
            params: np.ndarray,
            reparam: bool = False,
            xi_pos: bool = False) -> float:
        
        if params.size != self.required_params_num:
            raise Exception(f"{self.required_params_num} parameters are required, {params.size} were passed")
        
        cond_intensity_params = params[:self.ground_process_params_num]
        mark_params = params[-self.mark_params_num:]
        return -self.ll_gp(params=cond_intensity_params,
                           reparam=reparam) - self.ll_marks(mark_params=mark_params,
                                                            cond_intensity_params=cond_intensity_params,
                                                            reparam=reparam,
                                                            xi_pos=xi_pos)
    
    def compensator_old(self,
                    start: float,
                    end: float,
                    params: np.ndarray,
                    reparam: bool = False) -> float:
        if reparam:
            log_mu, log_K_0 = params[0], params[1]
        else:
            mu, K_0 = params[0], params[1]
            log_mu, log_K_0 = np.log(mu), np.log(K_0)

        arrival_times_before_end = self.arrival_times[self.arrival_times < end]
        marks_before_end = self.marks[self.arrival_times < end]

        if self.SE_fun == "Exp":
            if reparam:
                log_beta = params[2]
                beta = np.exp(params[2])
            else:
                beta = params[2]
                log_beta = np.log(beta)
            u_i = -beta * (end - arrival_times_before_end)
            v_i = -beta * (np.maximum(start, arrival_times_before_end) - arrival_times_before_end)
            if self.impact_fun is not None:
                if reparam:
                    alpha = np.exp(params[3])
                else:
                    alpha = params[3]
                if self.impact_fun == "Exp":
                    u_i += alpha * (marks_before_end - self.threshold)
                    v_i += alpha * (marks_before_end - self.threshold)
                else:
                    u_i += alpha * (np.log(marks_before_end) - np.log(self.threshold))
                    v_i += alpha * (np.log(marks_before_end) - np.log(self.threshold))
            return np.exp(logsumexp(a=[log_mu + np.log(end - start),
                                       log_K_0 - log_beta + logsumexp(u_i),
                                       log_K_0 - log_beta + logsumexp(v_i)],
                                    b=[1, -1, 1]))
        else:
            if reparam:
                log_gamma, log_omega = params[2:4]
                gamma, omega = np.exp(log_gamma), np.exp(log_omega)
            else:
                gamma, omega = params[2:4]
                log_gamma, log_omega = np.log(gamma), np.log(omega)
            u_i = -omega * np.log(gamma * (end - arrival_times_before_end) + 1)
            v_i = -omega * np.log(gamma * (np.maximum(start, arrival_times_before_end) - arrival_times_before_end) + 1)
            if self.impact_fun is not None:
                if reparam:
                    alpha = np.exp(params[4])
                else:
                    alpha = params[4]
                if self.impact_fun == "Exp":
                    u_i += alpha * (marks_before_end - self.threshold)
                    v_i += alpha * (marks_before_end - self.threshold)
                else:
                    u_i += alpha * (np.log(marks_before_end) - np.log(self.threshold))
                    v_i += alpha * (np.log(marks_before_end) - np.log(self.threshold))
            return np.exp(logsumexp(a=[log_mu + np.log(end - start),
                                       log_K_0 - log_gamma - log_omega + logsumexp(u_i),
                                       log_K_0 - log_gamma - log_omega + logsumexp(v_i)],
                                    b=[1, -1, 1]))
        
    def compensator(self,
                    starts: np.ndarray | float,
                    ends: np.ndarray | float,
                    params: np.ndarray,
                    reparam: bool = False) -> float:
        
        starts = np.asarray(a=starts, 
                            dtype=float).reshape(-1)
        ends = np.asarray(a=ends, 
                          dtype=float).reshape(-1)

        if starts.size != ends.size:
            raise Exception(f"Error, {"starts"} and {"ends"} must contain the same number of elements.")
        
        if reparam:
            log_mu, log_K_0 = params[0], params[1]
        else:
            mu, K_0 = params[0], params[1]
            log_mu, log_K_0 = np.log(mu), np.log(K_0)

        before_end_masks = self.arrival_times < ends[:, np.newaxis]

        ends_indicies, arrival_times_indicies = np.where(before_end_masks)

        flat_diffs_ends = ends[ends_indicies] - self.arrival_times[arrival_times_indicies]
        flat_diffs_starts = np.maximum(starts[ends_indicies], self.arrival_times[arrival_times_indicies]) - self.arrival_times[arrival_times_indicies]

        group_sizes = before_end_masks.sum(axis=1)
        split_indicies = np.cumsum(a=group_sizes)[:-1]


        diffs_ends = np.split(ary=flat_diffs_ends,
                              indices_or_sections=split_indicies)
        

        diffs_starts = np.split(ary=flat_diffs_starts,
                                indices_or_sections=split_indicies)
        
        marks_grouped = np.split(ary=self.marks[arrival_times_indicies],
                                 indices_or_sections=split_indicies)
        res = np.zeros(shape=(len(starts), ),
                       dtype=float)
        
        if self.SE_fun == "Exp":
            if reparam:
                log_beta = params[2]
                beta = np.exp(params[2])
            else:
                beta = params[2]
                log_beta = np.log(beta)
            u_i_s = [-beta * diff_ends for diff_ends in diffs_ends]
            v_i_s = [-beta * diff_starts for diff_starts in diffs_starts]

            if self.impact_fun is not None:
                if reparam:
                    alpha = np.exp(params[3])
                else:
                    alpha = params[3]
        else:
            if reparam:
                log_gamma, log_omega = params[2:4]
                gamma, omega = np.exp(log_gamma), np.exp(log_omega)
            else:
                gamma, omega = params[2:4]
                log_gamma, log_omega = np.log(gamma), np.log(omega)
            u_i_s = [-omega * np.log(gamma * diff_ends + 1) for diff_ends in diffs_ends]
            v_i_s = [-omega * np.log(gamma * diff_starts + 1) for diff_starts in diffs_starts]

            if self.impact_fun is not None:
                if reparam:
                    alpha = np.exp(params[4])
                else:
                    alpha = params[4]

        if self.impact_fun is not None:
            for i in range(len(u_i_s)):
                if self.impact_fun == "Exp":
                    u_i_s[i] += alpha * (marks_grouped[i] - self.threshold)
                    v_i_s[i] += alpha * (marks_grouped[i] - self.threshold)
                else:
                    u_i_s[i] += alpha * (np.log(marks_grouped[i]) - np.log(self.threshold))
                    v_i_s[i] += alpha * (np.log(marks_grouped[i]) - np.log(self.threshold))

        for i in range(len(res)):
                if np.array_equal(a1=ends[i],
                                  a2=starts[i]):
                    res[i] = 0.0
                else:
                    if self.SE_fun == "Exp":
                        res[i] = np.exp(logsumexp(a=[log_mu + np.log(ends[i] - starts[i]),
                                                     log_K_0 - log_beta + logsumexp(u_i_s[i]),
                                                     log_K_0 - log_beta + logsumexp(v_i_s[i])],
                                                  b=[1, -1, 1]))
                    else:
                        res[i] = np.exp(logsumexp(a=[log_mu + np.log(ends[i] - starts[i]),
                                                     log_K_0 - log_gamma - log_omega + logsumexp(u_i_s[i]),
                                                     log_K_0 - log_gamma -log_omega + logsumexp(v_i_s[i])],
                                                  b=[1, -1, 1]))
        return res
        
    #********************************************************************************************************************************************************************************************************
    #********************************************************************************************************************************************************************************************************
    

    # Returns a list of bounds for the model parameters
    def set_bounds(self,
                   zero_lb: float = 1e-05,
                   ub: float = np.inf,
                   ub_power: float = np.inf,
                   xi_pos: bool = False):
        
        bounds = 3 * [(zero_lb, ub)]

        if self.SE_fun == "Pow":
            bounds += [(zero_lb, ub_power)]
        if self.impact_fun is not None:
            bounds += [(zero_lb, ub_power)]
        if self.mark_dist == "GPD":
            if xi_pos:
                bounds += [(zero_lb, ub), (zero_lb, ub)]
            else:
                bounds += [(-ub, ub), (zero_lb, ub)]
        else:
            if xi_pos:
                bounds += [(zero_lb, ub), (zero_lb, ub), (zero_lb, ub)]
            else:
                bounds += [(-ub, ub), (zero_lb, ub), (zero_lb, ub)]
        return bounds
        


    # Fits the ground process to the data via MLE using scipy's optimize function
    def fit_ground_process(self,
                           method: str | None = None,
                           x0: np.ndarray | None = None,
                           reparam: bool = False,
                           bounds: list | None = None):
        if bounds is None:
            bounds = self.set_bounds()[:self.ground_process_params_num]
        if x0 is None:
            x0 = np.ones(shape=(self.ground_process_params_num, ))

        nll_ground_process = lambda params: -self.ll_gp(params=params, 
                                                        reparam=reparam)
        if reparam:
            x0 = np.log(x0)
            res = minimize(fun=nll_ground_process,
                           x0=x0,
                           method=method)
            res.x = np.exp(res.x)
        else:
            res = minimize(fun=nll_ground_process,
                           x0=x0,
                           bounds=bounds,
                           method=method)
        return res
    
    # Fits the mark distribution to the data via MLE using scipy's optimize function
    def fit_mark_dist(self,
                      method: str | None = None,
                      x0: np.ndarray | None = None,
                      reparam: bool = False,
                      bounds: list | None = None,
                      xi_pos: bool = False):
        if bounds is None:
            bounds = self.set_bounds(xi_pos=xi_pos)[-self.mark_params_num:]
        if x0 is None:
            x0 = np.ones(shape=(self.mark_params_num, ))
    
        nll_marks = lambda mark_params: -self.ll_marks(mark_params=mark_params,
                                                       reparam=reparam,
                                                       xi_pos=xi_pos)
        if reparam and xi_pos == False:
            x0 = np.concatenate([[x0[0]],
                                 np.log(x0[1:])])
            res = minimize(fun=nll_marks,
                           x0=x0,
                           method=method)
            res.x = np.concatenate([[res.x[0]],
                                    np.exp(res.x[1:])])
        elif reparam and xi_pos:
            x0 = np.log(x0)
            res = minimize(fun=nll_marks,
                           x0=x0,
                           method=method)
            res.x = np.exp(res.x)
        else:                          
            res = minimize(fun=nll_marks,
                           x0=x0,
                           bounds=bounds,
                           method=method)
        return res
    
    # Fits the ground process and the mark distribution to the data via MLE using scipy's optimize function
    def fit_gp_mark_dist(self,
                         method: str | None = None,
                         x0: np.ndarray | None = None,
                         reparam: bool = False,
                         bounds: list | None = None,
                         xi_pos: bool = False):
        if bounds is None:
            bounds = self.set_bounds(xi_pos=xi_pos)
        if x0 is None:
            x0 = np.ones(shape=(self.required_params_num, ))

        nll_ = lambda params: self.nll(params=params,
                                       reparam=reparam,
                                       xi_pos=xi_pos)
        
        if reparam and xi_pos:
            x0 = np.log(x0)
            res = minimize(fun=nll_,
                           x0=x0,
                           method=method)
            res.x = np.exp(res.x)
        elif reparam and xi_pos == False:
            x0 = x0 = np.concatenate([np.log(x0[:self.ground_process_params_num]),
                                      [x0[-self.mark_params_num]],
                                      np.log(x0[-(self.mark_params_num - 1):])])
            res = minimize(fun=nll_,
                           x0=x0,
                           method=method)
            
            res.x = np.concatenate([np.exp(res.x[:self.ground_process_params_num]),
                                    [res.x[-self.mark_params_num]],
                                    np.exp(res.x[-(self.mark_params_num - 1):])])
        else:
            res = minimize(fun=nll_,
                           x0=x0,
                           bounds=bounds,
                           method=method)
        return res
        

    # Fits the specified model to the data using the functions self.fit_ground_process, self.fit_mark_dist and self.fit_gp_mark_dist
    def fit(self, 
            method: str | None = None,
            x0: np.ndarray | None = None,
            n_optim_restarts: int = 0,
            log_scale_samples: bool = False,
            reparam: bool = False,
            bounds: list | None = None,
            arrival_times: np.ndarray | None = None, 
            marks: np.ndarray | None = None,
            T: float | None = None,
            threshold: float | None = None,
            xi_pos: bool = False,
            seed: int | None = None):
        
        if arrival_times is not None:
            self.arrival_times = arrival_times
        if marks is not None:
            self.marks = marks
        if T is not None:
            self.T = T
        if threshold is not None:
            self.threshold = threshold
        if seed is not None:
            self.seed = seed
        if x0 is None:
            x0 = np.ones(shape=(self.required_params_num,))
        else:
            if self.required_params_num != len(x0):
                raise Exception(f"{self.required_params_num} initial values are required for x0, {len(x0)} were passed")
            
        rng = np.random.default_rng(self.seed)

        self.__check_for_missing_data()
        self.__check_dimensions()
        self.__check_temporal_order()

        if self.arrival_times.ndim == 2:
            self.arrival_times = self.arrival_times.flatten()
        if self.marks.ndim == 2:
            self.marks = self.marks.flatten()


        if bounds is None and reparam == True and n_optim_restarts == 0:
            bounds = None
        else:
            if bounds is None:
                bounds = self.set_bounds(zero_lb=1e-05,
                                         ub=10.0,
                                         ub_power=10.0,
                                         xi_pos=xi_pos)
            else:
                bounds = bounds

        x0s = [x0]

        if n_optim_restarts > 0:
            if np.any(np.isinf(bounds)):
                raise ValueError("Invalid bounds passed.\nFor multiple restart optimizations, the bounds must be finite.")
            if log_scale_samples:
                log_bounds = []
                log_trans = []
                for min_i, max_i in bounds:
                    if min_i > 0:
                        log_bounds.append((np.log(min_i).item(), np.log(max_i).item()))
                        log_trans.append(True)
                    else:
                        log_bounds.append((min_i, max_i))
                        log_trans.append(False)
                log_trans = np.array(log_trans)
                log_mins, log_maxs = zip(*log_bounds)
            else:
                mins, maxs = zip(*bounds)
            for _ in range(n_optim_restarts):
                if log_scale_samples:
                    x0_sample = rng.uniform(low=log_mins,
                                            high=log_maxs)
                    x0_sample = np.where(log_trans, np.exp(x0_sample), x0_sample)
                else:
                    x0_sample = rng.uniform(low=mins,
                                            high=maxs)
                x0s.append(x0_sample)


        current_min = OptimizeResult(fun=np.inf)
        current_min_x0 = x0s[0]
        if self.mark_dist == "GPD":
            current_gp_min = OptimizeResult(fun=np.inf)
            current_mark_dist_min = OptimizeResult(fun=np.inf)
            current_gp_min_x0 = x0s[0][:self.ground_process_params_num]
            current_mark_min_x0 = x0s[0][-self.mark_params_num:]
        for x0_i in x0s:
            if self.mark_dist == "GPD":
                if bounds is None:
                    res_gp = self.fit_ground_process(method=method,
                                                     x0=x0_i[:self.ground_process_params_num],
                                                     reparam=reparam)
                    res_mark_dist = self.fit_mark_dist(method=method,
                                                       x0=x0_i[-self.mark_params_num:],
                                                       reparam=reparam,
                                                       xi_pos=xi_pos)
                else:
                    if reparam == True:
                        res_gp = self.fit_ground_process(method=method,
                                                         x0=x0_i[:self.ground_process_params_num],
                                                         reparam=reparam)
                        res_mark_dist = self.fit_mark_dist(method=method,
                                                           x0=x0_i[-self.mark_params_num:],
                                                           reparam=reparam,
                                                           xi_pos=xi_pos)
                    else:
                        res_gp = self.fit_ground_process(method=method,
                                                         x0=x0_i[:self.ground_process_params_num],
                                                         reparam=reparam,
                                                         bounds=bounds[:self.ground_process_params_num])
                        res_mark_dist = self.fit_mark_dist(method=method,
                                                           x0=x0_i[-self.mark_params_num:],
                                                           reparam=reparam,
                                                           bounds=bounds[-self.mark_params_num:],
                                                           xi_pos=xi_pos)
                if res_gp.fun < current_gp_min.fun:
                    current_gp_min = res_gp
                    current_gp_min_x0 = x0_i[:self.ground_process_params_num]
                if res_mark_dist.fun < current_mark_dist_min.fun:
                    current_mark_dist_min = res_mark_dist
                    current_mark_min_x0 = x0_i[-self.mark_params_num:]
            else:
                if bounds is None:
                    res = self.fit_gp_mark_dist(method=method,
                                                x0=x0_i,
                                                reparam=reparam,
                                                xi_pos=xi_pos)
                else:
                    if reparam == True:
                        res = self.fit_gp_mark_dist(method=method,
                                                    x0=x0_i,
                                                    reparam=reparam,
                                                    xi_pos=xi_pos)
                    else:
                        res = self.fit_gp_mark_dist(method=method,
                                                    x0=x0_i,
                                                    reparam=reparam,
                                                    bounds=bounds,
                                                    xi_pos=xi_pos)
                if res.fun < current_min.fun:
                    current_min = res
                    current_min_x0 = x0_i
        if self.mark_dist == "GPD":
            self.x0_best = np.concatenate([current_gp_min_x0,
                                           current_mark_min_x0])
        else:
            self.x0_best = current_min_x0
        if self.mark_dist == "GPD":
            res = OptimizeResult(x=np.concatenate([current_gp_min.x, current_mark_dist_min.x]),
                                 fun=current_gp_min.fun + current_mark_dist_min.fun,
                                 success=current_gp_min.success and current_mark_dist_min.success,
                                 message=f"Ground process: {current_gp_min.message}, Mark distribution: {current_mark_dist_min.message}",
                                 nit=current_gp_min.nit + current_mark_dist_min.nit)
            return res
        else:
            return current_min
        
       
    def predict_old(self,
                start: float,
                end: float,
                params: np.ndarray) -> float:
        return 1 - np.exp(-self.compensator_old(start=start,
                                                end=end,
                                                params=params))
    
    def predict(self,
                starts: np.ndarray | float,
                ends: np.ndarray | float,
                params: np.ndarray) -> np.ndarray:
        
        starts = np.asarray(a=starts, 
                            dtype=float).reshape(-1)
        ends = np.asarray(a=ends, 
                          dtype=float).reshape(-1)
        
        if starts.size != ends.size:
            raise Exception(f"Error, {"starts"} and {"ends"} must contain the same number of elements.")
        
        return 1 - np.exp(-self.compensator(starts=starts,
                                            ends=ends,
                                            params=params))
    
    def EWS(self,
            starts: np.ndarray | float,
            days: float,
            params: np.ndarray) -> np.ndarray:
        prob_event_over_next_days = self.predict(starts=starts,
                                                 ends=starts + days,
                                                 params=params)
        if prob_event_over_next_days > 0.5:
            return 1.0
        else:
            return 0.0

    def updata_arrival_times_marks(self,
                                   new_arrival_times: np.ndarray | float,
                                   new_marks: np.ndarray | float,
                                   new_T: float):
        
        new_arrival_times = np.asarray(a=new_arrival_times, 
                                       dtype=float).reshape(-1)
        new_marks = np.asarray(a=new_marks, 
                               dtype=float).reshape(-1)
        
        if new_arrival_times.size != new_marks.size:
            raise Exception(f"Error, {"new_arrival_times"} and {"new_marks"} must contain the same number of elements.")
        
        if new_arrival_times.size != 0:
            if np.max(new_arrival_times) > new_T:
                raise Exception(f"Error, {"new_T"} must be greater than all the values in {"new_arrival_times"}")
        
        self.T = new_T
        self.arrival_times = np.concatenate([self.arrival_times,
                                             new_arrival_times])
        self.marks = np.concatenate([self.marks,
                                     new_marks])
        
    def rolling_forecast(self,
                         start: float,
                         days: int,
                         params: np.ndarray,
                         test_arrival_times: np.ndarray,
                         test_marks: np.ndarray,
                         test_T: float):

        times_between_start_test_T = np.arange(start=start,
                                               stop=test_T + 1,
                                               dtype=float)
        
        times_chunks = [times_between_start_test_T[i : i + days] for i in range(0, len(times_between_start_test_T), days)]

        arrival_times_chunks = [test_arrival_times[np.isin(element=test_arrival_times,
                                                           test_elements=times_chunk)] for times_chunk in times_chunks]
        marks_chunks = [test_marks[np.isin(element=test_arrival_times,
                                           test_elements=times_chunk)] for times_chunk in times_chunks]
        
        crash_in_chunks = np.array([np.isin(element=test_arrival_times,
                                            test_elements=times_chunk).any().item() for times_chunk in times_chunks])

        starts = np.arange(start,
                           stop=test_T + 1,
                           step=days)
        
        ends = np.minimum(starts + days, test_T)

        starts_ends_differ_mask = starts != ends

        preds = np.zeros(shape=(int(starts_ends_differ_mask.sum()), ))

        crash_in_chunks = crash_in_chunks[:int(starts_ends_differ_mask.sum())]

        for i, (start_i, end_i) in enumerate(zip(starts, ends)):
            if start_i != end_i:
                preds[i] = self.predict(starts=start_i,
                                        ends=end_i,
                                        params=params).item()
                self.updata_arrival_times_marks(new_arrival_times=arrival_times_chunks[i],
                                                new_marks=marks_chunks[i],
                                                new_T=times_chunks[i][-1])
        return preds, crash_in_chunks
        

                
        
    

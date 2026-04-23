import numpy as np
import pandas as pd
from scipy.optimize import minimize 

class SEPOT():
    # arrival_times should be a (n, ) dimensional array containing the arrival times of the events
    # marks should be a (n, ) dimensional array containing the marks of the point process
    # threshold specifies the threshold value
    # SE_fun specifies what self-exciting function should be used
    # impact_fun specifies what impact function should be used
    # mark_dist specifies the what distribution should be used for the marks
    # random_state specifies the random seed
    def __init__(self,
                 arrival_times: np.ndarray | None = None,
                 marks: np.ndarray | None = None,
                 threshold: float | None = None,
                 SE_fun: str = "Pow",
                 impact_fun: str = "Pow",
                 mark_dist: str = "SE GPD",
                 random_state: int | None = None):
        self.arrival_times = arrival_times
        self.marks = marks
        self.threshold = threshold
        self.SE_fun = SE_fun
        self.impact_fun = impact_fun    
        self.mark_dist = mark_dist
        self.random_state = random_state

    def cond_intensity(self,
                       params: np.ndarray,
                       t: float,
                       arrival_times: np.ndarray | None = None,
                       marks: np.ndarray | None = None,
                       threshold: float | None = None) -> float:
        self.arrival_times, self.marks, self.threshold = arrival_times, marks, threshold

        if self.arrival_times == None or self.marks == None or self.threshold:
            raise Exception("No or insufficient data has been passed")
        if self.arrival_times.shape != self.marks.shape:
            raise Exception("Mismatched dimension of arrival times and marks")
        if np.array_equal(self.arrival_times, np.sort(self.arrival_times)):
            raise Exception("Arrival times are not in ascending order")
        if self.threshold == None:
            raise Exception("Threshold value has not been specified")
        if self.arrival_times.ndim > 2 or self.marks.ndim > 2:
            raise Exception("The dimensions of arrival_times or marks is invalid")
        
        if self.arrival_times.ndim == 2:
            self.arrival_times = self.arrival_times.flatten()
        if self.marks.ndim == 2:
            self.marks = self.marks.flatten()
        
        arrival_times_before_t = arrival_times[arrival_times < t]
        marks_before_t = marks[arrival_times < t]

        if self.SE_fun == "Pow" and self.impact_fun == "Pow":
            required_params_num = 5
            if params.size != required_params_num:
                raise Exception(f"{required_params_num} parameters are required, {params.size} were passed")
            else:
                mu, K_0, gamma, omega, alpha = params

                if arrival_times_before_t.shape[0] == 0:
                    return mu
                
                def g(K_0, gamma, omega, alpha):
                    return lambda t, t_i, m_i: K_0 / (gamma * (t - t_i) + 1) ** (1 + omega) * (m_i / self.threshold) ** alpha
                
                g = g(K_0, gamma, omega, alpha)
                    
                return mu + g(t, arrival_times_before_t, marks_before_t).sum()  



    def ll(self,
            params: np.ndarray,
            arrival_times: np.ndarray | None = None,
            marks: np.ndarray | None = None,
            threshold: float | None = None):
        # Computes and returns the log-likelihood
        self.arrival_times, self.marks, self.threshold = arrival_times, marks, threshold

        if self.arrival_times == None or self.marks == None or self.threshold == None:
            raise Exception("No or insufficient data has been passed")
        
        ll = 0
        
        if self.SE_fun == "Pow" and self.impact_fun == "Pow" and self.mark_dist == "SE GPD":
            required_params_num = 8
            if params.size != required_params_num:
                raise Exception(f"{required_params_num} parameters are required, {params.size} were passed")
            
            mu, K_0, gamma, omega, alpha, xi, phi, eta = params

            cond_intensity_params = np.array([mu, K_0, gamma, omega, alpha])

            sigma_t = phi

            for t in arrival_times:
                ll += np.log(self.cond_intensity(params=cond_intensity_params,
                                                  t = t, 
                                                  arrival_times=arrival_times, 
                                                  marks=marks))
                sigma_t += eta * (self.cond_intensity(params=cond_intensity_params,
                                                      t = t,
                                                      arrival_times=arrival_times,
                                                      marks=marks) - mu)
            ll -= np.log(sigma_t)

            mark_dist_part = lambda m_i: (1 + 1/xi) * np.log(1 + xi * (m_i - self.threshold)/sigma_t)

            ll += mark_dist_part(self.mark_dist).sum()




    def fit(self, arrival_times, marks):
        self.arrival_times, self.marks = arrival_times, marks

        if self.SE_fun == None:
            self.SE_fun = "Pow"
        if self.impact_fun == None:
            self.impact_fun = "Pow"

        # Compute likelihood






    
    
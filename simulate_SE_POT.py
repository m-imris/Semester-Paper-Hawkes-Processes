import numpy as np
from scipy.stats import expon, genpareto, uniform

class simulate_SE_POT():

    def __init__(self,
                 T: float,
                 threshold: float,
                 params: np.ndarray,
                 SE_fun: str = "Exp",
                 impact_fun: str | None = None,
                 mark_dist: str = "GPD",
                 discrete_time: bool = True,
                 seed: int | None = None):
        
        self.T = T
        self.threshold = threshold
        self.params = params
        self.SE_fun = SE_fun
        self.impact_fun = impact_fun
        self.mark_dist = mark_dist
        self.discrete_time = discrete_time
        self.rng = np.random.default_rng(seed=seed)
        self.required_params_num = 0

        self.__set_required_params_num()
        
        if params.size != self.required_params_num:
            raise Exception(f"{self.required_params_num} parameters are required, {params.size} were passed")
    

    # Specifies the number of required parameters for the specified model
    def __set_required_params_num(self):
        if self.SE_fun == "Exp":
            if self.impact_fun is None:
                if self.mark_dist == "GPD":
                    #mu, K_0, beta, xi, sigma
                    self.required_params_num = 5
                else:
                    #mu, K_0, beta, xi, phi, eta
                    self.required_params_num = 6
            else:
                if self.mark_dist == "GPD":
                    #mu, K_0, beta, alpha, xi, sigma
                    self.required_params_num = 6
                else:
                    #mu, K_0, beta, alpha, xi, phi, eta
                    self.required_params_num = 7
        
        else:
            if self.impact_fun is None:
                if self.mark_dist == "GPD":
                    #mu, K_0, gamma, omega, xi, sigma
                    self.required_params_num = 6
                else:
                    #mu, K_0, gamma, omega, xi, phi, eta
                    self.required_params_num = 7
            else:
                if self.mark_dist == "GPD":
                    #mu, K_0, gamma, omega, alpha, xi, sigma
                    self.required_params_num = 7
                else:
                    #mu, K_0, gamma, omega, alpha, xi, phi, eta
                    self.required_params_num = 8

    
    # Computes the values of the impact function c(m_i)
    def c(self,
          m_i: float):
        
        if self.SE_fun == "Exp":
            alpha = self.params[3]
        else:
            alpha = self.params[4]
        
        if self.impact_fun == "Exp":
            return np.exp(alpha * (m_i - self.threshold))
        else:
            return (m_i / self.threshold) ** alpha

    
    # Computes the values of the triggering function g(s, m_i)
    def g(self,
          s: float,
          m_i: float):
        
        if self.SE_fun == "Exp":
            K_0, beta = self.params[1:3]

            if self.impact_fun is None:
                return K_0 * np.exp(-beta * s)
            else:
                return K_0 * np.exp(-beta * s) * self.c(m_i=m_i)
        
        else:
            K_0, gamma, omega = self.params[1:4]
            
            if self.impact_fun is None:
                return K_0/(gamma * s + 1) ** (1 + omega)
            else:
                return K_0/(gamma * s + 1) ** (1 + omega) * self.c(m_i=m_i)


    # Computes the conditional intensity at the time points times given arrival_times and marks
    def cond_intensity(self,
                       times: np.ndarray | float,
                       arrival_times: np.ndarray,
                       marks: np.ndarray) -> np.ndarray:

        if isinstance(times, float):
            times = np.array([times])
        res = np.zeros_like(times)

        indicies = np.searchsorted(arrival_times, times)
        arrival_times_before_times = [arrival_times[:idx] for idx in indicies]
        marks_before_times = [marks[:idx] for idx in indicies]

        mu = self.params[0]

        for i, t in enumerate(times):
            if arrival_times_before_times[i].size == 0:
                res[i] = mu
            else:
                res[i] = mu + self.g(t - arrival_times_before_times[i], marks_before_times[i]).sum()
        return res
    

    # Simulates a sample trajectory of the specified marked Hawkes process in discrete time
    def simulate_discrete(self):
        arrival_times = np.array([])
        marks = np.array([])

        mu = self.params[0]

        t_n = expon.rvs(loc=0,
                        scale=mu,
                        random_state=self.rng)
        
        if t_n > self.T:
            return np.array(arrival_times), np.array(marks)
        else:
            t_n = np.ceil(t_n)
            if self.mark_dist == "GPD":
                xi, sigma = self.params[-2:]
                m_n = genpareto.rvs(c=xi,
                                    loc=self.threshold,
                                    scale=sigma,
                                    random_state=self.rng)
            else:
                xi, phi = self.params[-3:-1]
                m_n = genpareto.rvs(c=xi,
                                    loc=self.threshold,
                                    scale=phi,
                                    random_state=self.rng)
            arrival_times = np.append(arrival_times, t_n)
            marks = np.append(marks, m_n)

        while t_n < self.T:
            mu, K_0 = self.params[:2]
            if self.SE_fun == "Exp":
                beta = self.params[2]
                if self.impact_fun is None:
                    summation = ((np.exp(-beta * (t_n - arrival_times)) - np.exp(-beta * (t_n + 1 - arrival_times))) / beta).sum()
                else:
                    summation = (self.c(m_i=marks) * (np.exp(-beta * (t_n - arrival_times)) - np.exp(-beta * (t_n + 1 - arrival_times))) / beta).sum()
            else:
                gamma, omega = self.params[2:4]
                if self.impact_fun is None:
                    summation = (((gamma * (t_n - arrival_times) + 1) ** (-omega) - (gamma * (t_n + 1 - arrival_times) + 1) ** (-omega)) / (omega * gamma)).sum()
                else:
                    summation = (self.c(m_i=marks) * ((gamma * (t_n - arrival_times) + 1) ** (-omega) - (gamma * (t_n + 1 - arrival_times) + 1) ** (-omega)) / (omega * gamma)).sum()
            integral_cond_int = np.exp(- mu - K_0 * summation)
            uniform_rv = uniform.rvs(loc=0,
                                     scale=1,
                                     random_state=self.rng)
            t_n += 1
            if uniform_rv > integral_cond_int:
                if self.mark_dist == "GPD":
                    xi, sigma = self.params[-2:]
                    m_n = genpareto.rvs(c=xi,
                                        loc=self.threshold,
                                        scale=sigma,
                                        random_state=self.rng)
                else:
                    xi, phi, eta = self.params[-3:]
                    sigma_t = phi + eta * (self.cond_intensity(times=t_n,
                                                               arrival_times=arrival_times,
                                                               marks=marks) - mu)
                    m_n = genpareto.rvs(c=xi,
                                        loc=self.threshold,
                                        scale=sigma_t,
                                        random_state=self.rng)
                
                arrival_times = np.append(arrival_times, t_n)
                marks = np.append(marks, m_n)

        return np.array(arrival_times), np.array(marks)
    

    # Simulates a sample trajectory of the specified marked Hawkes process in continuous time
    def simulate_continuous(self):
        pass
    

    # Simulates a sample trajectory of the specified marked Hawkes process
    def simulate(self,
                 discrete_time: bool | None = None):
        if discrete_time is not None:
            self.discrete_time = discrete_time

        if self.discrete_time:
            return self.simulate_discrete()
        else:
            return self.simulate_continuous()
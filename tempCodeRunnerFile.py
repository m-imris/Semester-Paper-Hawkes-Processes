   
    def predict(self,
                start: float,
                end: float,
                params: np.ndarray) -> float:
        return 1 - np.exp(-self.compensator(start=start,
                                            end=end,
                                            params=params))
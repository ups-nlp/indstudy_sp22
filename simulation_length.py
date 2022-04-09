class simulation_length:
    def __init__(self):
        self.sim_length = 10

    def get_length(self):
        return self.sim_length

    def initialize_agent(self, initial_length):
        self.sim_length = initial_length
        return self.sim_length
    
    def adjust_sim_length(self,scale):
        self.sim_length = self.sim_length*scale
        return self.sim_length

    
        



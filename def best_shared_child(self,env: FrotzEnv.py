    def best_shared_child(self,env: FrotzEnv, score_list:list, count_list:list, num_actions:int):
        #take in list of the values received from each action and calculate the score
        action_values = {}*num_actions
        action_counts = {}*num_actions
        max = 0
        max_act = ""
        for act in env.get_valid_actions:
            action_values[act], action_counts[act] = self.calculate_action_values(score_list, count_list, act)
            if action_values[act]/action_counts[act] > max:
                max = action_values[act]/action_counts[act]
                max_act = act
        return max_act

    def calculate_action_values(self, score_list:list, count_list:list, act):
        score = 0
        count = 0
        for i in range(len(score_list)):
            score_dict = score_list[i]
            count_dict = count_list[i]
            score = score + (score_dict.get(act)*count_dict.get(act))
            count = count + count_dict.get(act)
        return score, count
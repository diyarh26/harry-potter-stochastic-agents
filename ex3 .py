import itertools
import copy
from itertools import product
from copy import deepcopy
from functools import lru_cache

ids = ["213932338", "214034621"]

DESTROY_HORCRUX_REWARD = 2
RESET_REWARD = -2
DEATH_EATER_CATCH_REWARD = -1

def gather_actions(actions_per_wizard):
    actions_lists = [actions_per_wizard[wizard] for wizard in sorted(actions_per_wizard)]
    all_combinations = list(itertools.product(*actions_lists)) + ["reset"]
    return all_combinations
class OptimalWizardAgent:
    def __init__(self, initial):
        self.initial_state = initial
        self.map = initial["map"]
        self.wizards = initial['wizards']
        self.horcruxes = initial['horcrux']
        self.death_eaters = initial['death_eaters']
        self.turns_to_go = initial['turns_to_go']
        self.immediate_reward = 0
        self.V = 0
        self.Q = {}
        self.value_iteration()

    def create_tuple_state(self, d):
        new_d = []
        for k, v in d.items():
            if k in ('turns_to_go', 'map', 'optimal'):
                continue
            if isinstance(v, dict):
                p_or_t_info = [(k1, self.create_tuple_state(v1)) if isinstance(v1, dict) else (
                    k1, tuple(v1) if isinstance(v1, list) else v1) for k1, v1 in v.items()]
                new_d.append((k, tuple(p_or_t_info)))
            elif isinstance(v, list):
                converted_list = []
                for item in v:
                    if isinstance(item, dict):
                        converted_list.append(self.create_tuple_state(item))
                    elif isinstance(item, list):
                        converted_list.append(tuple(item))
                    else:
                        converted_list.append(item)
                new_d.append((k, tuple(converted_list)))
            else:
                new_d.append((k, v))

        return tuple(new_d)

    @lru_cache(maxsize=None)
    def nestedTuple_to_nestedDict(self, s, depth=0):
        if isinstance(s, tuple) and depth == 0:
            d = {}
            for key, value in s:
                if key in ['wizards', 'horcrux', 'death_eaters']:
                    nested_dict = {}
                    for nested_key_value in value:
                        nested_key, nested_value = nested_key_value[0], nested_key_value[1]
                        nested_dict[nested_key] = {pv[0]: pv[1] for pv in nested_value}
                    d[key] = nested_dict
                else:
                    d[key] = value
            return d

    def all_states(self, state):
        map = self.map
        num_of_wizards = len(state['wizards'])

        death_eater_all_possible_indices = {}
        for death_eater in state['death_eaters']:
            death_eater_all_possible_indices[death_eater] = []
            for possible_index in range(len(state['death_eaters'][death_eater]['path'])):
                death_eater_all_possible_indices[death_eater].append(possible_index)

        horcrux_all_possible_loc = {}
        for horcrux in state['horcrux']:
            hloc = state['horcrux'][horcrux]['location']
            horcrux_all_possible_loc[horcrux] = []
            if state['horcrux'][horcrux]['prob_change_location'] > 0:
                for possible_loc in state['horcrux'][horcrux]['possible_locations']:
                    horcrux_all_possible_loc[horcrux].append(possible_loc)
            else:
                horcrux_all_possible_loc[horcrux].append(hloc)

        passable_tiles = []
        impassable_tiles = []
        for i in range(len(map)):
            for j in range(len(map[0])):
                if map[i][j] != 'I':
                    passable_tiles.append((i, j))
                else:
                    impassable_tiles.append((i, j))

        keys, values = zip(*death_eater_all_possible_indices.items())
        self.death_eaters_possible_indices = [dict(zip(keys, v)) for v in product(*values)]

        keys, values = zip(*horcrux_all_possible_loc.items())
        self.horcruxes_possible_loc = [dict(zip(keys, v)) for v in product(*values)]

        keys, values = zip(*{wizard: passable_tiles for wizard in state['wizards']}.items())
        self.wizards_possible_loc = [
            dict(zip(keys, v))
            for v in product(passable_tiles, repeat=num_of_wizards)
        ]

        S = []
        cstate = copy.deepcopy(state)
        for i, plocs in enumerate(self.wizards_possible_loc):
            for wizard in cstate['wizards']:
                cstate['wizards'][wizard]['location'] = plocs[wizard]
            for poss_marine in self.death_eaters_possible_indices:
                for marine in cstate['death_eaters']:
                    cstate['death_eaters'][marine]['index'] = poss_marine[marine]
                for t, tlocs in enumerate(self.horcruxes_possible_loc):
                    for treasure in cstate['horcrux']:
                        cstate['horcrux'][treasure]['location'] = tlocs[treasure]
                    S.append(cstate)
                    cstate = copy.deepcopy(cstate)

        return S

    @lru_cache(maxsize=None)
    def all_possible_actions(self, state):
        state = self.nestedTuple_to_nestedDict(state)
        actions_per_wizard = {}
        for wizard_name in state['wizards']:
            (x, y) = state['wizards'][wizard_name]['location']
            possible_actions = []
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < len(self.map) and 0 <= new_y < len(self.map[0]) and self.map[new_x][
                    new_y] != 'I':
                    possible_actions.append(('move', wizard_name, (new_x, new_y)))

            for horcrux_name in state['horcrux']:
                (hx, hy) = state['horcrux'][horcrux_name]['location']
                if abs(hx - x) + abs(hy - y) == 0:  # check if wizard is near the horcrux
                    possible_actions.append(('destroy', wizard_name, horcrux_name))

            # Add wait action
            possible_actions.append(('wait', wizard_name))

            actions_per_wizard[wizard_name] = possible_actions

        return gather_actions(actions_per_wizard)


    @lru_cache(maxsize=None)
    def all_possible_next_states(self, state, action):
        state = self.nestedTuple_to_nestedDict(state)
        RESET_PENALTY = 2
        COLLECT_HORCRUX_REWARD = 2
        DEATH_EATER_COLLISION_PENALTY = 1

        score = 0
        if action == "reset":
            state = self.initial_state
            return [{'state': state, 'prob': 1, 'reward': -RESET_PENALTY}]
        next_state = copy.deepcopy(state)


        for atomic_action in action:
            wizard_name = atomic_action[1]
            if atomic_action[0] == 'move':
                next_state['wizards'][wizard_name]['location'] = atomic_action[2]
            elif atomic_action[0] == 'destroy':
                horcrux_name = atomic_action[2]
                score += COLLECT_HORCRUX_REWARD

            # --- Handle Horcrux Movement (Probabilistic) ---
            horcrux_possible_loc = {}  # store all possible location for each horcrux
            for horcrux_name in next_state['horcrux']:
                horcrux_possible_loc[horcrux_name] = []
                if next_state['horcrux'][horcrux_name]['prob_change_location'] > 0:
                    possible_locations = next_state['horcrux'][horcrux_name]['possible_locations']
                    for loc in possible_locations:
                        horcrux_possible_loc[horcrux_name].append(loc)
                else:
                    horcrux_possible_loc[horcrux_name].append(next_state['horcrux'][horcrux_name]['location'])

            # --- Move Death Eaters (Probabilistic) ---
            death_eater_possible_indices = {}  # to store all possible indices for each death eaters

            for death_eater_name in next_state['death_eaters']:
                current_index = next_state['death_eaters'][death_eater_name]['index']
                path = next_state['death_eaters'][death_eater_name]['path']
                death_eater_possible_indices[death_eater_name] = []
                if len(path) > 1:
                    if current_index == 0:
                        death_eater_possible_indices[death_eater_name].extend([current_index, current_index + 1])
                    elif current_index == len(path) - 1:
                        death_eater_possible_indices[death_eater_name].extend([current_index, current_index - 1])
                    else:
                        death_eater_possible_indices[death_eater_name].extend(
                            [current_index, current_index - 1, current_index + 1])
                else:
                    death_eater_possible_indices[death_eater_name].append(current_index)

        # --- Collision Check (Deterministic) ---
        for wizard_name in next_state['wizards']:
            wizard_location = next_state['wizards'][wizard_name]['location']
            for death_eater_name in next_state['death_eaters']:
                death_eater_location = next_state['death_eaters'][death_eater_name]['path'][
                    next_state['death_eaters'][death_eater_name]['index']]
                if wizard_location == death_eater_location:
                    score -= DEATH_EATER_COLLISION_PENALTY

        # --- Calculate Possible Next States and Probabilities ---
        keys, values = zip(*horcrux_possible_loc.items())
        horcruxes_possible_loc = [dict(zip(keys, v)) for v in product(*values)]
        keys, values = zip(*death_eater_possible_indices.items())
        death_eaters_possible_indices = [dict(zip(keys, v)) for v in product(*values)]

        res = []
        for possible_death_eater_indices in death_eaters_possible_indices:

            death_eater_prob = 1.0
            temp_state_de = deepcopy(next_state)  # New temporary state for each death eater combination

            for de_name in possible_death_eater_indices:
                new_index = possible_death_eater_indices[de_name]
                temp_state_de['death_eaters'][de_name]['index'] = new_index
                num_possible_indices = len(death_eater_possible_indices[de_name])
                death_eater_prob *= (1.0 / num_possible_indices)

            for possible_horcrux_loc in horcruxes_possible_loc:

                temp_state = deepcopy(temp_state_de)  # Create a working copy of the state
                horcrux_prob = 1.0

                for horcrux_name in possible_horcrux_loc:
                    new_loc = possible_horcrux_loc[horcrux_name]
                    temp_state['horcrux'][horcrux_name]['location'] = new_loc
                    if temp_state['horcrux'][horcrux_name]['prob_change_location'] > 0:
                        num_possible_locations = len(temp_state['horcrux'][horcrux_name]['possible_locations'])
                        if new_loc == next_state['horcrux'][horcrux_name]['location']:
                            horcrux_prob *= (1 - next_state['horcrux'][horcrux_name]['prob_change_location']) + ((
                                                                                                                 next_state[
                                                                                                                     'horcrux'][
                                                                                                                     horcrux_name][
                                                                                         'prob_change_location']) / num_possible_locations)
                        else:
                            horcrux_prob *= (next_state['horcrux'][horcrux_name][
                                'prob_change_location']) / num_possible_locations

                # Collision Check (within each state)
                collision_score = 0
                for wizard_name in temp_state['wizards']:
                    wizard_location = temp_state['wizards'][wizard_name]['location']
                    for death_eater_name in temp_state['death_eaters']:
                        death_eater_location = temp_state['death_eaters'][death_eater_name]['path'][
                            temp_state['death_eaters'][death_eater_name]['index']]
                        if wizard_location == death_eater_location:
                            collision_score -= DEATH_EATER_COLLISION_PENALTY
                res.append(
                    {'state': temp_state, 'prob': horcrux_prob * death_eater_prob, 'reward': score + collision_score})

        return res

    @lru_cache(maxsize=None)
    def value_iteration(self):
        realS = self.all_states(self.initial_state)  # list of all states as dictionaries
        s_a_prob = {}  # To store next states and their probabilities for each (s, a) pair
        max_iter = realS[0]['turns_to_go']  # Maximum number of iterations (turns)
        S = [self.create_tuple_state(s) for s in realS]  # list of all states as tuple of tuples
        self.V = {s: 0 for s in S}  # Initialize V(s) to 0 for all states
        self.Q = {(s, i): 0 for s in S for i in range(max_iter + 1)}  # Initialize Q(s, i) to 0

        for i in range(1, max_iter + 1):  # Iterate through time steps (turns)
            newV = {s: 0 for s in S}  # Initialize V^(t)(s) for the current time step

            for h, s in enumerate(S):  # Iterate through all states
                max_val = float('-inf')  # Initialize max value for the current state
                A = self.all_possible_actions(S[h])  # Get all possible actions in the current state
                best_a = None  # Initialize the best action for the current state

                for j, a in enumerate(A):  # Iterate through all actions
                    s_a_prob[(s, a)] = self.all_possible_next_states(S[h], a)  # Get next states, probabilities, rewards
                    val = 0
                    immediate_reward = 0  # Initialize immediate reward

                    for state_prob_r in s_a_prob[(s, a)]:  # Iterate through all possible next states
                        immediate_reward = state_prob_r['reward']  # Get the immediate reward
                        next_state_tuple = self.create_tuple_state(state_prob_r['state'])
                        next_state_value = self.V.get(next_state_tuple, 0)  # Default to 0 if state is missing
                        transition_prob = state_prob_r['prob']  # Get the transition probability

                        val += transition_prob * next_state_value  # Bellman update without immediate reward

                    val = immediate_reward + val  # Add the immediate reward to the value

                    if val > max_val:  # Update max value and best action
                        max_val = val
                        best_a = a

                newV[s] = max_val  # Update V^(t)(s) with the max value
                self.Q[(s, i)] = best_a  # Save the best action

            self.V = newV  # Update V(s) for the next iteration
        return self.V, self.Q
    def act(self, state):
        """Chooses the best action based on the current state and policy."""
        if (self.create_tuple_state(state), state['turns_to_go']) not in self.Q:
            return "terminate"
        a = self.Q[(self.create_tuple_state(state), state['turns_to_go'])]
        return a


class WizardAgent:
    def __init__(self, initial):
        """
        A wizard agent that uses:
          - Probability/distance scoring to pick a cell to move toward
          - Guaranteed DE collision avoidance
          - Only calls "destroy" if wizard == horcrux official location
        """
        self.initial = deepcopy(initial)
        self.map = self.initial['map']
        self.n = len(self.map)
        self.m = len(self.map[0])
        self.turns_to_go = self.initial['turns_to_go']

    def act(self, state):
        """
        Steps:
          1) If no horcrux left -> 'terminate'
          2) For each wizard:
             a) If wizard == official horcrux location => "destroy"
             b) Else, do BFS from wizard location to find all reachable cells,
                ignoring impassable 'I' tiles and guaranteed DE positions.
                Then pick the cell c that maximizes:
                   probability(c) / (1 + distance(wizard, c))
             c) If the best cell is wizard's own cell => "wait"
                Otherwise, move one step in that BFS path
          3) Return the combined set of wizard actions as a tuple
        """
        if not state['horcrux']:
            return "terminate"

        actions = []
        wizards_dict = state['wizards']
        death_eaters_dict = state['death_eaters']
        horcruxes_dict = state['horcrux']

        # Identify guaranteed Death Eater positions next turn
        guaranteed_de_positions = self.compute_guaranteed_DE_positions(death_eaters_dict)

        for wizard_name, wizard_info in wizards_dict.items():
            w_loc = tuple(wizard_info['location'])

            # 1) Check if we can destroy a horcrux here (exact official location)
            horcrux_to_destroy = self.check_can_destroy_here(w_loc, horcruxes_dict)
            if horcrux_to_destroy is not None:
                actions.append(('destroy', wizard_name, horcrux_to_destroy))
                continue

            # 2) BFS from the wizard's location to find reachable cells
            #    and record distance to each reachable cell
            dist_map, parents = self.bfs_distances(w_loc, guaranteed_de_positions)

            # 3) Among all reachable cells, pick the cell that maximizes
            #    probability(c) / (1 + dist_map[c])
            best_cell = None
            best_score = -float('inf')
            prob_map = self.build_probability_map(horcruxes_dict)  # Probability of each cell next turn

            for c in dist_map:  # every reachable cell c
                d = dist_map[c]
                # Score = prob(c) / (1 + distance)
                score_c = prob_map.get(c, 0.0) / (1 + d)
                if score_c > best_score:
                    best_score = score_c
                    best_cell = c

            if best_cell is None:
                # nothing reachable or no data => wait
                actions.append(('wait', wizard_name))
                continue

            if best_cell == w_loc:
                # best option is to "camp" here
                actions.append(('wait', wizard_name))
            else:
                # Move one step toward best_cell using the BFS parents map
                next_step = self.reconstruct_next_step(parents, w_loc, best_cell)
                if next_step is None:
                    # can't get there? => wait
                    actions.append(('wait', wizard_name))
                else:
                    actions.append(('move', wizard_name, next_step))

        if not actions:
            return "terminate"

        return tuple(actions)

    ############################################################################
    #                           HELPER METHODS
    ############################################################################

    def compute_guaranteed_DE_positions(self, death_eaters):
        """
        Returns a set of board cells where a Death Eater is definitely located
        next turn (i.e. it has no branching path).
        """
        guaranteed_positions = set()

        for _, de_info in death_eaters.items():
            path = de_info['path']
            idx = de_info['index']
            if len(path) == 1:
                # Only one spot: guaranteed
                guaranteed_positions.add(path[idx])
            else:
                possible_indices = set()
                possible_indices.add(idx)         # can always stay
                if idx > 0:
                    possible_indices.add(idx - 1) # can move back
                if idx < len(path) - 1:
                    possible_indices.add(idx + 1) # can move forward

                if len(possible_indices) == 1:
                    # exactly one possible next index => guaranteed
                    next_i = possible_indices.pop()
                    guaranteed_positions.add(path[next_i])

        return guaranteed_positions

    def check_can_destroy_here(self, wizard_loc, horcruxes_dict):
        """
        If wizard_loc matches any horcrux's official location, return that horcrux name.
        Otherwise return None.
        """
        for hx_name, hx_info in horcruxes_dict.items():
            curr_loc = tuple(hx_info['location'])
            if wizard_loc == curr_loc:
                return hx_name
        return None

    def build_probability_map(self, horcruxes_dict):
        """
        For each cell c, sum the probability that c will be the horcrux next turn.
        Probability that a horcrux stays in curr_loc = (1 - p) + p/len(possible_locations)
        Probability that it teleports to some other loc = p/len(possible_locations)
        Return a dict: { cell: probability_sum, ... }
        """
        from collections import defaultdict
        cell_prob = defaultdict(float)
        for _, hx_info in horcruxes_dict.items():
            p = hx_info['prob_change_location']
            poss_locs = hx_info['possible_locations']
            curr_loc = tuple(hx_info['location'])
            k = len(poss_locs)

            stay_prob = (1 - p) + (p / k)
            cell_prob[curr_loc] += stay_prob

            # For the other possible locations:
            for loc in poss_locs:
                loc = tuple(loc)
                if loc != curr_loc:
                    cell_prob[loc] += (p / k)

        return dict(cell_prob)

    def bfs_distances(self, start, forbidden_positions):
        """
        Standard BFS from 'start', ignoring 'I' cells and 'forbidden_positions'.
        Returns:
          dist_map   = { cell: distance_from_start, ... }
          parents    = { cell: parent_cell_in_path, ... }
        """
        dist_map = {}
        parents = {}
        visited = set()
        queue = [(start, 0)]
        visited.add(start)
        dist_map[start] = 0
        parents[start] = None

        directions = [(1,0), (-1,0), (0,1), (0,-1)]
        while queue:
            (r, c), d = queue.pop(0)
            for dr, dc in directions:
                nr, nc = r+dr, c+dc
                if 0 <= nr < self.n and 0 <= nc < self.m:
                    # Must not be blocked or forbidden
                    if self.map[nr][nc] != 'I' and (nr, nc) not in forbidden_positions:
                        if (nr, nc) not in visited:
                            visited.add((nr, nc))
                            dist_map[(nr, nc)] = d + 1
                            parents[(nr, nc)] = (r, c)
                            queue.append(((nr, nc), d+1))

        return dist_map, parents

    def reconstruct_next_step(self, parents, start, goal):
        """
        Given BFS parent map, reconstruct the path from goal -> start,
        then reverse it. Return path[1] if it exists.
        """
        # If goal not in parents => not reachable
        if goal not in parents:
            return None

        # Rebuild path from goal -> start
        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = parents[cur]
        path.reverse()

        if len(path) < 2:
            return None
        return path[1]

import agentpy as ap
import numpy as np
import matplotlib.pyplot as plt
import IPython
import json
from import_bitmap import *
from A_star import *


class Road(ap.Agent):
    def setup(self):
        self.AType = 0

class Station(ap.Agent):
    def setup(self):
        self.AType = 2



def step_distance(a, b):
    # Takes two positions, returns the number of time steps it would
    # take to advance to that position going only up, down, left, and
    # right.
    return abs(b[0] - a[0]) + abs(b[1] - a[1])


class Train(ap.Agent):
    def setup(self):
        self.AType = 1
        self.rounds = 0

        # Progress in following the A* route
        self.progress = 0

    def next_roads(self):
        # Filters the grid's `.neighbors` output to only the 4
        # directions.
        t = self.model.tracks
        pos = t.positions[self]
        n_pos = list() # Neighbor positions
        for n in t.neighbors(self):
            n_pos.append(t.positions[n])

        return list(filter(four_way_filter(pos), n_pos))


    def follow_A_star(self):
        route = self.model.routes[self]

        # Avoid an IndexError if already in position or there is no route
        if len(route) <= self.progress:
            return
        
        next_pos = route[self.progress]
        self.model.tracks.move_to(self, next_pos)
        self.progress = self.progress + 1


    def remaining_route(self):
        return self.model.routes[self][self.progress:]


    def adjust_route(self, collision):
        """Adjustment for route to avoid collision with other train.
        Attempts to move out of the way using an intersection. It is
        more efficient than waiting out, since it does keep advancing,
        only at the intersection does it have to wait out.

        Returns a tuple with the place to insert the new position,
        said position, and the time to remain in that position.

        """
        other = None
        t = collision[0] # Time to collision
        if collision[1][0] is self:
            other = collision[1][1]
        else:
            other = collision[1][0]

        rem = self.remaining_route()
        orem = other.remaining_route()

        assert (rem[t] == orem[t] or (rem[t+1] == orem[t] and rem[t] == orem[t+1])), f"remaining_route[t] is not a collision {rem[t]} != {orem[t]}"

        for i, p in enumerate(rem[:t+1][::-1]):
            # if not (p in orem[t:]):
            #     # There is a point before an intersection
            #     print(f"Intersection not needed, can be waited out!")
            if p in self.model.intersections:
                # If the collision is in the same intersection but they don't go the same way
                if i == 0 and (rem[t-1] != orem[t+1]):
                    return (t-1, rem[t-1], 1)

                inter = self.model.intersections[p]
                possible_spots = [self.model.tracks.positions[n] for n in FWF_neighbors(inter) ]
                #print(possible_spots)
                for poss in possible_spots:
                    if not poss in orem:
                        # if it is a "swap" collision, add 1 to the time waiting.
                        return (t - i - 1, poss, i + 1 + (rem[t+1] == orem[t] and rem[t] == orem[t+1]))
        else:
            print(f"Found no way to avoid {collision} as {self}")


    def give_way(self):
        pass


    def priority_func(self):
        rem = self.remaining_route()
        P = len(rem)


def image_to_TrainModel(model, matrix, mapping):
    # Applies the values of an image to create the environment in Model
    def apply_mapping(col):
        if not col in mapping:
            return -1
        return mapping[col]

    mapped = np.vectorize(apply_mapping)(matrix)
    # Muestra la interpretacion de la imagen y los colores
    #print(mapped)

    trains = list()
    tracks = list()
    stations = list()

    for y_i in range(matrix.shape[0]):
        for x_i in range(matrix.shape[1]):
            v = mapped[y_i, x_i]
            #print(v)
            if v < 0 or v > 2:
                continue

            if v == 0: # Road
                tracks.append( (Road(model), (y_i, x_i)) )
            elif v == 1: # Train'
                tracks.append((Road(model), (y_i, x_i))) # Trains obviously are on a road (track)
                trains.append((Train(model), (y_i, x_i)))
            elif v == 2: # Station
                stations.append((Station(model), (y_i, x_i)))

    # UnZip and add the agents to their respective part of the model
    agents, positions = list(zip(*trains)) # First trains
    model.trains = ap.AgentList(model, objs=list(agents))
    model.tracks.add_agents(agents, positions=positions)

    agents, positions = list(zip(*tracks)) # Then tracks
    model.trackList = ap.AgentList(model, objs=list(agents))
    model.tracks.add_agents(agents, positions=positions)

    agents, positions = list(zip(*stations)) # Finally train stations
    model.stations = ap.AgentList(model, objs=list(agents))
    model.tracks.add_agents(agents, positions=positions)
    


class TrainModel(ap.Model):
    def setup(self):
        arr = ppm_to_array(self.p.image)

        self.tracks = ap.Grid(self, arr.shape)

        image_to_TrainModel(self, arr, self.p.im_map)
        self.routes = dict()
        self.routes_hist = dict()

        # Choose a random station to be the target of a train
        stations = self.stations.random(n=len(self.trains))
        for i in range(len(self.trains)):
            t = self.trains[i]
            goal = self.stations[i]
            route = A_Star(t, goal)
            #print(f"Route for train in {self.tracks.positions[t]}:\n{route}\n{'='*50}\n")
            self.routes[t] = route
            self.routes_hist[t] = [route]


        # Fill list of intersections
        self.intersections = dict()
        # {self.tracks.positions[r]: r for r in self.trackList if len(self.tracks.neighbors(r)) >= 3}
        for r in self.trackList:
            n_iter = self.tracks.neighbors(r)
            only_roads = len([a for a in n_iter if not a in self.trains])
            if only_roads >= 3:
                self.intersections[self.tracks.positions[r]] = r

        print(f"Expected collisions:\n {self.future_collisions()}")
                

    def update(self):

        routes_change = False
        # Assign new random goal if the train still has rounds to do
        for t in self.trains:
            pos = self.tracks.positions[t]
            t.record('pos', pos)
            rounds_finished = t.rounds

            # If this train has finished all its rounds, there is no
            # need for it to get assigned another station/goal.
            if rounds_finished == self.p.rounds - 1 or t.progress != len(self.routes[t]):
                continue

            routes_change = True

            # Make sure it is not assigned the station it is already in
            while pos == self.routes[t][-1]:
                new_goal = list(self.stations.random())[0]
                new_route = A_Star(t, new_goal)
                self.routes[t] = new_route
                self.routes_hist[t].append(new_route)
                t.rounds = rounds_finished + 1
                t.progress = 0

            print(f"{t} assigned station in {self.routes[t][-1]} for round #{t.rounds + 1} (t={self.t})")
            #print(f"Route for train in {pos}:\n{self.routes[t]}\n{'='*50}\n")

        all_finished = True
        for t in self.trains:
            # Some train still has not finished
            if t.rounds != self.p.rounds - 1:
                all_finished = False
                break
            
        if all_finished: self.stop()

        # Detect collisions
        positions = [self.tracks.positions[t] for t in self.trains]
        
        for i in range(len(positions)-1):
            for j in range(i+1, len(positions)):
                if positions[i] == positions[j]:
                    print(f"{self.trains[i]} and {self.trains[j]} collisioned in {positions[i]} (t={self.t})")

        if routes_change:
            self.resolve_collisions()


    def resolve_collisions(self):
        collisions = self.future_collisions()
        while collisions:
            coll = collisions.pop()
            
            a = coll[1][0]
            b = coll[1][1]
            yielder = a
            
            adj = a.adjust_route(coll)
            if adj is None:
                print(f"{a} could not avoid, trying {b}")
                adj = b.adjust_route(coll)
                yielder = b
            if adj is None:
                print(f"{coll} not avoidable!")
                continue

            t = adj[0]
            for i in range(adj[2]):
                self.routes[a].insert(t+a.progress, adj[1])
            print(f"Adding {adj} to route of {yielder}, as in {a.remaining_route()[t-1:t+2]}")
            collisions = self.future_collisions()


    def future_collisions(self):
        f_routes = self.trains.remaining_route()
        stations = [self.tracks.positions[s] for s in self.stations]
        fr_len = [len(r) for r in f_routes]
        collisions = list()

        # Time in future steps from now
        for t in range(1, max(fr_len)):
            f_pos = [f_routes[i][t] if t < len(f_routes[i]) else None for i in range(len(f_routes))]
            
            # position in t-1, to also recognize collitions where
            # trains swap places, not just share it.
            f_m1_pos = [f_routes[i][t-1] if t > 0 and (t-1) < len(f_routes[i]) else None for i in range(len(f_routes))]

            for i in range(0,len(f_routes)-1):
                i_pos = f_pos[i]
                i_pos_tm1 = f_m1_pos[i]
                if i_pos is None or i_pos in stations: continue
                for j in range(i+1, len(f_routes)):
                    j_pos = f_pos[j]
                    j_pos_tm1 = f_m1_pos[j]
                    if j_pos is None or j_pos in stations: continue
                    
                    if i_pos == j_pos:
                        # Add to return value the collision and when it would happen
                        collisions.append( (t, (self.trains[i], self.trains[j])) )
                    elif i_pos == j_pos_tm1 and j_pos == i_pos_tm1:
                        collisions.append( (t-1, (self.trains[i], self.trains[j])) )

        return collisions
            

    def step(self):
        self.trains.follow_A_star()


def animation_plot(model, ax):
    attr_grid = model.tracks.attr_grid("AType", otypes='f')
    # Forces priority showing the train
    for train in model.trains:
        pos = model.tracks.positions[train]
        attr_grid[pos[0]][pos[1]] = 1

    color_dict = {0:'#ffbd33', 1:'#ff5733', 2:'#12ff33', None:'#d5e5d5'}
    ap.gridplot(attr_grid, ax=ax, color_dict=color_dict, convert=True)


fig, ax = plt.subplots()
        
parameters = {'image': "Europa_70x90_trains_n_stations.ppm",
              'im_map': {0x00ff00: 1, 0x00: 0, 0xff0000: 2},
              'rounds': 10}

model = TrainModel(parameters)
#model.run(display=False)

animation = ap.animate(model, fig, ax, animation_plot)

name_video = "video_demo.mp4"

animation.save(name_video, fps=2)
print(f"Generated video {name_video}")

#                                  # Generating JSON First #
# #add file coordinates of all tracks
# json_dict = {"tracks": [model.tracks.positions[t] for t in model.trackList]}
# # Then stations
# json_dict["stations"] = [model.tracks.positions[s] for s in model.stations]
# # Then all trains and their routes historically
# json_dict["trains"] = {t.id: t.log['pos'] for t in model.trains}

# json_dump = json.dumps(json_dict, sort_keys=True)
# #print(json_dump)

# # Writing the json output, that will be used by Unity through a http server
# with open("output.json", 'w') as f:
#     f.write(json_dump)
#     f.close()

# Useful for debuggin pathfinding algo's
# for t in model.trains:
#     print("Route of train with id", t.id)
#     print(t.log, '\n')


#IPython.display.HTML(animation.to_jshtml(fps=3))
#model.run()

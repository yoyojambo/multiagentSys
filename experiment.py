import agentpy as ap
import numpy as np
import matplotlib.pyplot as plt
import IPython
import json
from import_bitmap import *


class Road(ap.Agent):
    def setup(self):
        self.AType = 0

class Station(ap.Agent):
    def setup(self):
        self.AType = 2


def four_way_filter(cur_pos):
    # Returns the function to be passed to `filter`.  Only keeps the
    # positions that are (exclusive) either to the 'north', 'south',
    # 'east' or 'west' of the current position.
    def f(track_pos):
        x_diff = track_pos[0] - cur_pos[0]
        y_diff = track_pos[1] - cur_pos[1]

        return (abs(x_diff) + abs(y_diff)) == 1

    return f

def step_distance(a, b):
    # Takes two positions, returns the number of time steps it would
    # take to advance to that position going only up, down, left, and
    # right.
    return abs(b[0] - a[0]) + abs(b[1] - a[1])


class Train(ap.Agent):
    def setup(self):
        self.AType = 1

    def next_roads(self):
        # Filters the grid's `.neighbors` output to only the 4
        # directions.
        t = self.model.tracks
        pos = t.positions[self]
        n_pos = list() # Neighbor positions
        for n in t.neighbors(self):
            n_pos.append(t.positions[n])

        return list(filter(four_way_filter(pos), n_pos))

    def go_to_goal(self):
        t = self.model.tracks
        goal = self.p.goal
        pos = t.positions[self]
        cur_dist = step_distance(pos, goal)

        # Gets the track 'agents' in the grid that would allow for
        # movement.
        for n_pos in self.next_roads():
            if step_distance(n_pos, goal) < cur_dist and not n_pos in self.log['pos']:
                t.move_to(self, n_pos)
                return

        if cur_dist == 0: return

        # This runs if it is stuck, it plays hug-the-wall to get unstuck
        for n_pos in self.next_roads():
            if not n_pos in self.log['pos']:
                t.move_to(self, n_pos)
                return


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


    def update(self):
        if self.t > 1000:
            print("stopped after too many steps!")
            self.stop()

        all_in_goal = True
        all_stuck = True
        for t in self.trains:
            pos = self.tracks.positions[t]
            t.record('pos', pos)

            if self.p.goal != pos: all_in_goal = False
            if self.t < 2 or pos != t.log['pos'][-2]: all_stuck = False
            
        if all_in_goal or all_stuck: self.stop()


    def step(self):
        self.trains.go_to_goal()


def animation_plot(model, ax):
    attr_grid = model.tracks.attr_grid("AType", otypes='f')
    # Forces priority showing the train
    for train in model.trains:
        pos = model.tracks.positions[train]
        attr_grid[pos[0]][pos[1]] = 1

    color_dict = {0:'#ffbd33', 1:'#ff5733', 2:'#12ff33', None:'#d5e5d5'}
    ap.gridplot(attr_grid, ax=ax, color_dict=color_dict, convert=True)


fig, ax = plt.subplots()

        
parameters = {'goal': (4, 14),
              'image': "test_with_trains.ppm",
              'im_map': {0x00ff00: 1, 0x00: 0, 0xff0000: 2}}

model = TrainModel(parameters)


animation = ap.animate(model, fig, ax, animation_plot)

name_video = "video_demo.mp4"
print(f"Generating video {name_video}")
animation.save(name_video, fps=2)

                                 # Generating JSON file #
# First add coordinates of all tracks
json_dict = {"tracks": [model.tracks.positions[t] for t in model.trackList]}
# Then stations
json_dict["stations"] = [model.tracks.positions[s] for s in model.stations]
# Then all trains and their routes historically
json_dict["trains"] = dict()
for t in model.trains:
    json_dict["trains"][t.id] = t.log['pos']

json_dump = json.dumps(json_dict, sort_keys=True)
#print(json_dump)

# Writing the json output, that will be used by Unity through a http server
with open("output.json", 'w') as f:
    f.write(json_dump)
    f.close()


# Useful for debuggin pathfinding algo's
# for t in model.trains:
#     print("Route of train with id", t.id)
#     print(t.log, '\n')


#IPython.display.HTML(animation.to_jshtml(fps=3))
#model.run()

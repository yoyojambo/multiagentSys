import agentpy as ap
import matplotlib.pyplot as plt
import IPython


class Road(ap.Agent):
    def setup(self):
        self.AType = 0


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
            if step_distance(n_pos, goal) < cur_dist:
                t.move_to(self, n_pos)
                return


track_positions = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (1, 4)]


class TrainModel(ap.Model):
    def setup(self):
        self.trains = ap.AgentList(self)
        self.trains.append(Train(self))

        self.tracks = ap.Grid(self, [5, 5])

        self.trackList = list()
        for i in range(len(track_positions)):
            self.trackList.append(Road(self))

        self.tracks.add_agents(self.trackList, positions=track_positions)

        self.tracks.add_agents(self.trains, positions=[(0, 0)])

    def update(self):
        # print(self.tracks)
        # neighbors = self.tracks.neighbors(self.trains[0])
        # print("neighbors: ", neighbors)
        # for n in neighbors:
        #     print(n)
        # print("=====")
        # print(list(self.tracks.grid))
        print("position", self.tracks.positions[self.trains[0]])
        print("=====")
        print("Tracks next to location:", self.trains[0].next_roads())
        # if self.t > 2: self.stop()
        if self.p.goal == self.tracks.positions[self.trains[0]]:
            self.stop()


    def step(self):
        self.trains.go_to_goal()


def animation_plot(model, ax):
    attr_grid = model.tracks.attr_grid("AType", otypes='f')
    color_dict = {0:'#ffbd33', 1:'#ff5733', None:'#d5e5d5'}
    ap.gridplot(attr_grid, ax=ax, color_dict=color_dict, convert=True)


fig, ax = plt.subplots()
        
parameters = {'goal': (1, 4)}

model = TrainModel(parameters)
animation = ap.animate(model, fig, ax, animation_plot)
IPython.display.HTML(animation.to_jshtml(fps=15))
#model.run()

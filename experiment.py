import agentpy as ap
import matplotlib


class Road(ap.Agent):
    pass

class TrainModel(ap.Model):
    def setup(self):
        n_trains = 1
        self.trains = ap.AgentList(self, n_trains)


        self.tracks = ap.Grid(self, [5, 5])

        trackList = list()
        for i in range(10):
            trackList.append(Road(self))

        self.tracks.add_agents(trackList, positions=[(0,0), (0,1), (0,2), (1,2), (1,2)])

        self.tracks.add_agents(self.trains, positions=[(0,0)])

    def update(self):
        print(self.tracks)
        neighbors = self.tracks.neighbors(self.trains[0])
        print("neighbors: ", neighbors)
        for n in neighbors:
            print(n)
        print("=====")
        print(list(self.tracks.grid))
        print("position", self.tracks.positions[self.trains[0]])
        self.stop()


model = TrainModel()
model.run()



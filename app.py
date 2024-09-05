import experiment

def create_JSON(rounds=5, seed=12351):
    parameters = {'image': "Europa_2.ppm",
                  'im_map': {0x00ff00: 1, 0x00: 0, 0xff0000: 2},
                  'rounds': rounds,
                  'seed' : seed }

    model = experiment.TrainModel(parameters)
    model.run()

    json_dict = dict()
    json_dict["tracks"] = [model.tracks.positions[t] for t in model.trackList]
    json_dict["stations"] = [model.tracks.positions[s] for s in model.stations]
    json_dict["trains"] = {t.id: t.log['pos'] for t in model.trains}

    return json_dict


from flask import Flask
app = Flask(__name__)

@app.route("/")
def call_experiment():
    return create_JSON()

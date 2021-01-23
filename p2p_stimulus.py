import numpy as np
import pulse2percept as p2p

input_video = "depth/sample.mp4"
output_folder = "p2p-samples/"

for RHO in [100, 300, 500]:
    for LAM in [0, 100, 200]:

        if LAM == 0:
            model = p2p.models.ScoreboardModel(xrange=(-10, 10), yrange=(-10, 10), rho=RHO)
        else:
            model = p2p.models.AxonMapModel(xrange=(-10, 10), yrange=(-10, 10), rho=RHO, axlambda=LAM)
        model.build()

        grid_sizes = [(8, 8), (16, 16), (32, 32)]
        implants = {}
        for gsize in grid_sizes:
            # Fit all electrodes into (-2000, 2000):
            spacing = 4000 / gsize[0]
            # Sensible radius might be 1/5th of spacing:
            radius = spacing / 5
            egrid = p2p.implants.ElectrodeGrid(gsize, spacing,
                                               etype=p2p.implants.DiskElectrode,
                                               r=radius)
            implants['%dx%d' % gsize] = p2p.implants.ProsthesisSystem(egrid)

        current_video = p2p.stimuli.VideoStimulus(input_video, as_gray=True)
        for gsize in grid_sizes:
            res = gsize[0]
            implant_key = str(res) + "x" + str(res)
            implant = implants[implant_key]
            implant.stim = current_video.resize(implant.earray.shape)
            percept = model.predict_percept(implant)
            percept.save(output_folder + "sample_{}({},{})".format(res, RHO, LAM) + ".mp4", fps=20)  # You can control the frame rate with fps=

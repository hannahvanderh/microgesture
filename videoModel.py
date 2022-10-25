from collections import OrderedDict

class Video():
    def __init__(self, gestureIndex, participantIndex, videoIndex):
        self.gestureIndex = gestureIndex
        self.participantIndex = participantIndex
        self.videoIndex = videoIndex
        self.frames = {}

    def addFrame(self, index, path):
        self.frames[index] = path

    def trimFrames(self):
        #TODO trim frames down based on indexes
        return OrderedDict(sorted(self.frames.items()))
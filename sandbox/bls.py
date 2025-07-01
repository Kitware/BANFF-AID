"""BLS class.

The BLS class inherits from the BanffLesionScore class, but its initializer makes it
easier to test and debug locally.
"""

import json
import os

from banff_aid.cli.BanffAid.banff_lesion_score.banff_lesion_score import BanffLesionScore
from girder_client import GirderClient


class BLS(BanffLesionScore):
    def __init__(self) -> None:
        # I think we're just going to have to do this the longer way
        annotations_directory = "data/annotations"
        files = sorted(os.listdir(annotations_directory))

        # Arteries
        with open(f"{annotations_directory}/{files[0]}") as file:
            art = json.load(file)
        # Cortical Interstitium
        with open(f"{annotations_directory}/{files[1]}") as file:
            cortex = json.load(file)
        # GSG
        with open(f"{annotations_directory}/{files[2]}") as file:
            gsg = json.load(file)
        # Non GSG
        with open(f"{annotations_directory}/{files[3]}") as file:
            nongsg = json.load(file)
        # Tubules
        with open(f"{annotations_directory}/{files[4]}") as file:
            tubules = json.load(file)

        self.arteries_annotation = art
        self.cortical_interstitium_annotation = cortex
        self.gsg_annotation = gsg
        self.non_gsg_annotation = nongsg
        self.tubules_annotation = tubules

        # Here's an example of how you could instantiate the other important attributes.
        # You will want to add an actual username and password as well as the actual
        # image ID you would get from Girder
        # self.gc = GirderClient(apiUrl="http://localhost:8080/api/v1")
        # self.gc.authenticate(username="admin", password="password")
        # self.results_folder = "67c9d3e77f87a3883fb5b43e"
        # self.image_id = "67dc170811ebeeb3121f9c23"
        # self.image_filepath = "images/example.svs"

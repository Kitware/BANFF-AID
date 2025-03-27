Scripts for converting OBX files
================================

This consists of:

- A java script to parse the OBX file and convert it to json.  The OBX file is
  a stored java ObjectStream.

- A python script to convert the output from the OBX parser to formats useful
  for the DSA or further analysis.

Parse OBX
---------

Download all the required jar files:

```
curl -OLJ https://repo1.maven.org/maven2/com/fasterxml/jackson/core/jackson-core/2.18.2/jackson-core-2.18.2.jar
curl -OLJ https://repo1.maven.org/maven2/com/fasterxml/jackson/core/jackson-annotations/2.18.2/jackson-annotations-2.18.2.jar
curl -OLJ https://repo1.maven.org/maven2/com/fasterxml/jackson/core/jackson-databind/2.18.2/jackson-databind-2.18.2.jar
curl -OLJ https://github.com/icbm-iupui/volumetric-tissue-exploration-analysis/releases/download/v1.2.3/vtea_1.2.3-vtea.jar
curl -OLJ https://imagej.net/ij/download/jars/ij146r.jar
```

Compile:

```
javac -cp .:jackson-core-2.18.2.jar:jackson-databind-2.18.2.jar ObjectStreamParser.java
```

Run 
```
java -cp .:jackson-core-2.18.2.jar:jackson-databind-2.18.2.jar:vtea_1.2.3-vtea.jar:ij146r.jar:jackson-annotations-2.18.2.jar ObjectStreamParser /mnt/data2/KPMP/23-0147_Segmentation.obx > segmentation.json
```

Convert OBX json to Annotations and data tables
-----------------------------------------------

Install prerequisites for python script:

```
pip install ijson numpy pandas shapely zarr
```

Convert the output to geojson annotations that can be loaded in HistomicsUI 
and to zarr and/or csv files with the tabular data per segmented area:

```
python jsontoannotation.py segmentation.json --geojson=segmentation.geojson --zarr=segmentation.zarr --csv=segmentation.csv
```

You can upload the geojson to the DSA via the annotation upload button on the
Girder item page


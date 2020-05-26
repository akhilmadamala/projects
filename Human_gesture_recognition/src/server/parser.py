import json
positions=[]
import pandas as pd
with open("C:\\Users\\Vasu\\Downloads\\key_points.json",'r') as f:
    data = json.load(f)
    for frame in data:
        for key,val in frame.items():
            if key=="keypoints":
                frameno=[]
                for subframe in val:
                    for key1,val1 in subframe.items():
                        if key1=="position":
                            frameno.extend(list(val1.values()))
                positions.append(frameno)
print(positions)
print(data)
print(len(data))
import json

with open("/home/janne/ros_ws/latte/src/data/datalatte_100k_lf.json", "r") as f:
    data = json.load(f)


change_type = []
for r in data:
    c = data[r]["change_type"]
    if c not in change_type:
        change_type.append(c)
    # for head in data[r]:
        # print(head)

print(change_type)
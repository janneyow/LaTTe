import json

#################################
# Extracting text from data set #
#################################

# with open("/home/janne/ros_ws/latte/src/data/datalatte_100k_lf.json", "r") as f:
#     data = json.load(f)


# corpus = []
# for d in data:
#     text = data[d]["text"]
#     change = data[d]["change_type"]
#     obj = data[d]["obj_in_text"]
#     data_dict = {"text": text, "change_type": change, "object": obj}
#     corpus.append(data_dict)

# with open("/home/janne/ros_ws/latte/src/data/data_latte_100k_text.json", "w") as f:
#     json.dump(corpus, f)

#####################################
# Extracting objects from text data #
#####################################

# with open("/home/janne/ros_ws/latte/src/data/data_latte_100k_text.json", "r") as f:
#     data = json.load(f)


# obj_list = []
# for d in data:
#     obj = d["object"]
#     if obj not in obj_list:
#         obj_list.append(obj)

# obj_list = sorted(obj_list)
# print(obj_list)

# with open("/home/janne/ros_ws/latte/src/data/data_latte_100k_obj_list.json", "w") as f:
#     json.dump(obj_list, f)

#########################################
# Checking is object exists in data set #
#########################################
with open("/home/janne/ros_ws/latte/src/data/data_latte_100k_obj_list.json", "r") as f:
    obj_list = json.load(f)
    
# bottle - beer bottle, soda bottle, water bottle, wine bottle
objects = ["cup", "beer bottle", "laptop"]
for obj in objects:
    if obj in obj_list:
        print(obj, "in list")
    else:
        print(obj, "NOT in list")
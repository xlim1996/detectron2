'''
Author: Xiaolin Lin xlim1996@outlook.com
Date: 2023-04-02 12:54:24
LastEditors: Xiaolin Lin xlim1996@outlook.com
LastEditTime: 2023-04-07 15:26:32
FilePath: /hiwi/detectron2/detectron2/data/datasets/lvis_own_categories.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# Copyright (c) Facebook, Inc. and its affiliates.
# Autogen with
# with open("lvis_v1_val.json", "r") as f:
#     a = json.load(f)
# c = a["categories"]
# for x in c:
#     del x["image_count"]
#     del x["instance_count"]
# LVIS_CATEGORIES = repr(c) + "  # noqa"
# with open("/tmp/lvis_categories.py", "wt") as f:
#     f.write(f"LVIS_CATEGORIES = {LVIS_CATEGORIES}")
# Then paste the contents of that file below

# fmt: off
LVIS_CATEGORIES = [{'frequency': 'f', 'synset': 'air_conditioner.n.01', 'synonyms': ['air_conditioner'], 'id': 2, 'def': 'a machine that keeps air cool and dry', 'name': 'air_conditioner'}, {'frequency': 'f', 'synset': 'alarm_clock.n.01', 'synonyms': ['alarm_clock'], 'id': 4, 'def': 'a clock that wakes a sleeper at some preset time', 'name': 'alarm_clock'}, {'frequency': 'f', 'synset': 'apple.n.01', 'synonyms': ['apple'], 'id': 12, 'def': 'fruit with red or yellow or green skin and sweet to tart crisp whitish flesh', 'name': 'apple'}, {'frequency': 'f', 'synset': 'armchair.n.01', 'synonyms': ['armchair'], 'id': 19, 'def': 'chair with a support on each side for arms', 'name': 'armchair'}, {'frequency': 'r', 'synset': 'armoire.n.01', 'synonyms': ['armoire'], 'id': 20, 'def': 'a large wardrobe or cabinet', 'name': 'armoire'}, {'frequency': 'f', 'synset': 'ashcan.n.01', 'synonyms': ['trash_can', 'garbage_can', 'wastebin', 'dustbin', 'trash_barrel', 'trash_bin'], 'id': 23, 'def': 'a bin that holds rubbish until it is collected', 'name': 'trash_can'}, {'frequency': 'f', 'synset': 'avocado.n.01', 'synonyms': ['avocado'], 'id': 27, 'def': 'a pear-shaped fruit with green or blackish skin and rich yellowish pulp enclosing a single large seed', 'name': 'avocado'}, {'frequency': 'f', 'synset': 'banana.n.02', 'synonyms': ['banana'], 'id': 45, 'def': 'elongated crescent-shaped yellow fruit with soft sweet flesh', 'name': 'banana'}, {'frequency': 'f', 'synset': 'bed.n.01', 'synonyms': ['bed'], 'id': 77, 'def': 'a piece of furniture that provides a place to sleep', 'name': 'bed'}, {'frequency': 'f', 'synset': 'bedspread.n.01', 'synonyms': ['bedspread', 'bedcover', 'bed_covering', 'counterpane', 'spread'], 'id': 79, 'def': 'decorative cover for a bed', 'name': 'bedspread'}, {'frequency': 'f', 'synset': 'beer_bottle.n.01', 'synonyms': ['beer_bottle'], 'id': 83, 'def': 'a bottle that holds beer', 'name': 'beer_bottle'}, {'frequency': 'c', 'synset': 'beer_can.n.01', 'synonyms': ['beer_can'], 'id': 84, 'def': 'a can that holds beer', 'name': 'beer_can'}, {'frequency': 'f', 'synset': 'bell_pepper.n.02', 'synonyms': ['bell_pepper', 'capsicum'], 'id': 87, 'def': 'large bell-shaped sweet pepper in green or red or yellow or orange or black varieties', 'name': 'bell_pepper'}, {'frequency': 'f', 'synset': 'blanket.n.01', 'synonyms': ['blanket'], 'id': 110, 'def': 'bedding that keeps a person warm in bed', 'name': 'blanket'}, {'frequency': 'f', 'synset': 'blueberry.n.02', 'synonyms': ['blueberry'], 'id': 116, 'def': 'sweet edible dark-blue berries of blueberry plants', 'name': 'blueberry'}, {'frequency': 'f', 'synset': 'book.n.01', 'synonyms': ['book'], 'id': 127, 'def': 'a written work or composition that has been published', 'name': 'book'}, {'frequency': 'c', 'synset': 'bookcase.n.01', 'synonyms': ['bookcase'], 'id': 128, 'def': 'a piece of furniture with shelves for storing books', 'name': 'bookcase'}, {'frequency': 'f', 'synset': 'bottle.n.01', 'synonyms': ['bottle'], 'id': 133, 'def': 'a glass or plastic vessel used for storing drinks or other liquids', 'name': 'bottle'}, {'frequency': 'f', 'synset': 'bowl.n.03', 'synonyms': ['bowl'], 'id': 139, 'def': 'a dish that is round and open at the top for serving foods', 'name': 'bowl'}, {'frequency': 'c', 'synset': 'bread-bin.n.01', 'synonyms': ['bread-bin', 'breadbox'], 'id': 149, 'def': 'a container used to keep bread or cake in', 'name': 'bread-bin'}, {'frequency': 'f', 'synset': 'bread.n.01', 'synonyms': ['bread'], 'id': 150, 'def': 'food made from dough of flour or meal and usually raised with yeast or baking powder and then baked', 'name': 'bread'}, {'frequency': 'f', 'synset': 'broccoli.n.01', 'synonyms': ['broccoli'], 'id': 154, 'def': 'plant with dense clusters of tight green flower buds', 'name': 'broccoli'}, {'frequency': 'c', 'synset': 'bunk_bed.n.01', 'synonyms': ['bunk_bed'], 'id': 170, 'def': 'beds built one above the other', 'name': 'bunk_bed'}, {'frequency': 'f', 'synset': 'butter.n.01', 'synonyms': ['butter'], 'id': 175, 'def': 'an edible emulsion of fat globules made by churning milk or cream; for cooking and table use', 'name': 'butter'}, {'frequency': 'f', 'synset': 'cabinet.n.01', 'synonyms': ['cabinet'], 'id': 181, 'def': 'a piece of furniture resembling a cupboard with doors and shelves and drawers', 'name': 'cabinet'}, {'frequency': 'r', 'synset': 'cabinet.n.03', 'synonyms': ['locker', 'storage_locker'], 'id': 182, 'def': 'a storage compartment for clothes and valuables; usually it has a lock', 'name': 'locker'}, {'frequency': 'f', 'synset': 'cake.n.03', 'synonyms': ['cake'], 'id': 183, 'def': 'baked goods made from or based on a mixture of flour, sugar, eggs, and fat', 'name': 'cake'}, {'frequency': 'f', 'synset': 'can.n.01', 'synonyms': ['can', 'tin_can'], 'id': 192, 'def': 'airtight sealed metal container for food or drink or paint etc.', 'name': 'can'}, {'frequency': 'f', 'synset': 'cap.n.02', 'synonyms': ['bottle_cap', 'cap_(container_lid)'], 'id': 204, 'def': 'a top (as for a bottle)', 'name': 'bottle_cap'}, {'frequency': 'f', 'synset': 'carrot.n.01', 'synonyms': ['carrot'], 'id': 217, 'def': 'deep orange edible root of the cultivated carrot plant', 'name': 'carrot'}, {'frequency': 'f', 'synset': 'chair.n.01', 'synonyms': ['chair'], 'id': 232, 'def': 'a seat for one person, with a support for the back', 'name': 'chair'}, {'frequency': 'r', 'synset': 'chaise_longue.n.01', 'synonyms': ['chaise_longue', 'chaise', 'daybed'], 'id': 233, 'def': 'a long chair; for reclining', 'name': 'chaise_longue'}, {'frequency': 'r', 'synset': 'clementine.n.01', 'synonyms': ['clementine'], 'id': 266, 'def': 'a variety of mandarin orange', 'name': 'clementine'}, {'frequency': 'f', 'synset': 'coffee_table.n.01', 'synonyms': ['coffee_table', 'cocktail_table'], 'id': 285, 'def': 'low table where magazines can be placed and coffee or cocktails are served', 'name': 'coffee_table'}, {'frequency': 'f', 'synset': 'cookie.n.01', 'synonyms': ['cookie', 'cooky', 'biscuit_(cookie)'], 'id': 303, 'def': "any of various small flat sweet cakes (`biscuit' is the British term)", 'name': 'cookie'}, {'frequency': 'f', 'synset': 'cucumber.n.02', 'synonyms': ['cucumber', 'cuke'], 'id': 342, 'def': 'cylindrical green fruit with thin green rind and white flesh eaten as a vegetable', 'name': 'cucumber'}, {'frequency': 'f', 'synset': 'cup.n.01', 'synonyms': ['cup'], 'id': 344, 'def': 'a small open container usually used for drinking; usually has a handle', 'name': 'cup'}, {'frequency': 'f', 'synset': 'curtain.n.01', 'synonyms': ['curtain', 'drapery'], 'id': 350, 'def': 'hanging cloth used as a blind (especially for a window)', 'name': 'curtain'}, {'frequency': 'f', 'synset': 'cushion.n.03', 'synonyms': ['cushion'], 'id': 351, 'def': 'a soft bag filled with air or padding such as feathers or foam rubber', 'name': 'cushion'}, {'frequency': 'f', 'synset': 'dining_table.n.01', 'synonyms': ['dining_table'], 'id': 367, 'def': 'a table at which meals are served', 'name': 'dining_table'}, {'frequency': 'f', 'synset': 'dish.n.01', 'synonyms': ['dish'], 'id': 369, 'def': 'a piece of dishware normally used as a container for holding or serving food', 'name': 'dish'}, {'frequency': 'f', 'synset': 'drawer.n.01', 'synonyms': ['drawer'], 'id': 390, 'def': 'a boxlike container in a piece of furniture; made so as to slide in and out', 'name': 'drawer'}, {'frequency': 'c', 'synset': 'drawers.n.01', 'synonyms': ['underdrawers', 'boxers', 'boxershorts'], 'id': 391, 'def': 'underpants worn by men', 'name': 'underdrawers'}, {'frequency': 'f', 'synset': 'dresser.n.05', 'synonyms': ['dresser'], 'id': 395, 'def': 'a cabinet with shelves', 'name': 'dresser'}, {'frequency': 'f', 'synset': 'electric_refrigerator.n.01', 'synonyms': ['refrigerator'], 'id': 421, 'def': 'a refrigerator in which the coolant is pumped around by an electric motor', 'name': 'refrigerator'}, {'frequency': 'c', 'synset': 'file.n.03', 'synonyms': ['file_cabinet', 'filing_cabinet'], 'id': 438, 'def': 'office furniture consisting of a container for keeping papers in order', 'name': 'file_cabinet'}, {'frequency': 'f', 'synset': 'fork.n.01', 'synonyms': ['fork'], 'id': 469, 'def': 'cutlery used for serving and eating food', 'name': 'fork'}, {'frequency': 'f', 'synset': 'frying_pan.n.01', 'synonyms': ['frying_pan', 'frypan', 'skillet'], 'id': 477, 'def': 'a pan used for frying foods', 'name': 'frying_pan'}, {'frequency': 'f', 'synset': 'glass.n.02', 'synonyms': ['glass_(drink_container)', 'drinking_glass'], 'id': 498, 'def': 'a container for holding liquids while drinking', 'name': 'glass_(drink_container)'}, {'frequency': 'f', 'synset': 'grape.n.01', 'synonyms': ['grape'], 'id': 510, 'def': 'any of various juicy fruit with green or purple skins; grow in clusters', 'name': 'grape'}, {'frequency': 'c', 'synset': 'hammock.n.02', 'synonyms': ['hammock'], 'id': 531, 'def': 'a hanging bed of canvas or rope netting (usually suspended between two trees)', 'name': 'hammock'}, {'frequency': 'f', 'synset': 'handle.n.01', 'synonyms': ['handle', 'grip', 'handgrip'], 'id': 540, 'def': 'the appendage to an object that is designed to be held in order to use or move it', 'name': 'handle'}, {'frequency': 'f', 'synset': 'kitchen_sink.n.01', 'synonyms': ['kitchen_sink'], 'id': 609, 'def': 'a sink in a kitchen', 'name': 'kitchen_sink'}, {'frequency': 'r', 'synset': 'kitchen_table.n.01', 'synonyms': ['kitchen_table'], 'id': 610, 'def': 'a table in the kitchen', 'name': 'kitchen_table'}, {'frequency': 'c', 'synset': 'kiwi.n.03', 'synonyms': ['kiwi_fruit'], 'id': 613, 'def': 'fuzzy brown egg-shaped fruit with slightly tart green flesh', 'name': 'kiwi_fruit'}, {'frequency': 'f', 'synset': 'knife.n.01', 'synonyms': ['knife'], 'id': 615, 'def': 'tool with a blade and point used as a cutting instrument', 'name': 'knife'}, {'frequency': 'f', 'synset': 'laptop.n.01', 'synonyms': ['laptop_computer', 'notebook_computer'], 'id': 631, 'def': 'a portable computer small enough to use in your lap', 'name': 'laptop_computer'}, {'frequency': 'f', 'synset': 'lemon.n.01', 'synonyms': ['lemon'], 'id': 639, 'def': 'yellow oval fruit with juicy acidic flesh', 'name': 'lemon'}, {'frequency': 'c', 'synset': 'love_seat.n.01', 'synonyms': ['loveseat'], 'id': 656, 'def': 'small sofa that seats two people', 'name': 'loveseat'}, {'frequency': 'f', 'synset': 'magazine.n.02', 'synonyms': ['magazine'], 'id': 658, 'def': 'a paperback periodic publication', 'name': 'magazine'}, {'frequency': 'r', 'synset': 'masher.n.02', 'synonyms': ['masher'], 'id': 674, 'def': 'a kitchen utensil used for mashing (e.g. potatoes)', 'name': 'masher'}, {'frequency': 'f', 'synset': 'microwave.n.02', 'synonyms': ['microwave_oven'], 'id': 687, 'def': 'kitchen appliance that cooks food by passing an electromagnetic wave through it', 'name': 'microwave_oven'}, {'frequency': 'f', 'synset': 'monitor.n.04', 'synonyms': ['monitor_(computer_equipment) computer_monitor'], 'id': 698, 'def': 'a computer monitor', 'name': 'monitor_(computer_equipment) computer_monitor'}, {'frequency': 'f', 'synset': 'mug.n.04', 'synonyms': ['mug'], 'id': 708, 'def': 'with handle and usually cylindrical', 'name': 'mug'}, {'frequency': 'c', 'synset': 'packet.n.03', 'synonyms': ['packet'], 'id': 742, 'def': 'a small package or bundle', 'name': 'packet'}, {'frequency': 'f', 'synset': 'pan.n.01', 'synonyms': ['pan_(for_cooking)', 'cooking_pan'], 'id': 751, 'def': 'cooking utensil consisting of a wide metal vessel', 'name': 'pan_(for_cooking)'}, {'frequency': 'c', 'synset': 'peach.n.03', 'synonyms': ['peach'], 'id': 774, 'def': 'downy juicy fruit with sweet yellowish or whitish flesh', 'name': 'peach'}, {'frequency': 'f', 'synset': 'pear.n.01', 'synonyms': ['pear'], 'id': 776, 'def': 'sweet juicy gritty-textured fruit available in many varieties', 'name': 'pear'}, {'frequency': 'f', 'synset': 'pen.n.01', 'synonyms': ['pen'], 'id': 781, 'def': 'a writing implement with a point from which ink flows', 'name': 'pen'}, {'frequency': 'f', 'synset': 'pencil.n.01', 'synonyms': ['pencil'], 'id': 782, 'def': 'a thin cylindrical pointed writing implement made of wood and graphite', 'name': 'pencil'}, {'frequency': 'f', 'synset': 'pillow.n.01', 'synonyms': ['pillow'], 'id': 804, 'def': 'a cushion to support the head of a sleeping person', 'name': 'pillow'}, {'frequency': 'f', 'synset': 'plate.n.04', 'synonyms': ['plate'], 'id': 818, 'def': 'dish on which food is served or from which food is eaten', 'name': 'plate'}, {'frequency': 'c', 'synset': 'platter.n.01', 'synonyms': ['platter'], 'id': 819, 'def': 'a large shallow dish used for serving food', 'name': 'platter'}, {'frequency': 'f', 'synset': 'pot.n.01', 'synonyms': ['pot'], 'id': 836, 'def': 'metal or earthenware cooking vessel that is usually round and deep; often has a handle and lid', 'name': 'pot'}, {'frequency': 'f', 'synset': 'potato.n.01', 'synonyms': ['potato'], 'id': 838, 'def': 'an edible tuber native to South America', 'name': 'potato'}, {'frequency': 'f', 'synset': 'remote_control.n.01', 'synonyms': ['remote_control'], 'id': 881, 'def': 'a device that can be used to control a machine or apparatus from a distance', 'name': 'remote_control'}, {'frequency': 'f', 'synset': 'scissors.n.01', 'synonyms': ['scissors'], 'id': 923, 'def': 'a tool having two crossed pivoting blades with looped handles', 'name': 'scissors'}, {'frequency': 'f', 'synset': 'sink.n.01', 'synonyms': ['sink'], 'id': 961, 'def': 'plumbing fixture consisting of a water basin fixed to a wall or floor and having a drainpipe', 'name': 'sink'}, {'frequency': 'f', 'synset': 'sofa.n.01', 'synonyms': ['sofa', 'couch', 'lounge'], 'id': 982, 'def': 'an upholstered seat for more than one person', 'name': 'sofa'}, {'frequency': 'r', 'synset': 'soup_bowl.n.01', 'synonyms': ['soup_bowl'], 'id': 987, 'def': 'a bowl for serving soup', 'name': 'soup_bowl'}, {'frequency': 'f', 'synset': 'spoon.n.01', 'synonyms': ['spoon'], 'id': 1000, 'def': 'a piece of cutlery with a shallow bowl-shaped container and a handle', 'name': 'spoon'}, {'frequency': 'f', 'synset': 'stool.n.01', 'synonyms': ['stool'], 'id': 1018, 'def': 'a simple seat without a back or arms', 'name': 'stool'}, {'frequency': 'f', 'synset': 'strawberry.n.01', 'synonyms': ['strawberry'], 'id': 1025, 'def': 'sweet fleshy red fruit', 'name': 'strawberry'}, {'frequency': 'f', 'synset': 'table.n.02', 'synonyms': ['table'], 'id': 1050, 'def': 'a piece of furniture having a smooth flat top that is usually supported by one or more vertical legs', 'name': 'table'}, {'frequency': 'c', 'synset': 'table_lamp.n.01', 'synonyms': ['table_lamp'], 'id': 1051, 'def': 'a lamp that sits on a table', 'name': 'table_lamp'}, {'frequency': 'f', 'synset': 'tablecloth.n.01', 'synonyms': ['tablecloth'], 'id': 1052, 'def': 'a covering spread over a dining table', 'name': 'tablecloth'}, {'frequency': 'c', 'synset': 'teacup.n.02', 'synonyms': ['teacup'], 'id': 1068, 'def': 'a cup from which tea is drunk', 'name': 'teacup'}, {'frequency': 'c', 'synset': 'teakettle.n.01', 'synonyms': ['teakettle'], 'id': 1069, 'def': 'kettle for boiling water to make tea', 'name': 'teakettle'}, {'frequency': 'f', 'synset': 'teapot.n.01', 'synonyms': ['teapot'], 'id': 1070, 'def': 'pot for brewing tea; usually has a spout and handle', 'name': 'teapot'}, {'frequency': 'f', 'synset': 'toy.n.03', 'synonyms': ['toy'], 'id': 1110, 'def': 'a device regarded as providing amusement', 'name': 'toy'}, {'frequency': 'f', 'synset': 'vase.n.01', 'synonyms': ['vase'], 'id': 1139, 'def': 'an open jar of glass or porcelain used as an ornament or to hold flowers', 'name': 'vase'}, {'frequency': 'c', 'synset': 'wall_clock.n.01', 'synonyms': ['wall_clock'], 'id': 1154, 'def': 'a clock mounted on a wall', 'name': 'wall_clock'}, {'frequency': 'f', 'synset': 'wall_socket.n.01', 'synonyms': ['wall_socket', 'wall_plug', 'electric_outlet', 'electrical_outlet', 'outlet', 'electric_receptacle'], 'id': 1155, 'def': 'receptacle providing a place in a wiring system where current can be taken to run electrical devices', 'name': 'wall_socket'}, {'frequency': 'f', 'synset': 'water_bottle.n.01', 'synonyms': ['water_bottle'], 'id': 1162, 'def': 'a bottle for holding water', 'name': 'water_bottle'}, {'frequency': 'r', 'synset': 'water_heater.n.01', 'synonyms': ['water_heater', 'hot-water_heater'], 'id': 1165, 'def': 'a heater and storage tank to supply heated water', 'name': 'water_heater'}, {'frequency': 'f', 'synset': 'watermelon.n.02', 'synonyms': ['watermelon'], 'id': 1172, 'def': 'large oblong or roundish melon with a hard green rind and sweet watery red or occasionally yellowish pulp', 'name': 'watermelon'}, {'frequency': 'f', 'synset': 'wineglass.n.01', 'synonyms': ['wineglass'], 'id': 1190, 'def': 'a glass that has a stem and in which wine is served', 'name': 'wineglass'}, {'frequency': 'c', 'synset': 'wok.n.01', 'synonyms': ['wok'], 'id': 1192, 'def': 'pan with a convex bottom; used for frying in Chinese cooking', 'name': 'wok'}, {'frequency': 'c', 'synset': 'yogurt.n.01', 'synonyms': ['yogurt', 'yoghurt', 'yoghourt'], 'id': 1200, 'def': 'a custard-like food made from curdled milk', 'name': 'yogurt'}, {'frequency': 'c', 'synset': 'zucchini.n.02', 'synonyms': ['zucchini', 'courgette'], 'id': 1203, 'def': 'small cucumber-shaped vegetable marrow; typically dark green', 'name': 'zucchini'}]
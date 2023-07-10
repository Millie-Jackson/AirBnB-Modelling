import math # For pi and squareroot


cuboid_height = 25 / 7
cuboid_width = 25 / 2
cuboid_length = 35
cuboid = {"Height": cuboid_height, "Width": cuboid_width, "Length": cuboid_length}

cone_height = 10
cone_radius = 3
cone = {"Height": cone_height, "Radius": cone_radius}

shape = ["cuboid", "cone"]


def cuboid_volume():

    cuboid_volume = cuboid["Length"] * cuboid["Width"] * cuboid["Height"]
    
    return cuboid_volume

def cuboid_surface_area():
    
    cuboid_surface_area = ((cuboid["Length"] * cuboid["Width"])
                            + (cuboid["Length"] * cuboid["Height"])
                             + (cuboid["Height"] * cuboid["Width"])) * 2
    
    return cuboid_surface_area

def cone_volume():

    cone_volume = math.pi * cone["Radius"] ** 2 * (cone["Height"] / 3)
    
    return cone_volume

def cone_surface_area():

    cone_surface_area = math.pi * cone["Radius"] * (cone["Radius"] + math.sqrt(cone["Height"] ** 2 + cone["Radius"] ** 2))
    
    return cone_surface_area


print(f"The volume of {shape[0]} is {cuboid_volume()} and the surface area is {cuboid_surface_area()}")
print(f"The volume of {shape[1]} is {cone_volume()} and the surface area is {cone_surface_area()}")
import numpy as np
import pathlib
import typing
import sys

sys.path.append((pathlib.Path(__file__).parents[1].resolve() / 'src').as_posix())
import room_and_object_types
from room_and_object_types import *


# ===================================================================================================

# NORTH_WALL = 'north_wall'
# SOUTH_WALL = 'south_wall'
# EAST_WALL = 'east_wall'
# WEST_WALL = 'west_wall'
NAMED_WALLS = [NORTH_WALL, SOUTH_WALL, EAST_WALL, WEST_WALL]

# DOOR = 'door'

# RUG = 'rug'
RUG_ALT_NAME = 'HighFrictionTrigger'

# ROOM_CENTER = 'room_center'

BUILDING_TYPE = BUILDING

FEW_OBJECTS_ROOM = FEW.lower()
MEDIUM_OBJECTS_ROOM = MEDIUM.lower()
MANY_OBJECTS_ROOM = MANY.lower()
ROOMS = [FEW_OBJECTS_ROOM, MEDIUM_OBJECTS_ROOM, MANY_OBJECTS_ROOM]

OBJECTS_SHARED_IN_ALL_ROOMS_BY_TYPE = {
    ALARM_CLOCK: [
        "AlarmClock|-01.41|+00.60|+00.45"
    ],
    BED: [
        "Bed|-02.46|00.00|-00.57"
    ],
    BLINDS: [
        "Blinds|+02.29|+02.07|-03.18",
        "Blinds|-01.00|+02.07|-03.18"
    ],
    BOOK: [
        "Book|+02.83|+00.41|-00.01"
    ],
    CD: [
        "CD|+02.99|+00.79|-00.37"
    ],
    CELLPHONE: [
        "CellPhone|+02.96|+00.79|-00.93"
    ],
    CHAIR: [
        "Chair|+02.73|00.00|-01.21",
        "Chair|+02.83|00.00|00.00"
    ],
    CREDIT_CARD: [
        "CreditCard|+02.99|+00.79|-01.24"
    ],
    DESK: [
        "Desk|+03.14|00.00|-01.41"
    ],
    LAMP: [
        "DeskLamp|+03.13|+00.79|-00.64"
    ],
    DESKTOP: [
        "Desktop|+03.10|+00.79|-01.24"
    ],
    DRAWER: [
        "Drawer|-01.52|+00.14|+00.35",
        "Drawer|-01.52|+00.41|+00.35"
    ],
    FLOOR: [
        "Floor|+00.00|+00.00|+00.00"
    ],
    KEY_CHAIN: [
        "KeyChain|-01.62|+00.60|+00.41"
    ],
    LAPTOP: [
        "Laptop|+03.04|+00.79|-02.28"
    ],
    MAIN_LIGHT_SWITCH: [
        "LightSwitch|-00.14|+01.33|+00.60"
    ],
    MIRROR: [
        "Mirror|+00.45|+01.49|+00.62"
    ],
    MUG: [
        "Mug|+03.14|+00.79|-00.87"
    ],
    PEN: [
        "Pen|+03.02|+00.80|-01.85"
    ],
    PENCIL: [
        "Pencil|+03.07|+00.79|-01.79"
    ],
    PILLOW: [
        "Pillow|-02.45|+00.66|+00.10"
    ],
    POSTER: [
        "Poster|+03.40|+01.70|-00.79",
        "Poster|+03.40|+01.86|-01.98",
    ],
    SHELF: [
        "Shelf|+00.62|+01.01|-02.82",
        "Shelf|+00.62|+01.51|-02.82",
        # "Shelf|+03.13|+00.63|-00.56",
        # "Shelf|+03.13|+00.63|-02.27",
        "Shelf|-02.97|+01.16|-01.72",
        "Shelf|-02.97|+01.16|-02.47",
        "Shelf|-02.97|+01.53|-01.72",
        "Shelf|-02.97|+01.53|-02.47"
    ],
    SHELF_DESK: [
        "Shelf|+03.13|+00.63|-00.56",
        "Shelf|+03.13|+00.63|-02.27"
    ],
    SIDE_TABLE: [
        "SideTable|-01.52|00.00|+00.41"
    ],
    SLIDING_DOOR: [
        "Window|+02.28|+00.93|-03.18",
        "Window|-01.02|+00.93|-03.19"
    ],
    WATCH: [
        "Watch|+03.07|+00.79|-00.45"
    ],
    # "window": [
    #     "Window|+02.28|+00.93|-03.18",
    #     "Window|-01.02|+00.93|-03.19"
    # ],
    WALL: [
        NORTH_WALL,
        SOUTH_WALL,
        EAST_WALL,
        WEST_WALL
    ],
    NORTH_WALL: [NORTH_WALL],
    SOUTH_WALL: [SOUTH_WALL],
    EAST_WALL: [EAST_WALL],
    WEST_WALL: [WEST_WALL],
    DOOR: [DOOR],
    RUG: [RUG],
    ROOM_CENTER: [ROOM_CENTER],
}


OBJECTS_BY_ROOM_AND_TYPE = {
    FEW_OBJECTS_ROOM: {
        CUBE_BLOCK: [
            "CubeBlock|+00.20|+00.10|-02.83",
            "CubeBlock|+00.20|+00.29|-02.83",
            "CubeBlock|-00.02|+00.10|-02.83",
            "CubeBlock|-00.02|+00.28|-02.83",
            "CubeBlock|-00.23|+00.28|-02.83",
            "CubeBlock|-00.24|+00.10|-02.83"
        ],
        CURVED_WOODEN_RAMP: [
            "CurvedRamp|-03.05|00.00|-02.78"
        ],
        DODGEBALL: [
            "Dodgeball|-02.95|+01.29|-02.61",
            "Dodgeball|-02.97|+01.29|-02.28"
        ],
        HEXAGONAL_BIN: [
            "GarbageCan|+00.95|-00.03|-02.68"
        ],
    },

    MEDIUM_OBJECTS_ROOM: {
        BASKETBALL: [
            "BasketBall|-02.58|+00.12|-01.93"
        ],
        BEACHBALL: [
            "Beachball|-02.93|+00.17|-01.99"
        ],
        BRIDGE_BLOCK: [
            "BridgeBlock|+00.63|+01.10|-02.91",
            "BridgeBlock|+01.03|+01.11|-02.88"
        ],
        CUBE_BLOCK: [
            "CubeBlock|+00.50|+01.61|-02.91",
            "CubeBlock|+00.70|+01.61|-02.91"
        ],
        CYLINDRICAL_BLOCK: [
            "CylinderBlock|+00.93|+01.61|-02.89",
            "CylinderBlock|+01.13|+01.61|-02.89"
        ],
        DODGEBALL: [
            "Dodgeball|-02.60|+00.13|-02.18"
        ],
        DOGGIE_BED: [
            "DogBed|+02.30|00.00|-02.85"
        ],
        FLAT_BLOCK: [
            "FlatRectBlock|+00.23|+01.66|-02.88",
            "FlatRectBlock|+00.24|+01.57|-02.89"
        ],
        HEXAGONAL_BIN: [
            "GarbageCan|-02.79|-00.03|-02.67"
        ],
        TALL_CYLINDRICAL_BLOCK: [
            "LongCylinderBlock|+00.12|+01.19|-02.89",
            "LongCylinderBlock|+00.31|+01.19|-02.89"
        ],
        PYRAMID_BLOCK: [
            "PyramidBlock|+00.93|+01.78|-02.89",
            "PyramidBlock|+01.13|+01.78|-02.89"
        ],
        TRIANGULAR_RAMP: [
            "SmallSlide|-00.97|+00.20|-03.02"
        ],
        TEDDY_BEAR: [
            "TeddyBear|-02.60|+00.60|-00.42"
        ],
    },

    MANY_OBJECTS_ROOM: {
        BEACHBALL: [
            "Beachball|+02.29|+00.19|-02.88"
        ],
        BRIDGE_BLOCK: [
            "BridgeBlock|-02.92|+00.09|-02.52",
            "BridgeBlock|-02.92|+00.26|-02.52",
            "BridgeBlock|-02.92|+00.43|-02.52"
        ],
        CUBE_BLOCK: [
            "CubeBlock|-02.96|+01.26|-01.72",
            "CubeBlock|-02.97|+01.26|-01.94",
            "CubeBlock|-02.99|+01.26|-01.49"
        ],
        CURVED_WOODEN_RAMP: [
            "CurvedRamp|-00.25|00.00|-02.98"
        ],
        CYLINDRICAL_BLOCK: [
            "CylinderBlock|-02.95|+01.62|-01.95",
            "CylinderBlock|-02.97|+01.62|-01.50",
            "CylinderBlock|-03.02|+01.62|-01.73"
        ],
        DODGEBALL: [
            "Dodgeball|+00.19|+01.13|-02.80",
            "Dodgeball|+00.44|+01.13|-02.80",
            "Dodgeball|+00.70|+01.11|-02.80"
        ],
        DOGGIE_BED: [
            "DogBed|+02.24|00.00|-02.85"
        ],
        FLAT_BLOCK: [
            "FlatRectBlock|-02.93|+00.05|-02.84",
            "FlatRectBlock|-02.93|+00.15|-02.84",
            "FlatRectBlock|-02.93|+00.25|-02.84"
        ],
        HEXAGONAL_BIN: [
            "GarbageCan|+00.75|-00.03|-02.74"
        ],
        GOLFBALL: [
            "Golfball|+00.96|+01.04|-02.70",
            "Golfball|+01.05|+01.04|-02.70",
            "Golfball|+01.14|+01.04|-02.70"
        ],
        TALL_CYLINDRICAL_BLOCK: [
            "LongCylinderBlock|-02.82|+00.19|-02.09",
            "LongCylinderBlock|-02.93|+00.19|-01.93",
            "LongCylinderBlock|-02.94|+00.19|-02.24"
        ],
        PILLOW: [
            "Pillow|-02.03|+00.68|-00.42",
        ],
        PYRAMID_BLOCK: [
            "PyramidBlock|-02.95|+01.61|-02.20",
            "PyramidBlock|-02.95|+01.61|-02.66",
            "PyramidBlock|-02.96|+01.61|-02.44"
        ],
        TRIANGULAR_RAMP: [
            "SmallSlide|-00.81|+00.14|-03.10",
            "SmallSlide|-01.31|+00.14|-03.10"
        ],
        TALL_RECTANGULAR_BLOCK: [
            "TallRectBlock|-02.95|+02.05|-02.31",
            "TallRectBlock|-02.95|+02.05|-02.52",
            "TallRectBlock|-02.95|+02.05|-02.72"
        ],
        TEDDY_BEAR: [
            "TeddyBear|-01.93|+00.60|+00.07",
            "TeddyBear|-02.60|+00.60|-00.42"
        ],
        TRIANGLE_BLOCK: [
            "TriangleBlock|-02.92|+01.23|-02.23",
            "TriangleBlock|-02.94|+01.23|-02.46",
            "TriangleBlock|-02.95|+01.23|-02.69"
        ],
    }
}


SHARED_SPECIFIC_NAMED_OBJECTS = {
    BOTTOM_DRAWER: ["Drawer|-01.52|+00.14|+00.35"],
    BOTTOM_SHELF: ["Shelf|+00.62|+01.01|-02.82"],
    EAST_SLIDING_DOOR: ["Window|+02.28|+00.93|-03.18"],
    TOP_DRAWER: ["Drawer|-01.52|+00.41|+00.35"],
    TOP_SHELF: ["Shelf|+00.62|+01.51|-02.82"],
    WEST_SLIDING_DOOR: ["Window|-01.02|+00.93|-03.19"],
}


SPECIFIC_NAMED_OBJECTS_BY_ROOM = {
    FEW_OBJECTS_ROOM: {
        CUBE_BLOCK_BLUE: ['CubeBlock|-00.02|+00.28|-02.83', 'CubeBlock|-00.24|+00.10|-02.83'],
        CUBE_BLOCK_TAN: ['CubeBlock|+00.20|+00.10|-02.83', 'CubeBlock|-00.23|+00.28|-02.83'],
        CUBE_BLOCK_YELLOW: ['CubeBlock|+00.20|+00.29|-02.83', 'CubeBlock|-00.02|+00.10|-02.83'],
        DODGEBALL_BLUE: ['Dodgeball|-02.95|+01.29|-02.61'],
        DODGEBALL_PINK: ['Dodgeball|-02.97|+01.29|-02.28'],
    },
    MEDIUM_OBJECTS_ROOM: {
        BRIDGE_BLOCK_GREEN: ['BridgeBlock|+01.03|+01.11|-02.88'],
        BRIDGE_BLOCK_TAN: ['BridgeBlock|+00.63|+01.10|-02.91'],
        CUBE_BLOCK_BLUE: ['CubeBlock|+00.50|+01.61|-02.91'],
        CUBE_BLOCK_YELLOW: ['CubeBlock|+00.70|+01.61|-02.91'],
        CYLINDRICAL_BLOCK_BLUE: ['CylinderBlock|+01.13|+01.61|-02.89'],
        CYLINDRICAL_BLOCK_GREEN: ['CylinderBlock|+00.93|+01.61|-02.89'],
        DODGEBALL_RED: ['Dodgeball|-02.60|+00.13|-02.18'],
        FLAT_BLOCK_GRAY: ['FlatRectBlock|+00.24|+01.57|-02.89'],
        FLAT_BLOCK_YELLOW: ['FlatRectBlock|+00.23|+01.66|-02.88'],
        PYRAMID_BLOCK_RED: ['PyramidBlock|+00.93|+01.78|-02.89'],
        PYRAMID_BLOCK_YELLOW: ['PyramidBlock|+01.13|+01.78|-02.89'],
        TALL_CYLINDRICAL_BLOCK_TAN: ['LongCylinderBlock|+00.12|+01.19|-02.89'],
        TALL_CYLINDRICAL_BLOCK_YELLOW: ['LongCylinderBlock|+00.31|+01.19|-02.89'],
        TRIANGULAR_RAMP_TAN: ['SmallSlide|-00.97|+00.20|-03.02'],
    },
    MANY_OBJECTS_ROOM: {
        BRIDGE_BLOCK_GREEN: ['BridgeBlock|-02.92|+00.43|-02.52'],
        BRIDGE_BLOCK_PINK: ['BridgeBlock|-02.92|+00.26|-02.52'],
        BRIDGE_BLOCK_TAN: ['BridgeBlock|-02.92|+00.09|-02.52'],
        CUBE_BLOCK_BLUE: ['CubeBlock|-02.99|+01.26|-01.49'],
        CUBE_BLOCK_TAN: ['CubeBlock|-02.96|+01.26|-01.72'],
        CUBE_BLOCK_YELLOW: ['CubeBlock|-02.97|+01.26|-01.94'],
        CYLINDRICAL_BLOCK_BLUE: ['CylinderBlock|-03.02|+01.62|-01.73'],
        CYLINDRICAL_BLOCK_GREEN: ['CylinderBlock|-02.97|+01.62|-01.50'],
        CYLINDRICAL_BLOCK_TAN: ['CylinderBlock|-02.95|+01.62|-01.95'],
        FLAT_BLOCK_GRAY: ['FlatRectBlock|-02.93|+00.15|-02.84'],
        FLAT_BLOCK_TAN: ['FlatRectBlock|-02.93|+00.25|-02.84'],
        FLAT_BLOCK_YELLOW: ['FlatRectBlock|-02.93|+00.05|-02.84'],
        DODGEBALL_BLUE: ['Dodgeball|+00.19|+01.13|-02.80'],
        DODGEBALL_PINK: ['Dodgeball|+00.70|+01.11|-02.80'],
        DODGEBALL_RED: ['Dodgeball|+00.44|+01.13|-02.80'],
        GOLFBALL_GREEN: ['Golfball|+01.05|+01.04|-02.70'],
        GOLFBALL_ORANGE: [ 'Golfball|+00.96|+01.04|-02.70'],
        GOLFBALL_WHITE: ['Golfball|+01.14|+01.04|-02.70'],
        PYRAMID_BLOCK_BLUE: ['PyramidBlock|-02.95|+01.61|-02.20'],
        PYRAMID_BLOCK_RED: ['PyramidBlock|-02.96|+01.61|-02.44'],
        PYRAMID_BLOCK_YELLOW: ['PyramidBlock|-02.95|+01.61|-02.66'],
        TALL_CYLINDRICAL_BLOCK_GREEN: ['LongCylinderBlock|-02.82|+00.19|-02.09'],
        TALL_CYLINDRICAL_BLOCK_TAN: ['LongCylinderBlock|-02.93|+00.19|-01.93'],
        TALL_CYLINDRICAL_BLOCK_YELLOW: ['LongCylinderBlock|-02.94|+00.19|-02.24'],
        TALL_RECTANGULAR_BLOCK_BLUE: ['TallRectBlock|-02.95|+02.05|-02.52'],
        TALL_RECTANGULAR_BLOCK_GREEN: ['TallRectBlock|-02.95|+02.05|-02.72'],
        TALL_RECTANGULAR_BLOCK_TAN: ['TallRectBlock|-02.95|+02.05|-02.31'],
        TRIANGLE_BLOCK_BLUE: ['TriangleBlock|-02.92|+01.23|-02.23'],
        TRIANGLE_BLOCK_GREEN: ['TriangleBlock|-02.95|+01.23|-02.69'],
        TRIANGLE_BLOCK_TAN: ['TriangleBlock|-02.94|+01.23|-02.46'],
        TRIANGULAR_RAMP_GREEN: ['SmallSlide|-00.81|+00.14|-03.10'],
        TRIANGULAR_RAMP_TAN: ['SmallSlide|-01.31|+00.14|-03.10'],
    }
}


for room_specific_named_objects in SPECIFIC_NAMED_OBJECTS_BY_ROOM.values():
    room_specific_named_objects.update(SHARED_SPECIFIC_NAMED_OBJECTS)


OBJECT_ID_TO_SPECIFIC_NAME_BY_ROOM = {
    room: {specific_object: name}
    for room, room_specific_objects in SPECIFIC_NAMED_OBJECTS_BY_ROOM.items()
    for name, specific_objects in room_specific_objects.items()
    for specific_object in specific_objects
}


# Add the shared objects to each of the room domains
for room_type in OBJECTS_BY_ROOM_AND_TYPE:
    for object_type, object_list in OBJECTS_SHARED_IN_ALL_ROOMS_BY_TYPE.items():
        if object_type in OBJECTS_BY_ROOM_AND_TYPE[room_type]:
            OBJECTS_BY_ROOM_AND_TYPE[room_type][object_type].extend(object_list)
        else:
            OBJECTS_BY_ROOM_AND_TYPE[room_type][object_type] = object_list[:]



# A list of all objects that can be referred to directly as variables inside of a game
# NAMED_OBJECTS = ["agent", "bed", "desk", "door", "floor", "main_light_switch", "side_table"]  # added the keys of the Pseudo Objects later
# NAMED_OBJECTS = ["agent", "bed", "desk", "desktop", "door", "floor", "main_light_switch", "side_table"]  # added the keys of the Pseudo Objects later
NAMED_OBJECTS = room_and_object_types.DIRECTLY_REFERRED_OBJECTS[:]

# A list of all the colors, which as a hack will also be mapped to themselves, as though they were named objects
COLORS = list(room_and_object_types.CATEGORIES_TO_TYPES[room_and_object_types.COLORS])
if COLOR in COLORS: COLORS.remove(COLOR)
ORIENTATIONS = list(room_and_object_types.CATEGORIES_TO_TYPES[room_and_object_types.ORIENTATIONS])
if ORIENTATION in ORIENTATIONS: ORIENTATIONS.remove(ORIENTATION)
SIDES = list(room_and_object_types.CATEGORIES_TO_TYPES[room_and_object_types.SIDES])
if SIDE in SIDES: SIDES.remove(SIDE)


NON_OBJECT_TYPES = [COLOR, ORIENTATION, SIDE]
ALL_NON_OBJECT_TYPES = set(COLORS + ORIENTATIONS + SIDES + NON_OBJECT_TYPES)

# Meta types compile objects from many other types (e.g. both beachballs and dodgeballs are balls)
META_TYPES = {BALL: [BEACHBALL, BASKETBALL, DODGEBALL, GOLFBALL],
              BLOCK: [BRIDGE_BLOCK, CUBE_BLOCK, CYLINDRICAL_BLOCK, FLAT_BLOCK, PYRAMID_BLOCK,
                      TALL_CYLINDRICAL_BLOCK, TALL_RECTANGULAR_BLOCK, TRIANGLE_BLOCK],
              RAMP: [CURVED_WOODEN_RAMP, TRIANGULAR_RAMP],
              COLOR: COLORS,
              ORIENTATION: ORIENTATIONS,
              SIDE: SIDES}

TYPES_TO_META_TYPE = {sub_type: meta_type for meta_type, sub_types in META_TYPES.items() for sub_type in sub_types}

# List of types that are *not* included in "game_object" -- easier than listing out all the types that are
# GAME_OBJECT_EXCLUDED_TYPES = ["bed", "blinds", "desk", "desktop", "lamp", "drawer", "floor", "main_light_switch", "mirror",
#                               "poster", "shelf", "side_table", "window", "wall", "agent"]

GAME_OBJECT_EXCLUDED_TYPES = list(CATEGORIES_TO_TYPES[FURNITURE])
GAME_OBJECT_EXCLUDED_TYPES.extend(list(CATEGORIES_TO_TYPES[ROOM_FEATURES]))
GAME_OBJECT_EXCLUDED_TYPES.append(AGENT)
GAME_OBJECT_EXCLUDED_TYPES.append(BUILDING)  # the fictional building type should not be included in game objects
GAME_OBJECT_EXCLUDED_TYPES.extend(COLORS)
GAME_OBJECT_EXCLUDED_TYPES.extend(ORIENTATIONS)
GAME_OBJECT_EXCLUDED_TYPES.extend(SIDES)
GAME_OBJECT_EXCLUDED_TYPES.extend(META_TYPES.keys())


ELIGILBLE_IN_OBJECT_TYPES = set(CATEGORIES_TO_TYPES[RECEPTACLES])
ELIGILBLE_IN_OBJECT_TYPES.add(MUG)

ELIGILBLE_IN_OBJECT_IDS = set()
for room_type_to_object_ids in OBJECTS_BY_ROOM_AND_TYPE.values():
    for object_type in ELIGILBLE_IN_OBJECT_TYPES:
        if object_type in room_type_to_object_ids:
            ELIGILBLE_IN_OBJECT_IDS.update(room_type_to_object_ids[object_type])

for room_type_to_object_ids in SPECIFIC_NAMED_OBJECTS_BY_ROOM.values():
    for object_type in ELIGILBLE_IN_OBJECT_TYPES:
        if object_type in room_type_to_object_ids:
            ELIGILBLE_IN_OBJECT_IDS.update(room_type_to_object_ids[object_type])


# Update the dictionary by mapping the agent and colors to themselves and grouping objects into meta types. Also group all
# of the objects that count as a "game_object"
for domain in ROOMS:
    OBJECTS_BY_ROOM_AND_TYPE[domain][AGENT] = [AGENT]
    for collection in COLORS, ORIENTATIONS, SIDES:
        OBJECTS_BY_ROOM_AND_TYPE[domain].update({item: [item] for item in collection})

    for meta_type, object_types in META_TYPES.items():
        OBJECTS_BY_ROOM_AND_TYPE[domain][meta_type] = []
        for object_type in object_types:
            if object_type in OBJECTS_BY_ROOM_AND_TYPE[domain]:
                OBJECTS_BY_ROOM_AND_TYPE[domain][meta_type] += OBJECTS_BY_ROOM_AND_TYPE[domain][object_type]

    OBJECTS_BY_ROOM_AND_TYPE[domain][GAME_OBJECT] = []
    for object_type in OBJECTS_BY_ROOM_AND_TYPE[domain]:
        if object_type not in GAME_OBJECT_EXCLUDED_TYPES:
            OBJECTS_BY_ROOM_AND_TYPE[domain][GAME_OBJECT] += OBJECTS_BY_ROOM_AND_TYPE[domain][object_type]

    OBJECTS_BY_ROOM_AND_TYPE[domain][GAME_OBJECT] = sorted(list(set(OBJECTS_BY_ROOM_AND_TYPE[domain][GAME_OBJECT])))

# A list of all object types (including meta types)
ALL_OBJECT_TYPES = list(set(list(OBJECTS_BY_ROOM_AND_TYPE[FEW_OBJECTS_ROOM].keys()) + \
                            list(OBJECTS_BY_ROOM_AND_TYPE[MEDIUM_OBJECTS_ROOM].keys()) + \
                            list(OBJECTS_BY_ROOM_AND_TYPE[MANY_OBJECTS_ROOM].keys())))


# Accounting for weird glitches/objects clipping
ON_EXCLUDED_OBJECT_TYPES = set([
    DOOR,
    MIRROR,
    WALL,
    NORTH_WALL,
    SOUTH_WALL,
    EAST_WALL,
    WEST_WALL,
    WEST_SLIDING_DOOR,
    SLIDING_DOOR,
    EAST_SLIDING_DOOR,
    WEST_SLIDING_DOOR,
    BLINDS,
    MAIN_LIGHT_SWITCH,
    POSTER,
])

# ===================================================================================================

class PseudoObject:
    identifiers: typing.List[str]
    object_id: str
    object_type: str
    name: str
    bbox_center: np.ndarray
    bbox_extents: np.ndarray
    position: np.ndarray
    rotation: np.ndarray

    def __init__(self, object_id: str, object_type: str, name: str, position: np.ndarray,
        extents: np.ndarray, rotation: np.ndarray, alternative_names: typing.Optional[typing.List[str]] = None):

        self.object_id = object_id
        self.object_type = object_type
        self.name = name
        self.position = position
        self.bbox_center = position
        self.bbox_extents = extents
        self.rotation = rotation
        if alternative_names is None:
            alternative_names = []
        self.identifiers = [self.object_id, self.object_type] + alternative_names


    def __getitem__(self, item: typing.Any) -> typing.Any:
        if item in self.__dict__:
            return self.__dict__[item]

        raise ValueError(f'PsuedoObjects have only a name and an id, not a {item}')

    def __contains__(self, item):
        return item in self.__dict__


WALL_ID = 'FP326:StandardWallSize.021'
WALL_TYPE = WALL

DOOR_ID = 'FP326:StandardDoor1.019'
DOOR_TYPE = DOOR

RUG_ID = 'FP326:Rug'
RUG_TYPE = RUG



# TODO: I think the ceiling also might be one, and maybe the floor or some other fixed furniture?
# Wall width is about 0.15, ceiling height is about 2.7
UNITY_PSEUDO_OBJECTS = {
        NORTH_WALL: PseudoObject(WALL_ID, WALL_TYPE, NORTH_WALL, position=np.array([0.1875, 1.35, 0.675]), extents=np.array([3.2875, 1.35, 0.075]), rotation=np.zeros(3)),           # has the door
        SOUTH_WALL: PseudoObject(WALL_ID, WALL_TYPE, SOUTH_WALL, position=np.array([0.1875, 1.35, -3.1]), extents=np.array([3.2875, 1.35, 0.075]), rotation=np.zeros(3)),            # has the window
        EAST_WALL: PseudoObject(WALL_ID, WALL_TYPE, EAST_WALL, position=np.array([3.475, 1.35, 1.2125]), extents=np.array([0.075, 1.35, 1.8875]), rotation=np.array([0, 90, 0])),   # has the desk
        WEST_WALL: PseudoObject(WALL_ID, WALL_TYPE, WEST_WALL, position=np.array([-3.1, 1.35, -1.2125]), extents=np.array([0.075, 1.35, 1.8875]), rotation=np.array([0, 90, 0])),   # has the bed

        ROOM_CENTER: PseudoObject(ROOM_CENTER, ROOM_CENTER, ROOM_CENTER, position=np.array([0.1875, 0, -1.2125]), extents=np.array([0.1, 0, 0.1]), rotation=np.zeros(3)),

        DOOR: PseudoObject(DOOR_ID, DOOR_TYPE, DOOR, position=np.array([0.448, 1.35, 0.675]), extents=np.array([0.423, 1.35, 0.075]), rotation=np.zeros(3)),

        RUG: PseudoObject(RUG_ID, RUG_TYPE, RUG, position=np.array([-1.485, 0.05, -1.673]), extents=np.array([1.338, 0.05, 0.926]), rotation=np.array([0, 90, 0]), alternative_names=[RUG_ALT_NAME]),
}

NAMED_OBJECTS.extend(UNITY_PSEUDO_OBJECTS.keys())

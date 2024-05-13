from collections import defaultdict
import argparse

from pandas import options

# -- CATEGOTIES --

AGENT = 'agent'
ANY_OBJECT = 'any_object'
BALLS = 'balls'
BLOCKS = 'blocks'
BUILDING = 'building'
COLORS = 'colors'
EMPTY_OBJECT = 'empty_object'
FURNITURE = 'furniture'
LARGE_OBJECTS = 'large_objects'
MEDIUM_OBJECTS = 'medium_objects'
ORIENTATIONS = 'orientations'
RAMPS = 'ramps'
RECEPTACLES = 'receptacles'
ROOM_FEATURES = 'room_features'
SIDES = 'sides'
SMALL_OBJECTS = 'small_objects'


# -- OBJECT TYPES --

# AGENT
AGENT = 'agent'

# ANY_OBJECT
GAME_OBJECT = 'game_object'

# BALLS
BALL = 'ball'
BASKETBALL = 'basketball'
BEACHBALL = 'beachball'
DODGEBALL = 'dodgeball'
DODGEBALL_BLUE = 'dodgeball_blue'
DODGEBALL_PINK = 'dodgeball_pink'
DODGEBALL_RED = 'dodgeball_red'
GOLFBALL = 'golfball'
GOLFBALL_GREEN = 'golfball_green'
GOLFBALL_ORANGE = 'golfball_orange'
GOLFBALL_WHITE = 'golfball_white'

# BLOCKS
BLOCK = 'block'
BRIDGE_BLOCK = 'bridge_block'
BRIDGE_BLOCK_GREEN = 'bridge_block_green'
BRIDGE_BLOCK_PINK = 'bridge_block_pink'
BRIDGE_BLOCK_TAN = 'bridge_block_tan'
CUBE_BLOCK = 'cube_block'
CUBE_BLOCK_BLUE = 'cube_block_blue'
CUBE_BLOCK_TAN = 'cube_block_tan'
CUBE_BLOCK_YELLOW = 'cube_block_yellow'
CYLINDRICAL_BLOCK = 'cylindrical_block'
CYLINDRICAL_BLOCK_BLUE = 'cylindrical_block_blue'
CYLINDRICAL_BLOCK_TAN = 'cylindrical_block_tan'
CYLINDRICAL_BLOCK_GREEN = 'cylindrical_block_green'
FLAT_BLOCK = 'flat_block'
FLAT_BLOCK_GRAY = 'flat_block_gray'
FLAT_BLOCK_TAN = 'flat_block_tan'
FLAT_BLOCK_YELLOW = 'flat_block_yellow'
PYRAMID_BLOCK = 'pyramid_block'
PYRAMID_BLOCK_BLUE = 'pyramid_block_blue'
PYRAMID_BLOCK_RED = 'pyramid_block_red'
PYRAMID_BLOCK_YELLOW = 'pyramid_block_yellow'
TALL_CYLINDRICAL_BLOCK = 'tall_cylindrical_block'
TALL_CYLINDRICAL_BLOCK_GREEN = 'tall_cylindrical_block_green'
TALL_CYLINDRICAL_BLOCK_TAN = 'tall_cylindrical_block_tan'
TALL_CYLINDRICAL_BLOCK_YELLOW = 'tall_cylindrical_block_yellow'
TALL_RECTANGULAR_BLOCK = 'tall_rectangular_block'
TALL_RECTANGULAR_BLOCK_BLUE = 'tall_rectangular_block_blue'
TALL_RECTANGULAR_BLOCK_GREEN = 'tall_rectangular_block_green'
TALL_RECTANGULAR_BLOCK_TAN = 'tall_rectangular_block_tan'
TRIANGLE_BLOCK = 'triangle_block'
TRIANGLE_BLOCK_BLUE = 'triangle_block_blue'
TRIANGLE_BLOCK_GREEN = 'triangle_block_green'
TRIANGLE_BLOCK_TAN = 'triangle_block_tan'


# COLORS
COLOR = 'color'
BLUE = 'blue'
BROWN = 'brown'
GRAY = 'gray'
GREEN = 'green'
ORANGE = 'orange'
PINK = 'pink'
PURPLE = 'purple'
RED = 'red'
TAN = 'tan'
WHITE = 'white'
YELLOW = 'yellow'

# EMPTY_OBJECT
EMPTY_OBJECT_OBJ = ''

# FURNITURE
BED = 'bed'
BLINDS = 'blinds'
DESK = 'desk'
SHELF_DESK = 'shelf_desk'
DRAWER = 'drawer'
MAIN_LIGHT_SWITCH = 'main_light_switch'
DESKTOP = 'desktop'
BOTTOM_DRAWER = 'bottom_drawer'
TOP_DRAWER = 'top_drawer'
SIDE_TABLE = 'side_table'

# BUILDING
BUILDING = 'building'

# LARGE_OBJECTS
BOOK = 'book'
CHAIR = 'chair'
LAPTOP = 'laptop'
PILLOW = 'pillow'
TEDDY_BEAR = 'teddy_bear'

# RAMPS
RAMP = 'ramp'
CURVED_WOODEN_RAMP = 'curved_wooden_ramp'
TRIANGULAR_RAMP = 'triangular_ramp'
TRIANGULAR_RAMP_GREEN = 'triangular_ramp_green'
TRIANGULAR_RAMP_TAN = 'triangular_ramp_tan'

# RECEPTACLES
DOGGIE_BED = 'doggie_bed'
HEXAGONAL_BIN = 'hexagonal_bin'

# ROOM_FEATURES
DOOR = 'door'
FLOOR = 'floor'
MIRROR = 'mirror'
POSTER = 'poster'
ROOM_CENTER = 'room_center'
RUG = 'rug'
SHELF = 'shelf'
BOTTOM_SHELF = 'bottom_shelf'
TOP_SHELF = 'top_shelf'
SLIDING_DOOR = 'sliding_door'
EAST_SLIDING_DOOR = 'east_sliding_door'
WEST_SLIDING_DOOR = 'west_sliding_door'
WALL = 'wall'
EAST_WALL = 'east_wall'
NORTH_WALL = 'north_wall'
SOUTH_WALL = 'south_wall'
WEST_WALL = 'west_wall'

# SMALL_OBJECTS
ALARM_CLOCK = 'alarm_clock'
CD = 'cd'
CELLPHONE = 'cellphone'
CREDIT_CARD = 'credit_card'
KEY_CHAIN = 'key_chain'
LAMP = 'lamp'
MUG = 'mug'
PEN = 'pen'
PENCIL = 'pencil'
WATCH = 'watch'

# SIDES
SIDE = 'side'
BACK = 'back'
FRONT = 'front'
LEFT = 'left'
RIGHT = 'right'

#  ORIENTATIONS
ORIENTATION = 'orientation'
DIAGONAL = 'diagonal'
SIDEWAYS = 'sideways'
UPRIGHT = 'upright'
UPSIDE_DOWN = 'upside_down'


# -- MAPPINGS --

SPECIFICALLY_REFERRED_OBJECTS = [
    BOTTOM_DRAWER, BOTTOM_SHELF,
    EAST_SLIDING_DOOR, EAST_WALL,
    NORTH_WALL, SOUTH_WALL,
    TOP_DRAWER, TOP_SHELF,
    WEST_SLIDING_DOOR, WEST_WALL
]

DIRECTLY_REFERRED_OBJECTS = [
    AGENT, BED,
    DESK, DOOR,
    FLOOR, MAIN_LIGHT_SWITCH, MIRROR,
    ROOM_CENTER, RUG, SIDE_TABLE,
] + SPECIFICALLY_REFERRED_OBJECTS

# -- MAPPINGS --

CATEGORIES_TO_TYPES = {
    AGENT: (
        AGENT,
    ),
    ANY_OBJECT: (
        GAME_OBJECT,
    ),
    BALLS: (
        BALL, BASKETBALL, BEACHBALL,
        DODGEBALL, DODGEBALL_BLUE, DODGEBALL_PINK, DODGEBALL_RED,
        GOLFBALL, GOLFBALL_GREEN, GOLFBALL_ORANGE, GOLFBALL_WHITE,
    ),
    BLOCKS: (
        BLOCK, BRIDGE_BLOCK, BRIDGE_BLOCK_GREEN, BRIDGE_BLOCK_PINK, BRIDGE_BLOCK_TAN,
        CUBE_BLOCK, CUBE_BLOCK_BLUE, CUBE_BLOCK_TAN, CUBE_BLOCK_YELLOW,
        CYLINDRICAL_BLOCK, CYLINDRICAL_BLOCK_BLUE, CYLINDRICAL_BLOCK_TAN, CYLINDRICAL_BLOCK_GREEN,
        FLAT_BLOCK, FLAT_BLOCK_GRAY, FLAT_BLOCK_TAN, FLAT_BLOCK_YELLOW,
        PYRAMID_BLOCK, PYRAMID_BLOCK_BLUE, PYRAMID_BLOCK_RED, PYRAMID_BLOCK_YELLOW,
        TALL_CYLINDRICAL_BLOCK, TALL_CYLINDRICAL_BLOCK_GREEN, TALL_CYLINDRICAL_BLOCK_TAN, TALL_CYLINDRICAL_BLOCK_YELLOW,
        TALL_RECTANGULAR_BLOCK, TALL_RECTANGULAR_BLOCK_BLUE, TALL_RECTANGULAR_BLOCK_GREEN, TALL_RECTANGULAR_BLOCK_TAN,
        TRIANGLE_BLOCK, TRIANGLE_BLOCK_BLUE, TRIANGLE_BLOCK_GREEN, TRIANGLE_BLOCK_TAN
    ),
    COLORS: (
        COLOR, BLUE, BROWN, GRAY, GREEN,
        ORANGE, PINK, PURPLE, RED,
        TAN, WHITE, YELLOW,
    ),
    EMPTY_OBJECT: (
        EMPTY_OBJECT_OBJ,
    ),
    FURNITURE: (
        BED, BLINDS, DESK, SHELF_DESK,
        MAIN_LIGHT_SWITCH, DESKTOP, SIDE_TABLE,
    ),
    BUILDING: (
        BUILDING,
    ),
    LARGE_OBJECTS: (
        BOOK, CHAIR, LAPTOP, PILLOW, TEDDY_BEAR,
    ),
    ORIENTATIONS: (
        ORIENTATION, DIAGONAL, SIDEWAYS, UPRIGHT, UPSIDE_DOWN,
    ),
    RAMPS: (
        RAMP, CURVED_WOODEN_RAMP, TRIANGULAR_RAMP, TRIANGULAR_RAMP_GREEN, TRIANGULAR_RAMP_TAN,
    ),
    RECEPTACLES: (
        DOGGIE_BED, HEXAGONAL_BIN, DRAWER, BOTTOM_DRAWER, TOP_DRAWER,
    ),
    ROOM_FEATURES: (
        DOOR, FLOOR, MIRROR, POSTER, RUG, ROOM_CENTER, SHELF, BOTTOM_SHELF,
        TOP_SHELF, SLIDING_DOOR, EAST_SLIDING_DOOR, WEST_SLIDING_DOOR,
        WALL, EAST_WALL, NORTH_WALL, SOUTH_WALL, WEST_WALL,
    ),
    SIDES: (
        SIDE, BACK, FRONT, LEFT, RIGHT,
    ),
    SMALL_OBJECTS: (
        ALARM_CLOCK, CD, CELLPHONE, CREDIT_CARD, KEY_CHAIN,
        LAMP, MUG, PEN, PENCIL, WATCH,
    ),
}

TYPES_TO_CATEGORIES = {type_name: cat for cat, type_names in CATEGORIES_TO_TYPES.items() for type_name in type_names}


FEW = 'Few'
MEDIUM = 'Medium'
MANY = 'Many'
ROOM_NAMES = (FEW, MEDIUM, MANY)

FULL_ROOMS_UNCATEGORIZED_OBJECTS = 'uncategorized_objects'

ABSTRACT_OBJECTS_IN_ALL_ROOMS = {
    AGENT: 1,
    GAME_OBJECT: 1,
    BUILDING: 1,
    EMPTY_OBJECT_OBJ: 1,
    ROOM_CENTER: 1,
    **{type_name: 1 for type_name in CATEGORIES_TO_TYPES[COLORS]},
    **{type_name: 1 for type_name in CATEGORIES_TO_TYPES[ORIENTATIONS]},
    **{type_name: 1 for type_name in CATEGORIES_TO_TYPES[SIDES]},
}

FULL_ROOMS_TO_OBJECTS = {
    FEW: {
        FULL_ROOMS_UNCATEGORIZED_OBJECTS: ABSTRACT_OBJECTS_IN_ALL_ROOMS,
        BALLS: {
            BALL: 1,
            DODGEBALL: {BLUE: 1, PINK: 1},
        },
        BLOCKS: {
            BLOCK: 1,
            CUBE_BLOCK: {BLUE: 2, TAN: 2, YELLOW: 2, },
        },
        FURNITURE: {
            BED: 1,
            BLINDS: 2,
            CHAIR: 2,
            DESK: 1,
            SHELF_DESK: 1,
            DESKTOP: 1,
            DRAWER: 1,
            MAIN_LIGHT_SWITCH: 1,
            SIDE_TABLE: 1,
        },
        LARGE_OBJECTS: {
            BOOK: 1,
            LAPTOP: 1,
            PILLOW: 1,
        },
        RAMPS: {
            RAMP: 1,
            CURVED_WOODEN_RAMP: 1,
        },
        RECEPTACLES: {
            BOTTOM_DRAWER: 1,
            HEXAGONAL_BIN: 1,
            TOP_DRAWER: 1,
        },
        ROOM_FEATURES: {
            BOTTOM_SHELF: 1,
            DOOR: 1,
            EAST_SLIDING_DOOR: 1,
            EAST_WALL: 1,
            FLOOR: 1,
            MIRROR: 1,
            NORTH_WALL: 1,
            POSTER: 2,
            ROOM_CENTER: 1,
            RUG: 1,
            SHELF: 4,
            SLIDING_DOOR: 2,
            SOUTH_WALL: 1,
            TOP_SHELF: 1,
            WALL: 1,
            WEST_SLIDING_DOOR: 1,
            WEST_WALL: 1,
        },
        SMALL_OBJECTS: {
            ALARM_CLOCK: 1,
            CD: 1,
            CELLPHONE: 1,
            CREDIT_CARD: 1,
            KEY_CHAIN: 1,
            LAMP: 1,
            MUG: 1,
            PEN: 1,
            PENCIL: 1,
            WATCH: 1,
        },
    },
    MEDIUM: {
        FULL_ROOMS_UNCATEGORIZED_OBJECTS: ABSTRACT_OBJECTS_IN_ALL_ROOMS,
        BALLS: {
            BALL: 1,
            BASKETBALL: 1,
            BEACHBALL: 1,
            DODGEBALL: {RED: 1, },
        },
        BLOCKS: {
            BLOCK: 1,
            BRIDGE_BLOCK: {GREEN: 1, TAN: 1},
            CUBE_BLOCK: {BLUE: 1, YELLOW: 1},
            CYLINDRICAL_BLOCK: {BLUE: 1, GREEN: 1},
            FLAT_BLOCK: {GRAY: 1, YELLOW: 1},
            PYRAMID_BLOCK: {RED: 1, YELLOW: 1},
            TALL_CYLINDRICAL_BLOCK: {TAN: 1, YELLOW: 1},
        },
        FURNITURE: {
            BED: 1,
            BLINDS: 2,
            CHAIR: 2,
            DESK: 1,
            SHELF_DESK: 1,
            DESKTOP: 1,
            DRAWER: 1,
            MAIN_LIGHT_SWITCH: 1,
            SIDE_TABLE: 1,
        },
        LARGE_OBJECTS: {
            BOOK: 1,
            LAPTOP: 1,
            PILLOW: 1,
            TEDDY_BEAR: 1,
        },
        RAMPS: {
            RAMP: 1,
            TRIANGULAR_RAMP: 1,
            TRIANGULAR_RAMP_TAN: 1,
        },
        RECEPTACLES: {
            BOTTOM_DRAWER: 1,
            DOGGIE_BED: 1,
            HEXAGONAL_BIN: 1,
            TOP_DRAWER: 1,
        },
        ROOM_FEATURES: {
            BOTTOM_SHELF: 1,
            DOOR: 1,
            EAST_SLIDING_DOOR: 1,
            EAST_WALL: 1,
            FLOOR: 1,
            MIRROR: 1,
            NORTH_WALL: 1,
            ROOM_CENTER: 1,
            RUG: 1,
            POSTER: 2,
            SHELF: 2,
            SLIDING_DOOR: 2,
            SOUTH_WALL: 1,
            TOP_SHELF: 1,
            WALL: 1,
            WEST_WALL: 1,
            WEST_SLIDING_DOOR: 1,
        },
        SMALL_OBJECTS: {
            ALARM_CLOCK: 1,
            CD: 1,
            CELLPHONE: 1,
            CREDIT_CARD: 1,
            KEY_CHAIN: 1,
            LAMP: 1,
            MUG: 1,
            PEN: 1,
            PENCIL: 1,
            WATCH: 1,
        },
    },
    MANY: {
        FULL_ROOMS_UNCATEGORIZED_OBJECTS: ABSTRACT_OBJECTS_IN_ALL_ROOMS,
        BALLS: {
            BALL: 1,
            BEACHBALL: 1,
            DODGEBALL: {BLUE: 1, PINK: 1, RED: 1, },
            GOLFBALL: {GREEN: 1, ORANGE: 1, WHITE: 1, },
        },
        BLOCKS: {
            BLOCK: 1,
            BRIDGE_BLOCK: {GREEN: 1, PINK: 1, TAN: 1, },
            CUBE_BLOCK: {BLUE: 1, TAN: 1, YELLOW: 1, },
            CYLINDRICAL_BLOCK: {BLUE: 1, GREEN: 1, TAN: 1},
            FLAT_BLOCK: {GRAY: 1, TAN: 1, YELLOW: 1},
            PYRAMID_BLOCK: {BLUE: 1, RED: 1, YELLOW: 1, },
            TALL_CYLINDRICAL_BLOCK: {GREEN: 1, TAN: 1, YELLOW: 1},
            TALL_RECTANGULAR_BLOCK: {BLUE: 1, GREEN: 1, TAN: 1, },
            TRIANGLE_BLOCK: {BLUE: 1, GREEN: 1, TAN: 1, },
        },
        FURNITURE: {
            BED: 1,
            BLINDS: 2,
            CHAIR: 2,
            DESK: 1,
            SHELF_DESK: 1,
            DESKTOP: 1,
            DRAWER: 1,
            MAIN_LIGHT_SWITCH: 1,
            SIDE_TABLE: 1,

        },
        LARGE_OBJECTS: {
            BOOK: 1,
            LAPTOP: 1,
            PILLOW: 2,
            TEDDY_BEAR: 2,
        },
        RAMPS: {
            RAMP: 1,
            CURVED_WOODEN_RAMP: 1,
            TRIANGULAR_RAMP: {GREEN: 1, TAN: 1},
        },
        RECEPTACLES: {
            BOTTOM_DRAWER: 1,
            HEXAGONAL_BIN: 1,
            TOP_DRAWER: 1,
        },
        ROOM_FEATURES: {
            BOTTOM_SHELF: 1,
            DOOR: 1,
            EAST_SLIDING_DOOR: 1,
            EAST_WALL: 1,
            FLOOR: 1,
            MIRROR: 1,
            NORTH_WALL: 1,
            POSTER: 2,
            ROOM_CENTER: 1,
            RUG: 1,
            SHELF: 2,
            SLIDING_DOOR: 2,
            SOUTH_WALL: 1,
            TOP_SHELF: 1,
            WALL: 1,
            WEST_WALL: 1,
            WEST_SLIDING_DOOR: 1,
        },
        SMALL_OBJECTS: {
            ALARM_CLOCK: 1,
            CD: 1,
            CELLPHONE: 1,
            CREDIT_CARD: 1,
            KEY_CHAIN: 1,
            LAMP: 1,
            MUG: 1,
            PEN: 1,
            PENCIL: 1,
            WATCH: 1,
        },
    },
}

ROOMS_TO_AVAILABLE_OBJECTS = {
    FEW: set([
        *ABSTRACT_OBJECTS_IN_ALL_ROOMS.keys(),
        BALL, DODGEBALL, DODGEBALL_BLUE, DODGEBALL_PINK,
        BLOCK, CUBE_BLOCK, CUBE_BLOCK_BLUE, CUBE_BLOCK_TAN, CUBE_BLOCK_YELLOW,
        COLOR, BLUE, BROWN, COLOR, GREEN, ORANGE, PINK, PURPLE, RED, TAN, WHITE, YELLOW,
        BED, BLINDS, CHAIR, DESK, SHELF_DESK, DESKTOP, DRAWER, MAIN_LIGHT_SWITCH, SIDE_TABLE, BOTTOM_DRAWER, TOP_DRAWER,
        BOOK, LAPTOP, PILLOW,
        CURVED_WOODEN_RAMP,
        HEXAGONAL_BIN,
        BOTTOM_SHELF, DOOR, EAST_SLIDING_DOOR, EAST_WALL, FLOOR, MIRROR, NORTH_WALL, POSTER, RUG, SHELF, SLIDING_DOOR, SOUTH_WALL, TOP_SHELF, WALL, WEST_WALL, WEST_SLIDING_DOOR,
        ALARM_CLOCK, CD, CELLPHONE, CREDIT_CARD, KEY_CHAIN, LAMP, MUG, PEN, PENCIL, WATCH,
    ]),
    MEDIUM: set([
        *ABSTRACT_OBJECTS_IN_ALL_ROOMS.keys(),
        BALL, BASKETBALL, BEACHBALL, DODGEBALL, DODGEBALL_RED,
        BLOCK, BRIDGE_BLOCK, CUBE_BLOCK, CUBE_BLOCK_BLUE, CUBE_BLOCK_YELLOW, CYLINDRICAL_BLOCK, FLAT_BLOCK, PYRAMID_BLOCK, PYRAMID_BLOCK_RED, PYRAMID_BLOCK_YELLOW, TALL_CYLINDRICAL_BLOCK,
        COLOR, BLUE, BROWN, COLOR, GREEN, ORANGE, PINK, PURPLE, RED, TAN, WHITE, YELLOW,
        BED, BLINDS, CHAIR, DESK, SHELF_DESK, DESKTOP, DRAWER, MAIN_LIGHT_SWITCH, SIDE_TABLE, BOTTOM_DRAWER, TOP_DRAWER,
        BOOK, LAPTOP, PILLOW, TEDDY_BEAR,
        TRIANGULAR_RAMP,
        DOGGIE_BED, HEXAGONAL_BIN,
        BOTTOM_SHELF, DOOR, EAST_SLIDING_DOOR, EAST_WALL, FLOOR, MIRROR, NORTH_WALL, POSTER, RUG, SHELF, SLIDING_DOOR, SOUTH_WALL, TOP_SHELF, WALL, WEST_WALL, WEST_SLIDING_DOOR,
        ALARM_CLOCK, CD, CELLPHONE, CREDIT_CARD, KEY_CHAIN, LAMP, MUG, PEN, PENCIL, WATCH,
    ]),
    MANY: set([
        *ABSTRACT_OBJECTS_IN_ALL_ROOMS.keys(),
        BALL, BEACHBALL, DODGEBALL, DODGEBALL_BLUE, DODGEBALL_PINK, DODGEBALL_RED, GOLFBALL, GOLFBALL_GREEN,
        BLOCK, BRIDGE_BLOCK, CUBE_BLOCK, CUBE_BLOCK_BLUE, CUBE_BLOCK_TAN, CUBE_BLOCK_YELLOW, CYLINDRICAL_BLOCK, FLAT_BLOCK, PYRAMID_BLOCK, PYRAMID_BLOCK_BLUE, PYRAMID_BLOCK_RED, PYRAMID_BLOCK_YELLOW, TALL_CYLINDRICAL_BLOCK, TRIANGLE_BLOCK,
        COLOR, BLUE, BROWN, COLOR, GREEN, ORANGE, PINK, PURPLE, RED, TAN, WHITE, YELLOW,
        BED, BLINDS, CHAIR, DESK, SHELF_DESK, DESKTOP, DRAWER, MAIN_LIGHT_SWITCH, SIDE_TABLE, BOTTOM_DRAWER, TOP_DRAWER,
        BOOK, LAPTOP, PILLOW, TEDDY_BEAR,
        CURVED_WOODEN_RAMP, TRIANGULAR_RAMP, TRIANGULAR_RAMP_GREEN,
        DOGGIE_BED, HEXAGONAL_BIN,
        BOTTOM_SHELF, DOOR, EAST_SLIDING_DOOR, EAST_WALL, FLOOR, MIRROR, NORTH_WALL, POSTER, RUG, SHELF, SLIDING_DOOR, SOUTH_WALL, TOP_SHELF, WALL, WEST_WALL, WEST_SLIDING_DOOR,
        ALARM_CLOCK, CD, CELLPHONE, CREDIT_CARD, KEY_CHAIN, LAMP, MUG, PEN, PENCIL, WATCH,
    ]),
}
TYPES_TO_SUB_TYPES = {
    # GENERICS
    BALL: [BEACHBALL, BASKETBALL, DODGEBALL, GOLFBALL],
    BLOCK: [BRIDGE_BLOCK, CUBE_BLOCK, CYLINDRICAL_BLOCK, FLAT_BLOCK,
            PYRAMID_BLOCK, TALL_CYLINDRICAL_BLOCK, TALL_RECTANGULAR_BLOCK, TRIANGLE_BLOCK],
    COLOR: CATEGORIES_TO_TYPES[COLORS],
    SIDE: CATEGORIES_TO_TYPES[SIDES],
    ORIENTATION: CATEGORIES_TO_TYPES[ORIENTATIONS],
    # BALLS
    DODGEBALL: [DODGEBALL_BLUE, DODGEBALL_PINK, DODGEBALL_RED],
    GOLFBALL: [GOLFBALL_GREEN, GOLFBALL_ORANGE, GOLFBALL_WHITE],
    # BLOCKS
    BRIDGE_BLOCK: [BRIDGE_BLOCK_GREEN, BRIDGE_BLOCK_PINK, BRIDGE_BLOCK_TAN],
    CUBE_BLOCK: [CUBE_BLOCK_BLUE, CUBE_BLOCK_TAN, CUBE_BLOCK_YELLOW],
    CYLINDRICAL_BLOCK: [CYLINDRICAL_BLOCK_BLUE, CYLINDRICAL_BLOCK_TAN, CYLINDRICAL_BLOCK_GREEN],
    FLAT_BLOCK: [FLAT_BLOCK_GRAY, FLAT_BLOCK_TAN, FLAT_BLOCK_YELLOW],
    PYRAMID_BLOCK: [PYRAMID_BLOCK_BLUE, PYRAMID_BLOCK_RED, PYRAMID_BLOCK_YELLOW],
    TALL_CYLINDRICAL_BLOCK: [TALL_CYLINDRICAL_BLOCK_GREEN, TALL_CYLINDRICAL_BLOCK_TAN, TALL_CYLINDRICAL_BLOCK_YELLOW],
    TALL_RECTANGULAR_BLOCK: [TALL_RECTANGULAR_BLOCK_BLUE, TALL_RECTANGULAR_BLOCK_GREEN, TALL_RECTANGULAR_BLOCK_TAN],
    TRIANGLE_BLOCK: [TRIANGLE_BLOCK_BLUE, TRIANGLE_BLOCK_GREEN, TRIANGLE_BLOCK_TAN],
    # RAMPS
    TRIANGULAR_RAMP: [TRIANGULAR_RAMP_GREEN, TRIANGULAR_RAMP_TAN],
    # RECEPTACLES
    DRAWER: [BOTTOM_DRAWER, TOP_DRAWER],
    # ROOM_FEATURES
    SHELF: [BOTTOM_SHELF, TOP_SHELF],
    SLIDING_DOOR: [EAST_SLIDING_DOOR, WEST_SLIDING_DOOR],
    WALL: [EAST_WALL, NORTH_WALL, SOUTH_WALL, WEST_WALL],
}


for supertype in TYPES_TO_SUB_TYPES:
    subtypes = TYPES_TO_SUB_TYPES[supertype]
    updated_subtypes = list(subtypes)
    if supertype in subtypes:
        updated_subtypes.remove(supertype)

    for subtype in subtypes:
        if subtype in TYPES_TO_SUB_TYPES:
            updated_subtypes.extend(TYPES_TO_SUB_TYPES[subtype])

    TYPES_TO_SUB_TYPES[supertype] = updated_subtypes

SUBTYPES_TO_TYPES = {t: m for m, ts in TYPES_TO_SUB_TYPES.items() for t in ts}
TYPES_TO_SUB_TYPES = {t: set(ts) for t, ts in TYPES_TO_SUB_TYPES.items()}



parser = argparse.ArgumentParser()
DEFAULT_START_TOKEN = '[CONTENTS]'
parser.add_argument('-s', '--start-token', default=DEFAULT_START_TOKEN)
DEFAULT_CATEGORIES_TO_SKIP = (FULL_ROOMS_UNCATEGORIZED_OBJECTS, COLORS)
parser.add_argument('-c', '--skip-categories', nargs='+', default=DEFAULT_CATEGORIES_TO_SKIP)
DEFAULT_TYPES_TO_SKIP = (BALL, BLOCK)
parser.add_argument('-t', '--skip-types', nargs='+', default=DEFAULT_TYPES_TO_SKIP)
parser.add_argument('--separator', default=', ')
parser.add_argument('-b', '--break-after-category', action='store_true', default=False)
parser.add_argument('-v', '--verbose', action='store_true')


def get_room_objects_naive(start_token, skip_categories, skip_types, separator, break_after_category=False, verbose=False):
    room_object_strs = {}
    for room, room_data in FULL_ROOMS_TO_OBJECTS.items():
        if verbose: print(f'\nRoom type {room}:')

        total_buffer = []
        for category, category_data in room_data.items():
            if category in skip_categories:
                continue

            category_buffer = []

            for obj_type, count in category_data.items():
                if obj_type in skip_types:
                    continue

                if isinstance(count, int):
                    if count == 1:
                        category_buffer.append(obj_type)
                    else:
                        category_buffer.append(f'{count} {obj_type}')

                elif isinstance(count, dict):
                    for color, color_count in count.items():
                        type_with_color = f'{color}_{obj_type}'
                        if color_count == 1:
                            category_buffer.append(type_with_color)
                        else:
                            category_buffer.append(f'{color_count} {type_with_color}')

                else:
                    raise ValueError(f'Unknown count type: {count}')

            category_str = separator.join(category_buffer)
            if break_after_category:
                category_str += '\n'

            else:
                category_str += ' '

            total_buffer.append(category_str)

        room_str = f'{start_token}: {" ".join(total_buffer)}'
        if verbose: print(room_str)
        room_object_strs[room] = room_str

    return room_object_strs


def get_room_objects_categories(start_token, skip_categories, skip_types, separator, break_after_category=False, verbose=False):
    room_object_strs = {}
    for room, room_data in FULL_ROOMS_TO_OBJECTS.items():
        if verbose: print(f'\nRoom type {room}:')

        total_buffer = []
        for category, category_data in room_data.items():
            if category in skip_categories:
                continue

            category_buffer = []

            for obj_type, count in category_data.items():
                if obj_type in skip_types:
                    continue

                if isinstance(count, int):
                    if count == 1:
                        category_buffer.append(obj_type)
                    else:
                        category_buffer.append(f'{count} {obj_type}')

                elif isinstance(count, dict):
                    for color, color_count in count.items():
                        type_with_color = f'{color}_{obj_type}'
                        if color_count == 1:
                            category_buffer.append(type_with_color)
                        else:
                            category_buffer.append(f'{color_count} {type_with_color}')

                else:
                    raise ValueError(f'Unknown count type: {count}')

            category_str = f'({category}): {separator.join(category_buffer)}'
            if break_after_category:
                category_str += '\n'

            else:
                category_str += ' '

            total_buffer.append(category_str)

        room_str = f'{start_token}: {" ".join(total_buffer)}'
        if verbose: print(room_str)
        room_object_strs[room] = room_str

    return room_object_strs


def get_room_objects_colors(start_token, skip_categories, skip_types, separator, break_after_category=False, verbose=False):
    room_object_strs = {}
    for room, room_data in FULL_ROOMS_TO_OBJECTS.items():
        print(f'\nRoom type {room}:')

        total_buffer = []
        for category, category_data in room_data.items():
            if category in skip_categories:
                continue

            category_buffer = []

            for obj_type, count in category_data.items():
                if obj_type in skip_types:
                    continue

                if isinstance(count, int):
                    if count == 1:
                        category_buffer.append(obj_type)
                    else:
                        category_buffer.append(f'{count} {obj_type}')

                elif isinstance(count, dict):
                    type_buffer = []
                    for color, color_count in count.items():

                        if color_count == 1:
                            type_buffer.append(color)
                        else:
                            type_buffer.append(f'{color_count} {color}')

                    category_buffer.append(f'{obj_type} ({separator.join(type_buffer)})')

                else:
                    raise ValueError(f'Unknown count type: {count}')

            category_str = f'({category}): {separator.join(category_buffer)}'
            if break_after_category:
                category_str += '\n'

            else:
                category_str += ' '

            total_buffer.append(category_str)

        room_str = f'{start_token}: {" ".join(total_buffer)}'
        if verbose: print(room_str)
        room_object_strs[room] = room_str

    return room_object_strs


NAIVE_MODE = 'naive'
CATEGORY_MODE = 'categories'
COLOR_MODE = 'colors'
MODES_TO_FUNCTIONS = {
    NAIVE_MODE: get_room_objects_naive,
    CATEGORY_MODE: get_room_objects_categories,
    COLOR_MODE: get_room_objects_colors,
}
parser.add_argument('-m', '--mode', choices=list(MODES_TO_FUNCTIONS.keys()))


def foo(**kwargs):
    print(kwargs)

if __name__ == '__main__':
    args = parser.parse_args()
    mode = args.mode
    delattr(args, 'mode')
    out = MODES_TO_FUNCTIONS[mode](**args.__dict__)

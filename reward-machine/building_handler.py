import numpy as np
import typing


from config import BUILDING_TYPE, OBJECTS_BY_ROOM_AND_TYPE, UNITY_PSEUDO_OBJECTS, ROOMS
from predicate_handler import ObjectState, _pred_in_motion
from utils import FullState, AgentState, BuildingPseudoObject

MAX_BUILDINGS = 35  # max in data so far

for domain in ROOMS:
    OBJECTS_BY_ROOM_AND_TYPE[domain][BUILDING_TYPE] = [f'building_{i}' for i in range(MAX_BUILDINGS)]

UNITY_PSEUDO_OBJECTS.update({f'building_{i}': BuildingPseudoObject(f'building_{i}') for i in range(MAX_BUILDINGS)})


class BuildingHandler:
    active_buildings: typing.Set[str]  # buildings that are currently active
    building_ids: typing.List[str]
    building_id_set: typing.Set[str]
    building_valid_objects: typing.Set[str]  # objects that have been held by the agent at some point or another
    currently_held_object_id: str
    objects_to_buildings: typing.Dict[str, str]
    object_building_ratios: typing.List[float]
    recently_moved_objects: typing.Set[str]
    domain: str
    max_buildings: int

    def __init__(self, domain: str, max_buildings: int = MAX_BUILDINGS):
        self.domain = domain
        self.max_buildings = max_buildings

        self.building_ids = [f'building_{i}' for i in range(max_buildings)]
        self.building_id_set = set(self.building_ids)
        OBJECTS_BY_ROOM_AND_TYPE[domain][BUILDING_TYPE] = self.building_ids
        UNITY_PSEUDO_OBJECTS.update({f'building_{i}': BuildingPseudoObject(f'building_{i}') for i in range(max_buildings)})

        self.currently_held_object_id = ''

        self.active_buildings = set()
        self.building_valid_objects = set()
        self.objects_to_buildings = {}
        self.recently_moved_objects = set()

        self.object_building_ratios = []

    def get_active_buildings(self) -> typing.Set[str]:
        return self.active_buildings

    def _add_object_to_building(self, obj: ObjectState, building_id: str) -> None:
        typing.cast(BuildingPseudoObject, UNITY_PSEUDO_OBJECTS[building_id]).add_object(obj)
        self.active_buildings.add(building_id)
        self.objects_to_buildings[obj.object_id] = building_id

    def _remove_object_from_building(self, obj: ObjectState, debug: bool = False) -> None:
        obj_id = obj.object_id
        if obj_id in self.objects_to_buildings:
            obj_building = typing.cast(BuildingPseudoObject, UNITY_PSEUDO_OBJECTS[self.objects_to_buildings[obj_id]])
            obj_building.remove_object(obj)
            self.objects_to_buildings.pop(obj_id)
            if len(obj_building.building_objects) == 0:
                self.active_buildings.remove(obj_building.object_id)

            if debug: print(f'Object {obj_id} removed from building {obj_building.object_id} because it is {"held" if obj_id == self.currently_held_object_id else "in motion"}')

    def _merge_buildings(self, touched_buildings: typing.Collection[str], debug: bool = False) -> None:
        min_building_id = min(touched_buildings)
        if debug: print(f'Merging buildings {touched_buildings} into {min_building_id}')

        for building_id in touched_buildings:
            if building_id == min_building_id:
                continue

            other_building = typing.cast(BuildingPseudoObject, UNITY_PSEUDO_OBJECTS[building_id])
            other_building_objects = other_building.building_objects.copy()

            for obj_id, obj in other_building_objects.items():
                other_building.remove_object(obj)
                self._add_object_to_building(obj, min_building_id)
                if debug: print(f'Moving {obj_id} from {building_id} to {min_building_id}')

            self.active_buildings.remove(building_id)


    def process(self, state: FullState, debug: bool = False) -> None:
        # if the currently held object isn't marked as active, mark it as active
        agent = state.agent_state
        if state.agent_state_changed:
            agent = typing.cast(AgentState, state.agent_state)
            self.currently_held_object_id = agent.held_object
            if self.currently_held_object_id:
                self.building_valid_objects.add(self.currently_held_object_id)

        # from the objects updated at this state intersected with the valid objects:
        current_object_ids = set([o.object_id for o in state.objects])
        current_object_ids.intersection_update(self.building_valid_objects)

        current_object_ids_list = list(sorted(current_object_ids))
        current_objects = {obj_id: [o for o in state.objects if o.object_id == obj_id][0]
            for obj_id in current_object_ids}
        current_objects_in_motion_or_held = {obj_id: _pred_in_motion(agent, [obj]) or obj_id == self.currently_held_object_id  # type: ignore
            for obj_id, obj in current_objects.items()
        }

        for obj_id in current_object_ids_list:  # type: ignore
            obj = current_objects[obj_id]

            # if the object is in motion, do we mark it immediately as no longer in a building?
            # or wait for it to settle? Let's try the former
            if current_objects_in_motion_or_held[obj_id]:
                self.recently_moved_objects.add(obj_id)
                if obj_id == self.currently_held_object_id:
                    self._remove_object_from_building(obj, debug=debug)

            # maintain collection of moving/held objects that we're monitoring?
            # if an object was in that collection and is no longer moving, check which objects it's touching
            elif obj_id in self.recently_moved_objects:
                # TODO (GD): test this logic, e.g. by printing buildings with more then one object
                # TODO (GD): consider if this logic would be simplified if we considered
                # an object in a building until it stops moving

                self.recently_moved_objects.remove(obj_id)
                # We only care about touched objects that are both (a) valid for building, and
                # (b) not currently being updated, as if they're updated they're either in motion/held
                # (and therefore not part of a building), or need to be updated at this step, in which
                # case they'll be dealt with after this object
                # touched_object_ids = list(filter(lambda o_id: o_id in self.building_valid_objects, obj['touchingObjects']))
                # touched_object_ids = list(filter(
                #     lambda o_id: o_id in self.building_valid_objects and (o_id not in current_object_ids
                #     or (not current_objects_in_motion_or_held[o_id] and current_object_ids_list.index(o_id) < current_object_ids_list.index(obj_id))),
                #     obj['touchingObjects']))

                # touched_buildings = set([self.objects_to_buildings[o_id]
                #         for o_id in touched_object_ids
                #         if o_id in self.objects_to_buildings])

                touched_buildings = set([self.objects_to_buildings[o_id]
                        for o_id in obj.touching_objects
                        if o_id in self.objects_to_buildings
                    ])

                # Doesn't touch any valid objects, create
                #  a new building, or continue the previous one
                if not touched_buildings:
                    # has a previous building, check if it can keep it
                    if obj_id in self.objects_to_buildings:
                        prev_building_id = self.objects_to_buildings[obj_id]
                        prev_building = typing.cast(BuildingPseudoObject, UNITY_PSEUDO_OBJECTS[prev_building_id])
                        if len(prev_building.building_objects) == 1:
                            # stays in the same building, so nothing to do here
                            continue

                    found = False
                    building_id = ''
                    for building_id in self.building_ids:
                        if building_id not in self.active_buildings:
                            found = True
                            break

                    if not found:
                        raise ValueError('No more buildings available')

                    if debug: print(f'Adding {obj_id} to new building {building_id}')
                    self._add_object_to_building(obj, building_id)

                # touches a single building, add the object to that building
                elif len(touched_buildings) == 1:
                    building_id = touched_buildings.pop()

                    if obj_id in self.objects_to_buildings:
                        prev_building_id = self.objects_to_buildings[obj_id]
                        if prev_building_id != building_id:
                            self._remove_object_from_building(obj, debug=debug)
                        else:
                            continue

                    self._add_object_to_building(obj, building_id)
                    if debug: print(f'Adding {obj_id} to existing building {building_id}')

                # touches more than one building, merge them
                else:   # len(touched_buildings) > 1:
                    min_building_id = min(touched_buildings)

                    if obj_id in self.objects_to_buildings:
                        prev_building_id = self.objects_to_buildings[obj_id]
                        if prev_building_id != min_building_id:
                            self._remove_object_from_building(obj, debug=debug)

                    self._add_object_to_building(obj, min_building_id)
                    if debug:
                        print(f'Adding {obj_id} to existing building {min_building_id}')

                    self._merge_buildings(touched_buildings, debug)

            else:
                if obj_id in self.objects_to_buildings:
                    obj_building = self.objects_to_buildings[obj_id]
                    touched_buildings = set([self.objects_to_buildings[o_id]
                        for o_id in obj.touching_objects
                        if o_id in self.objects_to_buildings
                    ])

                    if len(touched_buildings) > (obj_building in touched_buildings):
                        touched_buildings.add(obj_building)
                        self._merge_buildings(touched_buildings, debug)

        if debug:
            if self.active_buildings:
                self.object_building_ratios.append(len(self.objects_to_buildings) / len(self.active_buildings))
                print(f'Active buildings: {np.mean(self.object_building_ratios):.3f}')
                for building_id in self.active_buildings:
                    building = typing.cast(BuildingPseudoObject, UNITY_PSEUDO_OBJECTS[building_id])
                    print(f'Building {building_id} has {len(building.building_objects)} objects')

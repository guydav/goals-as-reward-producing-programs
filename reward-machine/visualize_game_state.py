import itertools
import json
import os
import typing

import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARNING)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D
import numpy as np
from tqdm import tqdm

from config import UNITY_PSEUDO_OBJECTS
from utils import FullState, _object_corners, _point_in_top_half
from predicate_handler import PREDICATE_LIBRARY_RAW, ON_DISTANCE_THRESHOLD
from building_handler import BuildingPseudoObject, BuildingHandler

class Visualizer():
    def __init__(self):
        self.agent_states_by_idx = []
        self.object_states_by_idx = []
        self.objects_to_track = []
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    def _plot_object_bounding_box(self, ax, object_state, color, alpha=0.5):

        x, z, y = object_state.bbox_center
        x_size, z_size, y_size = object_state.bbox_extents

        vertices = [
            (x - x_size, y - y_size, z - z_size),
            (x + x_size, y - y_size, z - z_size),
            (x + x_size, y + y_size, z - z_size),
            (x - x_size, y + y_size, z - z_size),
            (x - x_size, y - y_size, z + z_size),
            (x + x_size, y - y_size, z + z_size),
            (x + x_size, y + y_size, z + z_size),
            (x - x_size, y + y_size, z + z_size)
        ]

        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[0], vertices[3], vertices[7], vertices[4]],
            [vertices[1], vertices[2], vertices[6], vertices[5]]
        ]

        prism = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor=color)
        ax.add_collection3d(prism)

        return prism

    def _visualize_objects(self):

        prisms = []
        for obj_idx, object_id in enumerate(self.objects_to_track):

            if object_id in UNITY_PSEUDO_OBJECTS:
                object_state = UNITY_PSEUDO_OBJECTS[object_id]
            else:
                object_state = self.object_states_by_idx[self.visualization_index][object_id]

            if isinstance(object_state, BuildingPseudoObject):
                building_object_ids = list(object_state.building_objects.keys())
                print(f'Building {object_id} is composed of: {building_object_ids}')
                for building_object_id in building_object_ids:
                    building_object_state = self.object_states_by_idx[self.visualization_index][building_object_id]
                    prism = self._plot_object_bounding_box(self.ax, building_object_state, self.colors[obj_idx % len(self.colors)], alpha=0.5)
                    prisms.append(prism)

            else:
                prism = self._plot_object_bounding_box(self.ax, object_state, self.colors[obj_idx % len(self.colors)], alpha=0.5)
                prisms.append(prism)



        if self.zoom_to_objects:
            object_states = [UNITY_PSEUDO_OBJECTS[object_id] if object_id in UNITY_PSEUDO_OBJECTS else
                             self.object_states_by_idx[self.visualization_index][object_id] for object_id in self.objects_to_track]

            self.min_x = min([obj.bbox_center[0] - obj.bbox_extents[0] for obj in object_states])
            self.max_x = max([obj.bbox_center[0] + obj.bbox_extents[0] for obj in object_states])

            self.min_z = min([obj.bbox_center[1] - obj.bbox_extents[1] for obj in object_states])
            self.max_z = max([obj.bbox_center[1] + obj.bbox_extents[1] for obj in object_states])

            self.min_y = min([obj.bbox_center[2] - obj.bbox_extents[2] for obj in object_states])
            self.max_y = max([obj.bbox_center[2] + obj.bbox_extents[2] for obj in object_states])

            margin = 0.2

        else:
            self.min_x, self.min_y, self.min_z = -4, -4, -0.25
            self.max_x, self.max_y, self.max_z = 4, 4, 4
            margin = 0

        self.ax.set_xlim(self.min_x - margin, self.max_x + margin)
        self.ax.set_ylim(self.min_y - margin, self.max_y + margin)
        self.ax.set_zlim(self.min_z - margin, self.max_z + margin)

        legend_elements = [Line2D([0], [0], color=self.colors[i % len(self.colors)], lw=4, label=self.objects_to_track[i]) for i in range(len(self.objects_to_track))]
        self.ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

        if self.predicate is not None:

            if self.predicate_args is not None:
                args = self.predicate_args
            else:
                args = self.objects_to_track

            agent_state = [self.agent_states_by_idx[self.visualization_index]]
            object_states = [UNITY_PSEUDO_OBJECTS[object_id] if object_id in UNITY_PSEUDO_OBJECTS else
                             self.object_states_by_idx[self.visualization_index][object_id] for object_id in args]

            for i, object_id in enumerate(args):
                if object_id.startswith('building'):
                    building_state = typing.cast(BuildingPseudoObject, object_states[i])
                    building_object_ids = list(building_state.building_objects.keys())

                    for object_combination in itertools.permutations(building_object_ids, 2):
                        combination_object_states = [UNITY_PSEUDO_OBJECTS[object_id] if object_id in UNITY_PSEUDO_OBJECTS else
                                                     self.object_states_by_idx[self.visualization_index][object_id] for object_id in object_combination]
                        print(f'({self.predicate} {object_combination[0]} {object_combination[1]}) = {PREDICATE_LIBRARY_RAW[self.predicate](agent_state, combination_object_states)}')

                    print()


                    for j, other_object_id in enumerate(args):
                        if i != j:
                            print(f'Other object id {other_object_id} is in the building? {other_object_id in building_state.building_objects}')
                            print(f'({self.predicate} {object_id} {other_object_id}) = {PREDICATE_LIBRARY_RAW[self.predicate](agent_state, [object_states[i], object_states[j]])}')
                            print(f'({self.predicate} {other_object_id} {object_id} ) = {PREDICATE_LIBRARY_RAW[self.predicate](agent_state, [object_states[j], object_states[i]])}')


            predicate_value = PREDICATE_LIBRARY_RAW[self.predicate](agent_state, object_states)

            arg_names = " ".join([arg.name for arg in object_states])

            title = f"State {self.visualization_index}: ({self.predicate} {arg_names}) = {predicate_value}"

            # Special  case for debugging 'on':
            if self.predicate == "on":
                lower_object = object_states[0]
                upper_object = object_states[1]
                upper_object_bbox_center = upper_object.bbox_center
                upper_object_bbox_extents = upper_object.bbox_extents

                # Project a point slightly below the bottom center / corners of the upper object
                upper_object_corners = _object_corners(upper_object)

                test_points = [corner - np.array([0, upper_object_bbox_extents[1] + ON_DISTANCE_THRESHOLD, 0])  # type: ignore
                            for corner in upper_object_corners]
                test_points.append(upper_object_bbox_center - np.array([0, upper_object_bbox_extents[1] + ON_DISTANCE_THRESHOLD, 0]))
                test_points += upper_object_corners

                EXPANSION_FACTOR = 1
                for point in test_points:
                    u = np.linspace(0, 2 * np.pi, 5)
                    v = np.linspace(0, np.pi, 5)

                    if _point_in_top_half(point, lower_object):
                        EXPANSION_FACTOR *= 5

                    x = point[0] + (EXPANSION_FACTOR * ON_DISTANCE_THRESHOLD) * np.outer(np.cos(u), np.sin(v))
                    y = point[2] + (EXPANSION_FACTOR * ON_DISTANCE_THRESHOLD) * np.outer(np.sin(u), np.sin(v))
                    z = point[1] + (EXPANSION_FACTOR * ON_DISTANCE_THRESHOLD) * np.outer(np.ones(np.size(u)), np.cos(v))

                    if _point_in_top_half(point, lower_object):
                        EXPANSION_FACTOR /= 5

                    self.ax.plot_surface(x, y, z, color='r', alpha=1)

        else:
            title = f"State {self.visualization_index}"

        plt.title(title, fontsize=12)

    def _update_visualization(self, event):

        if event.key == "right":
            self.visualization_index = min(self.visualization_index + 1, len(self.agent_states_by_idx) - 1)
        elif event.key == "left":
            self.visualization_index = max(self.visualization_index - 1, 0)

        # Clear the axis and update the visualization
        self.ax.clear()
        self._visualize_objects()

        # Reset camera view
        self.ax.view_init(elev=self.elev, azim=self.azim)

    def _update_azim_elev(self, event):
        self.azim, self.elev = self.ax.azim, self.ax.elev

    def visualize(self, trace, objects_to_track, start_idx=0, predicate=None, domain=None, predicate_args=None,
                  zoom_to_objects=False):

        self.objects_to_track = objects_to_track
        self.visualization_index = start_idx
        self.predicate = predicate
        self.predicate_args = predicate_args
        self.zoom_to_objects = zoom_to_objects

        replay = trace['replay']
        replay_len = int(len(replay))

        # Stores the most recent state of the agent and of each object
        most_recent_agent_state = None
        most_recent_object_states = {}

        initial_object_states = {}

        if domain is None:
            print("No domain specified, using 'few' as the default domain")
            domain = "few"

        building_handler = BuildingHandler(domain)

        # Start by recording the states of objects we want to track
        for idx, state in tqdm(enumerate(replay), total=replay_len, desc=f"Processing replay", leave=False):
            state = FullState.from_state_dict(state)
            building_handler.process(state)

            # Track changes to the agent
            if state.agent_state_changed:
                most_recent_agent_state = state.agent_state

            # And to objects
            objects_with_initial_rotations = []
            for obj in state.objects:
                if obj.object_id not in initial_object_states:
                    initial_object_states[obj.object_id] = obj

                obj = obj._replace(initial_rotation=initial_object_states[obj.object_id].rotation)
                objects_with_initial_rotations.append(obj)
                most_recent_object_states[obj.object_id] = obj

            self.agent_states_by_idx.append(most_recent_agent_state)
            self.object_states_by_idx.append(most_recent_object_states.copy())

        # Plot figure
        self.fig = plt.figure(figsize=(10, 10))
        cid = self.fig.canvas.mpl_connect('key_press_event', self._update_visualization)
        cid2 = self.fig.canvas.mpl_connect('motion_notify_event', self._update_azim_elev)

        self.ax = self.fig.add_subplot(projection='3d')
        self._visualize_objects()

        # Matplotlib uses Z as the default "vertical" dimension, whereas the traces use Y
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Z')
        self.ax.set_zlabel('Y')

        plt.show()


if __name__ == "__main__":

    # TRACE_NAME = "ZSRiFalssazyXG8Qui4j-preCreateGame-rerecorded"
    # START_IDX = 476
    # PREDICATE = "on"
    # OBJECTS_TO_TRACK = ["north_wall", "CubeBlock|-02.99|+01.26|-01.49"]


    TRACE_NAME = "50OkWutbBlKsEJPMGbKu-gameplay-attempt-1-rerecorded"  # "qK8hfQE9E97kZMDdL4Hv-preCreateGame"  # -rerecorded"
    START_IDX = 77
    PREDICATE = "on"
    DOMAIN = "few"
    OBJECTS_TO_TRACK = ["Chair|+02.73|00.00|-01.21", "Desk|+03.14|00.00|-01.41"]
    # OBJECTS_TO_TRACK = ["Shelf|+00.62|+01.51|-02.82", "Laptop|+03.04|+00.79|-02.28"]
    # OBJECTS_TO_TRACK = ["Shelf|-02.97|+01.16|-01.72", "CubeBlock|+00.20|+00.29|-02.83"]
    # OBJECTS_TO_TRACK = ["Shelf|-02.97|+01.16|-02.47", "Dodgeball|-02.95|+01.29|-02.61"]


    trace_path = f"./reward-machine/traces/{TRACE_NAME}.json"
    if not os.path.exists(trace_path):
        trace_path = trace_path.replace('/traces/', '/traces/participant-traces/')

    trace = json.load(open(trace_path, 'r'))
    Visualizer().visualize(trace, objects_to_track=OBJECTS_TO_TRACK, start_idx=START_IDX, predicate=PREDICATE, domain=DOMAIN,
                           zoom_to_objects=True)

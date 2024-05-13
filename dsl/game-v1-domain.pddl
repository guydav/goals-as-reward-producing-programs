;Header and description

(define (domain game-v1)

;remove requirements that are not needed
(:requirements
    :strips
    :typing
    :equality
    :existential-preconditions  ; exists in preconditions
    :universal-preconditions  ; forall in preconditions
    :disjunctive-preconditions  ; or in preconditions
    :negative-preconditions  ; not in preconditions
    :conditional-effects  ; allows using (when <antecedent> <consequence>) in effects
    ;:durative-actions   ; enables actions with durations (PDDL 2.1)
    :timed-initial-literals  ; allows defining initializtions at points in time
    :numeric-fluents  ; enables the function block? but it seems to exist without it below?
    ;:duration-inequalities   ; allows using a inequality to specify action durations
    ;:fluents  ; old numeric supprt (PDDL 1.2)
    :preferences  ; specifying preferences for actions
    :constraints  ; specifying constraints for actions
)

(:types ;todo: enumerate types and their hierarchy here, e.g. car truck bus - vehicle
    side orientation color object perspective - type
    building structure game_object - object
    wall - structure
    stationary_object block ball - game_object
    block ball - pickupable
    dodgeball golfball beachball basketball - ball
    cube_block pyramid_block bridge_block flat_block - block
    tall_cylindrical_block short_cylindrical_block triangular_block - block
    chair shelf hexagonal_bin doggie_bed - game_object  ; TODO - figure out if these are pickupable or not?
    desktop laptop desk_lamp cd textbook pillow teddy_bear mug - pickupable
    curved_wooden_ramp triangular_ramp - pickupable
    ; do we actually want a category of all objects that can be picked up?
)

; un-comment following line if constants are needed
(:constants
    pickup_distance - num
    center front back left right top bottom - side
    face edge point upright upside_down - orientation
    room floor desk bed - structure
    east_wall west_wall north_wall south_wall - wall
    west_wall_shelf south_wall_shelf - shelf
    agent - object  ; since the agent can holds things
    rug poster - game_object
    green red blue yellow none - color
    pink_dodgeball blue_dodgeball - dodgeball
    looking_upside_down sideways eyes_closed - perspective
    building - object
)

(:predicates ;todo: define predicates here
    (agent_holds ?o - game-object) ; is the agent holding this game object
    (on ?o1 - object ?o2 - object)  ; object o2 on o1
    (adjacent ?o1 ?o2 - object)  ; objects o1 and o2 are adjacent
    ; side s1 of object o1 is adjacent to side s2 of o2
    ; (adjacent_side ?o1 -object ?s1 -side ?o2 -object ?s2 - side)
    (between ?o1 ?o2 ?o3 - object)  ; is o2 between o1 and o3?
    (touch ?o1 ?o2 - object)  ; are o1 and o2 touching?
    (object_orientation ?o - game_object ?r - orientation) ; is the orientation of the object as marked?
    (under ?o1 ?o2 - game-object) ; is o2 under o1? theoretically could also be a between with an object below, such as the floor
    ; TODO: do we want to handle the case of multiple identified buildings?
    (in ?b - building ?o - game_object) ; is the object part of this building?
    (building_fell ?b - building)  ; did this building just fall over?
    (building_size ?b)  ; how many objects are in the building?
    (object_color ?o - game-object ?c - color) ; is the object with this color
    (in_motion ?o - game-object)  ; is this game object still in motion?
    (thrown ?o - game-object)  ; has this object been thrown?
    (agent_perspective ?p - perspective)  ; what perspective is the agent looking in?
    (agent_finished_spin)  ; did the agent just finish a full spin?
    (is_rotating ?o - game-object) ; is this object currently rotating -- could maybe use for the above?
    (agent_terminated_episode)
)

(:functions ;todo: define numeric functions here
    (x_position ?o - object) - number
    (y_position ?o - object) - number
    (z_position ?o - object) - number
    (side ?o - object ?s - side) -object
    (side_x_position ?o - object ?s - side) - number
    (side_y_position ?o - object ?s - side) - number
    (side_z_position ?o - object ?s - side) - number
    (distance ?o1 ?o2 - object) - number
    (distance_side ?o1 -object ?s1 -side ?o2 -object ?s2 -side)- number
    (building_height) - number
    (max_building_height) - number
)

;define actions here
(:action pickup-object
    :parameters (?p - pickupable)
    :precondition (and
        (not (exists (?p2 - pickupable) (agent_holds ?p2)))
        (<= (distance agent ?p) 1)
    )
    :effect (and
        (agent_holds ?p)
    )
)

)

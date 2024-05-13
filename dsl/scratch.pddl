(and (not (agent_holds ?d)) (in_motion ?d)) ; in motion, not in hand until...

(and (on ?h ?d) (not (in_motion ?d)))

(:goal (or
    (and
        (minimum_time_reached)
        (agent_terminated_episode)
    )
    (maximum_time_reached)
))

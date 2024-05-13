; Human games
(define (game game-51) (:domain few-objects-room-v1)  ; 51

(:constraints (and
    (preference throwToBin
        (exists (?d - dodgeball ?h - hexagonal_bin)
            (then
                (once (agent_holds ?d))
                (hold (and (not (agent_holds ?d)) (in_motion ?d)))
                (once (and (not (in_motion ?d)) (in ?h ?d)))
            )
        )
    )
))
(:scoring
    (count throwToBin)
))

(define (game game-50) (:domain medium-objects-room-v1)  ; 50
(:setup
    (exists (?h - hexagonal_bin) (game-conserved (< (distance room_center ?h) 1)))
)
(:constraints (and
    (preference gameObjectToBin (exists (?g - game_object ?h - hexagonal_bin)
        (then
            (once (not (agent_holds ?g)))
            (hold (or (agent_holds ?g) (in_motion ?g)))
            (once (and (not (in_motion ?g)) (in ?h ?g)))
        )
    ))
))
(:scoring
    (count-once-per-objects gameObjectToBin)
))

(define (game game-52) (:domain few-objects-room-v1)  ; 52

(:constraints (and
    (preference blockFromRugToDesk (exists (?c - cube_block )
        (then
            (once (and (on rug agent) (agent_holds ?c)))
            (hold (and
                (on rug agent)
                (in_motion ?c)
                (not (agent_holds ?c))
                (not (exists (?o - (either lamp desktop laptop)) (or (broken ?o) (in_motion ?o))))
            ))
            (once (and (on rug agent) (on desk ?c) (not (in_motion ?c))))
        )
    ))
))
(:scoring
    (count-once-per-objects blockFromRugToDesk)
))

(define (game game-118) (:domain medium-objects-room-v1)  ; 118
(:constraints (and
    (forall (?x - color)
        (preference objectWithMatchingColor (exists (?o1 ?o2 - game_object)
            (at-end (and
                (same_color ?o1 ?o2)
                (same_color ?o1 ?x)
                (or
                    (on ?o1 ?o2)
                    (adjacent ?o1 ?o2)
                    (in ?o1 ?o2)
                )
            ))
        ))
    )
    (preference itemsTurnedOff
        (exists (?o - (either main_light_switch lamp))
            (at-end
                (not (toggled_on ?o))
            )
        )
    )
    (preference itemsBroken
        (exists (?o - game_object)
            (at-end
                (broken ?o)
            )
        )
    )
))
(:scoring (+
    (* 5 (count-once-per-objects objectWithMatchingColor))
    (* 5 (count-once-per-objects objectWithMatchingColor:green))
    (* 5 (count-once-per-objects objectWithMatchingColor:brown))
    (* 15 (count-once-per-objects itemsTurnedOff))
    (* -10 (count-once-per-objects itemsBroken))
)))


; Map elites games
;(define (game mapelites-1) (:domain medium-objects-room-v1)
;(:constraints
;  (and
;    (preference preference0
;      (exists (?v0 - bridge_block ?v1 - room_center)
;        (at-end
;          (and
;            (adjacent ?v1 ?v0)
;         )
;       )
;     )
;   )
;    (preference preference1
;      (exists (?v0 - dodgeball)
;        (then
;          (once (agent_holds ?v0))
;          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
;          (once (not (in_motion ?v0)))
;       )
;     )
;   )
; )
;)
;(:terminal
;  (or
;    (>= (count preference0) 2)
; )
;)
;(:scoring
;    (count preference1)
;))

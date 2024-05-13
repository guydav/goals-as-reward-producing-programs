(define (game game-51) (:domain few-objects-room-v1)
(:constraints
  (and
    (preference throwToBin
      (exists (?d - dodgeball ?h - hexagonal_bin)
        (then
          (once (agent_holds ?d) )
          (hold (and (not (agent_holds ?d) ) (in_motion ?d) ) )
          (once (and (not (in_motion ?d) ) (in ?h ?d) ) )
)))))
(:scoring
  (count throwToBin)
))

(define (game evo-8158-92-1) (:domain few-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - beachball ?v1 - hexagonal_bin)
        (then
          (once (agent_holds ?v0) )
          (hold (and (in_motion ?v0) (not (agent_holds ?v0) ) ) )
          (once (in ?v1 ?v0) )
)))))
(:terminal
  (or
    (>= (count preference0) 22 )
    (>= (count-once-per-objects preference0) 5 )
))
(:scoring
  (count preference0)
))

(define (game game-17) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference castleBuilt
      (exists (?b - bridge_block ?f - flat_block ?t - tall_cylindrical_block ?c - cube_block ?p - pyramid_block)
        (at-end
          (and
            (on ?b ?f)
            (on ?f ?t)
            (on ?t ?c)
            (on ?c ?p)
))))))
(:scoring
  (* 10 (count-once-per-objects castleBuilt) )
))

(define (game evo-8180-44-0) (:domain few-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - block ?v1 - bridge_block_green ?v2 - pyramid_block)
        (at-end
          (and
            (on ?v1 ?v2)
            (on ?v1 ?v0)
))))))
(:terminal
  (>= (count preference0) 12 )
)
(:scoring
  (count preference0)
))

(define (game game-114) (:domain medium-objects-room-v1)
(:setup
    (exists (?d - doggie_bed)
      (game-conserved
        (< (distance room_center ?d) 0.5)
)))
(:constraints
  (and
    (preference objectInBuilding
      (exists (?o - game_object ?d - doggie_bed ?b - building)
        (at-end
          (and
            (not
              (same_object ?o ?d)
            )
            (in ?b ?d)
            (in ?b ?o)
            (on floor ?d)
            (not
              (on floor ?o)
            )
            (not
              (exists (?w - wall)
                (touch ?w ?o)
))))))))
(:scoring
  (count-once-per-objects objectInBuilding)
))

(define (game evo-8111-143-0) (:domain few-objects-room-v1)
(:setup
    (exists (?v0 - hexagonal_bin)
      (game-conserved
        (< (distance east_wall ?v0) 0.4)
)))
(:constraints
  (and
    (preference preference0
      (exists (?v1 - cube_block_blue ?v0 - shelf)
        (at-end
          (and
            (adjacent west_wall ?v0)
            (on ?v0 ?v1)
    ))))
    (preference preference1
      (exists (?v2 - game_object)
        (at-end
          (and
            (same_color ?v2 orange)
            (< (distance rug ?v2) (distance door ?v2))
))))))
(:terminal
  (or
    (>= (count preference0) 23 )
    (>= (count-once-per-objects preference1) 20 )
))
(:scoring
  (count preference0)
))

(define (game evo-8170-346-1) (:domain medium-objects-room-v1)
(:setup
    (exists (?v0 - hexagonal_bin)
      (game-conserved
        (< (distance rug ?v0) 0.3)
)))
(:constraints
  (and
    (preference preference0
      (exists (?v1 - hexagonal_bin)
        (then
          (once (object_orientation ?v1 diagonal) )
          (hold (and (not (touch agent ?v1) ) (not (agent_holds ?v1) ) ) )
          (once (not (object_orientation ?v1 diagonal) ) )
)))))
(:terminal
  (>= (count preference0) 14 )
)
(:scoring
  (count preference0)
))

(define (game evo-8179-288-0) (:domain few-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - pillow ?v1 - bed)
        (at-end
          (and
            (on ?v1 ?v0)
    ))))
    (preference preference1
      (exists (?v1 - (either dodgeball beachball))
        (then
          (once (agent_holds ?v1) )
          (hold (and (not (agent_holds ?v1) ) (in_motion ?v1) ) )
          (once (not (in_motion ?v1) ) )
)))))
(:terminal
  (>= (count preference1) 5 )
)
(:scoring
  (count preference0)
))

(define (game evo-8174-339-0) (:domain few-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 ?v1 ?v2 ?v3 - game_object)
        (at-end
          (and
            (same_color ?v1 ?v2)
            (adjacent ?v0 ?v1)
            (in ?v3 ?v1)
    ))))
    (preference preference1
      (exists (?v2 - (either dodgeball golfball))
        (then
          (once (agent_holds ?v2) )
          (hold (and (not (agent_holds ?v2) ) (in_motion ?v2) ) )
          (once (not (in_motion ?v2) ) )
)))))
(:terminal
  (or
    (>= (count preference0) 2 )
    (>= (count-once-per-objects preference1) 3 )
))
(:scoring
  (count preference0)
))

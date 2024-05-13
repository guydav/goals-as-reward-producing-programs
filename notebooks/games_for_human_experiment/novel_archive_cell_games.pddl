; Key (1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0)
(define (game evo-8170-46-0) (:domain medium-objects-room-v1)
(:setup
  (exists (?v0 - doggie_bed)
    (game-conserved
      (near rug ?v0)
   )
 )
)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - game_object)
        (at-end
          (and
            (not
              (in_motion ?v0)
           )
            (on bed ?v0)
         )
       )
     )
   )
 )
)
(:scoring
  (count preference0)
)
)

; Key (1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0)
(define (game evo-8168-141-0) (:domain many-objects-room-v1)
(:setup
  (forall (?v0 - block)
    (game-conserved
      (on rug ?v0)
   )
 )
)
(:constraints
  (and
    (preference preference0
      (exists (?v1 - cube_block)
        (then
          (once (agent_holds ?v1))
          (hold (and (in_motion ?v1) (not (agent_holds ?v1))))
          (once (not (in_motion ?v1)))
       )
     )
   )
 )
)
(:scoring
  (count-once-per-objects preference0)
)
)

; Key (1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0)
(define (game evo-8188-71-0) (:domain many-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - (either credit_card cd) ?v1 - hexagonal_bin)
        (at-end
          (in ?v1 ?v0)
       )
     )
   )
 )
)
(:scoring
  (+ (* -1 (count preference0))
    (count preference0)
 )
)
)

; Key (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1)
(define (game evo-8153-162-1) (:domain few-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - game_object)
        (at-end
          (and
            (not
              (in_motion ?v0)
           )
            (near rug ?v0)
         )
       )
     )
   )
 )
)
(:terminal
  (>= (total-time) 180)
)
(:scoring
  (count preference0)
)
)

; Key (1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
(define (game evo-8191-96-0) (:domain many-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - dodgeball)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (not (in_motion ?v0)))
       )
     )
   )
 )
)
(:scoring
  (count preference0)
)
)

; Key (1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0)
(define (game evo-7988-64-1) (:domain many-objects-room-v1)
(:setup
  (forall (?v0 - bridge_block_green)
    (game-conserved
      (on rug ?v0)
   )
 )
)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - block ?v1 - cube_block ?v2 - flat_block ?v3 - cube_block)
        (at-end
          (and
            (on ?v0 ?v3)
            (on ?v0 ?v1)
            (on ?v0 ?v2)
            (same_type ?v0 ?v3)
         )
       )
     )
   )
 )
)
(:scoring
  (+ (* 5 (count preference0))
    (count preference0)
 )
)
)

; Key (1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0)
(define (game evo-8182-61-1) (:domain many-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - hexagonal_bin ?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (in ?v0 ?v1) (on ?v0 ?v1)))
       )
     )
   )
 )
)
(:scoring
  (count preference0)
)
)

; Key (1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0)
(define (game evo-8169-225-0) (:domain many-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - cube_block)
        (then
          (once (agent_holds ?v0))
          (hold (and (in_motion ?v0) (not (agent_holds ?v0))))
          (once (not (in_motion ?v0)))
       )
     )
   )
 )
)
(:scoring
  (count-once-per-objects preference0)
)
)

; Key (1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1)
(define (game evo-8087-311-0) (:domain few-objects-room-v1)
(:setup
  (forall (?v0 - dodgeball)
    (game-optional
      (near desk ?v0)
   )
 )
)
(:constraints
  (and
    (preference preference0
      (exists (?v1 - dodgeball ?v0 - teddy_bear)
        (then
          (once (agent_holds ?v1))
          (hold (and (in_motion ?v1) (not (agent_holds ?v1)) (not (agent_holds ?v0)) (touch south_wall ?v1)))
          (once (not (in_motion ?v1)))
       )
     )
   )
 )
)
(:terminal
  (>= (total-time) 60)
)
(:scoring
  (count preference0)
)
)

; Key (1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
(define (game evo-8140-86-0) (:domain many-objects-room-v1)
(:setup
  (forall (?v0 - dodgeball)
    (game-optional
      (near desk ?v0)
   )
 )
)
(:constraints
  (and
    (preference preference0
      (exists (?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (not (in_motion ?v1)))
       )
     )
   )
 )
)
(:scoring
  (count preference0)
)
)

; Key (1, 1, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0)
(define (game evo-8172-284-1) (:domain many-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - cube_block)
        (then
          (once (agent_holds ?v0))
          (hold (and (in_motion ?v0) (not (agent_holds ?v0))))
          (once (and (not (in_motion ?v0)) (on rug ?v0)))
       )
     )
   )
    (preference preference1
      (exists (?v0 - cube_block)
        (then
          (once (agent_holds ?v0))
          (hold (not (agent_holds ?v0)))
          (hold (agent_holds ?v0))
          (once (not (in_motion ?v0)))
       )
     )
   )
 )
)
(:scoring
  (+ (* 3 (count preference1))
    (count preference0)
 )
)
)

; Key (1, 0, 2, 0, 1, 0, 0, 0, 0, 1, 0, 0)
(define (game evo-8172-48-1) (:domain many-objects-room-v1)
(:setup
  (exists (?v0 - hexagonal_bin)
    (game-conserved
      (near north_wall ?v0)
   )
 )
)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - hexagonal_bin ?v1 - golfball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (on ?v0 ?v1) (not (in_motion ?v1)) (in ?v0 ?v1)))
       )
     )
   )
    (preference preference1
      (exists (?v0 - cube_block ?v2 - cube_block ?v3 - block ?v4 - cube_block)
        (at-end
          (and
            (on ?v3 ?v0)
            (on ?v3 ?v2)
            (on ?v3 ?v4)
            (same_type ?v4 ?v3)
         )
       )
     )
   )
 )
)
(:scoring
  (+ (* -4 (count preference0))
    (count preference1)
 )
)
)

; Key (1, 1, 2, 0, 0, 1, 0, 0, 0, 0, 0, 1)
(define (game evo-8177-63-0) (:domain many-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - game_object)
        (at-end
          (and
            (not
              (in_motion ?v0)
           )
            (near rug ?v0)
         )
       )
     )
   )
    (preference preference1
      (exists (?v1 - golfball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1) (touch rug ?v1)))
          (hold (and (in_motion ?v1) (not (agent_holds ?v1))))
          (once (not (in_motion ?v1)))
       )
     )
   )
 )
)
(:scoring
  (+ (count preference0) (count preference1))
)
)

; Key (1, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0)
(define (game evo-8115-124-0) (:domain many-objects-room-v1)
(:setup
  (exists (?v0 - hexagonal_bin)
    (game-conserved
      (near rug ?v0)
   )
 )
)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - chair ?v1 - desk)
        (at-end
          (and
            (on ?v1 ?v0)
         )
       )
     )
   )
    (preference preference1
      (exists (?v1 - hexagonal_bin ?v2 - basketball)
        (then
          (once (agent_holds ?v2))
          (hold (and (not (agent_holds ?v2)) (in_motion ?v2)))
          (once (and (not (in_motion ?v2)) (on ?v1 ?v2)))
       )
     )
   )
 )
)
(:scoring
  (+ (count preference0) (* 40 (count preference1))
 )
)
)

; Key (1, 0, 2, 1, 0, 0, 1, 0, 0, 0, 0, 0)
(define (game evo-8142-216-0) (:domain many-objects-room-v1)
(:setup
  (exists (?v0 - hexagonal_bin)
    (game-conserved
      (near door ?v0)
   )
 )
)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - (either credit_card cd) ?v1 - hexagonal_bin)
        (at-end
          (in ?v1 ?v0)
       )
     )
   )
    (preference preference1
      (exists (?v1 - golfball_white)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (not (in_motion ?v1)))
       )
     )
   )
 )
)
(:scoring
  (+ (count preference1) (* 40 (count preference0))
 )
)
)

; Key (1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0)
(define (game evo-8149-344-0) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - hexagonal_bin ?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (in ?v0 ?v1)))
       )
     )
   )
    (preference preference1
      (exists (?v0 - dodgeball ?v1 - doggie_bed)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (on ?v1 ?v0)))
       )
     )
   )
 )
)
(:scoring
  (+ (count preference1) (count preference0))
)
)

; Key (1, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0)
(define (game evo-7998-190-0) (:domain many-objects-room-v1)
(:setup
  (exists (?v0 - hexagonal_bin)
    (game-conserved
      (near rug ?v0)
   )
 )
)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - chair ?v1 - desk)
        (at-end
          (and
            (on ?v1 ?v0)
         )
       )
     )
   )
    (preference preference1
      (exists (?v2 - teddy_bear ?v1 - desk)
        (at-end
          (and
            (on ?v1 ?v2)
            (not
              (object_orientation ?v2 diagonal)
           )
         )
       )
     )
   )
 )
)
(:scoring
  (+ (* 80 (count preference0))
    (count preference1)
 )
)
)

; Key (1, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 1)
(define (game evo-8191-99-1) (:domain many-objects-room-v1)
(:setup
  (forall (?v0 - cube_block_blue)
    (game-conserved
      (on rug ?v0)
   )
 )
)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - game_object)
        (at-end
          (and
            (not
              (in_motion ?v0)
           )
            (near north_wall ?v0)
         )
       )
     )
   )
    (preference preference1
      (exists (?v0 - cube_block ?v1 - cube_block ?v2 - cube_block_blue ?v3 - cube_block_yellow)
        (at-end
          (and
            (on ?v1 ?v3)
            (on ?v3 ?v0)
            (same_type ?v3 ?v2)
         )
       )
     )
   )
 )
)
(:scoring
  (+ (* 5 (count preference0))
    (count preference1)
 )
)
)

; Key (1, 1, 2, 0, 0, 0, 1, 0, 1, 0, 0, 0)
(define (game evo-8177-58-0) (:domain many-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - cube_block)
        (then
          (once (agent_holds ?v0))
          (hold (and (in_motion ?v0) (not (agent_holds ?v0))))
          (once (not (in_motion ?v0)))
       )
     )
   )
    (preference preference1
      (exists (?v1 - (either credit_card cd) ?v0 - hexagonal_bin)
        (at-end
          (in ?v0 ?v1)
       )
     )
   )
 )
)
(:scoring
  (+ (* -1 (count preference0))
    (count preference1)
 )
)
)

; Key (1, 1, 2, 0, 1, 1, 0, 0, 0, 0, 0, 0)
(define (game evo-8168-60-0) (:domain many-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - dodgeball)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (on desk ?v0)))
       )
     )
   )
    (preference preference1
      (exists (?v1 - hexagonal_bin ?v0 - dodgeball)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (in ?v1 ?v0) (on ?v1 ?v0)))
       )
     )
   )
 )
)
(:scoring
  (+ (count preference0) (* 20 (count preference1))
 )
)
)

; Key (1, 1, 3, 1, 0, 0, 1, 0, 0, 0, 1, 0)
(define (game evo-8189-236-1) (:domain many-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - hexagonal_bin ?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (in ?v0 ?v1)))
       )
     )
   )
    (preference preference1
      (exists (?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (not (in_motion ?v1)))
       )
     )
   )
    (preference preference2
      (exists (?v2 - (either credit_card cd) ?v3 - hexagonal_bin)
        (at-end
          (in ?v3 ?v2)
       )
     )
   )
 )
)
(:scoring
  (+ (* 40 (count preference0) (count preference2))
    (count preference1)
 )
)
)

; Key (1, 1, 3, 0, 0, 2, 0, 0, 0, 0, 0, 0)
(define (game evo-8171-105-0) (:domain many-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - dodgeball)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (on side_table ?v0)))
       )
     )
   )
    (preference preference1
      (exists (?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (on bed ?v1)))
       )
     )
   )
    (preference preference2
      (exists (?v0 - hexagonal_bin ?v2 - dodgeball)
        (then
          (once (agent_holds ?v2))
          (hold (and (not (agent_holds ?v2)) (in_motion ?v2)))
          (once (and (not (in_motion ?v2)) (on ?v0 ?v2)))
       )
     )
   )
 )
)
(:scoring
  (+ (count preference0) (count preference2) (count preference1))
)
)

; Key (1, 1, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0)
(define (game evo-8127-244-0) (:domain many-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - cube_block ?v1 - cube_block ?v2 - tall_rectangular_block ?v3 - cube_block)
        (at-end
          (and
            (on ?v3 ?v0)
            (on ?v3 ?v2)
            (on ?v3 ?v1)
         )
       )
     )
   )
    (preference preference1
      (exists (?v0 - cube_block ?v1 - cube_block ?v2 - tall_rectangular_block ?v3 - cube_block)
        (at-end
          (and
            (on ?v2 ?v0)
            (on ?v1 ?v2)
            (on ?v2 ?v3)
            (same_type ?v1 ?v2)
         )
       )
     )
   )
    (preference preference2
      (exists (?v0 - cube_block ?v1 - cube_block ?v2 - tall_rectangular_block ?v3 - cube_block)
        (at-end
          (and
            (on ?v3 ?v2)
            (on ?v0 ?v3)
            (same_type ?v3 ?v1)
         )
       )
     )
   )
 )
)
(:scoring
  (+ (* 1 (count preference1) (count preference0))
    (count preference2)
 )
)
)

; Key (1, 1, 3, 0, 0, 0, 0, 2, 0, 0, 0, 0)
(define (game evo-8085-134-1) (:domain many-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - chair ?v1 - desk)
        (at-end
          (and
            (on ?v1 ?v0)
         )
       )
     )
   )
    (preference preference1
      (exists (?v2 - chair ?v1 - desktop)
        (at-end
          (and
            (on ?v1 ?v2)
            (not
              (agent_holds ?v2)
           )
            (in_motion ?v2)
         )
       )
     )
   )
    (preference preference2
      (exists (?v0 - chair)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (not (in_motion ?v0)))
       )
     )
   )
 )
)
(:terminal
  (>= (total-score) 10)
)
(:scoring
  (+ (count preference2) (* 4 (count preference1) (count preference0))
 )
)
)

; Key (1, 0, 3, 0, 0, 0, 0, 0, 1, 0, 0, 0)
(define (game evo-8176-150-0) (:domain many-objects-room-v1)
(:setup
  (exists (?v0 - hexagonal_bin)
    (game-conserved
      (near rug ?v0)
   )
 )
)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - hexagonal_bin ?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (on ?v0 ?v1)))
       )
     )
   )
    (preference preference1
      (exists (?v0 - hexagonal_bin ?v2 - cylindrical_block_tan)
        (then
          (once (agent_holds ?v2))
          (hold (and (in_motion ?v2) (not (agent_holds ?v2))))
          (once (and (not (in_motion ?v2)) (in ?v0 ?v2)))
       )
     )
   )
    (preference preference2
      (exists (?v3 - hexagonal_bin ?v0 - dodgeball)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (on ?v3 ?v0) (not (in_motion ?v0))))
       )
     )
   )
 )
)
(:scoring
  (+ (count preference0) (* 6 (count preference2))
    (count preference1)
 )
)
)

; Key (1, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0)
(define (game evo-8038-66-1) (:domain many-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - hexagonal_bin ?v1 - golfball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (on ?v0 ?v1)))
       )
     )
   )
    (preference preference1
      (exists (?v1 - (either basketball golfball_white))
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (on top_drawer ?v1)))
       )
     )
   )
    (preference preference2
      (exists (?v0 - hexagonal_bin ?v2 - basketball)
        (then
          (once (agent_holds ?v2))
          (hold (and (not (agent_holds ?v2)) (in_motion ?v2)))
          (once (and (on ?v0 ?v2) (not (in_motion ?v2))))
       )
     )
   )
 )
)
(:scoring
  (+ (count preference0) (* 40 (count preference1) (count preference0) (count preference2))
 )
)
)

; Key (1, 1, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0)
(define (game evo-8058-109-0) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - hexagonal_bin ?v1 - golfball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (on ?v0 ?v1)))
       )
     )
   )
    (preference preference1
      (exists (?v0 - hexagonal_bin ?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (on ?v0 ?v1) (not (in_motion ?v1)) (in ?v0 ?v1)))
       )
     )
   )
    (preference preference2
      (exists (?v2 - hexagonal_bin ?v0 - dodgeball)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (on ?v2 ?v0) (not (in_motion ?v0))))
       )
     )
   )
 )
)
(:scoring
  (+ (count preference2) (count preference1) (count preference0))
)
)

; Key (1, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2)
(define (game evo-8114-226-0) (:domain many-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - game_object)
        (at-end
          (and
            (not
              (in_motion ?v0)
           )
            (near rug ?v0)
         )
       )
     )
   )
    (preference preference1
      (exists (?v0 - game_object)
        (at-end
          (and
            (not
              (in_motion ?v0)
           )
            (near door ?v0)
         )
       )
     )
   )
    (preference preference2
      (exists (?v1 - game_object)
        (at-end
          (broken ?v1)
       )
     )
   )
 )
)
(:scoring
  (+ (* 0.7 (count preference2) (count preference0) (count preference1))
    (count preference2)
 )
)
)

; Key (1, 1, 3, 0, 0, 1, 0, 0, 1, 0, 0, 0)
(define (game evo-8142-307-1) (:domain many-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - triangle_block_green)
        (then
          (once (agent_holds ?v0))
          (hold (not (agent_holds ?v0)))
          (hold (agent_holds ?v0))
          (once (not (in_motion ?v0)))
       )
     )
   )
    (preference preference1
      (exists (?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (on desk ?v1)))
       )
     )
   )
    (preference preference2
      (exists (?v0 - hexagonal_bin ?v1 - basketball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (on ?v0 ?v1)))
       )
     )
   )
 )
)
(:scoring
  (+ (count preference0) (count preference2) (count preference1))
)
)

; Key (1, 1, 3, 1, 0, 0, 0, 0, 0, 2, 0, 0)
(define (game evo-8127-145-1) (:domain many-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - block ?v1 - cube_block ?v2 - cube_block_blue ?v3 - bridge_block)
        (at-end
          (and
            (on ?v2 ?v3)
            (on ?v3 ?v0)
            (on ?v1 ?v3)
         )
       )
     )
   )
    (preference preference1
      (exists (?v1 - dodgeball_red)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (not (in_motion ?v1)))
       )
     )
   )
    (preference preference2
      (exists (?v1 - cube_block ?v0 - cube_block ?v2 - flat_block_gray ?v3 - cube_block)
        (at-end
          (and
            (on ?v2 ?v1)
            (on ?v2 ?v0)
            (on ?v3 ?v2)
            (same_type ?v3 ?v2)
         )
       )
     )
   )
 )
)
(:scoring
  (+ (* 4 (count preference2))
    (count preference0)
    (count preference1)
 )
)
)

; Key (1, 0, 4, 0, 1, 1, 0, 1, 0, 1, 0, 0)
(define (game evo-8184-38-0) (:domain many-objects-room-v1)
(:setup
  (exists (?v0 - hexagonal_bin)
    (game-conserved
      (near rug ?v0)
   )
 )
)
(:constraints
  (and
    (preference preference0
      (exists (?v1 - chair ?v0 - desk)
        (at-end
          (and
            (on ?v0 ?v1)
         )
       )
     )
   )
    (preference preference1
      (exists (?v2 - golfball)
        (then
          (once (agent_holds ?v2))
          (hold (and (not (agent_holds ?v2)) (in_motion ?v2)))
          (once (and (not (in_motion ?v2)) (on desk ?v2)))
       )
     )
   )
    (preference preference2
      (exists (?v0 - hexagonal_bin ?v2 - dodgeball)
        (then
          (once (agent_holds ?v2))
          (hold (and (not (agent_holds ?v2)) (in_motion ?v2)))
          (once (and (not (in_motion ?v2)) (in ?v0 ?v2) (on ?v0 ?v2)))
       )
     )
   )
    (preference preference3
      (exists (?v0 - cube_block ?v2 - block ?v3 - block ?v1 - cube_block)
        (at-end
          (and
            (on ?v3 ?v0)
            (on ?v3 ?v2)
            (on ?v3 ?v1)
            (same_type ?v1 ?v3)
         )
       )
     )
   )
 )
)
(:scoring
  (+ (* 1 (count preference2) (count preference0) (count preference1))
    (count preference3)
 )
)
)

; Key (1, 1, 4, 0, 0, 0, 3, 0, 0, 0, 1, 0)
(define (game evo-8128-42-1) (:domain many-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - hexagonal_bin ?v1 - ball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (in ?v0 ?v1)))
       )
     )
   )
    (preference preference1
      (exists (?v2 - (either credit_card cd) ?v1 - hexagonal_bin)
        (at-end
          (not
            (in ?v1 ?v2)
         )
       )
     )
   )
    (preference preference2
      (exists (?v0 - (either credit_card cd) ?v2 - hexagonal_bin)
        (at-end
          (in ?v2 ?v0)
       )
     )
   )
    (preference preference3
      (exists (?v0 - (either credit_card cd pencil))
        (at-end
          (in top_drawer ?v0)
       )
     )
   )
 )
)
(:scoring
  (+ (* -3 (count preference0) (count preference1) (count preference3))
    (count preference2)
 )
)
)

; Key (1, 0, 4, 0, 0, 0, 0, 0, 3, 0, 0, 0)
(define (game evo-8174-187-1) (:domain many-objects-room-v1)
(:setup
  (exists (?v0 - hexagonal_bin)
    (game-conserved
      (near top_shelf ?v0)
   )
 )
)
(:constraints
  (and
    (preference preference0
      (exists (?v1 - triangle_block_green)
        (then
          (once (agent_holds ?v1))
          (hold (not (agent_holds ?v1)))
          (hold (agent_holds ?v1))
          (once (not (in_motion ?v1)))
       )
     )
   )
    (preference preference1
      (exists (?v2 - hexagonal_bin ?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (on ?v2 ?v1)))
       )
     )
   )
    (preference preference2
      (exists (?v0 - cube_block)
        (then
          (once (agent_holds ?v0))
          (hold (and (in_motion ?v0) (not (agent_holds ?v0))))
          (once (not (in_motion ?v0)))
       )
     )
   )
    (preference preference3
      (exists (?v1 - cube_block)
        (then
          (once (agent_holds ?v1))
          (hold (not (agent_holds ?v1)))
          (hold (and (in_motion ?v1) (not (agent_holds ?v1))))
          (once (not (in_motion ?v1)))
       )
     )
   )
 )
)
(:scoring
  (+ (* 3 (count preference3) (count preference0) (count preference2))
    (count preference1)
 )
)
)

; Key (1, 1, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0)
(define (game evo-8128-182-0) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - dodgeball ?v1 - doggie_bed)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (on ?v1 ?v0)))
       )
     )
   )
    (preference preference1
      (exists (?v1 - hexagonal_bin ?v0 - dodgeball)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (on ?v1 ?v0) (not (in_motion ?v0))))
       )
     )
   )
    (preference preference2
      (exists (?v0 - dodgeball)
        (then
          (once (agent_holds ?v0))
          (hold (and (in_motion ?v0) (not (agent_holds ?v0))))
          (once (not (in_motion ?v0)))
       )
     )
   )
    (preference preference3
      (exists (?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (not (in_motion ?v1)))
       )
     )
   )
 )
)
(:scoring
  (+ (count preference3) (count preference2) (count preference0) (count preference1))
)
)

; Key (1, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 3)
(define (game evo-8153-151-1) (:domain medium-objects-room-v1)
(:setup
  (exists (?v0 - hexagonal_bin)
    (game-conserved
      (near door ?v0)
   )
 )
)
(:constraints
  (and
    (preference preference0
      (exists (?v1 - game_object)
        (at-end
          (and
            (not
              (in_motion ?v1)
           )
            (near top_shelf ?v1)
         )
       )
     )
   )
    (preference preference1
      (exists (?v1 - game_object)
        (at-end
          (and
            (not
              (in_motion ?v1)
           )
            (near floor ?v1)
         )
       )
     )
   )
    (preference preference2
      (exists (?v0 - game_object)
        (at-end
          (and
            (not
              (agent_holds ?v0)
           )
            (not
              (in_motion ?v0)
           )
            (near north_wall ?v0)
         )
       )
     )
   )
    (preference preference3
      (exists (?v0 - hexagonal_bin ?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (on ?v0 ?v1)))
       )
     )
   )
 )
)
(:scoring
  (+ (count preference3) (count preference1) (count preference0) (count preference2))
)
)

; Key (1, 1, 4, 0, 0, 1, 1, 1, 0, 1, 0, 0)
(define (game evo-8164-141-0) (:domain many-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - chair ?v1 - desk)
        (at-end
          (and
            (on ?v1 ?v0)
         )
       )
     )
   )
    (preference preference1
      (exists (?v0 - (either pen cd) ?v1 - hexagonal_bin)
        (at-end
          (in ?v1 ?v0)
       )
     )
   )
    (preference preference2
      (exists (?v0 - dodgeball)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (on bed ?v0)))
       )
     )
   )
    (preference preference3
      (exists (?v0 - cube_block ?v1 - cube_block ?v2 - tall_rectangular_block ?v3 - cube_block)
        (at-end
          (and
            (on ?v2 ?v0)
            (on ?v2 ?v1)
            (on ?v3 ?v2)
            (same_type ?v2 ?v1)
         )
       )
     )
   )
 )
)
(:scoring
  (+ (* 2 (count preference1) (count preference0))
    (count preference2)
    (count preference3)
 )
)
)

; Key (1, 0, 4, 2, 0, 0, 0, 0, 0, 0, 0, 1)
(define (game evo-8176-302-1) (:domain medium-objects-room-v1)
(:setup
  (exists (?v0 - hexagonal_bin)
    (game-conserved
      (near door ?v0)
   )
 )
)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - game_object)
        (at-end
          (and
            (not
              (in_motion ?v0)
           )
            (near rug ?v0)
         )
       )
     )
   )
    (preference preference1
      (exists (?v0 - hexagonal_bin ?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (on ?v0 ?v1)))
       )
     )
   )
    (preference preference2
      (exists (?v0 - dodgeball)
        (then
          (once (agent_holds ?v0))
          (hold (and (in_motion ?v0) (not (agent_holds ?v0))))
          (once (not (in_motion ?v0)))
       )
     )
   )
    (preference preference3
      (exists (?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (not (in_motion ?v1)))
       )
     )
   )
 )
)
(:scoring
  (+ (count preference1) (count preference0) (count preference2) (count preference3))
)
)

; Key (1, 1, 4, 0, 2, 0, 0, 0, 1, 0, 0, 0)
(define (game evo-8169-311-1) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - hexagonal_bin ?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (on ?v0 ?v1) (not (in_motion ?v1)) (in ?v0 ?v1)))
       )
     )
   )
    (preference preference1
      (exists (?v1 - cube_block)
        (then
          (once (agent_holds ?v1))
          (hold (and (in_motion ?v1) (not (agent_holds ?v1))))
          (once (not (in_motion ?v1)))
       )
     )
   )
    (preference preference2
      (exists (?v1 - cube_block_yellow ?v2 - cube_block ?v3 - block ?v4 - cylindrical_block)
        (at-end
          (and
            (on ?v3 ?v1)
            (on ?v3 ?v2)
            (adjacent ?v3 ?v4)
         )
       )
     )
   )
    (preference preference3
      (exists (?v0 - hexagonal_bin ?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (in ?v0 ?v1) (on ?v0 ?v1)))
       )
     )
   )
 )
)
(:scoring
  (+ (count preference3) (count preference2) (count preference1) (count preference0))
)
)

; Key (1, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0)
(define (game evo-8182-43-0) (:domain many-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - hexagonal_bin ?v1 - golfball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (on ?v0 ?v1)))
       )
     )
   )
    (preference preference1
      (exists (?v1 - (either basketball golfball_white))
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (on top_drawer ?v1)))
       )
     )
   )
    (preference preference2
      (exists (?v0 - hexagonal_bin ?v2 - basketball)
        (then
          (once (agent_holds ?v2))
          (hold (and (not (agent_holds ?v2)) (in_motion ?v2)))
          (once (and (on ?v0 ?v2) (not (in_motion ?v2))))
       )
     )
   )
    (preference preference3
      (exists (?v3 - hexagonal_bin ?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (on ?v3 ?v1)))
       )
     )
   )
 )
)
(:scoring
  (+ (count preference0) (* 40 (count preference1) (count preference3) (count preference2))
 )
)
)

; Key (1, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0)
(define (game evo-8165-162-0) (:domain many-objects-room-v1)
(:setup
  (exists (?v0 - hexagonal_bin)
    (game-conserved
      (near rug ?v0)
   )
 )
)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - hexagonal_bin ?v1 - golfball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (on ?v0 ?v1)))
       )
     )
   )
    (preference preference1
      (exists (?v1 - (either basketball golfball_white))
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (on top_drawer ?v1)))
       )
     )
   )
    (preference preference2
      (exists (?v0 - hexagonal_bin ?v2 - basketball)
        (then
          (once (agent_holds ?v2))
          (hold (and (not (agent_holds ?v2)) (in_motion ?v2)))
          (once (and (on ?v0 ?v2) (not (in_motion ?v2))))
       )
     )
   )
    (preference preference3
      (exists (?v3 - hexagonal_bin ?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (on ?v3 ?v1)))
       )
     )
   )
 )
)
(:scoring
  (+ (count preference0) (* 40 (count preference1) (count preference3) (count preference2))
 )
)
)

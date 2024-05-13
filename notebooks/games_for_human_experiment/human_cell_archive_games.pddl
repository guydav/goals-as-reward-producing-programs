; Key (1, 0, 3, 0, 0, 0, 0, 0, 2, 1, 0, 0)
(define (game evo-8185-363-1) (:domain many-objects-room-v1)
(:setup
  (forall (?v0 - pyramid_block_red)
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
    (preference preference1
      (exists (?v2 - building ?v3 - cube_block_yellow)
        (then
          (once (agent_holds ?v3))
          (hold (in ?v2 ?v3))
          (once (not (in_motion ?v3)))
       )
     )
   )
    (preference preference2
      (exists (?v1 - cube_block_yellow ?v2 - cube_block ?v3 - block)
        (at-end
          (and
            (on ?v2 ?v3)
            (on ?v3 ?v1)
            (same_type ?v3 ?v1)
         )
       )
     )
   )
 )
)
(:scoring
  (+ (* 80 (count preference0) (count preference2))
    (count preference1)
 )
)
)

; Key (1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0)
(define (game evo-8141-73-1) (:domain medium-objects-room-v1)
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
 )
)
(:scoring
  (+ (count preference0) (count preference1))
)
)

; Key (1, 0, 2, 0, 0, 0, 0, 0, 0, 1, 1, 0)
(define (game evo-8171-370-1) (:domain many-objects-room-v1)
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
          (once (and (not (in_motion ?v1)) (in ?v0 ?v1)))
       )
     )
   )
    (preference preference1
      (exists (?v0 - cube_block ?v1 - cube_block ?v2 - bridge_block ?v3 - cube_block_blue)
        (at-end
          (and
            (on ?v2 ?v0)
            (on ?v2 ?v1)
            (on ?v2 ?v3)
            (same_type ?v3 ?v2)
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

; Key (1, 0, 2, 0, 0, 1, 0, 0, 0, 0, 1, 0)
(define (game evo-8178-6-0) (:domain medium-objects-room-v1)
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
      (exists (?v0 - dodgeball)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (on desk ?v0)))
       )
     )
   )
    (preference preference1
      (exists (?v0 - hexagonal_bin ?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (in ?v0 ?v1)))
       )
     )
   )
 )
)
(:scoring
  (+ (count preference1) (count preference0))
)
)

; Key (1, 1, 2, 1, 0, 0, 0, 0, 0, 0, 1, 0)
(define (game evo-8184-200-0) (:domain many-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - dodgeball ?v1 - doggie_bed)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (in ?v1 ?v0)))
       )
     )
   )
    (preference preference1
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
  (+ (* -3 (count preference0))
    (count preference1)
 )
)
)

; Key (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
(define (game evo-8109-371-1) (:domain many-objects-room-v1)
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
 )
)
(:scoring
  (count preference0)
)
)

; Key (1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0)
(define (game evo-8131-278-0) (:domain many-objects-room-v1)
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
      (exists (?v0 - hexagonal_bin ?v1 - basketball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (in ?v0 ?v1)))
       )
     )
   )
    (preference preference1
      (exists (?v0 - hexagonal_bin ?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (in_motion ?v1) (not (agent_holds ?v1))))
          (once (and (not (in_motion ?v1)) (in ?v0 ?v1)))
       )
     )
   )
 )
)
(:scoring
  (+ (count preference1) (* 0.1 (count preference0))
 )
)
)

; Key (1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0)
(define (game evo-8124-185-0) (:domain medium-objects-room-v1)
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
          (once (and (not (in_motion ?v1)) (in ?v0 ?v1)))
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

; Key (1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0)
(define (game evo-8187-124-0) (:domain many-objects-room-v1)
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
          (hold (and (in_motion ?v0) (not (agent_holds ?v0))))
          (once (and (not (in_motion ?v0)) (in ?v1 ?v0)))
       )
     )
   )
 )
)
(:scoring
  (+ (count preference0) (count preference1))
)
)

; Key (1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0)
(define (game evo-8158-236-1) (:domain medium-objects-room-v1)
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
 )
)
(:scoring
  (+ (count preference0) (count preference1))
)
)

; Key (1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0)
(define (game evo-8174-50-0) (:domain many-objects-room-v1)
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
      (exists (?v1 - hexagonal_bin ?v2 - dodgeball)
        (then
          (once (agent_holds ?v2))
          (hold (and (not (agent_holds ?v2)) (in_motion ?v2) (touch top_shelf ?v2)))
          (once (and (not (in_motion ?v2)) (on ?v1 ?v2)))
       )
     )
   )
 )
)
(:scoring
  (count preference0)
)
)

; Key (1, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0)
(define (game evo-8162-212-0) (:domain medium-objects-room-v1)
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
      (exists (?v2 - hexagonal_bin ?v3 - dodgeball)
        (then
          (once (agent_holds ?v3))
          (hold (and (not (agent_holds ?v3)) (in_motion ?v3)))
          (once (and (on ?v2 ?v3) (not (in_motion ?v3))))
       )
     )
   )
    (preference preference2
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
  (+ (count preference2) (count preference1) (count preference0))
)
)

; Key (1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0)
(define (game evo-8181-95-0) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - chair ?v1 - desk)
        (at-end
          (and
            (on ?v1 ?v0)
            (not
              (agent_holds ?v0)
           )
            (in_motion ?v0)
         )
       )
     )
   )
 )
)
(:terminal
  (>= (total-score) 30)
)
(:scoring
  (count preference0)
)
)

; Key (1, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0)
(define (game evo-8171-322-0) (:domain many-objects-room-v1)
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
      (exists (?v0 - hexagonal_bin ?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (on ?v0 ?v1) (not (in_motion ?v1)) (in ?v0 ?v1)))
       )
     )
   )
    (preference preference1
      (exists (?v2 - hexagonal_bin ?v3 - ball)
        (then
          (once (agent_holds ?v3))
          (hold (and (not (agent_holds ?v3)) (in_motion ?v3)))
          (once (and (not (in_motion ?v3)) (in ?v2 ?v3) (on ?v2 ?v3)))
       )
     )
   )
 )
)
(:scoring
  (+ (count preference1) (* 5 (count preference0))
 )
)
)

; Key (1, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1)
(define (game evo-8189-55-0) (:domain many-objects-room-v1)
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
      (exists (?v0 - dodgeball)
        (at-end
          (on rug ?v0)
       )
     )
   )
    (preference preference2
      (exists (?v1 - hexagonal_bin ?v2 - dodgeball)
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
  (+ (count preference2) (* 70 (count preference1) (count preference0))
 )
)
)

; Key (1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0)
(define (game evo-8180-66-0) (:domain medium-objects-room-v1)
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

; Key (1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0)
(define (game evo-8126-12-0) (:domain many-objects-room-v1)
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
  (count preference0)
)
)

; Key (1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0)
(define (game evo-8168-216-1) (:domain many-objects-room-v1)
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
      (exists (?v1 - hexagonal_bin ?v0 - dodgeball)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (in ?v1 ?v0)))
       )
     )
   )
 )
)
(:scoring
  (count preference0)
)
)

; Key (1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0)
(define (game evo-8175-229-0) (:domain few-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - dodgeball)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (on top_shelf ?v0)))
       )
     )
   )
 )
)
(:terminal
  (>= (total-time) 30)
)
(:scoring
  (count preference0)
)
)

; Key (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0)
(define (game evo-8182-124-0) (:domain many-objects-room-v1)
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
 )
)
(:scoring
  (count preference0)
)
)

; Key (1, 0, 2, 1, 0, 0, 0, 0, 0, 0, 1, 0)
(define (game evo-8188-337-0) (:domain many-objects-room-v1)
(:setup
  (exists (?v0 - hexagonal_bin)
    (game-conserved
      (near east_wall ?v0)
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
 )
)
(:scoring
  (+ (count preference1) (* 0.7 (count preference0))
 )
)
)

; Key (1, 0, 2, 1, 0, 1, 0, 0, 0, 0, 0, 0)
(define (game evo-8145-186-0) (:domain many-objects-room-v1)
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
      (exists (?v0 - hexagonal_bin ?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1) (touch rug ?v1)))
          (once (and (not (in_motion ?v1)) (on ?v0 ?v1)))
       )
     )
   )
    (preference preference1
      (exists (?v2 - dodgeball)
        (then
          (once (agent_holds ?v2))
          (hold (and (not (agent_holds ?v2)) (in_motion ?v2)))
          (once (not (in_motion ?v2)))
       )
     )
   )
 )
)
(:scoring
  (+ (count preference1) (* 5 (count preference0))
 )
)
)

; Key (1, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0)
(define (game evo-8183-271-1) (:domain many-objects-room-v1)
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
      (exists (?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (on desk ?v1)))
       )
     )
   )
    (preference preference1
      (exists (?v0 - hexagonal_bin ?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1) (touch rug ?v1)))
          (once (and (not (in_motion ?v1)) (on ?v0 ?v1)))
       )
     )
   )
 )
)
(:scoring
  (+ (count preference0) (* 70 (count preference1))
 )
)
)

; Key (1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0)
(define (game evo-8176-22-0) (:domain many-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - block ?v1 - block ?v2 - tall_rectangular_block ?v3 - block)
        (at-end
          (and
            (on ?v2 ?v0)
            (on ?v2 ?v1)
            (on ?v2 ?v3)
            (same_type ?v0 ?v2)
            (same_type ?v1 ?v2)
         )
       )
     )
   )
 )
)
(:scoring
  (+ (* 0.4 (count preference0))
    (count preference0)
 )
)
)

; Key (1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0)
(define (game evo-8176-180-0) (:domain medium-objects-room-v1)
(:setup
  (exists (?v0 - hexagonal_bin)
    (game-conserved
      (and
        (adjacent ?v0 rug)
     )
   )
 )
)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - (either cellphone cd) ?v1 - hexagonal_bin)
        (at-end
          (in ?v1 ?v0)
       )
     )
   )
 )
)
(:scoring
  (+ (* 1 (count preference0))
    (count preference0)
 )
)
)

; Key (1, 0, 4, 0, 0, 2, 0, 0, 0, 0, 1, 0)
(define (game evo-8191-71-1) (:domain many-objects-room-v1)
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
          (once (and (not (in_motion ?v1)) (in ?v0 ?v1)))
       )
     )
   )
    (preference preference1
      (exists (?v0 - dodgeball)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (on bed ?v0)))
       )
     )
   )
    (preference preference2
      (exists (?v2 - dodgeball)
        (then
          (once (agent_holds ?v2))
          (hold (and (not (agent_holds ?v2)) (in_motion ?v2)))
          (once (and (not (in_motion ?v2)) (on desk ?v2)))
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
  (+ (count preference2) (count preference0) (count preference3) (count preference1))
)
)

; Key (1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
(define (game evo-8147-296-1) (:domain many-objects-room-v1)
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
      (exists (?v1 - hexagonal_bin ?v0 - dodgeball)
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
  (count preference0)
)
)

; Key (1, 1, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0)
(define (game evo-8139-250-1) (:domain many-objects-room-v1)
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
  (+ (* 15 (count preference1))
    (count preference0)
 )
)
)

; Key (1, 0, 3, 1, 0, 0, 0, 0, 0, 0, 2, 0)
(define (game evo-8186-361-1) (:domain many-objects-room-v1)
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
      (exists (?v0 - doggie_bed ?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (not (agent_holds ?v1)) (in_motion ?v1)))
          (once (and (not (in_motion ?v1)) (in ?v0 ?v1)))
       )
     )
   )
    (preference preference1
      (exists (?v0 - hexagonal_bin ?v1 - dodgeball)
        (then
          (once (agent_holds ?v1))
          (hold (and (in_motion ?v1) (not (agent_holds ?v1))))
          (once (and (not (in_motion ?v1)) (in ?v0 ?v1)))
       )
     )
   )
    (preference preference2
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
  (+ (count preference2) (count preference0) (count preference1))
)
)

; Key (1, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0)
(define (game evo-8119-291-1) (:domain many-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - dodgeball ?v1 - doggie_bed)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (on ?v1 ?v0) (not (in_motion ?v0)) (in ?v1 ?v0)))
       )
     )
   )
    (preference preference1
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
  (+ (* -3 (count preference0))
    (count preference1)
 )
)
)

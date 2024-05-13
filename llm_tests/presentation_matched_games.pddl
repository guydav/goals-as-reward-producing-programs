;  (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)

(define (game game-33) (:domain many-objects-room-v1)
(:setup
  (forall (?g - game_object)
    (game-optional
      (not
        (in top_drawer ?g)
     )
   )
 )
)
(:constraints
  (and
    (preference itemInClosedDrawerAtEnd
      (exists (?g - game_object)
        (at-end
          (and
            (in top_drawer ?g)
            (not
              (open top_drawer)
           )
         )
       )
     )
   )
 )
)
(:scoring
  (count-once-per-objects itemInClosedDrawerAtEnd)
)
)

(define (game evo-7155-97-0) (:domain many-objects-room-v1)
(:setup
  (exists (?v0 - hexagonal_bin)
    (game-conserved
      (< (distance floor ?v0) 0.5)
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
              (same_type ?v1 pillow)
           )
            (< (distance mirror ?v1) (distance west_sliding_door ?v1))
         )
       )
     )
   )
 )
)
(:terminal
  (>= (total-time) 150)
)
(:scoring
  (count preference0)
)
)

; (1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0)

(define (game game-51) (:domain few-objects-room-v1)
(:constraints
  (and
    (preference throwToBin
      (exists (?d - dodgeball ?h - hexagonal_bin)
        (then
          (once (agent_holds ?d))
          (hold (and (not (agent_holds ?d)) (in_motion ?d)))
          (once (and (not (in_motion ?d)) (in ?h ?d)))
       )
     )
   )
 )
)
(:scoring
  (count throwToBin)
)
)

(define (game evo-7148-39-0) (:domain many-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - ball ?v1 - hexagonal_bin)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (in ?v1 ?v0)))
       )
     )
   )
 )
)
(:terminal
  (or
    (>= (total-score) -30)
    (>= (count preference0) 10)
    (>= (total-score) -1)
 )
)
(:scoring
  (count preference0)
)
)

;  (1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)

(define (game game-23) (:domain few-objects-room-v1)
(:constraints
  (and
    (preference throwBallToBin
      (exists (?d - dodgeball ?h - hexagonal_bin)
        (then
          (once (agent_holds ?d))
          (hold (and (not (agent_holds ?d)) (in_motion ?d)))
          (once (and (not (in_motion ?d)) (in ?h ?d)))
       )
     )
   )
    (preference throwAttempt
      (exists (?d - dodgeball)
        (then
          (once (agent_holds ?d))
          (hold (and (not (agent_holds ?d)) (in_motion ?d)))
          (once (not (in_motion ?d)))
       )
     )
   )
 )
)
(:scoring
  (+ (count throwBallToBin) (- (/ (count throwAttempt) 5))
 )
)
)

(define (game evo-7163-358-0) (:domain many-objects-room-v1)
(:constraints
  (and
    (preference preference0
      (exists (?v0 - ball ?v1 - hexagonal_bin)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (and (not (in_motion ?v0)) (in ?v1 ?v0)))
       )
     )
   )
    (preference preference1
      (exists (?v0 - ball ?v1 - hexagonal_bin)
        (then
          (once (agent_holds ?v0))
          (hold (and (in_motion ?v0) (not (agent_holds ?v0))))
          (once (and (not (in_motion ?v0)) (in ?v1 ?v0)))
       )
     )
   )
    (preference preference2
      (exists (?v0 - ball)
        (then
          (once (agent_holds ?v0))
          (hold (and (not (agent_holds ?v0)) (in_motion ?v0)))
          (once (not (in_motion ?v0)))
       )
     )
   )
    (preference preference3
      (exists (?v2 - ball)
        (then
          (once (agent_holds ?v2))
          (hold (and (in_motion ?v2) (not (agent_holds ?v2))))
          (once (not (in_motion ?v2)))
       )
     )
   )
 )
)
(:terminal
  (or
    (>= (count preference1) 7)
    (>= (count preference2) 18)
    (>= (count preference3) 10)
    (>= (count preference0) 6)
 )
)
(:scoring
  (count preference1)
)
)

;

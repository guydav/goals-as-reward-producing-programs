; Index #0 with key (1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0)
(define (game game-0) (:domain medium-objects-room-v1)
(:setup
  (exists (?h - hexagonal_bin ?r - triangular_ramp)
    (game-conserved
      (near ?h ?r)
   )
 )
)
(:constraints
  (and
    (preference throwToRampToBin
      (exists (?b - ball ?r - triangular_ramp ?h - hexagonal_bin)
        (then
          (once (agent_holds ?b))
          (hold-while (and (not (agent_holds ?b)) (in_motion ?b)) (touch ?b ?r))
          (once (and (not (in_motion ?b)) (in ?h ?b)))
       )
     )
   )
    (preference binKnockedOver
      (exists (?h - hexagonal_bin)
        (then
          (once (object_orientation ?h upright))
          (hold (and (not (touch agent ?h)) (not (agent_holds ?h))))
          (once (not (object_orientation ?h upright)))
       )
     )
   )
 )
)
(:terminal
  (>= (count-once binKnockedOver) 1)
)
(:scoring
  (count throwToRampToBin)
)
)

; Index #4 with key (1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0)
(define (game game-6) (:domain medium-objects-room-v1)
(:setup
  (and
    (exists (?h - hexagonal_bin)
      (game-conserved
        (adjacent ?h bed)
     )
   )
    (forall (?o - (either teddy_bear pillow))
      (game-conserved
        (not
          (on bed ?o)
       )
     )
   )
 )
)
(:constraints
  (and
    (forall (?b - ball)
      (and
        (preference throwBallToBin
          (exists (?h - hexagonal_bin)
            (then
              (once (and (agent_holds ?b) (adjacent desk agent)))
              (hold (and (not (agent_holds ?b)) (in_motion ?b)))
              (once (and (not (in_motion ?b)) (in ?h ?b)))
           )
         )
       )
     )
   )
    (preference failedThrowToBin
      (exists (?b - ball ?h - hexagonal_bin)
        (then
          (once (and (agent_holds ?b) (adjacent desk agent)))
          (hold (and (not (agent_holds ?b)) (in_motion ?b)))
          (once (and (not (in_motion ?b)) (not (in ?h ?b))))
       )
     )
   )
 )
)
(:scoring
  (+ (* 10 (count throwBallToBin:dodgeball))
    (* 20 (count throwBallToBin:basketball))
    (* 30 (count throwBallToBin:beachball))
    (- (count failedThrowToBin))
 )
)
)

; Index #6 with key (1, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0)
(define (game game-9) (:domain many-objects-room-v1)
(:setup
  (exists (?h - hexagonal_bin)
    (game-conserved
      (or
        (on bed ?h)
        (exists (?w - wall)
          (adjacent ?w ?h)
       )
     )
   )
 )
)
(:constraints
  (and
    (preference throwBallToBin
      (exists (?d - dodgeball ?h - hexagonal_bin)
        (then
          (once (and (agent_holds ?d) (or (on bed ?h) (exists (?w1 ?w2 - wall) (and (adjacent ?w1 ?h) (adjacent ?w2 agent) (opposite ?w1 ?w2))))))
          (hold (and (not (agent_holds ?d)) (in_motion ?d) (not (touch floor ?d))))
          (once (and (not (in_motion ?d)) (in ?h ?d)))
       )
     )
   )
    (preference bounceBallToBin
      (exists (?d - dodgeball ?h - hexagonal_bin)
        (then
          (once (and (agent_holds ?d) (or (on bed ?h) (exists (?w1 ?w2 - wall) (and (adjacent ?w1 ?h) (adjacent ?w2 agent) (opposite ?w1 ?w2))))))
          (hold-while (and (not (agent_holds ?d)) (in_motion ?d)) (touch floor ?d))
          (once (and (not (in_motion ?d)) (in ?h ?d)))
       )
     )
   )
 )
)
(:scoring
  (+ (count bounceBallToBin) (* 3 (count throwBallToBin))
 )
)
)

; Index #7 with key (1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0)
(define (game game-10) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference throwTeddyOntoPillow
      (exists (?t - teddy_bear ?p - pillow)
        (then
          (once (agent_holds ?t))
          (hold (and (not (agent_holds ?t)) (in_motion ?t)))
          (once (and (not (in_motion ?t)) (on ?p ?t)))
       )
     )
   )
    (preference throwAttempt
      (exists (?t - teddy_bear)
        (then
          (once (agent_holds ?t))
          (hold (and (not (agent_holds ?t)) (in_motion ?t)))
          (once (not (in_motion ?t)))
       )
     )
   )
 )
)
(:terminal
  (>= (count throwAttempt) 10)
)
(:scoring
  (count throwTeddyOntoPillow)
)
)

; Index #11 with key (1, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0)
(define (game game-14) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference throwInBin
      (exists (?b - ball ?h - hexagonal_bin)
        (then
          (once (and (on rug agent) (agent_holds ?b)))
          (hold (and (not (agent_holds ?b)) (in_motion ?b)))
          (once (and (not (in_motion ?b)) (in ?h ?b)))
       )
     )
   )
    (preference throwAttempt
      (exists (?b - ball)
        (then
          (once (agent_holds ?b))
          (hold (and (not (agent_holds ?b)) (in_motion ?b)))
          (once (not (in_motion ?b)))
       )
     )
   )
 )
)
(:terminal
  (>= (count throwAttempt) 10)
)
(:scoring
  (count throwInBin)
)
)

; Index #14 with key (1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0)
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
         )
       )
     )
   )
 )
)
(:scoring
  (* 10 (count-once-per-objects castleBuilt))
)
)

; Index #17 with key (1, 0, 4, 0, 0, 2, 0, 0, 0, 0, 1, 0)
(define (game game-21) (:domain few-objects-room-v1)
(:setup
  (exists (?c - chair)
    (game-conserved
      (and
        (near room_center ?c)
        (not
          (faces ?c desk)
       )
        (not
          (faces ?c bed)
       )
     )
   )
 )
)
(:constraints
  (and
    (preference ballThrownToBin
      (exists (?d - dodgeball ?h - hexagonal_bin)
        (then
          (once (and (agent_holds ?d) (adjacent desk agent)))
          (hold (and (not (agent_holds ?d)) (in_motion ?d)))
          (once (and (not (in_motion ?d)) (in ?h ?d)))
       )
     )
   )
    (preference ballThrownToBed
      (exists (?d - dodgeball)
        (then
          (once (and (agent_holds ?d) (adjacent desk agent)))
          (hold (and (not (agent_holds ?d)) (in_motion ?d)))
          (once (and (not (in_motion ?d)) (on bed ?d)))
       )
     )
   )
    (preference ballThrownToChair
      (exists (?d - dodgeball ?c - chair)
        (then
          (once (and (agent_holds ?d) (adjacent desk agent)))
          (hold (and (not (agent_holds ?d)) (in_motion ?d)))
          (once (and (not (in_motion ?d)) (on ?c ?d) (is_setup_object ?c)))
       )
     )
   )
    (preference ballThrownMissesEverything
      (exists (?d - dodgeball)
        (then
          (once (and (agent_holds ?d) (adjacent desk agent)))
          (hold (and (not (agent_holds ?d)) (in_motion ?d)))
          (once (and (not (in_motion ?d)) (not (exists (?h - hexagonal_bin) (in ?h ?d))) (not (on bed ?d)) (not (exists (?c - chair) (and (on ?c ?d) (is_setup_object ?c))))))
       )
     )
   )
 )
)
(:terminal
  (>= (total-score) 10)
)
(:scoring
  (+ (* 5 (count ballThrownToBin))
    (count ballThrownToBed)
    (count ballThrownToChair)
    (- (count ballThrownMissesEverything))
 )
)
)

; Index #23 with key (1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0)
(define (game game-29) (:domain few-objects-room-v1)
(:constraints
  (and
    (preference objectOnBed
      (exists (?g - game_object)
        (at-end
          (and
            (not
              (same_type ?g pillow)
           )
            (on bed ?g)
         )
       )
     )
   )
 )
)
(:scoring
  (count objectOnBed)
)
)

; Index #26 with key (1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0)
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

; Index #28 with key (1, 0, 2, 1, 0, 0, 0, 0, 0, 0, 1, 0)
(define (game game-36) (:domain few-objects-room-v1)
(:setup
  (and
    (exists (?h - hexagonal_bin)
      (game-conserved
        (on bed ?h)
     )
   )
    (forall (?d - dodgeball)
      (game-optional
        (on desk ?d)
     )
   )
 )
)
(:constraints
  (and
    (preference throwToBin
      (exists (?d - dodgeball ?h - hexagonal_bin)
        (then
          (once (and (agent_holds ?d) (adjacent desk agent)))
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
(:terminal
  (>= (count throwAttempt) 5)
)
(:scoring
  (count throwToBin)
)
)

; Index #31 with key (1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0)
(define (game game-39) (:domain many-objects-room-v1)
(:constraints
  (and
    (preference ballThrownToWallToAgent
      (exists (?b - ball ?w - wall)
        (then
          (once (agent_holds ?b))
          (hold-while (and (not (agent_holds ?b)) (in_motion ?b)) (touch ?w ?b))
          (once (or (agent_holds ?b) (touch agent ?b)))
       )
     )
   )
 )
)
(:scoring
  (count ballThrownToWallToAgent)
)
)

; Index #32 with key (1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0)
(define (game game-40) (:domain many-objects-room-v1)
(:setup
  (exists (?r - curved_wooden_ramp)
    (game-conserved
      (adjacent ?r rug)
   )
 )
)
(:constraints
  (and
    (forall (?x - color)
      (and
        (preference ballRolledOnRampToRug
          (exists (?b - beachball ?r - curved_wooden_ramp)
            (then
              (once (agent_holds ?b))
              (hold-while (and (not (agent_holds ?b)) (in_motion ?b)) (on ?r ?b))
              (once (and (not (in_motion ?b)) (on rug ?b) (rug_color_under ?b ?x)))
           )
         )
       )
     )
   )
 )
)
(:scoring
  (+ (count ballRolledOnRampToRug:pink) (* 2 (count ballRolledOnRampToRug:yellow))
    (* 3 (count ballRolledOnRampToRug:orange))
    (* 3 (count ballRolledOnRampToRug:green))
    (* 4 (count ballRolledOnRampToRug:purple))
    (- (count ballRolledOnRampToRug:white))
 )
)
)

; Index #35 with key (1, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0)
(define (game game-43) (:domain medium-objects-room-v1)
(:setup
  (exists (?d - doggie_bed)
    (game-conserved
      (near room_center ?d)
   )
 )
)
(:constraints
  (and
    (forall (?b - ball)
      (and
        (preference throwBallToDoggieBed
          (exists (?d - doggie_bed)
            (then
              (once (agent_holds ?b))
              (hold (and (not (agent_holds ?b)) (in_motion ?b) (not (exists (?w - wall) (touch ?w ?b)))))
              (once (and (not (in_motion ?b)) (on ?d ?b)))
           )
         )
       )
        (preference throwBallToDoggieBedOffWall
          (exists (?d - doggie_bed ?w - wall)
            (then
              (once (agent_holds ?b))
              (hold-while (and (not (agent_holds ?d)) (in_motion ?b)) (touch ?w ?b))
              (once (and (not (in_motion ?b)) (on ?d ?b)))
           )
         )
       )
     )
   )
 )
)
(:scoring
  (+ (count throwBallToDoggieBed:basketball) (* 2 (count throwBallToDoggieBed:beachball))
    (* 3 (count throwBallToDoggieBed:dodgeball))
    (* 2 (count throwBallToDoggieBedOffWall:basketball))
    (* 3 (count throwBallToDoggieBedOffWall:beachball))
    (* 4 (count throwBallToDoggieBedOffWall:dodgeball))
 )
)
)

; Index #37 with key (1, 0, 2, 1, 0, 1, 0, 0, 0, 0, 0, 0)
(define (game game-46) (:domain few-objects-room-v1)
(:setup
  (exists (?c - curved_wooden_ramp)
    (game-conserved
      (near room_center ?c)
   )
 )
)
(:constraints
  (and
    (preference ballThrownToRampToBed
      (exists (?d - dodgeball_pink ?c - curved_wooden_ramp)
        (then
          (once (and (agent_holds ?d) (faces agent ?c)))
          (hold-while (and (in_motion ?d) (not (agent_holds ?d))) (touch ?d ?c))
          (once (and (not (in_motion ?d)) (on bed ?d)))
       )
     )
   )
    (preference ballThrownHitsAgent
      (exists (?d - dodgeball_pink ?c - curved_wooden_ramp)
        (then
          (once (and (agent_holds ?d) (faces agent ?c)))
          (hold-while (and (in_motion ?d) (not (agent_holds ?d))) (touch ?d ?c))
          (once (and (touch ?d agent) (not (agent_holds ?d))))
       )
     )
   )
 )
)
(:scoring
  (+ (count ballThrownToRampToBed) (- (count ballThrownHitsAgent))
 )
)
)

; Index #40 with key (1, 0, 2, 0, 0, 1, 0, 0, 0, 0, 1, 0)
(define (game game-49) (:domain many-objects-room-v1)
(:setup
  (exists (?g - golfball_green)
    (and
      (game-conserved
        (near door ?g)
     )
      (forall (?d - dodgeball)
        (game-optional
          (near ?d ?g)
       )
     )
   )
 )
)
(:constraints
  (and
    (forall (?d - dodgeball)
      (and
        (preference dodgeballThrownToBin
          (exists (?h - hexagonal_bin ?g - golfball_green)
            (then
              (once (and (adjacent ?g agent) (adjacent door agent) (agent_holds ?d)))
              (hold (and (in_motion ?d) (not (agent_holds ?d))))
              (once (and (not (in_motion ?d)) (in ?h ?d)))
           )
         )
       )
        (preference throwAttemptFromDoor
          (exists (?g - golfball_green)
            (then
              (once (and (adjacent ?g agent) (adjacent door agent) (agent_holds ?d)))
              (hold (and (in_motion ?d) (not (agent_holds ?d))))
              (once (not (in_motion ?d)))
           )
         )
       )
     )
   )
 )
)
(:terminal
  (or
    (> (external-forall-maximize (count throwAttemptFromDoor)) 1)
    (>= (count-once-per-objects throwAttemptFromDoor) 3)
 )
)
(:scoring
  (* 10 (count-once-per-objects dodgeballThrownToBin))
)
)

; Index #41 with key (1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
(define (game game-50) (:domain medium-objects-room-v1)
(:setup
  (exists (?h - hexagonal_bin)
    (game-conserved
      (near room_center ?h)
   )
 )
)
(:constraints
  (and
    (preference gameObjectToBin
      (exists (?g - game_object ?h - hexagonal_bin)
        (then
          (once (not (agent_holds ?g)))
          (hold (or (agent_holds ?g) (in_motion ?g)))
          (once (and (not (in_motion ?g)) (in ?h ?g)))
       )
     )
   )
 )
)
(:scoring
  (count-once-per-objects gameObjectToBin)
)
)

; Index #42 with key (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0)
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

; Index #45 with key (1, 1, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0)
(define (game game-54) (:domain few-objects-room-v1)
(:constraints
  (and
    (forall (?b - building)
      (and
        (preference blockPlacedInBuilding
          (exists (?l - cube_block)
            (then
              (once (agent_holds ?l))
              (hold (not (agent_holds ?l)))
              (hold (in ?b ?l))
              (once (or (not (in ?b ?l)) (game_over)))
           )
         )
       )
     )
   )
    (forall (?l - cube_block)
      (and
        (preference blockPickedUp
          (then
            (once (not (agent_holds ?l)))
            (hold (agent_holds ?l))
            (once (not (agent_holds ?l)))
         )
       )
     )
   )
 )
)
(:terminal
  (>= (external-forall-maximize (count blockPickedUp)) 3)
)
(:scoring
  (external-forall-maximize
    (count-overlapping blockPlacedInBuilding)
 )
)
)

; Index #49 with key (1, 0, 3, 0, 0, 0, 0, 0, 2, 1, 0, 0)
(define (game game-58) (:domain medium-objects-room-v1)
(:setup
  (and
    (forall (?l - block ?s - shelf)
      (game-optional
        (not
          (on ?s ?l)
       )
     )
   )
    (exists (?b - building)
      (game-conserved
        (and
          (= (building_size ?b) 6)
          (forall (?l - block)
            (or
              (in ?b ?l)
              (exists (?l2 - block)
                (and
                  (in ?b ?l2)
                  (not
                    (same_object ?l ?l2)
                 )
                  (same_type ?l ?l2)
               )
             )
           )
         )
       )
     )
   )
 )
)
(:constraints
  (and
    (preference gameBlockFound
      (exists (?l - block)
        (then
          (once (game_start))
          (hold (not (exists (?b - building) (and (in ?b ?l) (is_setup_object ?b)))))
          (once (agent_holds ?l))
       )
     )
   )
    (preference towerFallsWhileBuilding
      (exists (?b - building ?l1 ?l2 - block)
        (then
          (once (and (in ?b ?l1) (agent_holds ?l2) (not (is_setup_object ?b))))
          (hold-while (and (not (agent_holds ?l1)) (in ?b ?l1) (or (agent_holds ?l2) (in_motion ?l2))) (touch ?l1 ?l2))
          (hold (and (in_motion ?l1) (not (agent_holds ?l1))))
          (once (not (in_motion ?l1)))
       )
     )
   )
    (preference matchingBuildingBuilt
      (exists (?b1 ?b2 - building)
        (at-end
          (and
            (is_setup_object ?b1)
            (not
              (is_setup_object ?b2)
           )
            (forall (?l1 ?l2 - block)
              (or
                (not
                  (in ?b1 ?l1)
               )
                (not
                  (in ?b1 ?l2)
               )
                (not
                  (on ?l1 ?l2)
               )
                (exists (?l3 ?l4 - block)
                  (and
                    (in ?b2 ?l3)
                    (in ?b2 ?l4)
                    (on ?l3 ?l4)
                    (same_type ?l1 ?l3)
                    (same_type ?l2 ?l4)
                 )
               )
             )
           )
         )
       )
     )
   )
 )
)
(:scoring
  (+ (* 5 (count-once-per-objects gameBlockFound))
    (* 100 (count-once matchingBuildingBuilt))
    (* -10 (count towerFallsWhileBuilding))
 )
)
)

; Index #51 with key (1, 0, 2, 0, 0, 0, 0, 0, 0, 1, 1, 0)
(define (game game-61) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (and
      (exists (?f - flat_block)
        (on rug ?f)
     )
      (forall (?p - pyramid_block)
        (on floor ?p)
     )
      (exists (?p1 - pyramid_block_yellow ?p2 - pyramid_block_red ?p3 - pyramid_block_blue ?h - hexagonal_bin)
        (and
          (> (distance ?h ?p2) (distance ?h ?p1))
          (> (distance ?h ?p3) (distance ?h ?p2))
       )
     )
   )
 )
)
(:constraints
  (and
    (forall (?p - pyramid_block)
      (and
        (preference dodgeballFromBlockToBin
          (exists (?d - dodgeball ?h - hexagonal_bin)
            (then
              (once (and (agent_holds ?d) (adjacent ?p agent)))
              (hold (and (not (agent_holds ?d)) (in_motion ?d)))
              (once (and (not (in_motion ?d)) (in ?h ?d)))
           )
         )
       )
     )
   )
    (preference cubeBlockInBuilding
      (exists (?b - building ?l - cube_block ?f - flat_block)
        (at-end
          (and
            (is_setup_object ?f)
            (in ?b ?f)
            (in ?b ?l)
         )
       )
     )
   )
 )
)
(:scoring
  (+ (* 10 (count dodgeballFromBlockToBin:pyramid_block_yellow))
    (* 25 (count dodgeballFromBlockToBin:pyramid_block_red))
    (* 50 (count dodgeballFromBlockToBin:pyramid_block_blue))
    (* 100 (= (count-once-per-objects dodgeballFromBlockToBin:pyramid_block_blue) 3)
   )
    (* 10 (count-once-per-objects cubeBlockInBuilding))
    (* 100 (= (count-once-per-objects cubeBlockInBuilding) 3)
   )
 )
)
)

; Index #52 with key (1, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1)
(define (game game-62) (:domain medium-objects-room-v1)
(:constraints
  (and
    (preference bigObjectThrownToBed
      (exists (?o - (either chair laptop doggie_bed))
        (then
          (once (and (agent_holds ?o) (adjacent desk agent)))
          (hold (and (not (agent_holds ?o)) (in_motion ?o)))
          (once (and (not (in_motion ?o)) (on bed ?o)))
       )
     )
   )
    (preference smallObjectThrownToBed
      (exists (?o - game_object)
        (then
          (once (and (agent_holds ?o) (adjacent desk agent) (not (exists (?o2 - (either chair laptop doggie_bed)) (same_object ?o ?o2)))))
          (hold (and (not (agent_holds ?o)) (in_motion ?o)))
          (once (and (not (in_motion ?o)) (on bed ?o)))
       )
     )
   )
    (preference failedThrowAttempt
      (exists (?o - game_object)
        (then
          (once (and (agent_holds ?o) (adjacent desk agent)))
          (hold (and (not (agent_holds ?o)) (in_motion ?o)))
          (once (and (not (in_motion ?o)) (not (on bed ?o))))
       )
     )
   )
 )
)
(:scoring
  (+ (count smallObjectThrownToBed) (* 5 (count bigObjectThrownToBed))
    (* -5 (count failedThrowAttempt))
 )
)
)

; Index #55 with key (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
(define (game game-65) (:domain many-objects-room-v1)
(:constraints
  (and
    (preference ballOnBedAtEnd
      (exists (?b - ball)
        (at-end
          (on bed ?b)
       )
     )
   )
 )
)
(:scoring
  (count-once-per-objects ballOnBedAtEnd)
)
)

; Index #58 with key (1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0)
(define (game game-69) (:domain many-objects-room-v1)
(:setup
  (exists (?c - curved_wooden_ramp ?h - hexagonal_bin)
    (game-conserved
      (adjacent ?c ?h)
   )
 )
)
(:constraints
  (and
    (preference ballThrownThroughRampToBin
      (exists (?d - dodgeball ?c - curved_wooden_ramp ?h - hexagonal_bin)
        (then
          (once (agent_holds ?d))
          (hold-while (and (not (agent_holds ?d)) (in_motion ?d)) (touch ?d ?c))
          (once (and (not (in_motion ?d)) (in ?h ?d)))
       )
     )
   )
 )
)
(:scoring
  (count ballThrownThroughRampToBin)
)
)

; Index #59 with key (1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0)
(define (game game-70) (:domain many-objects-room-v1)
(:setup
  (and
    (forall (?c - chair)
      (game-conserved
        (not
          (adjacent_side desk front ?c)
       )
     )
   )
    (exists (?h - hexagonal_bin ?c - curved_wooden_ramp)
      (game-conserved
        (and
          (adjacent_side desk front ?c)
          (adjacent_side ?h front ?c back)
       )
     )
   )
    (forall (?o - (either golfball dodgeball triangle_block pyramid_block))
      (game-optional
        (near side_table ?o)
     )
   )
 )
)
(:constraints
  (and
    (forall (?o - (either golfball dodgeball triangle_block pyramid_block))
      (and
        (preference objectLandsInBin
          (exists (?h - hexagonal_bin)
            (then
              (once (and (adjacent bed agent) (agent_holds ?o)))
              (hold (and (in_motion ?o) (not (agent_holds ?o))))
              (once (and (not (in_motion ?o)) (in ?h ?o)))
           )
         )
       )
        (preference thrownObjectHitsComputer
          (exists (?c - (either desktop laptop))
            (then
              (once (and (adjacent bed agent) (agent_holds ?o)))
              (hold (and (in_motion ?o) (not (agent_holds ?o))))
              (once (touch ?o ?c))
           )
         )
       )
     )
   )
    (preference golfballLandsInBinThroughRamp
      (exists (?g - golfball ?c - curved_wooden_ramp ?h - hexagonal_bin)
        (then
          (once (and (adjacent bed agent) (agent_holds ?g)))
          (hold-while (and (in_motion ?g) (not (agent_holds ?g))) (touch ?c ?g))
          (once (and (not (in_motion ?g)) (in ?h ?g)))
       )
     )
   )
 )
)
(:scoring
  (+ (count objectLandsInBin:triangle_block) (* 2 (count objectLandsInBin:pyramid_block))
    (* 2 (count objectLandsInBin:dodgeball))
    (* 3 (count objectLandsInBin:golfball))
    (* 6 (count golfballLandsInBinThroughRamp))
    (- (count thrownObjectHitsComputer))
 )
)
)

; Index #64 with key (1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0)
(define (game game-75) (:domain few-objects-room-v1)
(:constraints
  (and
    (preference ballDroppedInBin
      (exists (?b - ball ?h - hexagonal_bin)
        (then
          (once (and (adjacent ?h agent) (agent_holds ?b)))
          (hold (and (in_motion ?b) (not (agent_holds ?b))))
          (once (and (not (in_motion ?b)) (in ?h ?b)))
       )
     )
   )
    (preference dropAttempt
      (exists (?b - ball ?h - hexagonal_bin)
        (then
          (once (and (adjacent ?h agent) (agent_holds ?b)))
          (hold (and (in_motion ?b) (not (agent_holds ?b))))
          (once (not (in_motion ?b)))
       )
     )
   )
 )
)
(:terminal
  (or
    (>= (count dropAttempt) 5)
    (>= (count ballDroppedInBin) 1)
 )
)
(:scoring
  (* 5 (count ballDroppedInBin))
)
)

; Index #74 with key (1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0)
(define (game game-87) (:domain few-objects-room-v1)
(:setup
  (exists (?h - hexagonal_bin ?w - wall)
    (game-conserved
      (and
        (on desk ?h)
        (adjacent ?h ?w)
     )
   )
 )
)
(:constraints
  (and
    (forall (?o - (either dodgeball block))
      (and
        (preference basketMadeFromRug
          (exists (?h - hexagonal_bin)
            (then
              (once (and (agent_holds ?o) (on rug agent)))
              (hold (and (in_motion ?o) (not (agent_holds ?o))))
              (once (and (not (in_motion ?o)) (in ?h ?o)))
           )
         )
       )
     )
   )
 )
)
(:scoring
  (+ (count basketMadeFromRug:dodgeball) (* 2 (count basketMadeFromRug:block))
 )
)
)

; Index #88 with key (1, 1, 2, 1, 0, 0, 0, 0, 0, 0, 1, 0)
(define (game game-106) (:domain few-objects-room-v1)
(:constraints
  (and
    (preference throwInBin
      (exists (?b - ball ?h - hexagonal_bin)
        (then
          (once (agent_holds ?b))
          (hold (and (not (agent_holds ?b)) (in_motion ?b)))
          (once (and (not (in_motion ?b)) (in ?h ?b)))
       )
     )
   )
    (preference throwAttempt
      (exists (?b - ball)
        (then
          (once (agent_holds ?b))
          (hold (and (not (agent_holds ?b)) (in_motion ?b)))
          (once (not (in_motion ?b)))
       )
     )
   )
 )
)
(:terminal
  (or
    (>= (total-score) 6)
    (>= (count throwAttempt) 15)
 )
)
(:scoring
  (count throwInBin)
)
)

; Index #90 with key (1, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0)
(define (game game-109) (:domain many-objects-room-v1)
(:constraints
  (and
    (preference ballThrownToBin
      (exists (?b - ball ?h - hexagonal_bin)
        (then
          (once (agent_holds ?b))
          (hold (and (not (agent_holds ?b)) (in_motion ?b)))
          (once (and (not (in_motion ?b)) (in ?h ?b)))
       )
     )
   )
    (preference cubeBlockThrownToTopShelf
      (exists (?c - cube_block)
        (then
          (once (agent_holds ?c))
          (hold (and (not (agent_holds ?c)) (in_motion ?c)))
          (once (and (not (in_motion ?c)) (on top_shelf ?c)))
       )
     )
   )
    (preference pillowThrownToDoggieBed
      (exists (?p - pillow ?d - doggie_bed)
        (then
          (once (agent_holds ?p))
          (hold (and (not (agent_holds ?p)) (in_motion ?p)))
          (once (and (not (in_motion ?p)) (on ?d ?p)))
       )
     )
   )
 )
)
(:scoring
  (+ (count-once-per-objects ballThrownToBin) (count-once-per-objects cubeBlockThrownToTopShelf) (count-once-per-objects pillowThrownToDoggieBed))
)
)

; Index #94 with key (1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0)
(define (game game-115) (:domain medium-objects-room-v1)
(:setup
  (and
    (exists (?c - chair ?r - triangular_ramp ?t - teddy_bear ?h - hexagonal_bin)
      (and
        (game-conserved
          (and
            (near room_center ?r)
            (adjacent_side ?r front ?c)
            (between ?h ?c ?r)
            (forall (?b - ball)
              (near ?b ?h)
           )
         )
       )
        (game-optional
          (and
            (on ?c ?t)
         )
       )
     )
   )
 )
)
(:constraints
  (and
    (preference teddyBearLandsInBin
      (exists (?t - teddy_bear ?h - hexagonal_bin ?c - chair)
        (then
          (once (on ?c ?t))
          (hold (agent_holds ?t))
          (hold (and (not (agent_holds ?t)) (in_motion ?t)))
          (once (and (not (in_motion ?t)) (in ?h ?t)))
       )
     )
   )
    (preference teddyBearHitsBall
      (exists (?t - teddy_bear ?b - ball ?c - chair)
        (then
          (once (on ?c ?t))
          (hold (agent_holds ?t))
          (hold (and (not (agent_holds ?t)) (in_motion ?t)))
          (once (touch ?t ?b))
       )
     )
   )
 )
)
(:scoring
  (+ (* 5 (count teddyBearLandsInBin))
    (count teddyBearHitsBall)
 )
)
)

; Index #96 with key (1, 0, 3, 1, 0, 0, 0, 0, 0, 0, 2, 0)
(define (game game-117) (:domain medium-objects-room-v1)
(:setup
  (exists (?h - hexagonal_bin ?r - triangular_ramp)
    (game-conserved
      (and
        (near ?h ?r)
        (not
          (adjacent ?h ?r)
       )
     )
   )
 )
)
(:constraints
  (and
    (preference redDodgeballThrownToBinWithoutTouchingFloor
      (exists (?h - hexagonal_bin ?r - dodgeball_red)
        (then
          (once (agent_holds ?r))
          (hold (and (not (agent_holds ?r)) (in_motion ?r) (not (touch floor ?r))))
          (once (and (not (in_motion ?r)) (in ?h ?r)))
       )
     )
   )
    (preference redDodgeballThrownToBin
      (exists (?h - hexagonal_bin ?r - dodgeball_red)
        (then
          (once (agent_holds ?r))
          (hold (and (not (agent_holds ?r)) (in_motion ?r)))
          (once (and (not (in_motion ?r)) (in ?h ?r)))
       )
     )
   )
    (preference throwAttempt
      (exists (?r - dodgeball_red)
        (then
          (once (agent_holds ?r))
          (hold (and (not (agent_holds ?r)) (in_motion ?r)))
          (once (not (in_motion ?r)))
       )
     )
   )
 )
)
(:terminal
  (or
    (>= (count throwAttempt) 10)
    (>= (count-once redDodgeballThrownToBinWithoutTouchingFloor) 1)
    (>= (count-once redDodgeballThrownToBin) 1)
 )
)
(:scoring
  (+ (* 5 (count-once redDodgeballThrownToBin))
    (* 3 (= (count throwAttempt) 1)
      (count-once redDodgeballThrownToBinWithoutTouchingFloor)
   )
    (* 2 (< (count throwAttempt) 5)
      (count-once redDodgeballThrownToBinWithoutTouchingFloor)
   )
 )
)
)

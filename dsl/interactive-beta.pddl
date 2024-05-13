
(define (game game-0) (:domain medium-objects-room-v1)  ; 0
(:setup
    (exists (?h - hexagonal_bin ?r - triangular_ramp)
        (game-conserved (near ?h ?r))
    )
)
(:constraints (and
    (preference throwToRampToBin
        (exists (?b - ball ?r - triangular_ramp ?h - hexagonal_bin)
            (then
                (once (agent_holds ?b))
                (hold-while
                    (and (not (agent_holds ?b)) (in_motion ?b))
                    (touch ?b ?r)
                )
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
))
(:terminal (>= (count-once binKnockedOver) 1)
)
(:scoring (count throwToRampToBin)
))

; 1 is invalid

(define (game game-2) (:domain many-objects-room-v1)  ; 2
(:setup
    (game-conserved (open top_drawer))
)
(:constraints (and
    (forall (?b - (either dodgeball golfball) ?t - (either top_drawer hexagonal_bin))
        (preference throwToDrawerOrBin
            (then
                (once (and (agent_holds ?b) (adjacent door agent)))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (not (in_motion ?b)) (in ?t ?b)))
            )
        )
    )
    (preference throwAttempt
        (exists (?b - (either dodgeball golfball))
            (then
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (not (in_motion ?b)))
            )
        )
    )
))
(:terminal (>= (count-once-per-objects throwAttempt) 6)
)
(:scoring (+
    (count-once-per-objects throwToDrawerOrBin:golfball:hexagonal_bin)
    (* 2 (count-once-per-objects throwToDrawerOrBin:dodgeball:hexagonal_bin))
    (* 3 (count-once-per-objects throwToDrawerOrBin:golfball:top_drawer))
    (+ (count-once-per-objects throwToDrawerOrBin) (- (count-once-per-objects throwAttempt)))  ; as a way to encode -1 for each missed throw
)))

; 3 says "figures", but their demonstration only uses blocks, so I'm guessing that's what they meant
(define (game game-3) (:domain many-objects-room-v1)  ; 3
(:constraints (and
    (forall (?b - building)
        (preference blockInTowerAtEnd (exists (?l - block)
            (at-end (in ?b ?l))
        ))
    )
))
(:scoring (external-forall-maximize
    (count-once-per-objects blockInTowerAtEnd)
)))

; 4 is invalid -- woefully underconstrained

(define (game game-5) (:domain few-objects-room-v1)  ; 5
(:constraints (and
    (preference throwBallToBin
        (exists (?d - dodgeball ?h - hexagonal_bin)
            (then
                (once (and (agent_holds ?d) (= (distance ?h agent) 1)))
                (hold (and (not (agent_holds ?d)) (in_motion ?d)))
                (once (and (not (in_motion ?d)) (in ?h ?d)))
            )
        )
    )
))
(:scoring (count throwBallToBin)
))


(define (game game-6) (:domain medium-objects-room-v1)  ; 6
(:setup (and
    (exists (?h - hexagonal_bin) (game-conserved (adjacent ?h bed)))
    (forall (?o - (either teddy_bear pillow)) (game-conserved (not (on bed ?o))))
))
(:constraints (and
    (forall (?b - ball)
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
    (preference failedThrowToBin
        (exists (?b - ball ?h - hexagonal_bin)
            (then
                (once (and (agent_holds ?b) (adjacent desk agent)))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (not (in_motion ?b)) (not (in ?h ?b))))
            )
        )
    )
))
(:scoring (+
    (* 10 (count throwBallToBin:dodgeball))
    (* 20 (count throwBallToBin:basketball))
    (* 30 (count throwBallToBin:beachball))
    (- (count failedThrowToBin))
)))

; 7 is invalid -- vastly under-constrained -- I could probably make some guesses but leaving alone

(define (game game-8) (:domain few-objects-room-v1)  ; 8
(:setup
    (exists (?c - curved_wooden_ramp)
        (game-conserved (on floor ?c))
    )
)
(:constraints (and
    (preference throwOverRamp
        (exists (?d - dodgeball ?c - curved_wooden_ramp)
            (then
                (once (and
                    (agent_holds ?d)
                    (< (distance_side ?c front agent) (distance_side ?c back agent))
                ))
                (hold-while
                    (and (not (agent_holds ?d)) (in_motion ?d))
                    (on ?c ?d)
                )
                (once (and
                    (not (in_motion ?d))
                    (< (distance_side ?c back ?d) (distance_side ?c front ?d))
                ))
            )
        )
    )
    (preference throwAttempt
        (exists (?b - dodgeball)
            (then
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (not (in_motion ?b)))
            )
        )
    )
))
(:terminal (>= (count-once throwOverRamp) 1)
)
(:scoring (+
    (* 3 (= (count throwAttempt) 1) (count-once throwOverRamp))
    (* 2 (= (count throwAttempt) 2) (count-once throwOverRamp))
    (* (>= (count throwAttempt) 3) (count-once throwOverRamp))
)))


; Taking the first game this participant provided
(define (game game-9) (:domain many-objects-room-v1)  ; 9
(:setup
    (exists (?h - hexagonal_bin)
        (game-conserved (or
            (on bed ?h)
            (exists (?w - wall) (adjacent ?w ?h))
        ))
    )
)
(:constraints (and
    (preference throwBallToBin
        (exists (?d - dodgeball ?h - hexagonal_bin)
            (then
                (once (and
                    (agent_holds ?d)
                    (or
                        (on bed ?h)
                        (exists (?w1 ?w2 - wall) (and (adjacent ?w1 ?h) (adjacent ?w2 agent) (opposite ?w1 ?w2)))
                    )
                ))
                (hold (and (not (agent_holds ?d)) (in_motion ?d) (not (touch floor ?d))))
                (once (and (not (in_motion ?d)) (in ?h ?d)))
            )
        )
    )
    (preference bounceBallToBin
        (exists (?d - dodgeball ?h - hexagonal_bin)
            (then
                (once (and
                    (agent_holds ?d)
                    (or
                        (on bed ?h)
                        (exists (?w1 ?w2 - wall) (and (adjacent ?w1 ?h) (adjacent ?w2 agent) (opposite ?w1 ?w2)))
                    )
                ))
                (hold-while
                    (and (not (agent_holds ?d)) (in_motion ?d))
                    (touch floor ?d)
                )
                (once (and (not (in_motion ?d)) (in ?h ?d)))
            )
        )
    )
))
(:scoring (+
    (count bounceBallToBin)
    (* 3 (count throwBallToBin))
)))

(define (game game-10) (:domain medium-objects-room-v1)  ; 10

(:constraints (and
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
))
(:terminal
    (>= (count throwAttempt) 10)
)
(:scoring (count throwTeddyOntoPillow)
))

(define (game game-11) (:domain many-objects-room-v1)  ; 11
(:constraints (and
    (forall (?b - building) (and
        (preference baseBlockInTowerAtEnd (exists (?l - block)
            (at-end (and
                (in ?b ?l)
                (on floor ?l)
            ))
        ))
        (preference blockOnBlockInTowerAtEnd (exists (?l - block)
            (at-end
                (and
                    (in ?b ?l)
                    (not (exists (?o - game_object) (and (not (same_type ?o block)) (touch ?o ?l))))
                    (not (on floor ?l))
                )
            )
        ))
        (preference pyramidBlockAtopTowerAtEnd (exists (?p - pyramid_block)
            (at-end
                (and
                    (in ?b ?p)
                    (not (exists (?l - block) (on ?p ?l)))
                    (not (exists (?o - game_object) (and (not (same_type ?o block)) (touch ?o ?p))))
                )
            )
        ))
    ))
))
(:scoring (external-forall-maximize (*
    (count-once pyramidBlockAtopTowerAtEnd)
    (count-once baseBlockInTowerAtEnd)
    (+
        (count-once-per-objects blockOnBlockInTowerAtEnd)
        1
    )
))))

; 12 requires quantifying based on position -- something like

(define (game game-12) (:domain medium-objects-room-v1)  ; 12
(:setup
    (exists (?h - hexagonal_bin)
        (game-conserved (near room_center ?h))
    )
)
(:constraints (and
    (preference throwToRampToBin
        (exists (?r - triangular_ramp ?d - dodgeball ?h - hexagonal_bin)
            (then
                (once (and (agent_holds ?d) (adjacent door agent) (agent_crouches))) ; ball starts in hand
                (hold-while
                    (and (not (agent_holds ?d)) (in_motion ?d))
                    (touch ?r ?d)
                )
                (once  (and (in ?h ?d) (not (in_motion ?d)))) ; touches wall before in bin
            )
        )
    )
))
(:scoring
    (count-unique-positions throwToRampToBin)
))



(define (game game-13) (:domain many-objects-room-v1)  ; 13
(:setup
    (exists (?h - hexagonal_bin ?r - triangular_ramp) (game-conserved
        (and
            (= (distance ?h ?r) 1)
            (near room_center ?r)
        )
    ))
)
(:constraints (and
    (forall (?d - (either dodgeball golfball))
        (preference throwToRampToBin
            (exists (?r - triangular_ramp ?h - hexagonal_bin)
                (then
                    (once (and (agent_holds ?d) (adjacent door agent) (agent_crouches))) ; ball starts in hand
                    (hold-while
                        (and (not (agent_holds ?d)) (in_motion ?d))
                        (touch ?r ?d)
                    )
                    (once (and (in ?h ?d) (not (in_motion ?d)))) ; touches ramp before in bin
                )
            )
        )
    )
))
(:scoring (+
    (* 6 (count throwToRampToBin:dodgeball))
    (* 3 (count throwToRampToBin:golfball))
)))

(define (game game-14) (:domain medium-objects-room-v1)  ; 14

(:constraints (and
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
))
(:terminal
    (>= (count throwAttempt) 10)
)
(:scoring (count throwInBin)
))

(define (game game-15) (:domain few-objects-room-v1)  ; 15
(:setup
    (exists (?h - hexagonal_bin ?b - building) (and
        (game-conserved (adjacent ?h bed))
        (game-conserved (object_orientation ?h upside_down))
        (game-optional (on ?h ?b)) ; optional since building might cease to exist in game
        (forall (?c - cube_block) (game-optional (in ?b ?c)))
        (exists (?c1 ?c2 ?c3 ?c4 ?c5 ?c6 - cube_block) (game-optional (and ; specifying the pyramidal structure
           (on ?h ?c1)
           (on ?h ?c2)
           (on ?h ?c3)
           (on ?c1 ?c4)
           (on ?c2 ?c5)
           (on ?c4 ?c6)
        )))
    ))
)

(:constraints (and
    (preference blockInTowerKnockedByDodgeball (exists (?b - building ?c - cube_block
        ?d - dodgeball ?h - hexagonal_bin ?r - chair)
        (then
            (once (and
                (agent_holds ?d)
                (adjacent ?r agent)
                (on ?h ?b)
                (in ?b ?c)
            ))
            (hold-while
                (and (not (agent_holds ?d)) (in_motion ?d) (not (or (agent_holds ?c) (touch agent ?c))))
                (in_motion ?c)
            )
            (once (not (or (in_motion ?c) (in_motion ?d))))
        )
    ))
    (preference throwAttempt
        (exists (?d - dodgeball)
            (then
                (once (agent_holds ?d))
                (hold (and (not (agent_holds ?d)) (in_motion ?d)))
                (once (not (in_motion ?d)))
            )
        )
    )
))
(:terminal
    (>= (count-once-per-objects throwAttempt) 2)
)
(:scoring (count-once-per-objects blockInTowerKnockedByDodgeball)
))


(define (game game-16) (:domain few-objects-room-v1)  ; 16
(:setup
    (exists (?c - curved_wooden_ramp ?h - hexagonal_bin ?b1 ?b2 ?b3 ?b4 - block)
        (game-conserved (and
            (adjacent_side ?h front ?c back)
            (on floor ?b1)
            (adjacent_side ?h left ?b1)
            (on ?b1 ?b2)
            (on floor ?b3)
            (adjacent_side ?h right ?b3)
            (on ?b3 ?b4)
        ))
    )
)
(:constraints (and
    (preference rollBallToBin
        (exists (?d - dodgeball ?r - curved_wooden_ramp ?h - hexagonal_bin)
            (then
                (once (agent_holds ?d))
                (hold-while
                    (and (not (agent_holds ?d)) (in_motion ?d))
                    (on ?r ?d)
                )
                (once (and (in ?h ?d) (not (in_motion ?d))))
            )
        )
    )
))
(:scoring (count rollBallToBin)
))

; 18 is a dup of 17


(define (game game-17) (:domain medium-objects-room-v1)  ; 17/18

(:constraints (and
    (preference castleBuilt (exists (?b - bridge_block ?f - flat_block ?t - tall_cylindrical_block ?c - cube_block ?p - pyramid_block)
        (at-end
            (and
                (on ?b ?f)
                (on ?f ?t)
                (on ?t ?c)
                (on ?c ?p)
            )
        )
    ))
))
(:scoring
    (* 10 (count-once-per-objects castleBuilt))
))

(define (game game-19) (:domain medium-objects-room-v1)  ; 19
(:setup
    (forall (?b - ball)
        (game-optional (near door ?b))
    )
)
(:constraints (and
    (forall (?b - ball ?t - (either doggie_bed hexagonal_bin))
        (preference ballThrownIntoTarget
            (then
                (once (and (agent_holds ?b) (near door agent)))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (in ?t ?b) (not (in_motion ?b))))
            )
        )
    )
    (forall (?b - ball)
        (preference ballThrownOntoTarget
            (exists (?t - doggie_bed)
                (then
                    (once (and (agent_holds ?b) (near door agent)))
                    (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                    (once (and (on ?t ?b) (not (in_motion ?b))))
                )
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
))
(:terminal
    (>= (count-once-per-objects throwAttempt) 3)
)
(:scoring (+
    (* 6 (count-once-per-objects ballThrownIntoTarget:dodgeball:hexagonal_bin))
    (* 5 (count-once-per-objects ballThrownIntoTarget:beachball:hexagonal_bin))
    (* 4 (count-once-per-objects ballThrownIntoTarget:basketball:hexagonal_bin))
    (* 5 (count-once-per-objects ballThrownIntoTarget:dodgeball:doggie_bed))
    (* 4 (count-once-per-objects ballThrownIntoTarget:beachball:doggie_bed))
    (* 3 (count-once-per-objects ballThrownIntoTarget:basketball:doggie_bed))
    (* 5 (count-once-per-objects ballThrownOntoTarget:dodgeball))
    (* 4 (count-once-per-objects ballThrownOntoTarget:beachball))
    (* 3 (count-once-per-objects ballThrownOntoTarget:basketball))
)))


(define (game game-20) (:domain medium-objects-room-v1) ; 20
(:constraints (and
    (forall (?b - building) (and
        (preference blockInTowerAtEnd (exists (?l - block)
            (at-end
                (in ?b ?l)
            )
        ))
        (preference blockInTowerKnockedByDodgeball (exists (?l - block ?d - dodgeball)
            (then
                (once (and (in ?b ?l) (agent_holds ?d)))
                (hold-while
                    (and (in ?b ?l) (not (agent_holds ?d)) (in_motion ?d))
                    (touch ?d ?b)
                )
                (hold (in_motion ?l))
                (once (not (in_motion ?l)))
            )
        ))

    ))
    (preference towerFallsWhileBuilding (exists (?b - building ?l1 ?l2 - block)
        (then
            (once (and (in ?b ?l1) (agent_holds ?l2)))
            (hold-while
                (and
                    (not (agent_holds ?l1))
                    (in ?b ?l1)
                    (or
                        (agent_holds ?l2)
                        (in_motion ?l2)  ; (and (not (agent_holds ?l2)) -- used to be here, redundant with the first if clause
                    )
                )
                (touch ?l1 ?l2)
            )
            (hold (and
                (in_motion ?l1)
                (not (agent_holds ?l1))
            ))
            (once (not (in_motion ?l1)))
        )
    ))
))
(:scoring (+
    (external-forall-maximize (+
        (count-once-per-objects blockInTowerAtEnd)
        (* 2 (count-once-per-objects blockInTowerKnockedByDodgeball))
    ))
    (- (count towerFallsWhileBuilding))
)))

(define (game game-21) (:domain few-objects-room-v1)  ; 21
(:setup
    (exists (?c - chair) (game-conserved (and
        (near room_center ?c)
        (not (faces ?c desk))
        (not (faces ?c bed))
    )))
)
(:constraints (and
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
                (once (and
                    (not (in_motion ?d))
                    (not (exists (?h - hexagonal_bin) (in ?h ?d)))
                    (not (on bed ?d))
                    (not (exists (?c - chair) (and (on ?c ?d) (is_setup_object ?c))))
                ))
            )
        )
    )
))
(:terminal
    (>= (total-score) 10)
)
(:scoring (+
    (* 5 (count ballThrownToBin))
    (count ballThrownToBed)
    (count ballThrownToChair)
    (- (count ballThrownMissesEverything))
)))

(define (game game-22) (:domain medium-objects-room-v1)  ; 22
(:setup (and
    (exists (?h - hexagonal_bin) (game-conserved (adjacent bed ?h)))
    (forall (?b - ball) (game-optional (on rug ?b)))
    (game-optional (not (exists (?g - game_object) (on desk ?g))))
))
(:constraints (and
    (forall (?b - ball ?x - (either red yellow pink))
        (preference throwBallToBin
            (exists (?h - hexagonal_bin)
                (then
                    (once (and (agent_holds ?b) (on rug agent) (rug_color_under agent ?x)))
                    (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                    (once (and (not (in_motion ?b)) (in ?h ?b)))
                )
            )
        )
    )
    (preference throwAttempt
        (exists (?b - ball)
            (then
                (once (and (agent_holds ?b) (on rug agent)))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (not (in_motion ?b)))
            )
        )
    )
))
(:terminal
    (>= (count throwAttempt) 8)
)
(:scoring (+
    (* 2 (count throwBallToBin:dodgeball:red))
    (* 3 (count throwBallToBin:basketball:red))
    (* 4 (count throwBallToBin:beachball:red))
    (* 3 (count throwBallToBin:dodgeball:pink))
    (* 4 (count throwBallToBin:basketball:pink))
    (* 5 (count throwBallToBin:beachball:pink))
    (* 4 (count throwBallToBin:dodgeball:yellow))
    (* 5 (count throwBallToBin:basketball:yellow))
    (* 6 (count throwBallToBin:beachball:yellow))
)))


(define (game game-23) (:domain few-objects-room-v1)  ; 23

(:constraints (and
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
))
(:scoring (+
    (count throwBallToBin)
    (- (/ (count throwAttempt) 5))
)))




(define (game game-24) (:domain few-objects-room-v1)  ; 24
(:setup
    (exists (?c - chair ?h - hexagonal_bin) (game-conserved (on ?c ?h)))
)
(:constraints (and
    (forall (?d - dodgeball ?x - color)
        (preference throwBallToBin
            (exists (?h - hexagonal_bin)
                (then
                    (once (and (agent_holds ?d) (on rug agent) (rug_color_under agent ?x)))
                    (hold (and (not (agent_holds ?d)) (in_motion ?d)))
                    (once (and (not (in_motion ?d)) (in ?h ?d)))
                )
            )
        )
    )
))
(:terminal
    (>= (total-score) 300)
)
(:scoring (+
    (* 5 (count throwBallToBin:dodgeball_blue:red))
    (* 10 (count throwBallToBin:dodgeball_pink:red))
    (* 10 (count throwBallToBin:dodgeball_blue:pink))
    (* 20 (count throwBallToBin:dodgeball_pink:pink))
    (* 15 (count throwBallToBin:dodgeball_blue:orange))
    (* 30 (count throwBallToBin:dodgeball_pink:orange))
    (* 15 (count throwBallToBin:dodgeball_blue:green))
    (* 30 (count throwBallToBin:dodgeball_pink:green))
    (* 20 (count throwBallToBin:dodgeball_blue:purple))
    (* 40 (count throwBallToBin:dodgeball_pink:purple))
    (* 20 (count throwBallToBin:dodgeball_blue:yellow))
    (* 40 (count throwBallToBin:dodgeball_pink:yellow))
)))

; 25 and 26 are the same participant and are invalid -- hiding games

(define (game game-27) (:domain few-objects-room-v1)  ; 27
(:setup (and
    (forall (?d - (either dodgeball cube_block)) (game-optional (not (exists (?s - shelf) (on ?s ?d)))))
    (game-optional (toggled_on main_light_switch))
    (forall (?e - desktop) (game-optional (toggled_on ?e)))
))
(:constraints (and
    (preference dodgeballsInPlace
        (exists (?d - dodgeball ?h - hexagonal_bin)
            (at-end (in ?h ?d))
        )
    )
    (preference blocksInPlace
        (exists (?c - cube_block ?s - shelf)
            (at-end (and
                (adjacent ?s west_wall)
                (on ?s ?c)
            ))
        )
    )
    (preference laptopAndBookInPlace
        (exists (?o - (either laptop book) ?s - shelf)
            (at-end (and
                (adjacent ?s south_wall)
                (on ?s ?o)
            ))
        )
    )
    (preference smallItemsInPlace
        (exists (?o - (either cellphone key_chain) ?d - drawer)
            (at-end
                (in ?d ?o)
            )
        )
    )
    (preference itemsTurnedOff
        (exists (?o - (either main_light_switch desktop laptop))
            (at-end
                (not (toggled_on ?o))
            )
        )
    )
))
(:scoring (+
    (* 5 (+
        (count-once-per-objects dodgeballsInPlace)
        (count-once-per-objects blocksInPlace)
        (count-once-per-objects laptopAndBookInPlace)
        (count-once-per-objects smallItemsInPlace)
    ))
    (* 3 (count-once-per-objects itemsTurnedOff))
)))


(define (game game-28) (:domain few-objects-room-v1)  ; 28
(:setup
    (forall (?c - cube_block) (game-conserved (on rug ?c)))
)
(:constraints (and
    (forall (?x - color)
        (preference thrownBallHitsBlock
            (exists (?d - dodgeball ?b - cube_block)
                (then
                    (once (and (agent_holds ?d) (not (on rug agent))))
                    (hold (and (not (agent_holds ?d)) (in_motion ?d)))
                    (once (and (on rug ?b) (touch ?b ?d) (rug_color_under ?b ?x)))
                )
            )
        )
    )
    (preference thrownBallReachesEnd
            (exists (?d - dodgeball)
                (then
                    (once (and (agent_holds ?d) (not (on rug agent))))
                    (hold-while
                        (and
                            (not (agent_holds ?d))
                            (in_motion ?d)
                            (not (exists (?b - cube_block) (touch ?d ?b)))
                        )
                        (above rug ?d)
                    )
                    (once (or (touch ?d bed) (touch ?d west_wall)))
                )
            )
        )
))
(:terminal (or
    (>= (total-time) 180)
    (>= (total-score) 50)
))
(:scoring (+
    (* 10 (count thrownBallReachesEnd))
    (* -5 (count thrownBallHitsBlock:red))
    (* -3 (count thrownBallHitsBlock:green))
    (* -3 (count thrownBallHitsBlock:pink))
    (- (count thrownBallHitsBlock:yellow))
    (- (count thrownBallHitsBlock:purple))
)))

(define (game game-29) (:domain few-objects-room-v1)  ; 29

(:constraints (and
    (preference objectOnBed
        (exists (?g - game_object)
            (at-end (and
                (not (same_type ?g pillow))
                (on bed ?g)
            ))
        )
    )
))
(:scoring
    (count objectOnBed)
))


; 30 is invalid --  rather underdetermined, I could try, but it would take some guesswork


(define (game game-31) (:domain few-objects-room-v1)  ; 31
(:setup (and
    (exists (?h - hexagonal_bin) (game-conserved (and
        (adjacent desk ?h)
        (forall (?b - cube_block) (adjacent ?h ?b))
    )))
    (forall (?o - (either alarm_clock cellphone mug key_chain cd book ball))
        (game-optional (or
            (on side_table ?o)
            (on bed ?o)
        ))
    )
))
(:constraints (and
    (forall (?s - (either bed side_table))
        (preference objectThrownFromRug
            (exists (?o - (either alarm_clock cellphone mug key_chain cd book ball) ?h - hexagonal_bin)
                (then
                    (once (on ?s ?o))
                    (hold (and (agent_holds ?o) (on rug agent)))
                    (hold (and (not (agent_holds ?o)) (in_motion ?o)))
                    (once (and (not (in_motion ?o)) (in ?h ?o)))
                )
            )
        )
    )
))
(:scoring (+
    (count objectThrownFromRug:side_table)
    (* 2 (count objectThrownFromRug:bed))
)))


(define (game game-32) (:domain many-objects-room-v1)  ; 32
(:setup (and
    (exists (?b1 ?b2 ?b3 ?b4 ?b5 ?b6 - (either cube_block cylindrical_block pyramid_block)) (game-optional (and ; specifying the pyramidal structure
        (on desk ?b1)
        (on desk ?b2)
        (on desk ?b3)
        (on ?b1 ?b4)
        (on ?b2 ?b5)
        (on ?b4 ?b6)
    )))
    (exists (?w1 ?w2 - wall ?h - hexagonal_bin)
        (game-conserved (and
            (adjacent ?h ?w1)
            (adjacent ?h ?w2)
        ))
    )
))
(:constraints (and
    (forall (?b - (either dodgeball golfball))
        (preference ballThrownToBin (exists (?h - hexagonal_bin)
            (then
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (not (in_motion ?b)) (in ?h ?b)))
            )
        ))
        (preference blockInTowerKnocked (exists (?bl - building ?c - (either cube_block cylindrical_block pyramid_block))
            (then
                (once (and
                    (agent_holds ?b)
                    (on desk ?bl)
                    (in ?bl ?c)
                ))
                (hold-while
                    (and (not (agent_holds ?b)) (in_motion ?b) (not (or (agent_holds ?c) (touch agent ?c))))
                    (in_motion ?c)
                )
                (once (not (in_motion ?c)))
            )
        ))
        (preference throwAttempt
            (then
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (not (in_motion ?b)))
            )
        )
        (preference ballNeverThrown
            (then
                (once (game_start))
                (hold (not (agent_holds ?b)))
                (once (game_over))
            )
        )
    )
))
(:terminal (or
    (> (external-forall-maximize (count throwAttempt)) 2)
    (>= (count throwAttempt) 12)
))
(:scoring (*
    (>=
        (+
            (count ballThrownToBin:dodgeball)
            (* 2 (count ballThrownToBin:golfball))
        )
        2
    )
    (+
        (count-once-per-objects blockInTowerKnocked)
        (count-once-per-objects ballNeverThrown:golfball)
        (* 2 (count-once-per-objects ballNeverThrown:dodgeball))
    )
)))


(define (game game-33) (:domain many-objects-room-v1)  ; 33
(:setup
    (forall (?g - game_object) (game-optional
        (not (in top_drawer ?g))
    ))
)
(:constraints (and
    (preference itemInClosedDrawerAtEnd (exists (?g - game_object)
        (at-end (and
            (in top_drawer ?g)
            (not (open top_drawer))
        ))
    ))
))
(:scoring
    (count-once-per-objects itemInClosedDrawerAtEnd)
))

; 34 is invalid, another hiding game


(define (game game-35) (:domain few-objects-room-v1)  ; 35

(:constraints (and
    (forall (?b - (either book dodgeball))
        (preference throwObjectToBin
            (exists (?h - hexagonal_bin)
                (then
                    (once (agent_holds ?b))
                    (hold (and (not (agent_holds ?b)) (in_motion ?b) (not (exists (?g - (either game_object floor wall)) (touch ?g ?b )))))
                    (once (and (not (in_motion ?b)) (in ?h ?b)))
                )
            )
        )
    )
    (preference throwBallToBinOffObject
        (exists (?d - dodgeball ?h - hexagonal_bin ?g - (either game_object floor wall))
            (then
                (once (agent_holds ?d))
                (hold-while
                    (and (not (agent_holds ?d)) (in_motion ?d))
                    (touch ?g ?d)
                )
                (once (and (not (in_motion ?d)) (in ?h ?d)))
            )
        )
    )
    (preference throwMissesBin
        (exists (?b - dodgeball ?h - hexagonal_bin)
            (then
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (not (in_motion ?b)) (not (in ?h ?b))))
            )
        )
    )
))
(:terminal (or
    (>= (total-score) 10)
    (<= (total-score) -30)
))
(:scoring (+
    (count throwObjectToBin:dodgeball)
    (* 10 (count-once throwObjectToBin:book))
    (* 2 (count throwBallToBinOffObject))
    (- (count throwMissesBin))
)))


(define (game game-36) (:domain few-objects-room-v1)  ; 36
(:setup (and
    (exists (?h - hexagonal_bin) (game-conserved (on bed ?h)))
    (forall (?d - dodgeball) (game-optional (on desk ?d)))
))
(:constraints (and
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
))
(:terminal
    (>= (count throwAttempt) 5)
)
(:scoring
    (count throwToBin)
))


(define (game game-37) (:domain many-objects-room-v1)  ; 37

(:constraints (and
    (preference throwToBinFromOppositeWall
        (exists (?d - dodgeball ?h - hexagonal_bin ?w1 ?w2 - wall)
            (then
                (once (and
                    (agent_holds ?d)
                    (adjacent agent ?w1)
                    (opposite ?w1 ?w2)
                    (adjacent ?h ?w2)
                ))
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
))
(:terminal
    (>= (count throwAttempt) 10)
)
(:scoring
    (count throwToBinFromOppositeWall)
))

; projected 38 onto the space of feasible games

(define (game game-38) (:domain medium-objects-room-v1)  ; 38

(:constraints (and
    (preference throwToBin
        (exists (?d - dodgeball ?h - hexagonal_bin)
            (then
                (once (and (agent_holds ?d) (adjacent desk agent)))
                (hold (and (not (agent_holds ?d)) (in_motion ?d)))
                (once (and (not (in_motion ?d)) (in ?h ?d)))
            )
        )
    )
))
(:scoring
    (* 5 (count throwToBin))
))


(define (game game-39) (:domain many-objects-room-v1)  ; 39
(:constraints (and
    (preference ballThrownToWallToAgent
        (exists (?b - ball ?w - wall)
            (then
                (once (agent_holds ?b))
                (hold-while
                    (and (not (agent_holds ?b)) (in_motion ?b))
                    (touch ?w ?b)
                )
                (once (or (agent_holds ?b) (touch agent ?b)))
            )
        )
    )
))
(:scoring
    (count ballThrownToWallToAgent)
))


(define (game game-40) (:domain many-objects-room-v1)  ; 40
(:setup
    (exists (?r - curved_wooden_ramp) (game-conserved (adjacent ?r rug)))
)
(:constraints (and
    (forall (?x - color)
        (preference ballRolledOnRampToRug
            (exists (?b - beachball ?r - curved_wooden_ramp)
                (then
                    (once (agent_holds ?b))
                    (hold-while
                        (and (not (agent_holds ?b)) (in_motion ?b))
                        (on ?r ?b)
                    )
                    (once (and (not (in_motion ?b)) (on rug ?b) (rug_color_under ?b ?x)))
                )
            )
        )
    )
))
(:scoring (+
    (count ballRolledOnRampToRug:pink)
    (* 2 (count ballRolledOnRampToRug:yellow))
    (* 3 (count ballRolledOnRampToRug:orange))
    (* 3 (count ballRolledOnRampToRug:green))
    (* 4 (count ballRolledOnRampToRug:purple))
    (- (count ballRolledOnRampToRug:white))
)))


(define (game game-41) (:domain many-objects-room-v1)  ; 41
(:setup
    (exists (?w1 ?w2 - wall) (and
        (game-conserved (opposite ?w1 ?w2))
        (forall (?b - bridge_block) (game-conserved (and
            (on floor ?b)
            (= (distance ?w1 ?b) (distance ?w2 ?b))
        )))
        (forall (?g - game_object) (game-optional (or
            (same_type ?g bridge_block)
            (> (distance ?w1 ?g) (distance ?w2 ?g))
        )))
    ))
)
(:constraints (and
    (forall (?w1 ?w2 - wall)
        (preference objectMovedRoomSide (exists (?g - game_object)
            (then
                (once (and
                    (not (agent_holds ?g))
                    (not (in_motion ?g))
                    (not (same_type ?g bridge_block))
                    (> (distance ?w1 ?g) (distance ?w2 ?g))
                ))
                (hold (or
                    (agent_holds ?g)
                    (in_motion ?g)
                ))
                (once (and
                    (not (in_motion ?g))
                    (< (distance ?w1 ?g) (distance ?w2 ?g))
                ))
            )
        ))
    )
))
(:terminal
    (>= (total-time) 30)
)
(:scoring (external-forall-maximize
    (count-once-per-objects objectMovedRoomSide)
)))


(define (game game-42) (:domain few-objects-room-v1)  ; 42
(:setup
    (exists (?h - hexagonal_bin) (and
        (forall (?g - game_object) (game-optional (or
            (same_object ?h ?g)
            (not (near ?h ?g))
        )))
        (forall (?d - dodgeball) (game-optional (and
            (> (distance ?h ?d) 2)
            (< (distance ?h ?d) 6)
        )))
    ))
)
(:constraints (and
    (preference throwBallFromOtherBallToBin
        (exists (?d1 ?d2 - dodgeball ?h - hexagonal_bin)
            (then
                (once (and (agent_holds ?d1) (adjacent ?d2 agent)))
                (hold (and (not (agent_holds ?d1)) (in_motion ?d1)))
                (once (and (not (in_motion ?d1)) (in ?h ?d1)))
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
))
(:terminal
    (>= (count throwAttempt) 5)
)
(:scoring
    (count-same-positions throwBallFromOtherBallToBin)
))


(define (game game-43) (:domain medium-objects-room-v1)  ; 43
(:setup
    (exists (?d - doggie_bed) (game-conserved (near room_center ?d)))
)
(:constraints (and
    (forall (?b - ball) (and
        (preference throwBallToDoggieBed
            (exists (?d - doggie_bed)
                (then
                    (once (agent_holds ?b))
                    (hold (and (not (agent_holds ?b)) (in_motion ?b) (not (exists (?w - wall) (touch ?w ?b )))))
                    (once (and (not (in_motion ?b)) (on ?d ?b)))
                )
            )
        )
        (preference throwBallToDoggieBedOffWall
            (exists (?d - doggie_bed ?w - wall)
                (then
                    (once (agent_holds ?b))
                    (hold-while
                        (and (not (agent_holds ?d)) (in_motion ?b))
                        (touch ?w ?b)
                    )
                    (once (and (not (in_motion ?b)) (on ?d ?b)))
                )
            )
        )
    ))
))
(:scoring (+
    (count throwBallToDoggieBed:basketball)
    (* 2 (count throwBallToDoggieBed:beachball))
    (* 3 (count throwBallToDoggieBed:dodgeball))
    (* 2 (count throwBallToDoggieBedOffWall:basketball))
    (* 3 (count throwBallToDoggieBedOffWall:beachball))
    (* 4 (count throwBallToDoggieBedOffWall:dodgeball))
)))


; 44 is another find the hidden object game

(define (game game-45) (:domain many-objects-room-v1)  ; 45
(:setup
    (exists (?t1 ?t2 - teddy_bear) (game-optional (and
        (on floor ?t1)
        (on bed ?t2)
        (equal_z_position ?t1 ?t2)
        (equal_z_position bed ?t1)
    )))
)
(:constraints (and
    (forall (?b - (either golfball dodgeball)) (and
        (preference throwKnocksOverBear (exists (?t - teddy_bear ?s - sliding_door)
            (then
                (once (and
                    (agent_holds ?b)
                    (adjacent desk agent)
                    (adjacent ?s agent)
                    (equal_z_position bed ?t)
                    ; (= (z_position ?t) (z_position bed))
                ))
                (hold-while
                    (and (in_motion ?b) (not (agent_holds ?b)))
                    (touch ?b ?t)
                )
                (once (in_motion ?t))
            )
        ))
        (preference throwAttempt (exists (?s - sliding_door)
            (then
                (once (and (agent_holds ?b) (adjacent desk agent) (adjacent ?s agent)))
                (hold (and (in_motion ?b) (not (agent_holds ?b))))
                (once (not (in_motion ?b)))
            )
        ))
    ))
))
(:terminal (or
    (> (external-forall-maximize (count throwAttempt)) 1)
    (>= (count-once-per-objects throwAttempt) 6)
))
(:scoring (+
    (count-once-per-objects throwKnocksOverBear:dodgeball)
    (* 2 (count-once-per-objects throwKnocksOverBear:golfball))
)))

(define (game game-46) (:domain few-objects-room-v1)  ; 46
(:setup
    (exists (?c - curved_wooden_ramp) (game-conserved
        (near room_center ?c)
    ))
)
(:constraints (and
    (preference ballThrownToRampToBed (exists (?d - dodgeball_pink ?c - curved_wooden_ramp)
        (then
            (once (and (agent_holds ?d) (faces agent ?c)))
            (hold-while
                (and (in_motion ?d) (not (agent_holds ?d)))
                (touch ?d ?c)
            )
            (once (and (not (in_motion ?d)) (on bed ?d)))
        )
    ))
    (preference ballThrownHitsAgent (exists (?d - dodgeball_pink ?c - curved_wooden_ramp)
        (then
            (once (and (agent_holds ?d) (faces agent ?c)))
            (hold-while
                (and (in_motion ?d) (not (agent_holds ?d)))
                (touch ?d ?c)
            )
            (once (and (touch ?d agent) (not (agent_holds ?d))))
        )
    ))
))
(:scoring (+
    (count ballThrownToRampToBed)
    (- (count ballThrownHitsAgent))
)))


(define (game game-47) (:domain many-objects-room-v1)  ; 47

(:constraints (and
    (forall (?x - color)
        (preference beachballBouncedOffRamp
            (exists (?b - beachball ?r - triangular_ramp_green)
                (then
                    (once (and (agent_holds ?b) (not (on rug agent)) (faces agent ?r)))
                    (hold-while
                        (and (in_motion ?b) (not (agent_holds ?b)))
                        (touch ?b ?r)
                    )
                    (once (and (not (in_motion ?b)) (on rug ?b) (rug_color_under ?b ?x)))
                )
            )
        )
    )
))
(:scoring (+
    (count beachballBouncedOffRamp:red)
    (* 3 (count beachballBouncedOffRamp:pink))
    (* 10 (count beachballBouncedOffRamp:green))
)))

(define (game game-48) (:domain medium-objects-room-v1)  ; 48
(:setup
    (exists (?b - building ?h - hexagonal_bin) (game-conserved (and
        (in ?b ?h)
        (>= (building_size ?b) 4)
        (not (exists (?g - game_object) (and (in ?b ?g) (on ?h ?g))))
        (near room_center ?b)
    )))
)
(:constraints (and
    (forall (?d - (either dodgeball basketball beachball))
        (preference ballThrownToBin (exists (?b - building ?h - hexagonal_bin)
            (then
                (once (agent_holds ?d))
                (hold (and (in_motion ?d) (not (agent_holds ?d))))
                (once (and (not (in_motion ?d)) (or (in ?h ?d) (on ?h ?d)) (or (in ?b ?h) (on ?b ?h))))
            )
        ))
    )
    (preference itemsHidingScreens
        (exists (?s - (either desktop laptop) ?o - (either pillow doggie_bed teddy_bear))
            (at-end (on ?s ?o))
        )
    )
    (preference objectsHidden
        (exists (?o - (either alarm_clock cellphone) ?d - drawer)
            (at-end (in ?d ?o))
        )
    )
    (preference blindsOpened
        (exists (?b - blinds)
            (at-end (open ?b))  ; blinds being open = they were pulled down
        )
    )
    (preference objectMoved
        (exists (?g - game_object)
            (then
                (once (and
                    (not (in_motion ?g))
                    (not (same_type ?g ball))
                    (not (same_type ?g drawer))
                    (not (same_type ?g blinds))
                ))
                (hold (in_motion ?g))
                (once (not (in_motion ?g)))
            )
        )
    )
))
(:scoring (+
    (* 5 (count ballThrownToBin:dodgeball))
    (* 7 (count ballThrownToBin:basketball))
    (* 15 (count ballThrownToBin:beachball))
    (* 10 (count-once-per-objects itemsHidingScreens))
    (* 10 (count-once-per-objects objectsHidden))
    (* 10 (count-once-per-objects blindsOpened))
    (* -5 (count objectMoved))
)))

(define (game game-49) (:domain many-objects-room-v1)  ; 49
(:setup
    (exists (?g - golfball_green) (and
        (game-conserved (near door ?g))
        (forall (?d - dodgeball) (game-optional (near ?d ?g)))
    ))
)
(:constraints (and
    (forall (?d - dodgeball) (and
        (preference dodgeballThrownToBin (exists (?h - hexagonal_bin ?g - golfball_green)
            (then
                (once (and
                    (adjacent ?g agent)
                    (adjacent door agent)
                    (agent_holds ?d)
                ))
                (hold (and (in_motion ?d) (not (agent_holds ?d))))
                (once (and (not (in_motion ?d)) (in ?h ?d)))
            )
        ))
        (preference throwAttemptFromDoor (exists (?g - golfball_green)
            (then
                (once (and
                    (adjacent ?g agent)
                    (adjacent door agent)
                    (agent_holds ?d)
                ))
                (hold (and (in_motion ?d) (not (agent_holds ?d))))
                (once (not (in_motion ?d)))
            )
        ))
    ))
))
(:terminal (or
    (> (external-forall-maximize (count throwAttemptFromDoor)) 1)
    (>= (count-once-per-objects throwAttemptFromDoor) 3)
))
(:scoring
    (* 10 (count-once-per-objects dodgeballThrownToBin))
))

(define (game game-50) (:domain medium-objects-room-v1)  ; 50
(:setup
    (exists (?h - hexagonal_bin) (game-conserved (near room_center ?h)))
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


(define (game game-53) (:domain few-objects-room-v1)  ; 53

(:constraints (and
    (preference dodgeballsInPlace
        (exists (?d - dodgeball ?h - hexagonal_bin ?w1 ?w2 - wall)
            (at-end (and
                (in ?h ?d)
                (adjacent ?h ?w1)
                (adjacent ?h ?w2)
            ))
        )
    )
    (preference blocksInPlace
        (exists (?c - cube_block ?s - shelf)
            (at-end
                (on ?s ?c)
            )
        )
    )
    (preference smallItemsInPlace
        (exists (?o - (either cellphone key_chain mug credit_card cd watch alarm_clock) ?d - drawer)
            (at-end
                (in ?d ?o)
            )
        )
    )
))
(:scoring (+
    (* 5 (count-once-per-objects dodgeballsInPlace))
    (* 5 (count-once-per-objects blocksInPlace))
    (* 5 (count-once-per-objects smallItemsInPlace))
)))


(define (game game-54) (:domain few-objects-room-v1)  ; 54

(:constraints (and
    (forall (?b - building)
        (preference blockPlacedInBuilding (exists (?l - cube_block)
            (then
                (once (agent_holds ?l))
                (hold (not (agent_holds ?l)))
                (hold (in ?b ?l))
                (once (or (not (in ?b ?l)) (game_over)))
            )
        ))
    )
    (forall (?l - cube_block)
        (preference blockPickedUp
            (then
                (once (not (agent_holds ?l)))
                (hold (agent_holds ?l))
                (once (not (agent_holds ?l)))
            )
        )
    )
))
(:terminal
    (>= (external-forall-maximize (count blockPickedUp)) 3)
)
(:scoring (external-forall-maximize
    (count-overlapping blockPlacedInBuilding)
)))


(define (game game-55) (:domain few-objects-room-v1)  ; 55
(:setup
    (exists (?h - hexagonal_bin)
        (game-conserved (near room_center ?h))
    )
)
(:constraints (and
    (preference objectToBinOnFirstTry (exists (?o - game_object ?h - hexagonal_bin)
        (then
            (once (game_start))
            (hold (not (agent_holds ?o)))
            (hold (agent_holds ?o))
            (hold (and (in_motion ?o) (not (agent_holds ?o))))
            (once (and (not (in_motion ?o)) (in ?h ?o)))
        )
    ))
))
(:scoring
    (count-once-per-objects objectToBinOnFirstTry)
))

(define (game game-56) (:domain few-objects-room-v1)  ; 56

(:constraints (and
    (preference throwFromDoorToBin (exists (?d - dodgeball ?h - hexagonal_bin)
        (then
            (once (and (agent_holds ?d) (adjacent door agent)))
            (hold (and (not (agent_holds ?d)) (in_motion ?d)))
            (once (and (not (in_motion ?d)) (in ?h ?d)))
        )
    ))
    (preference throwAttempt (exists (?d - dodgeball)
        (then
            (once (agent_holds ?d))
            (hold (and (not (agent_holds ?d)) (in_motion ?d)))
            (once (not (in_motion ?d)))
        )
    ))
))
(:terminal
    (>= (count throwAttempt) 3)
)
(:scoring
    (count throwFromDoorToBin)
))

(define (game game-57) (:domain medium-objects-room-v1)  ; 57

(:constraints (and
    (preference bookOnDeskShelf (exists (?b - book ?d - shelf_desk)
        (at-end (and
            (on ?d ?b)
            (not (exists (?o - (either pencil pen cd)) (on ?d ?o)))
        ))
    ))
    (preference otherObjectsOnDeskShelf (exists (?o - (either pencil pen cd) ?d - shelf_desk)
        (at-end (and
            (on ?d ?o)
            (not (exists (?b - book) (on ?d ?b)))
        ))
    ))
    (preference dodgeballAndBasketballInBin (exists (?b - (either dodgeball basketball) ?h - hexagonal_bin)
        (at-end (in ?h ?b))
    ))
    (preference beachballOnRug (exists (?b - beachball)
        (at-end (on rug ?b))
    ))
    (preference smallItemsInPlace (exists (?o - (either cellphone key_chain cd) ?d - drawer)
        (at-end (in ?d ?o))
    ))
    (preference watchOnShelf (exists (?w - watch ?s - shelf)
        (at-end (on ?s ?w))
    ))
))
(:scoring (+
    (count-once-per-objects bookOnDeskShelf)
    (count-once-per-objects otherObjectsOnDeskShelf)
    (count-once-per-objects dodgeballAndBasketballInBin)
    (count-once-per-objects beachballOnRug)
    (count-once-per-objects smallItemsInPlace)
    (count-once-per-objects watchOnShelf)
)))

(define (game game-58) (:domain medium-objects-room-v1)  ; 58
(:setup (and
    (forall (?l - block ?s - shelf) (game-optional (not (on ?s ?l))))
    (exists (?b - building) (game-conserved (and
        (= (building_size ?b) 6)
        (forall (?l - block) (or
            (in ?b ?l)
            (exists (?l2 - block) (and
                (in ?b ?l2)
                (not (same_object ?l ?l2))
                (same_type ?l ?l2)
            ))
        ))
    )))
))
(:constraints (and
    (preference gameBlockFound (exists (?l - block)
        (then
            (once (game_start))
            (hold (not (exists (?b - building) (and (in ?b ?l) (is_setup_object ?b)))))
            (once (agent_holds ?l))
        )
    ))
    (preference towerFallsWhileBuilding (exists (?b - building ?l1 ?l2 - block)
        (then
            (once (and (in ?b ?l1) (agent_holds ?l2) (not (is_setup_object ?b))))
            (hold-while
                (and
                    (not (agent_holds ?l1))
                    (in ?b ?l1)
                    (or
                        (agent_holds ?l2)
                        (in_motion ?l2)
                    )
                )
                (touch ?l1 ?l2)
            )
            (hold (and
                (in_motion ?l1)
                (not (agent_holds ?l1))
            ))
            (once (not (in_motion ?l1)))
        )
    ))
    (preference matchingBuildingBuilt (exists (?b1 ?b2 - building)
        (at-end (and
            (is_setup_object ?b1)
            (not (is_setup_object ?b2))
            (forall (?l1 ?l2 - block) (or
                (not (in ?b1 ?l1))
                (not (in ?b1 ?l2))
                (not (on ?l1 ?l2))
                (exists (?l3 ?l4 - block) (and
                    (in ?b2 ?l3)
                    (in ?b2 ?l4)
                    (on ?l3 ?l4)
                    (same_type ?l1 ?l3)
                    (same_type ?l2 ?l4)
                ))
            ))
        ))
    ))
))
(:scoring (+
    (* 5 (count-once-per-objects gameBlockFound))
    (* 100 (count-once matchingBuildingBuilt))
    (* -10 (count towerFallsWhileBuilding))
)))

(define (game game-59) (:domain many-objects-room-v1)  ; 59
(:setup
    (exists (?h - hexagonal_bin) (game-conserved (near door ?h)))
)
(:constraints (and
    (forall (?b - (either golfball dodgeball beachball))
        (preference ballThrownToBin (exists (?h - hexagonal_bin)
            (then
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (not (in_motion ?b)) (in ?h ?b)))
            )
        ))
    )
))
(:scoring (+
    (* 2 (count ballThrownToBin:golfball))
    (* 3 (count ballThrownToBin:dodgeball))
    (* 4 (count ballThrownToBin:beachball))
)))

; 60 is invalid


(define (game game-61) (:domain many-objects-room-v1)  ; 61
(:setup (game-conserved (and
    (exists (?f - flat_block) (on rug ?f))
    (forall (?p - pyramid_block) (on floor ?p))
    (exists (?p1 - pyramid_block_yellow ?p2 - pyramid_block_red ?p3 - pyramid_block_blue ?h - hexagonal_bin)
        (and
            (> (distance ?h ?p2) (distance ?h ?p1))
            (> (distance ?h ?p3) (distance ?h ?p2))
        )
    )
)))
(:constraints (and
    (forall (?p - pyramid_block)
        (preference dodgeballFromBlockToBin (exists (?d - dodgeball ?h - hexagonal_bin)
            (then
                (once (and (agent_holds ?d) (adjacent ?p agent)))
                (hold (and (not (agent_holds ?d)) (in_motion ?d)))
                (once (and (not (in_motion ?d)) (in ?h ?d)))
            )
        ))
    )
    (preference cubeBlockInBuilding (exists (?b - building ?l - cube_block ?f - flat_block)
        (at-end (and
              (is_setup_object ?f)
              (in ?b ?f)
              (in ?b ?l)
        ))
    ))
))
(:scoring (+
    (* 10 (count dodgeballFromBlockToBin:pyramid_block_yellow))
    (* 25 (count dodgeballFromBlockToBin:pyramid_block_red))
    (* 50 (count dodgeballFromBlockToBin:pyramid_block_blue))
    (* 100 (= (count-once-per-objects dodgeballFromBlockToBin:pyramid_block_blue) 3))
    (* 10 (count-once-per-objects cubeBlockInBuilding))
    (* 100 (= (count-once-per-objects cubeBlockInBuilding) 3))
)))

(define (game game-62) (:domain medium-objects-room-v1)  ; 62

(:constraints (and
    (preference bigObjectThrownToBed (exists (?o - (either chair laptop doggie_bed))
        (then
            (once (and (agent_holds ?o) (adjacent desk agent)))
            (hold (and (not (agent_holds ?o)) (in_motion ?o)))
            (once (and (not (in_motion ?o)) (on bed ?o)))
        )
    ))
    (preference smallObjectThrownToBed (exists (?o - game_object)
        (then
            (once (and
                (agent_holds ?o)
                (adjacent desk agent)
                (not (exists (?o2 - (either chair laptop doggie_bed)) (same_object ?o ?o2)))
            ))
            (hold (and (not (agent_holds ?o)) (in_motion ?o)))
            (once (and (not (in_motion ?o)) (on bed ?o)))
        )
    ))
    (preference failedThrowAttempt (exists (?o - game_object)
        (then
            (once (and (agent_holds ?o) (adjacent desk agent)))
            (hold (and (not (agent_holds ?o)) (in_motion ?o)))
            (once (and (not (in_motion ?o)) (not (on bed ?o))))
        )
    ))
))
(:scoring (+
    (count smallObjectThrownToBed)
    (* 5 (count bigObjectThrownToBed))
    (* -5 (count failedThrowAttempt))
)))



(define (game game-63) (:domain medium-objects-room-v1)  ; 63

(:constraints (and
    (preference towerFallsWhileBuilding (exists (?b - building ?l1 ?l2 - block)
        (then
            (once (and (in ?b ?l1) (agent_holds ?l2) (not (is_setup_object ?b))))
            (hold-while
                (and
                    (not (agent_holds ?l1))
                    (in ?b ?l1)
                    (or
                        (agent_holds ?l2)
                        (in_motion ?l2)
                    )
                )
                (touch ?l1 ?l2)
            )
            (hold (and
                (in_motion ?l1)
                (not (agent_holds ?l1))
            ))
            (once (not (in_motion ?l1)))
        )
    ))
    (forall (?b - building) (and
        (preference blockPlacedInBuilding (exists (?l - block)
            (then
                (once (agent_holds ?l))
                (hold (not (agent_holds ?l)))
                (hold (in ?b ?l))
                (once (or (not (in ?b ?l)) (game_over)))
            )
        ))
        (preference nonBlockPlacedInBuilding (exists (?o - game_object)
            (then
                (once (and (agent_holds ?o) (not (same_type ?o block))))
                (hold (not (agent_holds ?o)))
                (hold (in ?b ?o))
                (once (or (not (in ?b ?o)) (game_over)))
            )
        ))
    ))
))
(:terminal
    (>= (count-once towerFallsWhileBuilding) 1)
)
(:scoring (external-forall-maximize (+
    (count-overlapping blockPlacedInBuilding)
    (* 2 (count-overlapping nonBlockPlacedInBuilding))
))))


(define (game game-64) (:domain many-objects-room-v1)  ; 64

(:constraints (and
    (forall (?o - (either hexagonal_bin rug wall))
        (preference ballThrownFromObjectToBin (exists (?d - dodgeball ?h - hexagonal_bin)
            (then
                (once (and (agent_holds ?d) (adjacent ?o agent)))
                (hold (and (not (agent_holds ?d)) (in_motion ?d)))
                (once (and (not (in_motion ?d)) (in ?h ?d)))
            )
        ))
    )
))
(:scoring (+
    (count ballThrownFromObjectToBin:hexagonal_bin)
    (* 2 (count ballThrownFromObjectToBin:rug))
    (* 3 (count ballThrownFromObjectToBin:wall))
)))


(define (game game-65) (:domain many-objects-room-v1)  ; 65

(:constraints (and
    (preference ballOnBedAtEnd (exists (?b - ball)
        (at-end
            (on bed ?b)
        )
    ))
))
(:scoring (count-once-per-objects ballOnBedAtEnd)
))


(define (game game-66) (:domain medium-objects-room-v1)  ; 66
(:setup (and
    (forall (?b - (either bridge_block cube_block))
        (game-conserved (near door ?b))
    )
    (forall (?b - (either cylindrical_block tall_cylindrical_block))
        (game-optional (on bottom_shelf ?b))
    )
    (forall (?b - (either flat_block pyramid_block))
        (game-conserved (not (exists (?s - shelf) (on ?s ?b))))
    )
    (exists (?d - doggie_bed ?r - triangular_ramp)
        (game-conserved (adjacent ?d ?r))
    )
))
(:constraints (and
    (forall (?b - (either cylindrical_block tall_cylindrical_block)) (and
        (preference blockCorrectlyPicked (exists (?d - dodgeball ?o - doggie_bed ?tb - (either bridge_block cube_block))
            (then
                (once (and
                    (agent_holds ?d)
                    (adjacent ?o agent)
                    (on top_shelf ?b)
                    (not (exists (?ob - block)
                        (and
                            (not (same_object ?b ?ob))
                            (on top_shelf ?ob)
                        )
                    ))
                ))
                (hold (and (not (agent_holds ?d)) (in_motion ?d) (not (agent_holds ?b))))
                (once (and
                    (not (in_motion ?d))
                    (not (exists (?ob - block) (< (distance ?d ?ob) (distance ?d ?tb))))
                    (same_color ?b ?tb)
                ))
            )
        ))
        (preference blockIncorrectlyPicked (exists (?d - dodgeball ?o - doggie_bed ?tb - (either bridge_block cube_block))
            (then
                (once (and
                    (agent_holds ?d)
                    (adjacent ?o agent)
                    (on top_shelf ?b)
                    (not (exists (?ob - block)
                        (and
                            (not (same_object ?b ?ob))
                            (on top_shelf ?ob)
                        )
                    ))
                ))
                (hold (and (not (agent_holds ?d)) (in_motion ?d) (not (agent_holds ?b))))
                (once (and
                    (not (in_motion ?d))
                    (not (exists (?ob - block) (< (distance ?d ?ob) (distance ?d ?tb))))
                    (not (same_color ?b ?tb))
                ))
            )
        ))
    ))
))
(:terminal
    (>= (count-once-per-external-objects blockCorrectlyPicked) 4)
)
(:scoring (+
    (* 10 (count-once-per-external-objects blockCorrectlyPicked))
    (- (count blockIncorrectlyPicked))
    ( * 100 (>= (count-once-per-external-objects blockCorrectlyPicked) 4))
)))

(define (game game-67) (:domain medium-objects-room-v1)  ; 67
(:setup (and
    (exists (?b1 ?b2 ?b3 ?b4 ?b5 ?b6 ?b7 ?b8 ?b9 ?b10 - (either tall_cylindrical_block bridge_block flat_block cube_block cylindrical_block))
        (game-optional (and
            (= (distance desk ?b1) (distance desk ?b2) (distance desk ?b3) (distance desk ?b4))
            (= (distance desk ?b5) (distance desk ?b6) (distance desk ?b7))
            (= (distance desk ?b8) (distance desk ?b9))
            (< (distance desk ?b10) 2)
            (< (distance desk ?b1) (distance desk ?b5))
            (< (distance desk ?b5) (distance desk ?b8))
            (< (distance desk ?b8) (distance desk ?b10))
        ))
    )
    (forall (?c - chair) (game-conserved (not (adjacent_side desk front ?c))))
))
(:constraints (and
    (forall (?b - ball) (and
        (preference ballKnocksBlockFromRug (exists (?l - block)
            (then
                (once (and (agent_holds ?b) (on rug agent) (is_setup_object ?l)))
                (hold-while
                    (and (not (agent_holds ?b)) (in_motion ?b))
                    (touch ?b ?l)
                    (in_motion ?l)
                )
                (once (not (in_motion ?b)))
            )
        ))
        (preference throwAttempt
            (then
                (once (and (agent_holds ?b) (on rug agent)))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (not (in_motion ?b)))
            )
        )
    ))
))
(:terminal
    (>= (count throwAttempt) 16)
)
(:scoring (+
    (count-once-per-objects ballKnocksBlockFromRug:dodgeball)
    (* 0.7 (count-once-per-objects ballKnocksBlockFromRug:basketball))
    (* 0.5 (count-once-per-objects ballKnocksBlockFromRug:beachball))
)))

; 68 has subjective scoring -- I could attempt to objectify, but it's hard

(define (game game-69) (:domain many-objects-room-v1)  ; 69
(:setup
    (exists (?c - curved_wooden_ramp ?h - hexagonal_bin) (game-conserved (adjacent ?c ?h)))
)
(:constraints (and
    (preference ballThrownThroughRampToBin (exists (?d - dodgeball ?c - curved_wooden_ramp ?h - hexagonal_bin)
        (then
            (once (agent_holds ?d))
            (hold-while
                (and (not (agent_holds ?d)) (in_motion ?d))
                (touch ?d ?c)
            )
            (once (and (not (in_motion ?d)) (in ?h ?d)))
        )
    ))
))
(:scoring
    (count ballThrownThroughRampToBin)
))

(define (game game-70) (:domain many-objects-room-v1)  ; 70
(:setup (and
    (forall (?c - chair) (game-conserved (not (adjacent_side desk front ?c))))
    (exists (?h - hexagonal_bin ?c - curved_wooden_ramp )
        (game-conserved (and
            (adjacent_side desk front ?c)
            (adjacent_side ?h front ?c back)
        ))
    )
    (forall (?o - (either golfball dodgeball triangle_block pyramid_block))
        (game-optional (near side_table ?o))
    )
))
(:constraints (and
    (forall (?o - (either golfball dodgeball triangle_block pyramid_block)) (and
        (preference objectLandsInBin (exists (?h - hexagonal_bin)
            (then
                (once (and (adjacent bed agent) (agent_holds ?o)))
                (hold (and (in_motion ?o) (not (agent_holds ?o))))
                (once (and (not (in_motion ?o)) (in ?h ?o)))
            )
        ))
        (preference thrownObjectHitsComputer (exists (?c - (either desktop laptop))
            (then
                (once (and (adjacent bed agent) (agent_holds ?o)))
                (hold (and (in_motion ?o) (not (agent_holds ?o))))
                (once (touch ?o ?c))
            )
        ))
    ))
    (preference golfballLandsInBinThroughRamp (exists (?g - golfball ?c - curved_wooden_ramp ?h - hexagonal_bin)
        (then
            (once (and (adjacent bed agent) (agent_holds ?g)))
            (hold-while
                (and (in_motion ?g) (not (agent_holds ?g)))
                (touch ?c ?g)
            )
            (once (and (not (in_motion ?g)) (in ?h ?g)))
        )
    ))
))
(:scoring (+
    (count objectLandsInBin:triangle_block)
    (* 2 (count objectLandsInBin:pyramid_block))
    (* 2 (count objectLandsInBin:dodgeball))
    (* 3 (count objectLandsInBin:golfball))
    (* 6 (count golfballLandsInBinThroughRamp))
    (- (count thrownObjectHitsComputer))
)))

(define (game game-71) (:domain many-objects-room-v1)  ; 71
(:setup (and
    (forall (?p - pillow) (game-conserved (on bed ?p)))
    (forall (?b - bridge_block) (game-conserved (on floor ?b)))
    (forall (?c - cylindrical_block) (game-conserved (exists (?o - (either pillow bridge_block)) (near ?c ?o))) )
))
(:constraints (and
    (preference dodgeballHitsPillowWithoutTouchingBlock (exists (?d - dodgeball ?p - pillow ?r - triangular_ramp)
        (then
            (once (and (adjacent ?r agent) (near desk ?r) (agent_holds ?d)))
            (hold-while
                (and (in_motion ?d) (not (agent_holds ?d)) (not (exists (?c - cylindrical_block) (touch ?c ?d) )) )
                (touch ?d ?p)
            )
            (once (not (in_motion ?d)))
        )
    ))
    (preference golfballUnderBridgeWithoutTouchingBlock (exists (?g - golfball ?b - bridge_block ?r - triangular_ramp)
        (then
            (once (and (adjacent ?r agent) (near desk ?r) (agent_holds ?g)))
            (hold-while
                (and (in_motion ?g) (not (agent_holds ?g)) (not (exists (?c - cylindrical_block) (touch ?c ?g) )) )
                (above ?g ?b)
            )
            (once (not (in_motion ?g)))
        )
    ))
))
(:scoring (+
    (count dodgeballHitsPillowWithoutTouchingBlock)
    (count golfballUnderBridgeWithoutTouchingBlock)
)))

(define (game game-72) (:domain many-objects-room-v1)  ; 72
(:setup (and
    (exists (?t - teddy_bear) (game-optional (and (on bed ?t) (object_orientation ?t upright))))
    (forall (?b - ball) (game-optional (near desk ?b)))
))
(:constraints (and
    (preference ballKnocksTeddy (exists (?b - ball ?t - teddy_bear ?c - chair)
        (then
            (once (and
                (adjacent ?c agent)
                (adjacent ?c desk)
                (agent_holds ?b)
                (object_orientation ?t upright)
            ))
            (hold-while
                (and (in_motion ?b) (not (agent_holds ?b)))
                (touch ?b ?t)
            )
            (once (not (object_orientation ?t upright)))
        )
    ))
))
(:terminal
    (>= (count ballKnocksTeddy) 7)
)
(:scoring
    (count ballKnocksTeddy)
))

(define (game game-73) (:domain many-objects-room-v1)  ; 73
(:setup (and
    (exists (?h - hexagonal_bin) (game-conserved (near room_center ?h)))
    (forall (?d - dodgeball) (game-optional (on desk ?d)))
))
(:constraints (and
    (preference dodgeballThrownToBinFromDesk (exists (?d - dodgeball ?h - hexagonal_bin)
        (then
            (once (and (adjacent desk agent) (agent_holds ?d)))
            (hold (and (in_motion ?d) (not (agent_holds ?d))))
            (once (and (not (in_motion ?d)) (in ?h ?d)))
        )
    ))
))
(:scoring
    (count dodgeballThrownToBinFromDesk)
))

(define (game game-74) (:domain many-objects-room-v1)  ; 74
(:setup
    (game-conserved (exists (?p - pillow) (near room_center ?p)))
)
(:constraints (and
    (preference golfballInBinFromPillow (exists (?g - golfball ?h - hexagonal_bin ?p - pillow)
        (then
            (once (and (adjacent ?p agent) (agent_holds ?g) (is_setup_object ?p) ))
            (hold (and (in_motion ?g) (not (agent_holds ?g))))
            (once (and (not (in_motion ?g)) (in ?h ?g)))
        )
    ))
    (preference throwAttempt (exists (?g - golfball)
        (then
            (once (agent_holds ?g))
            (hold (and (in_motion ?g) (not (agent_holds ?g))))
            (once (not (in_motion ?g)))
        )
    ))
))
(:terminal
    (>= (count throwAttempt) 10)
)
(:scoring
    (* 5 (count golfballInBinFromPillow))
))


(define (game game-75) (:domain few-objects-room-v1)  ; 75
(:constraints (and
    (preference ballDroppedInBin (exists (?b - ball ?h - hexagonal_bin)
        (then
            (once (and (adjacent ?h agent) (agent_holds ?b)))
            (hold (and (in_motion ?b) (not (agent_holds ?b))))
            (once (and (not (in_motion ?b)) (in ?h ?b)))
        )
    ))
    (preference dropAttempt (exists (?b - ball ?h - hexagonal_bin)
        (then
            (once (and (adjacent ?h agent) (agent_holds ?b)))
            (hold (and (in_motion ?b) (not (agent_holds ?b))))
            (once (not (in_motion ?b)))
        )
    ))
))
(:terminal (or
    (>= (count dropAttempt) 5)
    (>= (count ballDroppedInBin) 1)
))
(:scoring
    (* 5 (count ballDroppedInBin))
))


(define (game game-76) (:domain few-objects-room-v1)  ; 76
(:constraints (and
    (forall (?x - (either pink yellow)) (and
        (preference blockToBinFromRug (exists (?b - cube_block ?h - hexagonal_bin)
            (then
                (once (and (agent_holds ?b) (rug_color_under agent ?x)))
                (hold (and (in_motion ?b) (not (agent_holds ?b))))
                (once (and
                    (not (in_motion ?b))
                    (or
                        (in ?h ?b)
                        (exists (?bl - building) (and
                            (in ?bl ?b)
                            (in ?h ?bl)
                        ))
                    )
                ))
            )
        ))
        (preference blockThrowAttempt (exists (?b - cube_block)
            (then
                (once (and (agent_holds ?b) (rug_color_under agent ?x)))
                (hold (and (in_motion ?b) (not (agent_holds ?b))))
                (once (not (in_motion ?b)))
            )
        ))
    ))
    (preference blockKnockedFromBuildingInBin (exists (?d - dodgeball ?h - hexagonal_bin ?bl - building ?b - block)
        (then
            (once (and
                (agent_holds ?d)
                (rug_color_under agent yellow)
                (in ?bl ?b)
                (in ?h ?bl)
            ))
            (hold-while
                (and (in_motion ?d) (not (agent_holds ?d)))
                (touch ?d ?b)
                (in_motion ?b)
            )
            (once (and (not (in_motion ?d)) (not (in_motion ?b)) (not (in ?bl ?b))))
        )
    ))
    (preference ballThrowAttempt (exists (?d - dodgeball)
        (then
            (once (and (agent_holds ?d) (rug_color_under agent yellow)))
            (hold (and (in_motion ?d) (not (agent_holds ?d))))
            (once (not (in_motion ?d)))
        )
    ))
))
(:terminal (and
    (> (count blockThrowAttempt) 18)
    (>= (count ballThrowAttempt) 2)
))
(:scoring (+
    (* 10 (count-once-per-objects blockToBinFromRug:pink))
    (* 15 (count-once-per-objects blockToBinFromRug:yellow))
    (* 15 (= (count-once-per-objects blockToBinFromRug:yellow) 6))
    (* 15 (<= (count blockThrowAttempt) 18) (= (count-once-per-objects blockToBinFromRug) 6))
    (* 20 (count-once-per-objects blockKnockedFromBuildingInBin))
)))


(define (game game-77) (:domain many-objects-room-v1)  ; 77

(:constraints (and
    (preference throwToBinFromDistance (exists (?d - dodgeball ?h - hexagonal_bin)
        (then
            (once-measure (agent_holds ?d) (distance ?h agent))
            (hold (and (not (agent_holds ?d)) (in_motion ?d)))
            (once (and (not (in_motion ?d)) (in ?h ?d)))
        )
    ))
))
(:scoring (count-measure throwToBinFromDistance)
))


(define (game game-78) (:domain medium-objects-room-v1)  ; 78
(:setup (and
    (exists (?t - teddy_bear) (game-optional (and
        (adjacent_side bed front ?t)
        (adjacent_side bed left ?t)
        (object_orientation ?t upright)
    )))
    (exists (?b - beachball) (game-optional (and
        (< (distance_side bed front ?b) 1)
        (< (distance_side bed left ?b) 1)
        (on floor ?b)
    )))
    (forall (?o - (either hexagonal_bin basketball))
        (game-conserved (near side_table ?o))
    )
))
(:constraints (and
    (preference throwMovesBeachballWithoutKnockingTeddy (exists (?d - dodgeball ?b - beachball ?t - teddy_bear ?db - doggie_bed)
        (then
            (once (and (agent_holds ?d) (near ?db agent) (object_orientation ?t upright)))
            (hold-while
                (and (in_motion ?d) (not (agent_holds ?d)) (not (agent_holds ?t)))
                (touch ?d ?b)
                (in_motion ?b)
            )
            (once (and (not (in_motion ?d)) (not (in_motion ?b)) (object_orientation ?t upright)))
        )
    ))
    (preference throwKnocksOverBear (exists (?d - dodgeball ?b - beachball ?t - teddy_bear ?db - doggie_bed)
        (then
            (once (and (agent_holds ?d) (near ?db agent) (object_orientation ?t upright)))
            (hold (and (in_motion ?d) (not (agent_holds ?d)) (not (agent_holds ?t))))
            (once (and (not (in_motion ?d)) (not (in_motion ?b)) (not (object_orientation ?t upright))))
        )
    ))
))
(:scoring (+
    (* 3 (count throwMovesBeachballWithoutKnockingTeddy))
    (- (count throwKnocksOverBear))
)))


(define (game game-79) (:domain many-objects-room-v1)  ; 79
(:constraints (and
    (preference throwGolfballToBin (exists (?g - golfball ?h - hexagonal_bin)
        (then
            (once (agent_holds ?g))
            (hold (and (not (agent_holds ?g)) (in_motion ?g)))
            (once (and (not (in_motion ?g)) (in ?h ?g)))
        )
    ))
))
(:scoring (count throwGolfballToBin)
))


(define (game game-80) (:domain few-objects-room-v1)  ; 80
(:constraints (and
    (preference pinkObjectMovedToRoomCenter (exists (?o - game_object)
        (then
            (once (and (agent_holds ?o) (same_color ?o pink)))
            (hold (and (in_motion ?o) (not (agent_holds ?o))))
            (once (and (not (in_motion ?o)) (near room_center ?o)))
        )
    ))
    (preference blueObjectMovedToRoomCenter (exists (?o - game_object)
        (then
            (once (and (agent_holds ?o) (same_color ?o blue)))
            (hold (and (in_motion ?o) (not (agent_holds ?o))))
            (once (and (not (in_motion ?o)) (near room_center ?o)
                (exists (?o1 - game_object) (and
                    (same_color ?o1 pink) (near room_center ?o1)
                ))
            ))
        )
    ))
    (preference brownObjectMovedToRoomCenter (exists (?o - game_object)
        (then
            (once (and (agent_holds ?o) (same_color ?o brown)))
            (hold (and (in_motion ?o) (not (agent_holds ?o))))
            (once (and (not (in_motion ?o)) (near room_center ?o)
                (exists (?o1 ?o2 - game_object) (and
                    (same_color ?o1 pink) (near room_center ?o1)
                    (same_color ?o2 blue) (near room_center ?o2)
                ))
            ))
        )
    ))
    (preference pillowMovedToRoomCenter (exists (?o - pillow)
        (then
            (once (and (agent_holds ?o)))
            (hold (and (in_motion ?o) (not (agent_holds ?o))))
            (once (and (not (in_motion ?o)) (near room_center ?o)
                (exists (?o1 ?o2 ?o3 - game_object) (and
                    (same_color ?o1 pink) (near room_center ?o1)
                    (same_color ?o2 blue) (near room_center ?o2)
                    (same_color ?o3 brown) (near room_center ?o3)
                ))
            ))
        )
    ))
    (preference greenObjectMovedToRoomCenter (exists (?o - game_object)
        (then
            (once (and (agent_holds ?o) (same_color ?o green)))
            (hold (and (in_motion ?o) (not (agent_holds ?o))))
            (once (and (not (in_motion ?o)) (near room_center ?o)
                (exists (?o1 ?o2 ?o3 ?o4 - game_object) (and
                    (same_color ?o1 pink) (near room_center ?o1)
                    (same_color ?o2 blue) (near room_center ?o2)
                    (same_color ?o3 brown) (near room_center ?o3)
                    (same_type ?o4 pillow) (near room_center ?o4)
                ))
            ))
        )
    ))
    (preference tanObjectMovedToRoomCenter (exists (?o - game_object)
        (then
            (once (and (agent_holds ?o) (same_color ?o tan)))
            (hold (and (in_motion ?o) (not (agent_holds ?o))))
            (once (and (not (in_motion ?o)) (near room_center ?o)
                (exists (?o1 ?o2 ?o3 ?o4 ?o5 - game_object) (and
                    (same_color ?o1 pink) (near room_center ?o1)
                    (same_color ?o2 blue) (near room_center ?o2)
                    (same_color ?o3 brown) (near room_center ?o3)
                    (same_type ?o4 pillow) (near room_center ?o4)
                    (same_color ?o5 green) (near room_center ?o5)
                ))
            ))
        )
    ))
))
(:scoring (+
    (count-once pinkObjectMovedToRoomCenter)
    (count-once blueObjectMovedToRoomCenter)
    (count-once brownObjectMovedToRoomCenter)
    (count-once pillowMovedToRoomCenter)
    (count-once greenObjectMovedToRoomCenter)
    (count-once tanObjectMovedToRoomCenter)
)))

(define (game game-81) (:domain many-objects-room-v1)  ; 81
(:setup
    (exists (?h - hexagonal_bin ?r1 ?r2 - (either triangular_ramp curved_wooden_ramp))
        (game-conserved (and
            (adjacent ?h desk)
            (near ?h ?r1)
            (near ?h ?r2)
            (not (exists (?o - game_object) (adjacent_side ?h front ?o)))
        ))
    )
)
(:constraints (and
    (preference dodgeballFromRugToBin (exists (?d - dodgeball ?h - hexagonal_bin)
        (then
            (once (and (agent_holds ?d) (on rug agent)))
            (hold (and (in_motion ?d) (not (agent_holds ?d))))
            (once (and (not (in_motion ?d)) (in ?h ?d)))
        )
    ))
))
(:terminal
    (>= (count dodgeballFromRugToBin) 3)
)
(:scoring
    (count dodgeballFromRugToBin)
))

(define (game game-82) (:domain many-objects-room-v1)  ; 82
(:constraints (and
    (preference ballThrownToBin (exists (?b - ball ?h - hexagonal_bin)
        (then
            (once (agent_holds ?b))
            (hold (and (in_motion ?b) (not (agent_holds ?b))))
            (once (and (not (in_motion ?b)) (in ?h ?b)))
        )
    ))
))
(:terminal
    (>= (total-time) 300)
)
(:scoring
    (count ballThrownToBin)
))

(define (game game-83) (:domain many-objects-room-v1)  ; 83
(:setup
    (exists (?h - hexagonal_bin ?c1 ?c2 - chair) (game-conserved (and
        (object_orientation ?h sideways)
        (between ?c1 ?h ?c2)
    )))
)
(:constraints (and
    (forall (?b - (either dodgeball golfball))
        (preference ballToBinFromBed (exists (?h - hexagonal_bin)
            (then
                (once (and (agent_holds ?b) (adjacent bed agent)))
                (hold (and (in_motion ?b) (not (agent_holds ?b))))
                (once (and (not (in_motion ?b)) (in ?h ?b)))
            )
        ))
    )
))
(:scoring (+
    (count-once-per-objects ballToBinFromBed:dodgeball)
    (* (= (count-once-per-objects ballToBinFromBed:dodgeball) 3) (count-once-per-objects ballToBinFromBed:golfball))
)))

; 84 is a hiding game -- invalid

(define (game game-85) (:domain few-objects-room-v1)  ; 85
(:constraints (and
    (forall (?x - color)
        (preference cubeThrownToBin (exists (?h - hexagonal_bin ?b - cube_block)
            (then
                (once (and
                    (agent_holds ?b)
                    (rug_color_under agent pink)
                    (same_color ?b ?x)
                    (not (exists (?ob - cube_block) (in ?h ?ob)))
                ))
                (hold (and (in_motion ?b) (not (agent_holds ?b))))
                (once (and (not (in_motion ?b)) (in ?h ?b)))
            )
        ))
    )
    (forall (?b - cube_block)
        (preference throwAttempt
            (then
                (once (and
                    (agent_holds ?b)
                    (rug_color_under agent pink)
                ))
                (hold (and (in_motion ?b) (not (agent_holds ?b))))
                (once (not (in_motion ?b)))
            )
        )
    )
))
(:terminal (or
    (> (external-forall-maximize (count throwAttempt)) 1)
    (>= (count-once-per-objects throwAttempt) 6)
))
(:scoring (+
    (count-once-per-objects cubeThrownToBin:yellow)
    (* 2 (count-once-per-objects cubeThrownToBin:tan))
    (* 3 (count-once-per-objects cubeThrownToBin:blue))
    (- (count-once-per-objects throwAttempt))
)))

; 86 is a dup of 84 -- and is aldo invalid

(define (game game-87) (:domain few-objects-room-v1)  ; 87
(:setup
    (exists (?h - hexagonal_bin ?w - wall) (game-conserved (and
        (on desk ?h)
        (adjacent ?h ?w)
    )))
)
(:constraints (and
    (forall (?o - (either dodgeball block))
        (preference basketMadeFromRug (exists (?h - hexagonal_bin)
            (then
                (once (and (agent_holds ?o) (on rug agent)))
                (hold (and (in_motion ?o) (not (agent_holds ?o))))
                (once (and (not (in_motion ?o)) (in ?h ?o)))
            )
        ))
    )
))
(:scoring (+
    (count basketMadeFromRug:dodgeball)
    (* 2 (count basketMadeFromRug:block))
)))


(define (game game-88) (:domain few-objects-room-v1)  ; 88
(:setup
    (exists (?h - hexagonal_bin ?p - pillow ?b1 ?b2 ?b3 ?b4 ?b5 ?b6 - cube_block)
        (game-conserved (and
            (on bed ?h)
            (object_orientation ?p diagonal)
            (adjacent_side ?h left ?b1)
            (on bed ?b1)
            (on ?b1 ?b2)
            (on ?b2 ?b3)
            (adjacent_side ?h right ?b4)
            (on bed ?b4)
            (on ?b4 ?b5)
            (on ?b5 ?b6)
        ))
    )
)
(:constraints (and
    (preference throwFromEdgeOfRug (exists (?d - dodgeball ?h - hexagonal_bin)
        (then
            (once (and
                (agent_holds ?d)
                (on floor agent)
                (adjacent rug agent)
                (> (distance bed agent) 2)
            ))
            (hold (and (in_motion ?d) (not (agent_holds ?d))))
            (once (and (not (in_motion ?d)) (in ?h ?d)))
        )
    ))
    (preference throwAttempt (exists (?d - dodgeball)
        (then
            (once (and
                (agent_holds ?d)
                (on floor agent)
                (adjacent rug agent)
                (> (distance bed agent) 2)
            ))
            (hold (and (in_motion ?d) (not (agent_holds ?d))))
            (once (not (in_motion ?d)))
        )
    ))
    (preference throwAttemptKnocksBlock (exists (?d - dodgeball ?c - cube_block)
        (then
            (once (and
                (agent_holds ?d)
                (on floor agent)
                (adjacent rug agent)
                (> (distance bed agent) 2)
            ))
            (hold-while
                (and (in_motion ?d) (not (agent_holds ?d)))
                (touch ?d ?c)
                (in_motion ?c)
            )
            (once (not (in_motion ?d)))
        )
    ))
))
(:terminal (or
    (>= (count throwAttempt) 10)
    (>= (count-once throwAttemptKnocksBlock) 1)
    (>= (total-score) 5)
))
(:scoring
    (count throwFromEdgeOfRug)
))

(define (game game-89) (:domain medium-objects-room-v1)  ; 89
(:setup
    (exists (?d - desktop ?h - hexagonal_bin) (game-conserved (and
        (on desk ?h)
        (not (on desk ?d))
    )))
)
(:constraints (and
    (forall (?b - ball)
        (preference ballThrownFromRug (exists (?h - hexagonal_bin)
            (then
                (once (and (agent_holds ?b) (on rug agent)))
                (hold (and (in_motion ?b) (not (agent_holds ?b))))
                (once (and (not (in_motion ?b)) (in ?h ?b)))
            )
        ))
    )
))
(:terminal (or
    (>= (total-time) 180)
    (>= (total-score) 10)
))
(:scoring (+
    (count ballThrownFromRug:dodgeball)
    (* 2 (count ballThrownFromRug:basketball))
    (* 10 (count ballThrownFromRug:beachball))
)))

(define (game game-90) (:domain many-objects-room-v1)  ; 90
(:constraints (and
    (preference dodgeballBouncesOnceToDoggieBed (exists (?d - dodgeball ?b - doggie_bed)
        (then
            (once (agent_holds ?d))
            (hold (and (in_motion ?d) (not (agent_holds ?d)) (not (touch floor ?d))))
            (hold (and (in_motion ?d) (not (agent_holds ?d)) (touch floor ?d)))
            (hold (and (in_motion ?d) (not (agent_holds ?d)) (not (touch floor ?d))))
            (once (and (not (in_motion ?d)) (on ?b ?d)))
        )
    ))
))
(:scoring
    (count dodgeballBouncesOnceToDoggieBed)
))

; 91 is a dup of 89 with slightly different scoring numbers

; 92 is a hiding game -- invalid

(define (game game-93) (:domain many-objects-room-v1)  ; 93
(:constraints (and
    (preference throwBallToBin (exists (?d - dodgeball ?h - hexagonal_bin)
        (then
            (once (agent_holds ?d))
            (hold (and (not (agent_holds ?d)) (in_motion ?d)))
            (once (and (not (in_motion ?d)) (in ?h ?d)))
        )
    ))
))
(:scoring
    (count throwBallToBin)
))


(define (game game-94) (:domain many-objects-room-v1)  ; 94
(:constraints (and
    (forall (?b - (either dodgeball golfball)) (and
        (preference ballThrownFromDoor (exists (?h - hexagonal_bin)
            (then
                (once (and (agent_holds ?b) (adjacent door agent)))
                (hold (and (in_motion ?b) (not (agent_holds ?b))))
                (once (and (not (in_motion ?b)) (in ?h ?b)))
            )
        ))
        (preference throwAttemptFromDoor
            (then
                (once (and (agent_holds ?b) (adjacent door agent)))
                (hold (and (in_motion ?b) (not (agent_holds ?b))))
                (once (not (in_motion ?b)))
            )
        )
    ))
))
(:terminal
    (>= (count throwAttemptFromDoor) 8)
)
(:scoring (+
    (* 3 (count ballThrownFromDoor:dodgeball))
    (* 6 (count ballThrownFromDoor:golfball))
)))

; 95 requires counting something that happens during a preference

; 96 is underconstrainted -- I'm omitting it for now


(define (game game-97) (:domain medium-objects-room-v1)  ; 97
(:constraints (and
    (preference ballThrownToRug (exists (?d - dodgeball_red)
        (then
            (once (and (agent_holds ?d) (not (on rug agent))))
            (hold (and (in_motion ?d) (not (agent_holds ?d))))
            (once (and (not (in_motion ?d)) (on rug ?d)))
        )
    ))
))
(:terminal
    (>= (total-time) 60)
)
(:scoring
    (count ballThrownToRug)
))


(define (game game-98) (:domain medium-objects-room-v1)  ; 98
(:setup (and
    (exists (?h - hexagonal_bin) (game-conserved (not (exists (?s - shelf) (above ?h ?s)))))
    (forall (?b - ball) (game-optional (on bed ?b)))
))
(:constraints (and
    (forall (?b - ball)
        (preference ballThrownToBin (exists (?h - hexagonal_bin)
            (then
                (once (and (agent_holds ?b) (or (on bed agent) (adjacent bed agent))))
                (hold (and (in_motion ?b) (not (agent_holds ?b))))
                (once (and (not (in_motion ?b)) (in ?h ?b)))
            )
        ))
    )
))
(:terminal
    (>= (total-score) 6)
)
(:scoring (+
    (count-once-per-objects ballThrownToBin:beachball)
    (* 2 (count-once-per-objects ballThrownToBin:basketball))
    (* 3 (count-once-per-objects ballThrownToBin:dodgeball))
)))


(define (game game-99) (:domain few-objects-room-v1)  ; 99
(:constraints (and
    (preference cubeBlockFromBedToShelf (exists (?c - cube_block ?s - shelf)
        (then
            (once (and (agent_holds ?c) (adjacent bed agent)))
            (hold (and (in_motion ?c) (not (agent_holds ?c))))
            (once (and (not (in_motion ?c)) (on ?s ?c)))
        )
    ))
    (preference cubeBlockThrowAttempt (exists (?c - cube_block)
        (then
            (once (and (agent_holds ?c) (adjacent bed agent)))
            (hold (and (in_motion ?c) (not (agent_holds ?c))))
            (once (not (in_motion ?c)))
        )
    ))
))
(:terminal
    (>= (count cubeBlockThrowAttempt) 3)
)
(:scoring
    (count cubeBlockFromBedToShelf)
))

(define (game game-100) (:domain medium-objects-room-v1)  ; 100
(:setup
    (exists (?h - hexagonal_bin ?d - doggie_bed) (game-conserved (and
        (on floor ?d)
        (on bed ?h)
        (equal_z_position ?h ?d)
    )))
)
(:constraints (and
    (forall (?t - (either hexagonal_bin doggie_bed))
        (preference dodgeballFromDeskToTarget (exists (?d - dodgeball)
            (then
                (once (and (agent_holds ?d) (adjacent desk agent)))
                (hold (and (in_motion ?d) (not (agent_holds ?d))))
                (once (and (not (in_motion ?d)) (or (in ?t ?d) (on ?t ?d))))
            )
        ))
    )
))
(:scoring (+
    (* 2 (count dodgeballFromDeskToTarget:doggie_bed))
    (* 3 (count dodgeballFromDeskToTarget:hexagonal_bin))
)))


(define (game game-101) (:domain few-objects-room-v1)  ; 101
(:setup (and
    (exists (?h - hexagonal_bin) (game-conserved (on bed ?h)))
    (exists (?r - curved_wooden_ramp) (game-conserved (and (adjacent bed ?r) (faces ?r desk))))
    (exists (?c1 ?c2 - cube_block_blue ?c3 ?c4 - cube_block_yellow) (game-conserved (and
        (= (distance desk ?c1) 1)
        (= (distance desk ?c2) 1)
        (= (distance desk ?c3) 2)
        (= (distance desk ?c4) 2)
        (between desk ?c1 ?c3)
        (between desk ?c2 ?c4)
    )))
))
(:constraints (and
    (forall (?c - (either cube_block_blue cube_block_yellow)) (and
        (preference ballThrownFromBehindBlock (exists (?b - ball ?h - hexagonal_bin)
            (then
                (once (and
                    (agent_holds ?b)
                    (is_setup_object ?c)
                    (>= (distance ?h agent) (distance ?c ?h))
                ))
                (hold (and (in_motion ?b) (not (agent_holds ?b))))
                (once (and (not (in_motion ?b)) (in ?h ?b)))
            )
        ))
        (preference throwAttemptFromBehindBlock (exists (?b - ball ?h - hexagonal_bin)
            (then
                (once (and
                    (agent_holds ?b)
                    (is_setup_object ?c)
                    (>= (distance ?h agent) (distance ?c ?h))
                ))
                (hold (and (in_motion ?b) (not (agent_holds ?b))))
                (once (and (not (in_motion ?b))))
            )
        ))
    ))
))
(:terminal (or
    (>= (count throwAttemptFromBehindBlock) 2)
    (>= (total-score) 50)
))
(:scoring (+
    (* 10 (count ballThrownFromBehindBlock:cube_block_blue))
    (* 5 (count ballThrownFromBehindBlock:cube_block_yellow))
    (* 30 (= (count ballThrownFromBehindBlock:cube_block_blue) 2))
    (* 15 (= (count ballThrownFromBehindBlock:cube_block_yellow) 2))
)))


; 102 is almost a copy of 101 and same participant -- omit


(define (game game-103) (:domain few-objects-room-v1)  ; 103
(:setup
    (exists (?h - hexagonal_bin) (game-conserved (and
        (on bed ?h)
        (object_orientation ?h sideways)
    )))
)
(:constraints (and
    (preference dodgeballHitsBin (exists (?d - dodgeball ?h - hexagonal_bin)
        (then
            (once (agent_holds ?d))
            (hold-while
                (and (in_motion ?d) (not (agent_holds ?d)) (not (in ?h ?d)))
                (touch ?h ?d)
            )
            (once (and (not (in_motion ?d)) (not (in ?h ?d))))
        )
    ))
    (preference dodgeballHitsBinBottom (exists (?d - dodgeball ?h - hexagonal_bin)
        (then
            (once (agent_holds ?d))
            (hold-while
                (and (in_motion ?d) (not (agent_holds ?d)))
                (in ?h ?d)
            )
            (once (and (not (in_motion ?d))))
        )
    ))
    (preference throwAttempt (exists (?d - dodgeball)
        (then
            (once (agent_holds ?d))
            (hold (and (in_motion ?d) (not (agent_holds ?d))))
            (once (and (not (in_motion ?d))))
        )
    ))
))
(:terminal
    (>= (count throwAttempt) 10)
)
(:scoring (+
    (count dodgeballHitsBin)
    (* 2 (count dodgeballHitsBinBottom))
)))


(define (game game-104) (:domain few-objects-room-v1)  ; 104
(:setup
    (exists (?h - hexagonal_bin) (game-conserved (and
        (equal_x_position east_sliding_door ?h)
    )))
)
(:constraints (and
    (preference throwFromEdgeOfRug (exists (?d - dodgeball ?h - hexagonal_bin)
        (then
            (once (and (agent_holds ?d) (adjacent rug agent)))
            (hold (and (in_motion ?d) (not (agent_holds ?d))))
            (once (in ?h ?d))  ; participant specified that couning out is okay
        )
    ))
))
(:terminal
    (>= (total-time) 300)
)
(:scoring
    (count throwFromEdgeOfRug)
))


(define (game game-105) (:domain few-objects-room-v1)  ; 105
(:setup (and
    (forall (?c - chair) (game-conserved (or (on bed ?c) (adjacent bed ?c))))
    (forall (?c - cube_block) (game-optional (on rug ?c)))
    (game-optional (not (exists (?o - game_object) (above ?o desk))))
    (forall (?d - dodgeball) (game-conserved (not (exists (?s - shelf) (on ?s ?d)))))
))
(:constraints (and
    (preference woodenBlockMovedFromRugToDesk (exists (?b - cube_block_tan)
        (then
            (once (and
                (forall (?c - (either cube_block_blue cube_block_yellow)) (on rug ?c))
                (on rug ?b)
            ))
            (hold (forall (?c - (either cube_block_blue cube_block_yellow)) (or
                (on rug ?c)
                (agent_holds ?c)
                (in_motion ?c)
                (near desk ?c)
                (exists (?c2 - (either cube_block_blue cube_block_yellow)) (and
                    (not (same_object ?c ?c2))
                    (adjacent ?c ?c2)
                    (on floor ?c)
                    (on floor ?c2)
                ))
            )))
            (hold (forall (?c - (either cube_block_blue cube_block_yellow))
                (near desk ?c)
            ))
            (once (above ?b desk))
        )
    ))
))
(:scoring
    (count-once-per-objects woodenBlockMovedFromRugToDesk)
))


(define (game game-106) (:domain few-objects-room-v1)  ; 106
(:constraints (and
    (preference throwInBin (exists (?b - ball ?h - hexagonal_bin)
        (then
            (once (agent_holds ?b))
            (hold (and (not (agent_holds ?b)) (in_motion ?b)))
            (once (and (not (in_motion ?b)) (in ?h ?b)))
        )
    ))
    (preference throwAttempt (exists (?b - ball)
        (then
            (once (agent_holds ?b))
            (hold (and (not (agent_holds ?b)) (in_motion ?b)))
            (once (not (in_motion ?b)))
        )
    ))
))
(:terminal (or
    (>= (total-score) 6)
    (>= (count throwAttempt) 15)
))
(:scoring
    (count throwInBin)
))

; 107 and 109 are by the same participant, and 109 is actually mostly valid

(define (game game-108) (:domain medium-objects-room-v1)  ; 108
(:setup (and
    (exists (?h - hexagonal_bin ?b1 ?b2 - tall_cylindrical_block ?p1 ?p2 - pyramid_block ?b3 - cylindrical_block)
        (and
            (game-conserved (and
                (on side_table ?b3)
                (on bed ?b1)
                (on ?b1 ?p1)
                (on bed ?b2)
                (on ?p2 ?b2)
                (adjacent ?b1 north_wall)
                (between ?b1 ?h ?b2)
                (= (distance ?b1 ?h) (distance ?b2 ?h))
            ))
            (game-optional (and
                (on bed ?h)
                (equal_z_position bed ?h)
            ))
        )
    )
    (exists (?d - doggie_bed) (forall (?b - ball) (game-optional (or
        (on ?d ?b)
        (near ?d ?b)
    ))))
))
(:constraints (and
    (preference agentLeavesDogbedOrNoMoreBalls (exists (?d - doggie_bed)
        (then
            (once (not (near ?d agent)))
            (hold-while
                (near ?d agent)
                (exists (?b - ball) (agent_holds ?b))
            )
            (once (or
                (not (near ?d agent))
                (forall (?b - ball) (and
                    (not (in_motion ?b))
                    (not (near ?b agent))
                ))
            ))
        )
    ))
    (forall (?c - (either cylindrical_block tall_cylindrical_block pyramid_block))
        (preference throwKnocksBlock (exists (?b - ball ?d - doggie_bed)
            (then
                (once (and
                    (is_setup_object ?c)
                    (agent_holds ?b)
                    (near ?d agent)
                ))
                (hold-while
                    (and (in_motion ?b) (not (agent_holds ?b)))
                    (touch ?b ?c)
                    (in_motion ?c)
                )
                (once (not (in_motion ?b)))
            )
        ))
    )
    (forall (?b - ball)
        (preference ballInOrOnBin (exists (?d - doggie_bed ?h - hexagonal_bin)
            (then
                (once (and
                    (agent_holds ?b)
                    (near ?d agent)
                ))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (not (in_motion ?b)) (or (in ?h ?b) (on ?h ?b))))
            )
        ))
    )
))
(:terminal
    (>= (count-once agentLeavesDogbedOrNoMoreBalls) 1)
)
(:scoring (+
    (* 3 (count-once-per-external-objects throwKnocksBlock:pyramid_block))
    (* -3 (count-once-per-external-objects throwKnocksBlock:tall_cylindrical_block))
    (count-once-per-external-objects throwKnocksBlock:cylindrical_block)
    (* 2 (count-once-per-external-objects ballInOrOnBin:dodgeball))
    (* 2 (count-once-per-external-objects ballInOrOnBin:basketball))
    (* 4 (count-once-per-external-objects ballInOrOnBin:beachball))
)))

(define (game game-109) (:domain many-objects-room-v1)  ; 109
(:constraints (and
    (preference ballThrownToBin (exists (?b - ball ?h - hexagonal_bin)
        (then
            (once (agent_holds ?b))
            (hold (and (not (agent_holds ?b)) (in_motion ?b)))
            (once (and (not (in_motion ?b)) (in ?h ?b)))
        )
    ))
    (preference cubeBlockThrownToTopShelf (exists (?c - cube_block)
        (then
            (once (agent_holds ?c))
            (hold (and (not (agent_holds ?c)) (in_motion ?c)))
            (once (and (not (in_motion ?c)) (on top_shelf ?c)))
        )
    ))
    (preference pillowThrownToDoggieBed (exists (?p - pillow ?d - doggie_bed)
        (then
            (once (agent_holds ?p))
            (hold (and (not (agent_holds ?p)) (in_motion ?p)))
            (once (and (not (in_motion ?p)) (on ?d ?p)))
        )
    ))
))
(:scoring (+
    (count-once-per-objects ballThrownToBin)
    (count-once-per-objects cubeBlockThrownToTopShelf)
    (count-once-per-objects pillowThrownToDoggieBed)
)))

(define (game game-110) (:domain few-objects-room-v1)  ; 110
(:setup (and
    (forall (?c - chair) (game-conserved (equal_x_position door ?c)))
    (exists (?h - hexagonal_bin) (game-conserved (and
        (adjacent south_wall ?h)
        (adjacent west_wall ?h)
    )))
    (forall (?o - (either dodgeball cube_block alarm_clock book)) (game-optional (adjacent ?o desk)))
))
(:constraints (and
    (forall (?o - (either dodgeball cube_block alarm_clock book)) (and
        (preference throwFromBehindChairsInBin (exists (?h - hexagonal_bin)
            (then
                (once (and
                    (agent_holds ?o)
                    (forall (?c - chair) (> (x_position agent) (x_position ?c)))
                ))
                (hold (and (not (agent_holds ?o)) (in_motion ?o)))
                (once (and (not (in_motion ?o)) (in ?h ?o)))
            )
        ))
        (preference throwAttempt
            (then
                (once (and
                    (agent_holds ?o)
                    (forall (?c - chair) (> (x_position agent) (x_position ?c)))
                ))
                (hold (and (not (agent_holds ?o)) (in_motion ?o)))
                (once (not (in_motion ?o)))
            )
        )
    ))
))
(:terminal (or
    (> (external-forall-maximize (count throwAttempt:dodgeball)) 3)
    (> (external-forall-maximize (count throwAttempt:cube_block)) 1)
    (> (count throwAttempt:book) 1)
    (> (count throwAttempt:alarm_clock) 1)
))
(:scoring (+
    (* 8 (count throwFromBehindChairsInBin:dodgeball))
    (* 5 (count throwFromBehindChairsInBin:cube_block))
    (* 20 (count throwFromBehindChairsInBin:alarm_clock))
    (* 50 (count throwFromBehindChairsInBin:book))
)))

; 111 requires evaluation that one preference takes place before another preference is evaluated, and it's underconstrained

; 112 is definitely invalid and underdefined

(define (game game-113) (:domain few-objects-room-v1)  ; 113
(:setup
    (exists (?h - hexagonal_bin ?c1 ?c2 ?c3 ?c4 - cube_block ?r - curved_wooden_ramp) (game-conserved (and
        (adjacent_side ?h front ?c1)
        (adjacent ?c1 ?c3)
        (between ?h ?c1 ?c3)
        (on ?c1 ?c2)
        (on ?c3 ?c4)
        (adjacent_side ?r back ?c3)
        (between ?r ?c3 ?c1)
    )))
)
(:constraints (and
    (preference ballThrownThroughRampAndBlocksToBin (exists (?b - ball ?r - curved_wooden_ramp ?h - hexagonal_bin ?c1 ?c2 - cube_block)
        (then
            (once (agent_holds ?b))
            (hold-while
                (and (not (agent_holds ?b)) (in_motion ?b))
                (on ?r ?b)
                (on ?c1 ?b)
                (on ?c2 ?b)
            )
            (once (and (not (in_motion ?b)) (in ?h ?b)))
        )
    ))
))
(:scoring
    (count ballThrownThroughRampAndBlocksToBin)
))

(define (game game-114) (:domain medium-objects-room-v1)  ; 114
(:setup
    (exists (?d - doggie_bed) (game-conserved (near room_center ?d)))
)
(:constraints (and
    (preference objectInBuilding (exists (?o - game_object ?d - doggie_bed ?b - building)
        (at-end (and
            (not (same_object ?o ?d))
            (in ?b ?d)
            (in ?b ?o)
            (on floor ?d)
            (not (on floor ?o))
            (not (exists (?w - wall) (touch ?w ?o)))
        ))
    ))
))
(:scoring
    (count-once-per-objects objectInBuilding)
))

(define (game game-115) (:domain medium-objects-room-v1)  ; 115
(:setup (and
    (exists (?c - chair ?r - triangular_ramp ?t - teddy_bear ?h - hexagonal_bin) (and
        (game-conserved (and
            (near room_center ?r)
            (adjacent_side ?r front ?c)
            (between ?h ?c ?r)
            (forall (?b - ball) (near ?b ?h))
        ))
        (game-optional (and
            (on ?c ?t)
        ))
    ))
))
(:constraints (and
    (preference teddyBearLandsInBin (exists (?t - teddy_bear ?h - hexagonal_bin ?c - chair)
        (then
            (once (on ?c ?t))
            (hold (agent_holds ?t))
            (hold (and (not (agent_holds ?t)) (in_motion ?t)))
            (once (and (not (in_motion ?t)) (in ?h ?t)))
        )
    ))
    (preference teddyBearHitsBall (exists (?t - teddy_bear ?b - ball ?c - chair)
        (then
            (once (on ?c ?t))
            (hold (agent_holds ?t))
            (hold (and (not (agent_holds ?t)) (in_motion ?t)))
            (once (touch ?t ?b))
        )
    ))
))
(:scoring (+
    (* 5 (count teddyBearLandsInBin))
    (count teddyBearHitsBall)
)))

(define (game game-116) (:domain medium-objects-room-v1)  ; 116
(:setup
    (exists (?h - hexagonal_bin) (game-conserved (or (on bed ?h) (on desk ?h))))
)
(:constraints (and
    (forall (?b - (either basketball dodgeball)) (and
        (preference ballThrownToBin (exists (?h - hexagonal_bin)
            (then
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (not (in_motion ?b)) (in ?h ?b)))
            )
        ))
        (preference throwAttempt
            (then
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (not (in_motion ?b)))
            )
        )
    ))
))
(:terminal
    (> (external-forall-maximize (count throwAttempt)) 4)
)
(:scoring
    (count ballThrownToBin)
))


(define (game game-117) (:domain medium-objects-room-v1)  ; 117
(:setup
    (exists (?h - hexagonal_bin ?r - triangular_ramp) (game-conserved (and
        (near ?h ?r)
        (not (adjacent ?h ?r))
    )))
)
(:constraints (and
    (preference redDodgeballThrownToBinWithoutTouchingFloor (exists (?h - hexagonal_bin ?r - dodgeball_red)
        (then
            (once (agent_holds ?r))
            (hold (and (not (agent_holds ?r)) (in_motion ?r) (not (touch floor ?r))))
            (once (and (not (in_motion ?r)) (in ?h ?r)))
        )
    ))
    (preference redDodgeballThrownToBin (exists (?h - hexagonal_bin ?r - dodgeball_red)
        (then
            (once (agent_holds ?r))
            (hold (and (not (agent_holds ?r)) (in_motion ?r)))
            (once (and (not (in_motion ?r)) (in ?h ?r)))
        )
    ))
    (preference throwAttempt (exists (?r - dodgeball_red)
        (then
            (once (agent_holds ?r))
            (hold (and (not (agent_holds ?r)) (in_motion ?r)))
            (once (not (in_motion ?r)))
        )
    ))
))
(:terminal (or
    (>= (count throwAttempt) 10)
    (>= (count-once redDodgeballThrownToBinWithoutTouchingFloor) 1)
    (>= (count-once redDodgeballThrownToBin) 1)
))
(:scoring (+
    (* 5 (count-once redDodgeballThrownToBin))
    (* 3
        (= (count throwAttempt) 1)
        (count-once redDodgeballThrownToBinWithoutTouchingFloor)
    )
    (* 2
        (< (count throwAttempt) 5)
        (count-once redDodgeballThrownToBinWithoutTouchingFloor)
    )
)))

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

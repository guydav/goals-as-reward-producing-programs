(define (game few-objects-2) (:domain few-objects-room-v1)
(:setup
    (exists (?h - hexagonal_bin ?c - chair)
        (game-conserved (< (distance ?h ?c) 1))
    )

)
(:constraints (and
    ; TODO: think about these as a test case for having "predicate while other preference"
    (preference basketWithChairInTheWay
        (exists (?c - chair ?h - hexagonal_bin ?d - dodgeball)
            (then
                (once (agent_holds ?d))
                (hold (and (in_motion ?d) (not (agent_holds ?d)) (between agent ?c ?h)))
                (once (and (in ?h ?d) (not (agent_holds ?d))))
            )
        )
    )
    (preference basketMade
        (exists (?h - hexagonal_bin ?d - dodgeball)
            (then
                (once (agent_holds ?d))
                (hold (and (in_motion ?d) (not (agent_holds ?d))))
                (once (and (in ?h ?d) (not (agent_holds ?d))))
            )
        )
    )
))
(:scoring maximize (+
    (* 2 (count-nonoverlapping basketWithChairInTheWay))
    (* 1 (count-nonoverlapping chairBetweenAgentAndBall))
))
)

(define (game few-objects-3) (:domain few-objects-room-v1)
(:setup
    (forall (?c - (either desktop laptop))
        (game-conserved (not (on desk ?c)))
    )
)
(:constraints (and
    (preference cubeBlockOnDesk (exists (?c - cube_block ?d - desk)
        (at-end
            (and
                (in_building ?c)
                (or (object_orientation ?c edge) (object_orientation ?c point))
                (on ?d ?c)
            )
        )
    ))
    (preference cubeBlockOnCubeBlock (exists (?b ?c - cube_block)
        (at-end
            (and
                (in_building ?c)
                (or (object_orientation ?c edge) (object_orientation ?c point))
                (on ?b ?c)
            )
        )
    ))
))
(:scoring maximize (+
    (count-once cubeBlockOnDesk)
    (count-once-per-objects cubeBlockOnCubeBlock)
)))


(define (game few-objects-4) (:domain few-objects-room-v1)
(:setup (and
    (exists (?w - wall ?h - hexagonal_bin)
            (game-conserved (= (distance ?w ?h) 1))
    )
))
(:constraints (and
    (preference throwToWallToBin
        (exists (?d - dodgeball ?w - wall ?h - hexagonal_bin)
            (then
                (once (agent_holds ?d)) ; ball starts in hand
                (hold-while
                    (and (not (agent_holds ?d)) (in_motion ?d))
                    (touch ?w ?d)
                )
                (once  (and (in ?h ?d) (not (in_motion ?d)))) ; touches wall before in bin
            )
        )
    )
))
(:terminal
    (>= (total-time) 60)
)
(:scoring maximize (count-nonoverlapping throwToWallToBin))
)


(define (game few-objects-5) (:domain few-objects-room-v1)
(:setup (and
    (exists (?c - curved_wooden_ramp ?h - hexagonal_bin ?d - dodgeball ?t - textbook)
        (and
            (game-conserved (adjacent (side ?h front) (side ?c back)))
            (game-conserved (= (distance ?t (side ?c front)) 1))
            (game-optional (adjacent ?d ?t))
        )
    )
))
(:constraints (and
    (preference kickBallToBin
        (exists (?d - dodgeball ?r - curved_wooden_ramp ?h - hexagonal_bin ?t - textbook)
            (then
                ; agent starts by touching ball while next to the marking textbook
                (once (and (adjacent agent ?t) (touch agent ?d)))
                (hold-while
                    (and (not (agent_holds ?d)) (in_motion ?d)) ; in motion, not in hand until...
                    (on ?r ?d)   ; on ramp and then in bin -- should this be touch?
                )
                (once (and (in ?h ?d) (not (in_motion ?d))))
            )
        )
    )
    (preference ballKicked
        (exists (?d - dodgeball ?t - textbook)
            (then
                (once (and (adjacent agent ?t) (touch agent ?d)))
                (hold (and (not (agent_holds ?d)) (in_motion ?d)))
                (once (not (in_motion ?d)))
            )
        )
    )
))
(:terminal
    (>= (count-nonoverlapping ballKicked) 10)
)
(:scoring maximize (count-nonoverlapping throwToWallToBin))
)

;6 is invalid

(define (game few-objects-7) (:domain few-objects-room-v1)
(:setup (and
    (exists (?c - curved_wooden_ramp ?h - hexagonal_bin)
        (and
            (game-conserved (adjacent (side ?h front) (side ?c back)))
            (game-conserved (= (distance ?c room_center) 1))
        )
    )
))
(:constraints (and
    (preference bowlBallToBin
        (exists (?d - dodgeball ?r - curved_wooden_ramp ?h - hexagonal_bin)
            (then
                (once (agent_holds ?d)) ; agent starts by holding ball
                (hold-while
                    (and (not (agent_holds ?d)) (in_motion ?d)) ; in motion, not in hand until...
                    (on ?r ?d)  ; on ramp and then in bin -- should this be touch?
                )
                (once (and (in ?h ?d) (not (in_motion ?d))))
            )
    ))
))
(:scoring maximize (* 5 (count-nonoverlapping bowlBallToBin)))
)


(define (game few-objects-8) (:domain few-objects-room-v1)
(:setup (and
    (exists (?c - curved_wooden_ramp ?h - hexagonal_bin)
        (game-conserved (adjacent (side ?h front) (side ?c back)))
    )
))
(:constraints (and
    (preference rollBallToBin
        (exists (?d - dodgeball ?r - curved_wooden_ramp ?h - hexagonal_bin)
            (then
                (once (agent_holds ?d)) ; agent starts by holding ball
                (hold-while
                    (and (not (agent_holds ?d)) (in_motion ?d)) ; in motion, not in hand until...
                    (on ?r ?d) ; on ramp and then in bin -- should this be touch?
                )
                (once (and (in ?h ?d) (not (in_motion ?d))))
            )
        )
    )
    (preference bothBallsThrown
        (forall (?d - dodgeball)
            (then
                (once (agent_holds ?d)) ; agent starts by holding ball
                (hold (and (not (agent_holds ?d)) (in_motion ?d)))
                (once (not (in_motion ?d)))
            )
        )
    )
))
(:terminal
    (>= (count-nonoverlapping bothBallsThrown) 5)
)
(:scoring maximize (* 5 (count-nonoverlapping rollBallToBin)))
)

(define (game few-objects-9) (:domain few-objects-room-v1)
(:setup
)
(:constraints (and
    (forall (?o - (either cellphone textbook laptop))
        (preference objectThrownOnDoggieBed
            (exists (?d - doggie_bed)
                (then
                    (once (agent_holds ?o))
                    (hold (and (not (agent_holds ?o)) (in_motion ?o))) ; in motion, not in hand until...
                    (once (and (on ?d ?o) (not (in_motion ?o))))
                )
            )
        )
    )
))
(:scoring maximize (+
    (* 15 (count-nonoverlapping objectThrownOnDoggieBed:cellphone))
    (* 10 (count-nonoverlapping objectThrownOnDoggieBed:textbook))
    (* 5 (count-nonoverlapping objectThrownOnDoggieBed:laptop))
)))


(define (game few-objects-10) (:domain few-objects-room-v1)
(:setup
; no real setup for 10 unless we want to mark which objects are in the game
)
(:constraints (and
    (preference chairHitFromBed
        (exists (?c - chair ?o - (either doggie_bed pillow))
            (then
                (once (and (agent_holds ?o) (on bed agent)))
                (hold (and (not (agent_holds ?o)) (in_motion ?o)))
                (once (touch ?o ?c))
            )
        )
    )
))
(:scoring maximize (+
    ; TODO: ths instructions here say "20 score each pillow or doggie bed hit the 2chairs 10 times in a row"
    ; TODO: another test case for quantifying streaks?
    (* 20 (count-nonoverlapping chairHitFromBed))
)))

; 11 is invalid

(define (game few-objects-12) (:domain few-objects-room-v1)
(:setup
; no real setup for 12 since the bin moves
)
(:constraints (and
    (preference throwToBinOnBed
        (exists (?b - bed ?d - dodgeball ?h - hexagonal_bin)
            (then
                (once (agent_holds ?d))
                (hold (and (not (agent_holds ?d)) (in_motion ?d)))
                (once (and (on ?h ?d) (not (in_motion ?d)) (on ?b ?h)))
            )
        )
    )
    (preference throwToBinOnDesk
        (exists (?d - dodgeball ?e - desk ?h - hexagonal_bin)
            (then
                (once (agent_holds ?d))
                (hold (and (not (agent_holds ?d)) (in_motion ?d)))
                (once (and (on ?h ?d) (not (in_motion ?d)) (on ?e ?h)))
            )
        )
    )
    (preference throwToBinOnShelf
        (exists (?d - dodgeball ?h - hexagonal_bin ?s - shelf)
            (then
                (once (agent_holds ?d))
                (hold (and (not (agent_holds ?d)) (in_motion ?d)))
                (once (and (on ?h ?d) (not (in_motion ?d)) (on ?s ?h)))
            )
        )
    )
))
(:scoring maximize (+
    (* 1 (count-once throwToBinOnBed))
    (* 2 (count-once throwToBinOnBed) (count-once throwToBinOnDesk))
    (* 3 (count-once throwToBinOnBed) (count-once throwToBinOnDesk) (count-once throwToBinOnShelf))
)))

(define (game few-objects-13) (:domain few-objects-room-v1)
(:setup
; no real setup for 13
)
(:constraints (and
    (preference onChairFromWallToWall
        (exists (?c - chair ?w1 ?w2 - wall)
            (then
                (once (adjacent agent ?w1))
                (hold (on ?c agent))
                (once (and (adjacent agent ?w2) (opposite ?w1 ?w2)))
            )
        )
    )
))
(:scoring minimize (+
    (* 1 (count-shortest onChairFromWallToWall))
)))

(define (game few-objects-14) (:domain few-objects-room-v1)
(:setup
    ; is there a prettier way to do this?
    (exists (?f - floor ?r - rug
            ?w1 ?w2 - window ?c - chair ?x1 ?x2 - wall
            ?b1 ?b2 ?b3 ?b4 ?b5 ?b6 - cube_block)
        (game-conserved (and
            (not (on ?f ?r))
            (adjacent bed ?w1)
            (adjacent desk ?w2)
            (adjacent ?c ?x1)
            (opposite ?x1 ?x2)
            (=
                (distance ?x1 ?b1)
                (distance ?b1 ?b2)
                (distance ?b3 ?b4)
                (distance ?b4 ?b5)
                (distance ?b5 ?b6)
                (distance ?b6 ?x2)
            )
        ))
    )
)
(:constraints (and
    (preference onChairFromWallToBlock
        (exists (?c - chair ?w - wall ?b - cube_block)
            (then
                (once (adjacent agent ?w))
                (hold (on ?c agent))
                (once (adjacent agent ?b))
            )
        )
    )
))
(:scoring maximize (count-once-per-objects onChairFromWallToBlock)
))


(define (game few-objects-15) (:domain few-objects-room-v1)
(:setup
; no real setup for 15
)
(:constraints (and
    (preference throwToWallAndBack
        (exists (?d - dodgeball ?w - wall)
            (then
                (once (agent_holds ?d))
                (hold-while
                    (and (not (agent_holds ?d)) (in_motion ?d))
                    (touch ?d ?w)
                )
                (once-measure (agent_holds ?d) (distance agent ?w))
            )
        )
    )
))
(:scoring maximize (count-increasing-measure throwToWallAndBack)
))


(define (game few-objects-16) (:domain few-objects-room-v1)
(:setup
    (forall (?b - cube_block ?d - desk)
        (and
            (game-conserved (> (distance ?b ?d) 3))
            (not (exists (?b2 - cube_block)
                (game-optional (and
                    (not (= ?b ?b2))
                    (< (distance ?b ?b2) 0.5)
                ))
            ))
        )
    )
)
(:constraints (and
    (preference throwHitsBlock
        (exists (?d - dodgeball ?b - cube_block)
            (then
                (once (agent_holds ?d))
                (hold (and (not (agent_holds ?d)) (in_motion ?d)))
                (once (touch ?d ?b))
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
    (>= (count-nonoverlapping throwAttempt) 6)
)
(:scoring maximize
    (* 10 (count-once-per-objects throwHitsBlock))
))


(define (game few-objects-17) (:domain few-objects-room-v1)
(:setup
    (exists (?c - curved_wooden_ramp ?r - rug) (game-conserved (adjacent ?c ?r)))
)
(:constraints (and
    (preference ballLandsOnRed
        (exists (?d - dodgeball ?c - curved_wooden_ramp ?r - rug ?t - desktop)
            (then
                (once (and (agent_holds ?d) (< (distance agent ?t) 0.5)))
                (hold-while
                    (and (not (agent_holds ?d)) (in_motion ?d))
                    (on ?c ?d)
                )
                (once (and (on ?r ?d) (not (in_motion ?d)) (rug_color_under ?d red)))
            )
        )
    )
    (preference blueBallLandsOnPink
        (exists (?b - blue_dodgeball ?c - curved_wooden_ramp ?r - rug ?t - desktop)
            (then
                (once (and (agent_holds ?b) (< (distance agent ?t) 0.5)))
                (hold-while
                    (and (not (agent_holds ?b)) (in_motion ?b))
                    (on ?c ?b)
                )
                (once (and (on ?r ?b) (not (in_motion ?b)) (rug_color_under ?b pink)))
            )
        )
    )
    (preference pinkBallLandsOnPink
        (exists (?p - pink_dodgeball ?c - curved_wooden_ramp ?r - rug ?t - desktop)
            (then
                (once (and (agent_holds ?p) (< (distance agent ?t) 0.5)))
                (hold-while
                    (and (not (agent_holds ?p)) (in_motion ?p))
                    (on ?c ?p)
                )
                (once (and (on ?r ?p) (not (in_motion ?p)) (rug_color_under ?p pink)))
            )
        )
    )
    (preference ballLandsOnOrangeOrGreen
        (exists (?d - dodgeball ?c - curved_wooden_ramp ?r - rug ?t - desktop)
            (then
                (once (and (agent_holds ?d) (< (distance agent ?t) 0.5)))
                (hold-while
                    (and (not (agent_holds ?d)) (in_motion ?d))
                    (on ?c ?d)
                )
                (once (and (on ?r ?d) (not (in_motion ?d)) (or (rug_color_under ?d green) (rug_color_under ?d orange))))
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
    (> (count-nonoverlapping throwAttempt) (+ (/ (total-score) 30) 1))
)
(:scoring maximize (+
    (* 50 (count-nonoverlapping ballLandsOnRed))
    (* 10 (count-nonoverlapping blueBallLandsOnPink))
    (* 15 (count-nonoverlapping pinkBallLandsOnPink))
    (* 15 (count-nonoverlapping ballLandsOnOrangeOrGreen))
)))


(define (game few-objects-18) (:domain few-objects-room-v1)
(:setup
)
(:constraints (and
    (preference objectLandsOnRotatingChair
        (exists (?o - game_object ?c - chair)
            (then
                (once (and (agent_holds ?o) (is_rotating ?c)))
                (hold (and (not (agent_holds ?o)) (in_motion ?o) (is_rotating ?c)))
                (once (and
                    (is_rotating ?c)
                    (not (in_motion ?o))
                    (or
                        (on ?c ?o)
                        (exists (?o2 - game_object) (and
                            (not (= ?o ?o2))
                            (in_building ?o2)
                            (on ?o2 ?o)
                        ))
                    )
                ))
            )
        )
    )
    (preference chairStoppedRotating
        (exists (?c - chair)
            (then
                (once (not (is_rotating ?c)))
                (hold (is_rotating ?c))
                (once (not (is_rotating ?c)))
            )
        )
    )
))
(:terminal
    (> (count-once chairStoppedRotating) 0)
)
(:scoring maximize (* 10 (count-once-per-objects objectLandsOnRotatingChair))
))


(define (game few-objects-19) (:domain few-objects-room-v1)
(:setup
    (exists (?h - hexagonal_bin ?l - lamp) (game-conserved (on ?h ?l)))
)
(:constraints (and
    (preference ballHitsLamp
        (exists (?d - dodgeball ?l - lamp ?h - hexagonal_bin)
            (then
                (once (and (agent_holds ?d) (> (distance agent ?h) 10)))
                (hold (and (not (agent_holds ?d)) (in_motion ?d)))
                (once (and (on ?h ?l) (touch ?d ?l) (not (in_motion ?d))))
            )
        )
    )
))
(:scoring maximize (* 10 (count-nonoverlapping ballHitsLamp))
))

(define (game few-objects-20) (:domain few-objects-room-v1)
(:setup
    ; is there a prettier way to do this?
    (and
        (exists (?b1 ?b2 ?b3 ?b4 ?b5 ?b6 - cube_block)
            (game-optional (and
                (on bed ?b1)
                (on bed ?b2)
                (adjacent ?b1 ?b2)
                (or
                    (on ?b1 ?b3)
                    (on ?b2 ?b3)
                )
                (on bed ?b4)
                (on bed ?b5)
                (adjacent ?b4 ?b5)
                (or
                    (on ?b4 ?b6)
                    (on ?b5 ?b6)
                )
            ))
        )
        (forall (?d - dodgeball) (game-optional (on desk ?d)))
    )
)
(:constraints (and
    (preference bothBallsThrownFromDesk
        (then
            (forall-sequence (?d - dodgeball)
                (then
                    (once (and (agent_holds ?d) (adjancet agent desk)))
                    (hold (and (not (agent_holds ?d)) (in_motion ?d) (adjacent agent desk)))
                    (once (not (in_motion ?d)))
                    (hold (adjacent agent desk)) ; unti the second throw
                )
            )
            (once (forall (?b - cube_block) (not (in_motion ?b))))
        )
    )
    (forall (?b - cube_block)
        (preference allBlocksHit (exists (?d - dodgeball)
            (then
                (once (and (agent_holds ?d) (adjancet agent desk)))
                (hold-while
                    (and (not (touch agent ?b) (in_motion ?d)))
                    (touch ?b ?d)
                    (in_motion ?b)
                    (not (in_motion ?b))
                )
                (once (not (in_motion ?d)))
            )
        ))
    )
    (preference throwAttempt (exists (?d - dodgeball )
        (then
            (once (and (agent_hholds ?d)))
        )
    ))
))
(:terminal (or
    (> (count-once bothBallsThrownFromDesk) 0)
    (> (count-once allBlocksHit) 0)
))
(:scoring maximize (+
    (* 2 (count-once allBlocksHit:yellow_cube_block))
    (* 2 (count-once allBlocksHit:blue_cube_block))
    (* 2 (count-once allBlocksHit:brown_cube_block))
    (* 1 (count-once allBlocksHit)) ; the 7th point if all blocks are knocked over
    (* 3 (count-once allBlocksHit) (= (count-nonoverlapping throwAttempt) 1)) ; points 8-10 for doing it all in one throw
)))


(define (game few-objects-21) (:domain few-objects-room-v1)
(:setup
    ; is there a prettier way to do this?
    (and
        (exists (?b1 ?b2 ?b3 ?b4 ?b5 ?b6 - cube_block ?c - chair ?f - floor)
            (game-optional (and
                (on ?f ?b1)
                (on ?b1 ?b2)
                (on ?f ?b3)
                (on ?b3 ?b4)
                (on ?f ?b5)
                (on ?b5 ?b6)
                (= (distance ?b1 ?b3) (distance ?b1 ?b5))
                (= (distance ?b1 ?b3) (distance ?b3 ?b5))
                (< (distance ?c ?b1) (distance ?c ?b3))
                (= (distance ?c ?b3) (distance ?c ?b5))
            ))
        )
    )
)
(:constraints (and
    (preference blockKnockedFromBlock
        (exists (?d - dodgeball ?b1 ?b2 - cube_block ?c - chair ?f - floor)
            (then
                (once (and (agent_holds ?d) (on ?c agent) (on ?f ?b1) (on ?b1 ?b2)))
                (hold (and (not (agent_holds ?d)) (in_motion ?d) (not (touch agent ?b1)) (not (touch agent ?b2))))
                (once (and (on ?f ?b1) (on ?f ?b2)))
            )
        )
    )
))
(:scoring maximize (count-once-per-objects blockKnockedFromBlock))
)

; 22 is invalid

; 23 is a little inconsistent, but should work
(define (game few-objects-23) (:domain few-objects-room-v1)
(:setup
    (and
        (forall (?b - cube_block) (exists (?b1 ?b2 - cube_block)
            (game-conserved (and
                (on floor ?b)
                (not (= ?b ?b1))
                (not (= ?b ?b2))
                (touch ?b ?b1)
                (touch ?b ?b2)
            ))
        ))
    )
)
(:constraints (and
    (forall (?o - (either pillow cd dodgeball))
        (preference objectLandsInBlocks
            (exists (?b1 ?b2 - cube_block)
                (then
                    (once (and (agent_holds ?o) (forall (?b - cube_block) (> (distance agent ?b) 2))))
                    (hold (and (not (agent_holds ?o)) (in_motion ?o)))
                    (once (and
                        (not (in_motion ?o))
                        (on floor ?o)
                        (between ?b1 ?o ?b2)
                    ))
                )
            )
        )
    )
    (preference objectBouncesOut
        (exists (?p - (either pillow cd dodgeball) ?f - floor)
            (then
                (once (and (agent_holds ?p) (forall (?b - cube_block) (> (distance agent ?b) 2))))
                (hold-while
                    (and (not (agent_holds ?p)) (in_motion ?p))
                    (exists (?b1 ?b2 - cube_block) (between ?b1 ?p ?b2))
                    (and (touch ?f ?p) (not (exists (?b1 ?b2 - cube_block) (and (on ?f ?b1) (on ?f ?b2) (between ?b1 ?p ?b2 )))))
                )
            )
        )
    )
))
(:scoring maximize (+
    (* 5 (count-once-per-objects objectLandsInBlocks:pillow))
    (* 10 (count-once-per-objects objectLandsInBlocks:cd))
    (* 20 (count-once-per-objects objectLandsInBlocks:dodgeball))
    (* (- 5) (count-once-per-objects objectBouncesOut))
)))


(define (game few-objects-24) (:domain few-objects-room-v1)
(:setup
    (forall (?b - cube_block)
        (not (exists (?b2 - cube_block)
            (game-optional (and
                (not (= ?b ?b2))
                (< (distance ?b ?b2) 1)
            ))
        ))
    )
)
(:constraints (and
    (preference throwHitsBlock
        (exists (?d - dodgeball ?b - cube_block)
            (then
                (once (agent_holds ?d))
                (hold (and (not (agent_holds ?d)) (in_motion ?d)))
                (once (touch ?d ?b))
            )
        )
    )
    (preference throwHitsBlockAndFlips
        (exists (?d - dodgeball ?b - cube_block)
            (then
                (once (agent_holds ?d))
                (hold-while
                    (not (agent_holds ?d) (not (agent_holds ?b)))
                    (touch ?d ?b)
                    (not (object_orientation ?b face))
                )
            )
        )
    )
))
(:scoring maximize (+
    (* 5 (count-once-per-objects throwHitsBlock))
    (* 10 (count-once-per-objects throwHitsBlockAndFlips))
    ; Without accounting for the "if the cube blocks moves a long distance" bit
)))


; this ignores the "spin X times" bit
(define (game few-objects-25) (:domain few-objects-room-v1)
(:setup
    (exists (?b1 ?b2 ?b3 ?b4 ?b5 ?b6 - cube_block ?c - chair ?h - hexagonal_bin ?b - bed)
        (game-optional (and
            (adjacent ?b1 ?b2)
            (adjacent ?b1 ?c)
            (adjacent ?b2 ?c)
            (adjacent ?b3 ?b4)
            (adjacent ?b3 ?h)
            (adjacent ?b4 ?h)
            (adjacent ?b5 ?b6)
            (on ?b5 ?b)
            (on ?b6 ?b)
        ))
    )
)
(:constraints (and
    (preference blocksFromChairToShelf
        (exists (?c - chair ?b1 ?b2 - cube_block ?s - shelf)
            (then
                (once (adjcent agent ?s))
                (hold (and (agent_perspective eyes_closed) (adjacent ?b1 ?c) (adjacent ?b2 ?c)))
                (hold (and (agent_perspective eyes_closed) (agent_holds ?b1)))
                (hold (and (agent_perspective eyes_closed) (agent_holds ?b1) (agent_holds ?b2)))
                (hold (and (agent_perspective eyes_closed)
                    (or
                        (and (agent_holds ?b1) (on ?s ?b2) )
                        (and (on ?s ?b1) (agent_holds ?b2))
                    )
                ))
                (once (and (on ?s ?b1) (on ?s ?b2)))
            )
        )
    )
    (preference blocksFromBinToShelf
        (exists (?h - hexagonal_bin ?b1 ?b2 - cube_block ?s - shelf)
            (then
                (once (adjcent agent ?s))
                (hold (and (agent_perspective eyes_closed) (adjacent ?b1 ?h) (adjacent ?b2 ?h)))
                (hold (and (agent_perspective eyes_closed) (agent_holds ?b1)))
                (hold (and (agent_perspective eyes_closed) (agent_holds ?b1) (agent_holds ?b2)))
                (hold (and (agent_perspective eyes_closed)
                    (or
                        (and (agent_holds ?b1) (on ?s ?b2) )
                        (and (on ?s ?b1) (agent_holds ?b2))
                    )
                ))
                (once (and (on ?s ?b1) (on ?s ?b2)))
            )
        )
    )
    (preference blocksFromBedToShelf
        (exists (?b1 ?b2 - cube_block ?s - shelf ?b - bed)
            (then
                (once (adjcent agent ?s))
                (hold (and (agent_perspective eyes_closed) (on ?b ?b1) (on ?b ?b2)))
                (hold (and (agent_perspective eyes_closed) (agent_holds ?b1)))
                (hold (and (agent_perspective eyes_closed) (agent_holds ?b1) (agent_holds ?b2)))
                (hold (and (agent_perspective eyes_closed)
                    (or
                        (and (agent_holds ?b1) (on ?s ?b2) )
                        (and (on ?s ?b1) (agent_holds ?b2))
                    )
                ))
                (once (and (on ?s ?b1) (on ?s ?b2)))
            )
        )
    )
))
(:scoring maximize (+
    (* 20 (count-once blocksFromChairToShelf))
    (* 10 (count-once blocksFromBinToShelf))
    (* 5 (count-once blocksFromBedToShelf))
)))


(define (game few-objects-26) (:domain few-objects-room-v1)
(:setup
    (exists (?b1 ?b2 ?b3 ?b4 ?b5 ?b6 - cube_block ?c - chair ?h - hexagonal_bin)
        (and
            (game-conserved (and
                (on bed ?h)
                (adjacent ?c desk)
            ))
            (game-optional (and
                (adjacent ?b1 bed)
                (on ?b1 ?b2)
                (adjacent ?b3 ?b1)
                (on ?b3 ?b4)
                (adjacent ?b5 ?b3)
                (on ?b5 ?b6)
                (between ?b1 ?b3 ?b5)
            ))
        )
    )
)
(:constraints (and
    (preference throwHitsBlock
        (exists (?d - dodgeball ?b - cube_block ?c - chair)
            (then
                (once (and (on ?c agent) (agent_holds ?d) (is_rotating ?c)))
                (hold (and (not (agent_holds ?d)) (in_motion ?d)))
                (hold (touch ?d ?b))
                (once (in_motion ?b))
            )
        )
    )
    (preference throwInBin
        (exists (?d - dodgeball ?h - hexagonal_bin ?c - chair)
            (then
                (once (and (on ?c agent) (agent_holds ?d) (is_rotating ?c)))
                (hold (and (not (agent_holds ?d)) (in_motion ?d)))
                (once (and (in ?h ?d) (not (in_motion ?d))))
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
    (>= (count-once-per-objects throwAttempt) 2)
)
(:scoring maximize (+
    (* 1 (count-once-per-objects throwHitsBlock))
    (* 5 (count-once-per-objects throwInBin))
)))

(define (game few-objects-27) (:domain few-objects-room-v1)
(:setup
)
(:constraints (and
    ; Two valid ways of writing this -- one where I define all of the requirements
    ; in each preference, and another where I define it in the scoring.
    ; I'll actually try doing it in the scoring here, we'll see how well that works
    (preference bookOnChairs
        (exists (?c1 ?c2 - chair ?t - textbook)
            (at-end
                (and
                    (on ?c1 ?t)
                    (on ?c2 ?t)
                )
            )
        )
    )
    (preference firstLayerOfBlocks
        (exists (?t - textbook ?b1 ?b2 ?b3 - cube_block)
            (at-end
                (and
                    (on ?t ?b1)
                    (on ?t ?b2)
                    (on ?t ?b3)
                )
            )
        )
    )
    (preference secondLayerOfBlocks
        (exists (?b1 ?b2 ?b3 ?b4 ?b5 - cube_block)
            (at-end
                (and
                    (on ?b1 ?b4)
                    (on ?b2 ?b4)
                    (on ?b2 ?b5)
                    (on ?b3 ?b5)
                )
            )
        )
    )
    (preference thirdLayerOfBlocks
        (exists (?b1 ?b2 ?b3 ?b4 ?b5 ?b6 - cube_block)
            (at-end
                (and
                    (on ?b1 ?b4)
                    (on ?b2 ?b4)
                    (on ?b2 ?b5)
                    (on ?b3 ?b5)
                    (on ?b4 ?b6)
                    (on ?b5 ?b6)
                )
            )
        )
    )
    (preference mugOnTopOfPyrmaid
        (exists (?b1 ?b2 ?b3 ?b4 ?b5 ?b6 - cube_block ?m - mug)
            (at-end
                (and
                    (on ?b1 ?b4)
                    (on ?b2 ?b4)
                    (on ?b2 ?b5)
                    (on ?b3 ?b5)
                    (on ?b4 ?b6)
                    (on ?b5 ?b6)
                    (on ?b6 ?m)
                )
            )
        )
    )
    (preference dodgeballOnMug
        (exists (?m - mug ?d - dodgeball)
            (at-end
                (and
                    (on ?m ?d)
                )
            )
        )
    )
))
(:terminal
    (>= (total-time) 60)
)
(:scoring maximize (+
    (count-once bookOnChairs)
    (* (count-once bookOnChairs) (count-once firstLayerOfBlocks))
    (* (count-once bookOnChairs) (count-once firstLayerOfBlocks) (count-once secondLayerOfBlocks))
    (*
        (count-once bookOnChairs) (count-once firstLayerOfBlocks)
        (count-once secondLayerOfBlocks) (count-once thirdLayerOfBlocks)
    )
    (*
        (count-once bookOnChairs) (count-once firstLayerOfBlocks)
        (count-once secondLayerOfBlocks) (count-once thirdLayerOfBlocks)
        (count-once thirdLayerOfBlocks)
    )
    (*
        (count-once bookOnChairs) (count-once firstLayerOfBlocks)
        (count-once secondLayerOfBlocks) (count-once thirdLayerOfBlocks)
        (count-once thirdLayerOfBlocks) (count-once mugOnTopOfPyrmaid)
    )
    (*
        (count-once bookOnChairs) (count-once firstLayerOfBlocks)
        (count-once secondLayerOfBlocks) (count-once thirdLayerOfBlocks)
        (count-once thirdLayerOfBlocks) (count-once mugOnTopOfPyrmaid)
        (count-once dodgeballOnMug)
    )
)))


(define (game few-objects-28) (:domain few-objects-room-v1)
(:setup
    (and
        (forall (?b - cube_block)
            (or
                (game-optional (on bed ?b))
                (exists (?c - cube_block) (game-optional (and (not (= ?b ?c)) (on ?c ?b))))
            )
        )
        (exists (?d - dodgeball) (game-optional (on bed ?d)))
    )
)
(:constraints (and
    (preference allBlocksThrownToBinAndBallToChair
        (exists (?d - dodgeball ?h - hexagonal_bin ?c - chair)
            (then
                (once (on bed agent))
                (forall-sequence (?b - cube_block)
                    (then
                        (once (and (on bed agent) (agent_holds ?b)))
                        (hold (and (on bed agent) (not (agent_holds ?b)) (in_motion ?b)))
                        (once (and (on bed agent) (in ?h ?b) (not (in_motion ?b))))
                        (hold (on bed agent))  ; until picking up the next cube
                    )
                )
                (once (and (on bed agent) (agent_holds ?d)))
                (hold-while
                    (and (on bed agent) (not (agent_holds ?b)))
                    (touch ?c ?d)
                    (not (object_orientation ?c upright))
                )
            )
        )
    )
))
(:scoring maximize (+
    (* 0.5 (count-nonoverlapping allBlocksThrownToBinAndBallToChair))
)))


;29 is invalid


;30 is invalid, unless someone knows what "marabbalism" is ?


(define (game few-objects-31) (:domain few-objects-room-v1)
(:setup

)
(:constraints (and
    (preference blockThrownToGround
        (exists (?b - cube_block ?f - floor)
            (then
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (hold-to-end (and (on ?f ?b) (not (agent_holds ?b)) ))
            )
        )
    )
    (preference blockThrownToBlock
        (exists (?b1 ?b2 - cube_block)
            (then
                (once (agent_holds ?b1))
                (hold (and (not (agent_holds ?b1)) (in_motion ?b1)))
                (hold-to-end (and (on ?b1 ?b2) (not (agent_holds ?b1)) (not (agent_holds ?b2)) ))
            )
        )
    )
))
(:scoring maximize (+
    (* (count-once-per-objects blockThrownToGround)
        (+ 1 (count-once-per-objects blockThrownToBlock))
    )
)))

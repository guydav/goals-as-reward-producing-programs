
(define (game game-0) (:domain medium-objects-room-v1)  ; 0
; SETUP: place the triangular ramp near the bin for the entire game
(:setup (and
    (exists (?h - hexagonal_bin ?r - triangular_ramp)
        (game-conserved (< (distance ?h ?r) 1))
    )
))
(:constraints (and
    ; PREFERENCE: count throws of any ball on the ramp and then to the bin
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
    ; PREFERENCE: count how many times the bin is knocked over
    (preference binKnockedOver
        (exists (?h - hexagonal_bin)
            (then
                (hold (and (not (touch agent ?h)) (not (agent_holds ?h))))
                (once (not (object_orientation ?h upright)))
            )
        )
    )
))
; TERMINAL: the game ends when the bin is knocked over
(:terminal (>= (count-once binKnockedOver) 1)
)
; SCORING: one point for each throw from the ramp to the bin
(:scoring (count throwToRampToBin)
))

; 1 is invalid

(define (game game-2) (:domain many-objects-room-v1)  ; 2
; SETUP: keep the top drawer open for the entire game
(:setup (and
    (game-conserved (open top_drawer))
))
(:constraints (and
    (forall (?b - (either dodgeball golfball) ?t - (either top_drawer hexagonal_bin))
        ; PREFERENCE: count how many throws of either the dodgeball or golfball land in the drawer or bin
        (preference throwToDrawerOrBin
            (then
                (once (and (agent_holds ?b) (adjacent agent door)))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (not (in_motion ?b)) (in ?t ?b)))
            )
        )
    )
    ; PREFERENCE: count any throw attempt
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
; TERMINAL: the game ends when the agent has thrown each ball once
(:terminal (>= (count-once-per-objects throwAttempt) 6)
)
; SCORING: one point for each throw of the golfball to the bin, two points for each throw of the dodgeball to the bin, three points for each throw of the golfball to the drawer, and -1 point for each missed throw
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
        ; PREFERENCE: count in block in the building at the end of the game
        (preference blockInTowerAtEnd (exists (?l - block)
            (at-end (in ?b ?l))
        ))
    )
))
; SCORING: for the tower with the most blocks, one point for each block
(:scoring (external-forall-maximize
    (count-once-per-objects blockInTowerAtEnd)
)))

; 4 is invalid -- woefully underconstrained

(define (game game-5) (:domain few-objects-room-v1)  ; 5
(:constraints (and
    ; PREFERENCE: count throws of a dodgeball to the bin from one distance unit away
    (preference throwBallToBin
        (exists (?d - dodgeball ?h - hexagonal_bin)
            (then
                (once (and (agent_holds ?d) (= (distance agent ?h) 1)))
                (hold (and (not (agent_holds ?d)) (in_motion ?d)))
                (once (and (not (in_motion ?d)) (in ?h ?d)))
            )
        )
    )
))
; SCORING: one point for each throw from one distance unit away
(:scoring (count throwBallToBin)
))

(define (game game-6) (:domain medium-objects-room-v1)  ; 6
; SETUP: place a hexagonal bin next to the bed for the entire game. place all teddy bears or pillows off the bed for the entire game.
(:setup (and
    (exists (?h - hexagonal_bin) (game-conserved (adjacent ?h bed)))
    (forall (?x - (either teddy_bear pillow)) (game-conserved (not (on bed ?x))))
))
(:constraints (and
    (forall (?b - ball)
        ; PREFERENCE: count throws of any of the balls to the bin from the desk
        (preference throwBallToBin
            (exists (?h - hexagonal_bin)
                (then
                    (once (and (agent_holds ?b) (adjacent agent desk)))
                    (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                    (once (and (not (in_motion ?b)) (in ?h ?b)))
                )
            )
        )
    )
    ; PREFERENCE: count throws of any of the balls to the bin from the desk that miss
    (preference failedThrowToBin
        (exists (?b - ball ?h - hexagonal_bin)
            (then
                (once (and (agent_holds ?b) (adjacent agent desk)))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (not (in_motion ?b)) (not (in ?h ?b))))
            )
        )
    )
))
; SCORING: 10 points for each throw of the dodgeball to the bin, 20 points for each throw of the basketball to the bin, 30 points for each throw of the beachball to the bin, and -1 point for each missed throw
(:scoring (+
    (* 10 (count throwBallToBin:dodgeball))
    (* 20 (count throwBallToBin:basketball))
    (* 30 (count throwBallToBin:beachball))
    (- (count failedThrowToBin))
)))

; 7 is invalid -- vastly under-constrained -- I could probably make some guesses but leaving alone

(define (game game-8) (:domain few-objects-room-v1)  ; 8
; SETUP: place a curved wooden ramp on the floor for the entire game
(:setup (and
    (exists (?c - curved_wooden_ramp)
        (game-conserved (on floor ?c))
    )
))
(:constraints (and
    ; PREFERENCE: count throws of a dodgeball that pass over the curved wooden ramp and land on its other side
    (preference throwOverRamp  ; TODO: does this quanitfy over reasonably?
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
    ; PREFERENCE: count any throws of any of the balls
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
; TERMINAL: the game ends when the participant has thrown a ball over the curved wooden ramp at least once
(:terminal (>= (count-once throwOverRamp) 1)
)
; SCORING: 3 points for throwing a ball over the curved wooden ramp in one attempt, 2 points for throwing a ball over the curved wooden ramp in two attempts, and 1 point for throwing a ball over the curved wooden ramp in three or more attempts
(:scoring (+
    (* 3 (= (count throwAttempt) 1) (count-once throwOverRamp))
    (* 2 (= (count throwAttempt) 2) (count-once throwOverRamp))
    (* (>= (count throwAttempt) 3) (count-once throwOverRamp))
)))

; Taking the first game this participant provided
(define (game game-9) (:domain many-objects-room-v1)  ; 9
; SETUP: place a hexagonal bin either on the bed or next to a wall for the entire game
(:setup (and
    (exists (?h - hexagonal_bin)
        (game-conserved (or
            (on bed ?h)
            (exists (?w - wall) (adjacent ?w ?h))
        ))
    )
))
(:constraints (and
    ; PREFERENCE: count throws of a dodgeball that land in a hexagonal bin that is either on the bed, or next to a wall where the agent throws from the opposite wall
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
    ; PREFERENCE: count throws of a dodgeball that bounces on the floor and lands in a hexagonal bin that is either on the bed, or next to a wall where the agent throws from the opposite wall
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
; SCORING: 1 point for each throw that bounces on the floor and lands in the hexagonal bin, 3 points for each throw that lands in the hexagonal bin without bouncing on the floor
(:scoring (+
    (count bounceBallToBin)
    (* 3 (count throwBallToBin))
)))

(define (game game-10) (:domain medium-objects-room-v1)  ; 10

(:constraints (and
    ; PREFERENCE: count throws of a teddy bear that land on a pillow
    (preference throwTeddyOntoPillow
        (exists (?t - teddy_bear ?p - pillow)
            (then
                (once (agent_holds ?t))
                (hold (and (not (agent_holds ?t)) (in_motion ?t)))
                (once (and (not (in_motion ?t)) (on ?p ?t)))
            )
        )
    )
    ; PREFERENCE: count any throws of any of the teddy bears
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
; TERMINAL: the game ends when the participant has thrown a teddy bear at least 10 times
(:terminal
    (>= (count throwAttempt) 10)
)
; SCORING: 1 point for each throw that lands on a pillow
(:scoring (count throwTeddyOntoPillow)
))

(define (game game-11) (:domain many-objects-room-v1)  ; 11
(:constraints (and
    (forall (?b - building) (and
        ; PREFERENCE: for each building, count that blocks that are in the building and touch the floor
        (preference baseBlockInTowerAtEnd (exists (?l - block)
            (at-end (and
                (in ?b ?l)
                (on floor ?l)
            ))
        ))
        ; PREFERENCE: for each building, count how many blocks are in the building, and do not touch the floor or any non-block object
        (preference blockOnBlockInTowerAtEnd (exists (?l - block)
            (at-end
                (and
                    (in ?b ?l)
                    (not (exists (?o - game_object) (and (not (same_type ?o block)) (touch ?o ?l))))
                    (not (on floor ?l))
                )
            )
        ))
        ; PREFERENCE: for each building, count if there is a pyramid block in the building that is on top of the tower and does not touch any other object
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
; SCORING: for the single building with the highest score, count how many blocks are in the building, either on the floor or on another block
(:scoring (external-forall-maximize (*
    (count-once pyramidBlockAtopTowerAtEnd)
    (count-once baseBlockInTowerAtEnd)
    (+
        (count-once baseBlockInTowerAtEnd)
        (count-once-per-objects blockOnBlockInTowerAtEnd)
    )
))))

; 12 requires quantifying based on position -- something like

(define (game game-12) (:domain medium-objects-room-v1)  ; 12
; SETUP: place a hexagonal bin close to the center of the room for the entire game
(:setup (and
    (exists (?h - hexagonal_bin)
        (game-conserved (< (distance ?h room_center) 1))
    )
))
(:constraints (and
    ; PREFERENCE: count throws of a dodgeball when the agent is crouching next to the door that touch a triangular ramp before landing in the hexagonal bin
    (preference throwToRampToBin
        (exists (?r - triangular_ramp ?d - dodgeball ?h - hexagonal_bin)
            (then
                (once (and (agent_holds ?d) (adjacent agent door) (agent_crouches))) ; ball starts in hand
                (hold-while
                    (and (not (agent_holds ?d)) (in_motion ?d))
                    (touch ?r ?d)
                )
                (once  (and (in ?h ?d) (not (in_motion ?d)))) ; touches wall before in bin
            )
        )
    )
))
; SCORING: one point for each succesful throw for each unique position of the triangular ramp
(:scoring
    (count-unique-positions throwToRampToBin)
))


(define (game game-13) (:domain many-objects-room-v1)  ; 13
; SETUP: place a triangular ramp close to the center of the room, and place a hexagonal bin close to the ramp, for the entire game
(:setup (and
    (exists (?h - hexagonal_bin ?r - triangular_ramp) (game-conserved
        (and
            (< (distance ?h ?r) 1)
            (< (distance ?r room_center) 0.5)
        )
    ))
))
(:constraints (and
    (forall (?d - (either dodgeball golfball))
        ; PREFERENCE: count throws of a dodgeball when the agent is crouching next to the door that touch a triangular ramp before landing in the hexagonal bin
        (preference throwToRampToBin
            (exists (?r - triangular_ramp ?h - hexagonal_bin)
                (then
                    (once (and (agent_holds ?d) (adjacent agent door) (agent_crouches))) ; ball starts in hand
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
; SCORING: 6 points for each succesful throw of a dodgeball, 3 points for each succesful throw of a golfball
(:scoring (+
    (* 6 (count throwToRampToBin:dodgeball))
    (* 3 (count throwToRampToBin:golfball))
)))

(define (game game-14) (:domain medium-objects-room-v1)  ; 14
(:constraints (and
    ; PREFERENCE: count throws of any ball when the agent is on the rug that land in a hexagonal bin
    (preference throwInBin
        (exists (?b - ball ?h - hexagonal_bin)
            (then
                (once (and (on rug agent) (agent_holds ?b)))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (not (in_motion ?b)) (in ?h ?b)))
            )
        )
    )
    ; PREFERENCE: count any throws of any ball
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
; TERMINAL: end the game after 10 throws
(:terminal
    (>= (count throwAttempt) 10)
)
; SCORING: 1 point for each throw that lands in the bin
(:scoring (count throwInBin)
    ; TODO: how do we want to quantify streaks? some number of one preference without another preference?
))

(define (game game-15) (:domain few-objects-room-v1)  ; 15
; SETUP: place a hexagonal bin upside down and next to the bed for the entire game. Place six cube blocks in a pyramid building on top of the bin: three blocks on the bin, two blocks on the original three, and one block on the middle two.
(:setup (and
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
))

(:constraints (and
    ; PREFERENCE: count throws of a dodgeball from next to a chair that knock a cube block from the building on the hexagonal bin.
    (preference blockInTowerKnockedByDodgeball (exists (?b - building ?c - cube_block
        ?d - dodgeball ?h - hexagonal_bin ?r - chair)
        (then
            (once (and
                (agent_holds ?d)
                (adjacent agent ?r)
                (on ?h ?b)
                (in ?b ?c)
            ))
            (hold-while (and (not (agent_holds ?d)) (in_motion ?d))
                (or
                    (touch ?c ?d)
                    (exists (?c2 - cube_block) (touch ?c2 ?c))
                )
                (in_motion ?c)
            )
            (once (not (in_motion ?c)))
        )
    ))
    ; PREFERENCE: count any throws of a dodgeball
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
; TERMINAL: end the game after 2 throws
(:terminal
    (>= (count-once-per-objects throwAttempt) 2)
)
; SCORING: 1 point for each block that is knocked from the building
(:scoring (count-once-per-objects blockInTowerKnockedByDodgeball)
))


(define (game game-16) (:domain few-objects-room-v1)  ; 16
; SETUP: place a curved wooden ramp with its back to the front of the hexagonal bin for the entire game. Place a two blocks one on top of the other on each side of the bin for the entire game.
(:setup (and
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
))
(:constraints (and
    ; PREFERENCE: count throws of a dodgeball that pass on a curved wooden ramp and land in a hexagonal bin
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
; SCORING: 1 point for each throw that lands in the bin
(:scoring (count rollBallToBin)
))



(define (game game-17) (:domain medium-objects-room-v1)  ; 17/18

(:constraints (and
    ; PREFERENCE: build a building that includes, in order, a bridge block, a flat block, a tall cylindrical block, a cube block, and a pyramid block
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
; SCORING: 10 points for building the castle
(:scoring (+
    (* 10 (count-once-per-objects castleBuilt))
    ; (* 4 (or
    ;     (with (?b - green_bridge_block ?f - yellow_flat_block ?t - yellow_tall_cylindrical_block) (count-once-per-objects castleBuilt))
    ;     (with (?b - green_bridge_block ?f - yellow_flat_block ?c - green_cube_block) (count-once-per-objects castleBuilt))
    ;     (with (?b - green_bridge_block ?f - yellow_flat_block ?p - orange_pyramid_block) (count-once-per-objects castleBuilt))
    ;     (with (?f - yellow_flat_block ?t - yellow_tall_cylindrical_block ?c - green_cube_block) (count-once-per-objects castleBuilt))
    ;     (with (?f - yellow_flat_block ?t - yellow_tall_cylindrical_block ?p - orange_pyramid_block) (count-once-per-objects castleBuilt))
    ;     (with (?t - yellow_tall_cylindrical_block ?c - green_cube_block ?p - orange_pyramid_block) (count-once-per-objects castleBuilt))
    ; ))
    ; (* 3 (or
    ;     (with (?b - green_bridge_block ?f - yellow_flat_block ?t - yellow_tall_cylindrical_block ?c - green_cube_block) (count-once-per-objects castleBuilt))
    ;     (with (?b - green_bridge_block ?f - yellow_flat_block ?t - yellow_tall_cylindrical_block ?p - orange_pyramid_block) (count-once-per-objects castleBuilt))
    ;     (with (?b - green_bridge_block ?t - yellow_tall_cylindrical_block ?c - green_cube_block ?p - orange_pyramid_block) (count-once-per-objects castleBuilt))
    ;     (with (?f - yellow_flat_block ?t - yellow_tall_cylindrical_block ?c - green_cube_block ?p - orange_pyramid_block) (count-once-per-objects castleBuilt))
    ; ))
    ; (* 3 (with (?b - green_bridge_block ?f - yellow_flat_block ?t - yellow_tall_cylindrical_block ?c - green_cube_block ?p - orange_pyramid_block) (count-once-per-objects castleBuilt)))
    ; (* 4 (or
    ;     (with (?b - brown_bridge_block ?f - gray_flat_block ?t - brown_tall_cylindrical_block) (count-once-per-objects castleBuilt))
    ;     (with (?b - brown_bridge_block ?f - gray_flat_block ?c - blue_cube_block) (count-once-per-objects castleBuilt))
    ;     (with (?b - brown_bridge_block ?f - gray_flat_block ?p - red_pyramid_block) (count-once-per-objects castleBuilt))
    ;     (with (?f - gray_flat_block ?t - brown_tall_cylindrical_block ?c - blue_cube_block) (count-once-per-objects castleBuilt))
    ;     (with (?f - gray_flat_block ?t - brown_tall_cylindrical_block ?p - red_pyramid_block) (count-once-per-objects castleBuilt))
    ;     (with (?t - brown_tall_cylindrical_block ?c - blue_cube_block ?p - red_pyramid_block) (count-once-per-objects castleBuilt))
    ; ))
    ; (* 3 (or
    ;     (with (?b - brown_bridge_block ?f - gray_flat_block ?t - brown_tall_cylindrical_block ?c - blue_cube_block) (count-once-per-objects castleBuilt))
    ;     (with (?b - brown_bridge_block ?f - gray_flat_block ?t - brown_tall_cylindrical_block ?p - red_pyramid_block) (count-once-per-objects castleBuilt))
    ;     (with (?b - brown_bridge_block ?t - brown_tall_cylindrical_block ?c - blue_cube_block ?p - red_pyramid_block) (count-once-per-objects castleBuilt))
    ;     (with (?f - gray_flat_block ?t - brown_tall_cylindrical_block ?c - blue_cube_block ?p - red_pyramid_block) (count-once-per-objects castleBuilt))
    ; ))
    ; (* 3 (with (?b - brown_bridge_block ?f - gray_flat_block ?t - brown_tall_cylindrical_block ?c - blue_cube_block ?p - red_pyramid_block) (count-once-per-objects castleBuilt)))
)))

(define (game game-19) (:domain medium-objects-room-v1)  ; 19
; SETUP: place all balls near the door to start the game.
(:setup (and
    (forall (?b - ball)
        (game-optional (< (distance ?b door) 1))
    )
))
(:constraints (and
    (forall (?b - ball ?t - (either doggie_bed hexagonal_bin))
        ; PREFERENCE: count throws of any ball into either the doggie bed or the hexagonal bin
        (preference ballThrownIntoTarget
            (then
                (once (and (agent_holds ?b) (< (distance agent door) 1)))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (in ?t ?b) (not (in_motion ?b))))
            )
        )
    )
    (forall (?b - ball)
        ; PREFERENCE: count throws of any ball onto the doggie bed
        (preference ballThrownOntoTarget
            (exists (?t - doggie_bed)
                (then
                    (once (and (agent_holds ?b) (< (distance agent door) 1)))
                    (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                    (once (and (on ?t ?b) (not (in_motion ?b))))
                )
            )
        )
    )
    ; PREFERENCE: count throws of any ball
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
; TERMINAL: the game ends afer the agent throws 3 different balls
(:terminal
    (>= (count-once-per-objects throwAttempt) 3)
)
; SCORING: 6 points for throwing the dodgeball into the hexagonal bin, 5 points for throwing the beachball into the hexagonal bin, 4 points for throwing the basketball into the hexagonal bin, 5 points for throwing the dodgeball into the doggie bed, 4 points for throwing the beachball into the doggie bed, 3 points for throwing the basketball into the doggie bed, 5 points for throwing the dodgeball onto the doggie bed, 4 points for throwing the beachball onto the doggie bed, 3 points for throwing the basketball onto the doggie bed
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
        ; PREFERENCE: for any building, count blocks that are in the building at the end of the game
        (preference blockInTowerAtEnd (exists (?l - block)
            (at-end
                (and
                    (in ?b ?l)
                )
            )
        ))
        ; PREFERENCE: for any building, count blocks that were in the building at some point until they were knocked out by a dodgeball
        (preference blockInTowerKnockedByDodgeball (exists (?l - block ?d - dodgeball)
            (then
                (once (and (in ?b ?l) (agent_holds ?d)))
                (hold (and (in ?b ?l) (not (agent_holds ?d)) (in_motion ?d)))
                (once (and (in ?b ?l) (touch ?d ?b)))
                (hold (in_motion ?l))
                (once (not (in_motion ?l)))
            )
        ))

    ))
    ; PREFERENCE: count how many times a building falls while the agent is building it
    (preference towerFallsWhileBuilding (exists (?b - building ?l1 ?l2 - block)
        (then
            (once (and (in ?b ?l1) (agent_holds ?l2)))
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
))
; SCORING: for the single building with the highest score, 1 point for each block in the building, and 2 points for each block knocked by a dodgeball. Subtract 1 point for each time any building falls while the agent is building it.
(:scoring (+
    (external-forall-maximize (+
        (count-once-per-objects blockInTowerAtEnd)
        (* 2 (count-once-per-objects blockInTowerKnockedByDodgeball))
    ))
    (- (count towerFallsWhileBuilding))
)))

(define (game game-21) (:domain few-objects-room-v1)  ; 21
; SETUP: place a chair near the center of the room, and face it away from both the bed and the desk for the entire game.
(:setup (and
    (exists (?c - chair) (game-conserved (and
        (< (distance ?c room_center) 1)
        (not (faces ?c desk))
        (not (faces ?c bed))
    )))
))
(:constraints (and
    ; PREFERENCE: count throws of a dodgeball into a hexagonal bin from the desk
    (preference ballThrownToBin
        (exists (?d - dodgeball ?h - hexagonal_bin)
            (then
                (once (and (agent_holds ?d) (adjacent agent desk)))
                (hold (and (not (agent_holds ?d)) (in_motion ?d)))
                (once (and (not (in_motion ?d)) (in ?h ?d)))
            )
        )
    )
    ; PREFERENCE: count throws of a dodgeball onto the bed from the desk
    (preference ballThrownToBed
        (exists (?d - dodgeball)
            (then
                (once (and (agent_holds ?d) (adjacent agent desk)))
                (hold (and (not (agent_holds ?d)) (in_motion ?d)))
                (once (and (not (in_motion ?d)) (on bed ?d)))
            )
        )
    )
    ; PREFERENCE: count throws of a dodgeball onto a chair from the desk
    (preference ballThrownToChair
        (exists (?d - dodgeball ?c - chair)
            (then
                (once (and (agent_holds ?d) (adjacent agent desk)))
                (hold (and (not (agent_holds ?d)) (in_motion ?d)))
                (once (and (not (in_motion ?d)) (on ?c ?d) (is_setup_object ?c)))
            )
        )
    )
    ; PREFERENCE: count throws of a dodgeball from the desk that land on neither the hexagonal bin, nor the bed, nor the chair from the setup
    (preference ballThrownMissesEverything
        (exists (?d - dodgeball)
            (then
                (once (and (agent_holds ?d) (adjacent agent desk)))
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
; TERMINAL: end the game when the agent has scored 10 points or more
(:terminal
    (>= (total-score) 10)
)
; SCORING: 5 points for each throw of a dodgeball into a hexagonal bin from the desk, 1 point for each throw of a dodgeball onto the bed from the desk, 1 point for each throw of a dodgeball that misses everything
(:scoring (+
    (* 5 (count ballThrownToBin))
    (count ballThrownToBed)
    (count ballThrownToChair)
    (- (count ballThrownMissesEverything))
)))

(define (game game-22) (:domain medium-objects-room-v1)  ; 22
; SETUP: place a hexagonal bin next to the bed for the entire game. To start the game, place all balls on the rug, remove all objects from the desk.
(:setup (and
    (exists (?h - hexagonal_bin) (game-conserved (adjacent bed ?h)))
    (forall (?b - ball) (game-optional (on rug ?b)))
    (game-optional (not (exists (?g - game_object) (on desk ?g))))
))
(:constraints (and
    (forall (?b - ball ?c - (either red yellow pink))
        ; PREFERENCE: count throws of any ball into a hexagonal bin from the rug when the agent stands on a red, yellow, or pink spot
        (preference throwBallToBin
            (exists (?h - hexagonal_bin)
                (then
                    (once (and (agent_holds ?b) (on rug agent) (rug_color_under agent ?c)))
                    (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                    (once (and (not (in_motion ?b)) (in ?h ?b)))
                )
            )
        )
    )
    ; PREFERENCE: count any throws of a dodgeball from the rug
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
; TERMINAL: end the game when the agent has thrown 8 throws of any ball
(:terminal
    (>= (count throwAttempt) 8)
)
; SCORING: 2 points for throwing a dodgeball from a red spot, 3 points for throwing a basketball from a red spot, 4 points for throwing a beachball from a red spot, 3 points for throwing a dodgeball from a pink spot, 4 points for throwing a basketball from a pink spot, 5 points for throwing a beachball from a pink spot, 4 points for throwing a dodgeball from a yellow spot, 5 points for throwing a basketball from a yellow spot, 6 points for throwing a beachball from a yellow spot
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
    ; PREFERENCE: count throws of a dodgeball into a hexagonal bin
    (preference throwBallToBin
        (exists (?d - dodgeball ?h - hexagonal_bin)
            (then
                (once (agent_holds ?d))
                (hold (and (not (agent_holds ?d)) (in_motion ?d)))
                (once (and (not (in_motion ?d)) (in ?h ?d)))
            )
        )
    )
    ; PREFERENCE: count any throws of a dodgeball
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
; SCORING: 1 points for each throw of a dodgeball into a hexagonal bin, -1/5 point for each throw
(:scoring (+
    (count throwBallToBin)
    (- (/ (count throwAttempt) 5))
)))



(define (game game-24) (:domain few-objects-room-v1)  ; 24
; SETUP: place a hexagonal bin on the chair for the entire game.
(:setup (and
    (exists (?c - chair ?h - hexagonal_bin) (game-conserved (on ?c ?h)))
))
(:constraints (and
    (forall (?d - dodgeball ?c - color)
        ; PREFERENCE: count throws of any dodgeball into a hexagonal bin from the rug when the agent stands on any color spot
        (preference throwBallToBin
            (exists (?h - hexagonal_bin)
                (then
                    (once (and (agent_holds ?d) (on rug agent) (rug_color_under agent ?c)))
                    (hold (and (not (agent_holds ?d)) (in_motion ?d)))
                    (once (and (not (in_motion ?d)) (in ?h ?d)))
                )
            )
        )
    )
))
; TERMINAL: the game ends when the agent scored 300 points or more
(:terminal
    (>= (total-score) 300)
)
; SCORING: 5 points for throwing the blue dodgeball from a red spot, 10 points for throwing the pink dodgeball from a red spot, 10 points for throwing the blue dodgeball from a pink spot, 20 points for throwing the pink dodgeball from a pink spot, 15 points for throwing the blue dodgeball from a orange spot, 30 points for throwing the pink dodgeball from a orange spot, 15 points for throwing the blue dodgeball from a green spot, 30 points for throwing the pink dodgeball from a green spot, 20 points for throwing the blue dodgeball from a purple spot, 40 points for throwing the pink dodgeball from a purple spot, 20 points for throwing the blue dodgeball from a blue spot, 40 points for throwing the pink dodgeball from a blue spot,
(:scoring (+
    (* 5 (count throwBallToBin:blue_dodgeball:red))
    (* 10 (count throwBallToBin:pink_dodgeball:red))
    (* 10 (count throwBallToBin:blue_dodgeball:pink))
    (* 20 (count throwBallToBin:pink_dodgeball:pink))
    (* 15 (count throwBallToBin:blue_dodgeball:orange))
    (* 30 (count throwBallToBin:pink_dodgeball:orange))
    (* 15 (count throwBallToBin:blue_dodgeball:green))
    (* 30 (count throwBallToBin:pink_dodgeball:green))
    (* 20 (count throwBallToBin:blue_dodgeball:purple))
    (* 40 (count throwBallToBin:pink_dodgeball:purple))
    (* 20 (count throwBallToBin:blue_dodgeball:yellow))
    (* 40 (count throwBallToBin:pink_dodgeball:yellow))
)))

; 25 and 26 are the same participant and are invalid -- hiding games

(define (game game-27) (:domain few-objects-room-v1)  ; 27
;SETUP: to start the game, remove all dodgeballs and cube blocks from all shelves, and toggle on the main light switch and the desktop.
(:setup (and
    (forall (?d - (either dodgeball cube_block)) (game-optional (not (exists (?s - shelf) (on ?s ?d)))))
    (game-optional (toggled_on main_light_switch))
    (game-optional (toggled_on desktop))
))
(:constraints (and
    ; PREFERENCE: count the number of dodgeballs placed in the hexagonal bin at the end of the game
    (preference dodgeballsInPlace
        (exists (?d - dodgeball ?h - hexagonal_bin)
            (at-end (in ?h ?d))
        )
    )
    ; PREFERENCE: count the number of cube blocks placed on a shelf on the west wall at the end of the game.
    (preference blocksInPlace
        (exists (?c - cube_block ?s - shelf)
            (at-end (and
                (adjacent ?s west_wall)
                (on ?s ?c)
            ))
        )
    )
    ; PREFERENCE: count the number of laptops or books placed on a shelf on a south wall at the end of the game
    (preference laptopAndBookInPlace
        (exists (?o - (either laptop book) ?s - shelf)
            (at-end (and
                (adjacent ?s south_wall)
                (on ?s ?o)
            ))
        )
    )
    ; PREFERENCE: count the number of cellphones or key chains placed in a drawer at the end of the game
    (preference smallItemsInPlace
        (exists (?o - (either cellphone key_chain) ?d - drawer)
            (at-end (and
                (in ?d ?o)
            ))
        )
    )
    ; PREFERENCE: count the number of items that are either the main light switch or the desktop or laptop that are toggled off at the end of the game
    (preference itemsTurnedOff
        (exists (?o - (either main_light_switch desktop laptop))
            (at-end (and
                (not (toggled_on ?o))
            ))
        )
    )
))
; SCORING: 5 points for each item placed in the correct location, 3 points for each item turned off
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
; SETUP: place all cube blocks on the rug for the entire game
(:setup (and
    (forall (?c - cube_block) (game-conserved (on rug ?c)))
))
(:constraints (and
    (forall (?c - color)
        ; PREFERENCE: count the number of dodgeball throws from off the rug that touch a cube block on a spot of the rug of a particular color
        (preference thrownBallHitsBlock
            (exists (?d - dodgeball ?b - cube_block)
                (then
                    (once (and (agent_holds ?d) (not (on rug agent))))
                    (hold (and (not (agent_holds ?d)) (in_motion ?d)))
                    (once (and (on rug ?b) (touch ?b ?d) (rug_color_under ?b ?c)))
                )
            )
        )
    )
    ; PREFERENCE: count the number of throws from off the rug that reach the end of the rug by touching the bed or the west wall
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
; TERMINAL: end the game after 3 minutes or when the agent has scored 50 points
(:terminal (or
    (>= (total-time) 180)
    (>= (total-score) 50)
))
; SCORING: 10 points for each throw that reaches the end of the rug, -5 points for each throw that hits a red cube block, -3 points for each throw that hits a green or pink cube block, -1 point for each throw that hits a yellow cube block, -1 point for each throw that hits a purple cube block
(:scoring (+
    (* 10 (count thrownBallReachesEnd))
    (* (- 5) (count thrownBallHitsBlock:red))
    (* (- 3) (count thrownBallHitsBlock:green))
    (* (- 3) (count thrownBallHitsBlock:pink))
    (- (count thrownBallHitsBlock:yellow))
    (- (count thrownBallHitsBlock:purple))
)))


(define (game game-29) (:domain few-objects-room-v1)  ; 29
(:constraints (and
    ; PREFERENCE: count the number of objects on the bed at the end of the game that are not pillows
    (preference objectOnBed
        (exists (?g - game_object)
            (at-end (and
                (not (same_type ?g pillow))
                (on bed ?g)
            ))
        )
    )
))
; SCORING: 1 point for each object on the bed at the end of the game that is not a pillow
(:scoring
    (count objectOnBed)
))


(define (game game-31) (:domain few-objects-room-v1)  ; 31
; SETUP: place a hexagonal bin next to the desk, and place all cube blocks next to the hexagonal bin, for the enture game. To start the game, place any alarm clock, cell phone, mug, key chain, cd, book, or ball either on the side table or bed.
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
        ; PREFERENCE: count any throws of a alarm clock, cell phone, mug, key chain, cd, book, or ball that is picked up from the bed or side table, thrown from the rug, and lands in the hexagonal bin
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
; SCORING: 1 point for each object thrown that is picked up from the side table, 2 points for each object thrown that is picked up from the bed
(:scoring (+
    (count objectThrownFromRug:side_table)
    (* 2 (count objectThrownFromRug:bed))
)))


(define (game game-32) (:domain many-objects-room-v1)  ; 32
; SETUP: place a hexagonal bin in the room corner adjacent to two walls for the entire game. To start the game, build a pyramid on the desk from six cube blocks, cylindrical blocks, or pyramid blocks, by placing three blocks on the desk, two blocks on the first three blocks, and one block on the middle two blocks.
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
        ; PREFERENCE: count any throws of a dodgeball or golfball that land in the hexagonal bin
        (preference ballThrownToBin (exists (?h - hexagonal_bin)
            (then
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (not (in_motion ?b)) (in ?h ?b)))
            )
        ))
        ; PREFERENCE: count any throws of a dodgeball or golfball that knock a cube block, cylindrical block, or pyramid block from the building on the desk
        (preference blockInTowerKnocked (exists (?bl - building ?c - (either cube_block cylindrical_block pyramid_block))
            (then
                (once (and
                    (agent_holds ?b)
                    (on desk ?bl)
                    (in ?bl ?c)
                ))
                (hold-while
                    (and (not (agent_holds ?b)) (in_motion ?b))
                    (or
                        (touch ?c ?b)
                        (exists (?c2 - (either cube_block cylindrical_block pyramid_block)) (touch ?c2 ?c))
                    )
                    (in_motion ?c)
                )
                (once (not (in_motion ?c)))
            )
        ))
        ; PREFERENCE: count any throws of a dodgeball or a golfball
        (preference throwAttempt
            (then
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (not (in_motion ?b)))
            )
        )
        ; PREFERENCE: count any dodgeball or golfball that is never thrown by the agent
        (preference ballNeverThrown
            (then
                (once (game_start))
                (hold (not (agent_holds ?b)))
                (once (game_over))
            )
        )
    )
))
; TERMINAL: end the game if the agent throws a dodgeball or golfball more than twice, or if the agent throws all balls 12 times or more
(:terminal (or
    (> (external-forall-maximize (count throwAttempt)) 2)
    (>= (count throwAttempt) 12)
))
; SCORING: count how many dodgeballs or golfballs land in the bin. If at least one golfball or two dodgeballs land in the bin, the agent receives 1 point for each block knocked from the tower, 1 point for each golfball that is not thrown, and 2 points for each dodgeball that is not thrown.
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
; SETUP: To start the game, make sure there are no objects in the top drawer.
(:setup (and
    (forall (?g - game_object) (game-optional
        (not (in top_drawer ?g))
    ))
))
(:constraints (and
    ; PREFERENCE: count any objects that are in the top drawer at the end of the game if the top drawer is closed
    (preference itemInClosedDrawerAtEnd (exists (?g - game_object)
        (at-end (and
            (in top_drawer ?g)
            (not (open top_drawer))
        ))
    ))
))
; SCORING: 1 point for each object in the closed top drawer at the end of the game
(:scoring
    (count-once-per-objects itemInClosedDrawerAtEnd)
))

; 34 is invalid, another hiding game


(define (game game-35) (:domain few-objects-room-v1)  ; 35
(:constraints (and
    (forall (?b - (either book dodgeball))
        ; PREFERENCE: count any throws of a book or a dodgeball that land in the bin without touching any other object, floor, or wall
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
    ; PREFERENCE: count any throws of a dodgeball that land in the bin after touching another object, floor, or wall
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
    ; PREFERENCE: count any throws of a dodgeball that miss the bin
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
; TERMINAL: the game ends if the agent scores ten points or more or if the agent scores less than -30 points
(:terminal (or
    (>= (total-score) 10)
    (<= (total-score) -30)
))
; SCORING: 1 point for each dodgeball that lands in the bin without hitting another object, 10 points if a book lands in the bin without hitting another object at least once, 2 points for each dodgeball that lands in the bin after hitting another object, and -1 point for each dodgeball throw that misses the bin
(:scoring (+
    (count throwObjectToBin:dodgeball)
    (* 10 (count-once throwObjectToBin:book))
    (* 2 (count throwBallToBinOffObject))
    (- (count throwMissesBin))
)))


(define (game game-36) (:domain few-objects-room-v1)  ; 36
; SETUP: place the hexagonal bin on the bed for the entire game. To start the game, place all dodgeballs on the desk.
(:setup (and
    (exists (?h - hexagonal_bin) (game-conserved (on bed ?h)))
    (forall (?d - dodgeball) (game-optional (on desk ?d)))
))
(:constraints (and
    ; PREFERENCE: count any throws of a dodgeball that land in the bin
    (preference throwToBin
        (exists (?d - dodgeball ?h - hexagonal_bin)
            (then
                (once (and (agent_holds ?d) (adjacent agent desk)))
                (hold (and (not (agent_holds ?d)) (in_motion ?d)))
                (once (and (not (in_motion ?d)) (in ?h ?d)))
                ; TODO: do we do anything about "whenever you get a point you put one of the blocks on the shelf. (on any of the two, it doesn't matter)"??
            )
        )
    )
    ; PREFERENCE: count any throws of a dodgeball
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
; TERMINAL: the game ends if the agent throws a dodgeball five or more times
(:terminal
    (>= (count throwAttempt) 5)
)
; SCORING: 1 point for each dodgeball throw that lands in the bin
(:scoring
    (count throwToBin)
))


(define (game game-37) (:domain many-objects-room-v1)  ; 37
(:constraints (and
    ; PREFERENCE: count any throws of a dodgeball from one wall that lands in the bin adjacent to the opposite wall
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
    ; PREFERENCE: count any throws of a dodgeball
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
; TERMINAL: the game ends if the agent throws a dodgeball ten or more times
(:terminal
    (>= (count throwAttempt) 10)
)
; SCORING: 1 point for each dodgeball throw that lands in the bin from the opposite wall
(:scoring
    (count throwToBinFromOppositeWall)
))

; projected 38 onto the space of feasible games, but could also ignore

(define (game game-38) (:domain medium-objects-room-v1)  ; 38
(:constraints (and
    ; PREFERENCE: count any throws of a dodgeball from next to the desk that land in the bin
    (preference throwToBin
        (exists (?d - dodgeball ?h - hexagonal_bin)
            (then
                (once (and (agent_holds ?d) (adjacent agent desk)))
                (hold (and (not (agent_holds ?d)) (in_motion ?d)))
                (once (and (not (in_motion ?d)) (in ?h ?d)))
            )
        )
    )
))
; SCORING: 5 points for each dodgeball throw that lands in the bin
(:scoring
    (* 5 (count throwToBin))
))


(define (game game-39) (:domain many-objects-room-v1)  ; 39
(:constraints (and
    ; PREFERENCE: count any dodgeball throws that touch a wall and then touch the agent
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
; SCORING: 1 point for each dodgeball throw that touches a wall and then touches the agent
(:scoring
    (count ballThrownToWallToAgent)
))


(define (game game-40) (:domain many-objects-room-v1)  ; 40
; SETUP: place a curved wooden ramp next to the rug for the entire game.
(:setup (and
    (exists (?r - curved_wooden_ramp) (game-conserved (adjacent ?r rug)))
))
(:constraints (and
    (forall (?c - color)
        (preference ballRolledOnRampToRug
            ; PREFERENCE: count any balls that roll from the ramp to the rug to a spot with a particular color
            (exists (?b - beachball ?r - curved_wooden_ramp)
                (then
                    (once (agent_holds ?b))
                    (hold-while
                        (and (not (agent_holds ?b)) (in_motion ?b))
                        (on ?r ?b)
                    )
                    (once (and (not (in_motion ?b)) (on rug ?b) (rug_color_under ?b ?c)))
                )
            )
        )
    )
))
; SCORING: 1 point for each ball that lands on a pink spot on the rug, 2 points for each ball that lands on a yellow spot, 3 points for each ball that lands on an orange spot, 3 points for each ball that lands on a green spot, 4 points for each ball that lands on a purple spot, and -1 point for each ball that lands on a white spot.
(:scoring (+
    (count ballRolledOnRampToRug:pink)
    (* 2 (count ballRolledOnRampToRug:yellow))
    (* 3 (count ballRolledOnRampToRug:orange))
    (* 3 (count ballRolledOnRampToRug:green))
    (* 4 (count ballRolledOnRampToRug:purple))
    (- (count ballRolledOnRampToRug:white))
)))


(define (game game-41) (:domain many-objects-room-v1)  ; 41
; SETUP: place all bridge blocks on the floor and between two walls for the entire ame. To start playing the game, place all other objects closer to one of those walls than to the other.
(:setup (and
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
))
(:constraints (and
    (forall (?w1 ?w2 - wall)
        ; PREFERENCE: count any objects that move from one side of the room to the other
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
; TERMINAL: end the game after 30 seconds
(:terminal
    (>= (total-time) 30)
)
; SCORING: 1 point for each object that moves from one side of the room to the other
(:scoring (external-forall-maximize
    (count-once-per-objects objectMovedRoomSide)
)))


(define (game 5edc195a95d5090e1c3f91b-42) (:domain few-objects-room-v1)  ; 42
; SETUP: place the hexagonal bin at least 1 distance unit away from all other objects. To start the game, place all dodgeballs between 2 and 6 distance units away from the bin.
(:setup (and
    (exists (?h - hexagonal_bin) (and
        (forall (?g - game_object) (game-optional (or
            (same_object ?h ?g)
            (> (distance ?h ?g) 1)
        )))
        (forall (?d - dodgeball) (game-optional (and
            (> (distance ?h ?d) 2)
            (< (distance ?h ?d) 6)
        )))
    ))
))
(:constraints (and
    ; PREFERENCE: count throws of one dodgeball made while standing next to the other dodgeball that land in the bin.
    (preference throwBallFromOtherBallToBin
        (exists (?d1 ?d2 - dodgeball ?h - hexagonal_bin)
            (then
                (once (and (agent_holds ?d1) (adjacent agent ?d2)))
                (hold (and (not (agent_holds ?d1)) (in_motion ?d1)))
                (once (and (not (in_motion ?d1)) (in ?h ?d1)))
            )
        )
    )
    ; PREFERENCE: count all throws of a dodgeball
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
; TERMINAL: end the game after 5 throws
(:terminal
    (>= (count throwAttempt) 5)
)
; SCORING: 1 point for each throw of one dodgeball made while standing next to the other dodgeball that lands in the bin, without moving the other dodgeball.
(:scoring
    (count-same-positions throwBallFromOtherBallToBin)
))


(define (game game-43) (:domain medium-objects-room-v1)  ; 43
; SETUP: place the doggie bed om the center of the room.
(:setup (and
    (exists (?d - doggie_bed) (game-conserved (< (distance room_center ?d) 1)))
))
(:constraints (and
    (forall (?b - ball) (and
        ; PREFERENCE: count throws of any ball that land on the doggie bed without touching any walls.
        (preference throwBallToDoggieBed
            (exists (?d - doggie_bed)
                (then
                    (once (agent_holds ?b))
                    (hold (and (not (agent_holds ?b)) (in_motion ?b) (not (exists (?w - wall) (touch ?w ?b )))))
                    (once (and (not (in_motion ?b)) (on ?d ?b)))
                )
            )
        )
        ; PREFERENCE: count throws of any ball that land on the doggie bed after touching a wall.
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
; SCORING: 1 point for each basketball thrown onto the doggie bed without touching a wall, 2 points for each beachball, 3 points for each dodgeball, 2 points for each basketball thrown onto the doggie bed after touching a wall, 3 points for each beachball, 4 points for each dodgeball.
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
; SETUP: to start the game, place the two teddybears aligned to the middle of the bed, one on the bed and one of the floor.
(:setup (and
    (exists (?t1 ?t2 - teddy_bear) (game-optional (and
        (on floor ?t1)
        (on bed ?t2)
        ; TODO: is the below nicer than (= (z_position ?t1) (z_position ?T2))
        (equal_z_position ?t1 ?t2)
        (equal_z_position ?t1 bed)
    )))
))
(:constraints (and
    (forall (?b - (either golfball dodgeball)) (and
        ; PREFERENCE: count throws of any ball thrown with the agent adjacent to the sliding door and desk that knock a teddy bear over.
        (preference throwKnocksOverBear (exists (?t - teddy_bear ?s - sliding_door)
            (then
                (once (and
                    (agent_holds ?b)
                    (adjacent agent desk)
                    (adjacent agent ?s)
                    (equal_z_position ?t bed)
                    ; (= (z_position ?t) (z_position bed))
                ))
                (hold-while
                    (and (in_motion ?b) (not (agent_holds ?b)))
                    (touch ?b ?t)
                )
                (once (in_motion ?t))
            )
        ))
        ; PREFERENCE: count any throw of a ball thrown with the agent adjacent to the sliding door and desk
        (preference throwAttempt (exists (?s - sliding_door)
            (then
                (once (and (agent_holds ?b) (adjacent agent desk) (adjacent agent ?s)))
                (hold (and (in_motion ?b) (not (agent_holds ?b))))
                (once (not (in_motion ?b)))
            )
        ))
    ))
))
; TERMINAL: end the game after 6 throws or after any ball is thrown more than once.
(:terminal (or
    (> (external-forall-maximize (count throwAttempt)) 1)
    (>= (count-once-per-objects throwAttempt) 6)
))
; SCORING: 1 point for each throw of a dodgeball that knocks over a teddy bear, 2 points for each throw of a golfball that knocks over a teddy bear.
(:scoring (+
    (count-once-per-objects throwKnocksOverBear:dodgeball)
    (* 2 (count-once-per-objects throwKnocksOverBear:golfball))
)))


(define (game game-46) (:domain few-objects-room-v1)  ; 46
; SETUP: place the curved wooden ramp near the center of the room for the entire game.
(:setup (and
    (exists (?c - curved_wooden_ramp) (game-conserved
        (< (distance ?c room_center) 3)
    ))
))
(:constraints (and
    ; PREFERENCE: count throws of the pink dodgeball thrown with the agent facing the ramp that touch the ramp and land on the bed.
    (preference ballThrownToRampToBed (exists (?c - curved_wooden_ramp)
        (then
            (once (and (agent_holds pink_dodgeball) (faces agent ?c)))
            (hold-while
                (and (in_motion pink_dodgeball) (not (agent_holds pink_dodgeball)))
                (touch pink_dodgeball ?c)
            )
            (once (and (not (in_motion pink_dodgeball)) (on bed pink_dodgeball)))
        )
    ))
    ; PREFERENCE: count throws of the pink dodgeball thrown with the agent facing the ramp that touch the ramp and then touch the agent.
    (preference ballThrownHitsAgent (exists (?c - curved_wooden_ramp)
        (then
            (once (and (agent_holds pink_dodgeball) (faces agent ?c)))
            (hold-while
                (and (in_motion pink_dodgeball) (not (agent_holds pink_dodgeball)))
                (touch pink_dodgeball ?c)
            )
            (once (and (touch pink_dodgeball agent) (not (agent_holds pink_dodgeball))))
        )
    ))
))
; SCORING: 1 point for each throw of the pink dodgeball that lands on the bed, -1 point for each throw of the pink dodgeball that hits the agent.
(:scoring (+
    (count ballThrownToRampToBed)
    (- (count ballThrownHitsAgent))
)))


(define (game game-47) (:domain many-objects-room-v1)  ; 47
(:constraints (and
    (forall (?c - color)
        ; PREFERENCE: count any beachball throws made from off the rug that touch the green triangular ramp and land on a spot of a particular color on the rug.
        (preference beachballBouncedOffRamp
            (exists (?b - beachball ?r - green_triangular_ramp)
                (then
                    (once (and (agent_holds ?b) (not (on rug agent)) (faces agent ?r)))
                    (hold-while
                        (and (in_motion ?b) (not (agent_holds ?b)))
                        (touch ?b ?r)
                    )
                    (once (and (not (in_motion ?b)) (on rug ?b) (rug_color_under ?b ?c)))
                )
            )
        )
    )
))
; SCORING: 1 point for each beachball throw that lands on a red spot, 3 points for each beachball throw that lands on a pink spot, 10 points for each beachball throw that lands on a green spot.
(:scoring (+
    (count beachballBouncedOffRamp:red)
    (* 3 (count beachballBouncedOffRamp:pink))
    (* 10 (count beachballBouncedOffRamp:green))
)))

; TODO: this is a crude approximation of 48 -- let's hope it's reasonable?


(define (game game-48) (:domain medium-objects-room-v1)  ; 48
; SETUP: place the hexagonal bin on top of a building with at least three other items, with no items on the hexagonal bin, near the center of the room, for the entire game.
(:setup (and
    (exists (?b - building ?h - hexagonal_bin) (game-conserved (and
        (in ?b ?h)
        (>= (building_size ?b) 4) ; TODO: could also quantify out additional objects
        (not (exists (?g - game_object) (and (in ?b ?g) (on ?h ?g))))
        (< (distance ?b room_center) 1)
    )))
))
(:constraints (and
    (forall (?d - (either dodgeball basketball beachball))
        ; PREFERENCE: count throws of a dodgeball, basketball, or beachball that land in the bin with the bin remaining in or on the building.
        (preference ballThrownToBin (exists (?b - building ?h - hexagonal_bin)
            (then
                (once (agent_holds ?d))
                (hold (and (in_motion ?d) (not (agent_holds ?d))))
                (once (and (not (in_motion ?d)) (or (in ?h ?d) (on ?h ?d)) (or (in ?b ?h) (on ?b ?h))))
            )
        ))
    )
    ; PREFERENCE: count if the pillow, doggie bed, or teddy bear is on on the desktop or laptop at the end of the game.
    (preference itemsHidingScreens
        (exists (?s - (either desktop laptop) ?o - (either pillow doggie_bed teddy_bear))
            (at-end (on ?s ?o))
        )
    )
    ; PREFERENCE: count if the alarm clock or cellphone is in a drawer at the end of the game.
    (preference objectsHidden
        (exists (?o - (either alarm_clock cellphone) ?d - drawer)
            (at-end (in ?d ?o))
        )
    )
    ; PREFERENCE: count if the blinds are open at the end of the game.
    (preference blindsOpened
        (exists (?b - blinds)
            (at-end (open ?b))  ; blinds being open = they were pulled down
        )
    )
    ; PREFERENCE: count if any non-ball object is moved at any point during the game.
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
; SCORING: 5 points for each dodgeball that lands in the bin, 7 points for each basketball that lands in the bin, 15 points for each beachball that lands in the bin, 10 points for each item that hides a screen, 10 points for each object that is hidden in a drawer, 10 points for each set of blinds that are opened, and -5 points for each object that is moved.
(:scoring (+
    (* 5 (count ballThrownToBin:dodgeball))
    (* 7 (count ballThrownToBin:basketball))
    (* 15 (count ballThrownToBin:beachball))
    (* 10 (count-once-per-objects itemsHidingScreens))
    (* 10 (count-once-per-objects objectsHidden))
    (* 10 (count-once-per-objects blindsOpened))
    (* (- 5) (count objectMoved))
)))


(define (game game-49) (:domain many-objects-room-v1)  ; 49
; SETUP: place the green golfball near the door for the entire game. To start the game, place all dodgeballs within 1 meter of the green golfball.
(:setup (and
    (game-conserved (< (distance green_golfball door) 0.5))
    (forall (?d - dodgeball) (game-optional (< (distance green_golfball ?d) 1)))
))
(:constraints (and
    (forall (?d - dodgeball) (and
        ; PREFERENCE: count all dodgeball throws that are thrown from near the green golfball and door and land in the bin.
        (preference dodgeballThrownToBin (exists (?h - hexagonal_bin)
            (then
                (once (and
                    (adjacent agent green_golfball)
                    (adjacent agent door)
                    (agent_holds ?d)
                ))
                (hold (and (in_motion ?d) (not (agent_holds ?d))))
                (once (and (not (in_motion ?d)) (in ?h ?d)))
            )
        ))
        ; PREFERENCE: count all dodgeball throws that are thrown from near the green golfball and door.
        (preference throwAttemptFromDoor
            (then
                (once (and
                    (adjacent agent green_golfball)
                    (adjacent agent door)
                    (agent_holds ?d)
                ))
                (hold (and (in_motion ?d) (not (agent_holds ?d))))
                (once (not (in_motion ?d)))
            )
        )
    ))
))
; TERMINAL: end the game if the agent has thrown a dodgeball from near the green golfball more than once  or if the agent threw at least 3 dodgeballs from near the green golfball and door.
(:terminal (or
    (> (external-forall-maximize (count throwAttemptFromDoor)) 1)
    (>= (count-once-per-objects throwAttemptFromDoor) 3)
))
; SCORING: 10 points for each dodgeball that lands in the bin
(:scoring
    (* 10 (count-once-per-objects dodgeballThrownToBin))
))


(define (game game-50) (:domain medium-objects-room-v1)  ; 50
; SETUP: place the hexagonal bin near the center of the room for the entire game.
(:setup (and
    (exists (?h - hexagonal_bin) (game-conserved (< (distance room_center ?h) 1)))
))
(:constraints (and
    ; PREFERENCE: count any game object that is thrown into the bin.
    (preference gameObjectToBin (exists (?g - game_object ?h - hexagonal_bin)
        (then
            (once (not (agent_holds ?g)))
            (hold (or (agent_holds ?g) (in_motion ?g)))
            (once (and (not (in_motion ?g)) (in ?h ?g)))
        )
    ))
))
; SCORING: 1 point for each object that lands in the bin.
(:scoring
    (count-once-per-objects gameObjectToBin)
))

(define (game game-51) (:domain few-objects-room-v1)  ; 51
(:constraints (and
    ; PREFERENCE: count if the agent throws a dodgeball into the bin.
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
; SCORING: 1 point for each dodgeball that lands in the bin.
(:scoring
    (count throwToBin)
))


(define (game game-52) (:domain few-objects-room-v1)  ; 52
(:constraints (and
    ; PREFERENCE: count cube blocks thrown from the rug onto the desk without breaking a lamp, dekstop, or laptop
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
; SCORING: 1 point for each cube block thrown from the rug onto the desk without breaking a lamp, dekstop, or laptop
(:scoring
    (count-once-per-objects blockFromRugToDesk)
))


(define (game game-53) (:domain few-objects-room-v1)  ; 53
(:constraints (and
    ; PREFERENCE: count dodgeballs placed in a bin in a corner adjacent to two walls at the end of the game.
    (preference dodgeballsInPlace
        (exists (?d - dodgeball ?h - hexagonal_bin ?w1 ?w2 - wall)
            (at-end (and (in ?h ?d) (adjacent ?h ?w1) (adjacent ?h ?w2)))
        )
    )
    ; PREFERENCE: count cube blocks placed on a shelf at the end of the game.
    (preference blocksInPlace
        (exists (?c - cube_block ?s - shelf)
            (at-end (on ?s ?c))
        )
    )
    ; PREFERENCE: count any cellphones, key chains, mugs, credit cards, cds, watches, or alarm clocks placed in a drawer at the end of the game.
    (preference smallItemsInPlace
        (exists (?o - (either cellphone key_chain mug credit_card cd watch alarm_clock) ?d - drawer)
            (at-end (and
                (in ?d ?o)
            ))
        )
    )
))
; SCORING: 5 points for each dodgeball placed in its place, 5 points for each cube block placed on its shelf, and 5 points for each small item placed in its drawer.
(:scoring (+
    (* 5 (count-once-per-objects dodgeballsInPlace))
    (* 5 (count-once-per-objects blocksInPlace))
    (* 5 (count-once-per-objects smallItemsInPlace))
)))


(define (game game-54) (:domain few-objects-room-v1)  ; 54
(:constraints (and
    (forall (?b - building)
        ; PREFERENCE: count cube blocks placed in a building that remain in that building until the end of the game
        (preference blockPlacedInBuilding (exists (?l - cube_block)
            (then
                (once (agent_holds ?l))
                (hold (and (in_motion ?l) (not (agent_holds ?l))))
                (hold (in ?b ?l))
                (once (or (not (in ?b ?l)) (game_over)))
            )
        ))
    )
    (forall (?l - cube_block)
        ; PREFERENCE: count cube blocks picked up
        (preference blockPickedUp
            (then
                (once (not (agent_holds ?l)))
                (hold (agent_holds ?l))
                (once (not (agent_holds ?l)))
            )
        )
    )
))
; TERMINAL: the game ends when the agent picked up any particular cube block for the third time.
(:terminal
    (>= (external-forall-maximize (count blockPickedUp)) 3)
)
; SCORING: for the building with the highest score, 1 point for each cube block placed in that building that remains in that building until the end of the game.
(:scoring (external-forall-maximize
    (count-overlapping blockPlacedInBuilding)
)))


(define (game game-55) (:domain few-objects-room-v1)  ; 55
; SETUP: place a hexagonal bin in the center of the room for the entire game.
(:setup (and
    (exists (?h - hexagonal_bin)
        (game-conserved (< (distance ?h room_center) 1))
    )
))
(:constraints (and
    ; PREFERENCE: count throws of any object that succesfully land in the bin on the first time the agent picks them up and throws them.
    (preference objectToBinOnFirstTry (exists (?o - game_object ?h - hexagonal_bin)
        (then
            (once (game_start))
            (hold (not (agent_holds ?o)))
            (hold (agent_holds ?o))
            (hold (and (in_motion ?o) (not (agent_holds ?o))))
            (once (and (not (in_motion ?o)) (in ?h ?o)))
            (hold (not (agent_holds ?o)))
        )
    ))
))
; SCORING: 1 point for each object thrown into the bin on the first try.
(:scoring
    (count-once-per-objects objectToBinOnFirstTry)
))


(define (game game-56) (:domain few-objects-room-v1)  ; 56

(:constraints (and
    ; PREFERENCE: count throws of a dodgeball thrown from the door that land in a hexagonal bin
    (preference throwFromDoorToBin (exists (?d - dodgeball ?h - hexagonal_bin)
        (then
            (once (and (agent_holds ?d) (adjacent agent door)))
            (hold (and (not (agent_holds ?d)) (in_motion ?d)))
            (once (and (not (in_motion ?d)) (in ?h ?d)))
        )
    ))
    ; PREFERENCE: count any throws of a dodgeball
    (preference throwAttempt (exists (?d - dodgeball)
        (then
            (once (agent_holds ?d))
            (hold (and (not (agent_holds ?d)) (in_motion ?d)))
            (once (not (in_motion ?d)))
        )
    ))
))
; TERMINAL: the game ends when the agent has thrown a dodgeball at least three times.
(:terminal
    (>= (count throwAttempt) 3)
)
; SCORING: 1 point for each throw of a dodgeball from the door that lands in a hexagonal bin.
(:scoring
    (count throwFromDoorToBin)
))


(define (game game-57) (:domain medium-objects-room-v1)  ; 57
(:constraints (and
    ; PREFERENCE: count how many books are placed on a desk shelf at the end of the game with no pencils, pens, or cds on the desk shelf.
    (preference bookOnDeskShelf (exists (?b - book ?d - desk_shelf)
        (at-end (and
            (on ?d ?b)
            (not (exists (?o - (either pencil pen cd)) (on ?d ?o)))
        ))
    ))
    ; PREFERENCE: count how many pencils, pens, or cds are placed on a desk shelf at the end of the game with no books on the desk shelf.
    (preference otherObjectsOnDeskShelf (exists (?o - (either pencil pen cd) ?d - desk_shelf)
        (at-end (and
            (on ?d ?o)
            (not (exists (?b - book) (on ?d ?b)))
        ))
    ))
    ; PREFERENCE: count how many dodgeballs and basketballs are placed in a hexagonal bin at the end of the game.
    (preference dodgeballAndBasketballInBin (exists (?b - (either dodgeball basketball) ?h - hexagonal_bin)
        (at-end (in ?h ?b))
    ))
    ; PREFERENCE: count how many beachballs are placed on a rug at the end of the game.
    (preference beachballOnRug (exists (?b - beachball)
        (at-end (on rug ?b))
    ))
    ; PREFERENCE: count how many cellphones, key chains, or cds are placed in a drawer at the end of the game.
    (preference smallItemsInPlace (exists (?o - (either cellphone key_chain cd) ?d - drawer)
        (at-end (in ?d ?o))
    ))
    ; PREFERENCE: count how many watches are placed on a shelf at the end of the game.
    (preference watchOnShelf (exists (?w - watch ?s - shelf)
        (at-end (on ?s ?w))
    ))
))
; SCORING: 1 point for each object placed in the correct place at the end of the game as specified above.
(:scoring (+
    (count-once-per-objects bookOnDeskShelf)
    (count-once-per-objects otherObjectsOnDeskShelf)
    (count-once-per-objects dodgeballAndBasketballInBin)
    (count-once-per-objects beachballOnRug)
    (count-once-per-objects smallItemsInPlace)
    (count-once-per-objects watchOnShelf)
)))


(define (game game-58) (:domain medium-objects-room-v1)  ; 58
; SETUP: create a building with six different blocks, one of each type of block, for the entire game. To start the game, remove all blocks from the shelves.
(:setup (and
    (exists (?b - building) (and
        (game-conserved (= (building_size ?b) 6))
        (forall (?l - block) (or
            (game-conserved (and
                    (in ?b ?l)
                    (not (exists (?l2 - block) (and
                        (in ?b ?l2)
                        (not (same_object ?l ?l2))
                        (same_type ?l ?l2)
                    )))
            ))
            (game-optional (not (exists (?s - shelf) (on ?s ?l))))
        ))
    ))
))
(:constraints (and
    ; PREFERENCE: count how many times the agent picks up a block not in the setup building for the first time.
    (preference gameBlockFound (exists (?l - block)
        (then
            (once (game_start))
            (hold (not (exists (?b - building) (and (in ?b ?l) (is_setup_object ?b)))))
            (once (agent_holds ?l))
        )
    ))
    ; PREFERENCE: count how many times a building falls while the agent is attempting to build it.
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
    ; PREFERENCE: count if at the end of the game the agent has built a building that is the same as the setup building.
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
; SETUP: 5 points for each block found for the first time, 100 points for building a building that matches the setup building, and -10 points for each building that falls while the agent is attempting to build it.
(:scoring (+
    (* 5 (count-once-per-objects gameBlockFound))
    (* 100 (count-once matchingBuildingBuilt))
    (* (-10) (count towerFallsWhileBuilding))
)))


(define (game game-59) (:domain many-objects-room-v1)  ; 59
; SETUP: place a hexagonal bin near the door for the entire game.
(:setup (and
    (exists (?h - hexagonal_bin) (game-conserved (< (distance ?h door) 1)))
))
(:constraints (and
    (forall (?b - (either golfball dodgeball beachball))
        ; PREFERENCE: count throws of a golfball, dodgeball, or beachball that land in the hexagonal bin.
        (preference ballThrownToBin (exists (?h - hexagonal_bin)
            (then
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (not (in_motion ?b)) (in ?h ?b)))
            )
        ))
    )
))
; SCORING: 2 points for each golfball thrown into the bin, 3 points for each dodgeball thrown into the bin, and 4 points for each beachball thrown into the bin.
(:scoring (+
    (* 2 (count ballThrownToBin:golfball))
    (* 3 (count ballThrownToBin:dodgeball))
    (* 4 (count ballThrownToBin:beachball))
)))

; 60 is invalid


(define (game game-61) (:domain many-objects-room-v1)  ; 61
; SETUP: place a flat block on the rug for the entire game. Place all pyramid blocks on the floor for the entire game, with the yellow pyramind block closer to the bin than the red one, and the red one closer than the blue one.
(:setup (game-conserved (and
    (exists (?f - flat_block) (on rug ?f))
    (forall (?p - pyramid_block) (on floor ?p))
    (exists (?p1 - yellow_pyramid_block ?p2 - red_pyramid_block ?p3 - blue_pyramid_block ?h - hexagonal_bin)
        (and
            (> (distance ?h ?p2) (distance ?h ?p1))
            (> (distance ?h ?p3) (distance ?h ?p2))
        )
    )
)))
(:constraints (and
    (forall (?p - pyramid_block)
        ; PREFERENCE: count how many times a dodgeball is thrown with the agent adjacent to a pyramid block and lands in the hexagonal bin.
        (preference dodgeballFromBlockToBin (exists (?d - dodgeball ?h - hexagonal_bin)
            (then
                (once (and (agent_holds ?d) (adjacent agent ?p)))
                (hold (and (not (agent_holds ?d)) (in_motion ?d)))
                (once (and (not (in_motion ?d)) (in ?h ?d)))
            )
        ))
    )
    ; PREFERENCE: count how many times a cube block is in a building that includes the setup flat block at the end of the game.
    (preference cubeBlockInBuilding (exists (?b - building ?l - cube_block ?f - flat_block)
        (at-end (and
              (is_setup_object ?f)
              (in ?b ?f)
              (in ?b ?l)
        ))
    ))
))
; SCORING: 10 points for each dodgeball thrown from the yellow pyramid block, 25 points for each dodgeball thrown from the red pyramid block, 50 points for each dodgeball thrown from the blue pyramid block, 100 points for throwing three different dodgeballs from the blue pyramid block, 10 points for each cube block in a building that includes the setup flat block, and 100 points for building a building that includes the setup flat block and has three cube blocks in it.
(:scoring (+
    (* 10 (count dodgeballFromBlockToBin:yellow_pyramid_block))
    (* 25 (count dodgeballFromBlockToBin:red_pyramid_block))
    (* 50 (count dodgeballFromBlockToBin:blue_pyramid_block))
    (* 100 (= (count-once-per-objects dodgeballFromBlockToBin:blue_pyramid_block) 3))
    (* 10 (count-once-per-objects cubeBlockInBuilding))
    (* 100 (= (count-once-per-objects cubeBlockInBuilding) 3))
)))


(define (game game-62) (:domain medium-objects-room-v1)  ; 62
(:constraints (and
    ; PREFERENCE: count how many times a chair, laptop, or doggie bed is thrown from the desk onto the bed.
    (preference bigObjectThrownToBed (exists (?o - (either chair laptop doggie_bed))
        (then
            (once (and (agent_holds ?o) (adjacent agent desk)))
            (hold (and (not (agent_holds ?o)) (in_motion ?o)))
            (once (and (not (in_motion ?o)) (on bed ?o)))
        )
    ))
    ; PREFERENCE: count how many times an object that is not a chair, laptop, or doggie bed is thrown from the desk onto the bed.
    (preference smallObjectThrownToBed (exists (?o - game_object)
        (then
            (once (and
                (agent_holds ?o)
                (adjacent agent desk)
                (not (exists (?o2 - (either chair laptop doggie_bed)) (same_object ?o ?o2)))
            ))
            (hold (and (not (agent_holds ?o)) (in_motion ?o)))
            (once (and (not (in_motion ?o)) (on bed ?o)))
        )
    ))
    ; PREFERENCE: count how many times an object is thrown from the desk and does not land on the bed.
    (preference failedThrowAttempt (exists (?o - game_object)
        (then
            (once (and (agent_holds ?o) (adjacent agent desk)))
            (hold (and (not (agent_holds ?o)) (in_motion ?o)))
            (once (and (not (in_motion ?o)) (not (on bed ?o))))
        )
    ))
))
; SCORING: 1 point for each small object thrown onto the bed, 5 points for each chair, laptop, or doggie bed thrown onto the bed, and -5 points for each failed throw attempt.
(:scoring (+
    (count smallObjectThrownToBed)
    (* 5 (count bigObjectThrownToBed))
    (* (- 5) (count failedThrowAttempt))
)))



(define (game game-63) (:domain medium-objects-room-v1)  ; 63
(:constraints (and
    ; PREFERENCE: count how many times a building falls while the agent is attempting to build it
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
        ; PREFERENCE: count how many blocks are placed in a building until they fall or the game ends.
        (preference blockPlacedInBuilding (exists (?l - block)
            (then
                (once (agent_holds ?l))
                (hold (and (in_motion ?l) (not (agent_holds ?l))))
                (hold (in ?b ?l))
                (once (or (not (in ?b ?l)) (game_over)))
            )
        ))
        ; PREFERENCE: count how many non-block objects are placed in a building until they fall or the game ends.
        (preference nonBlockPlacedInBuilding (exists (?o - game_object)
            (then
                (once (and (agent_holds ?o) (not (same_type ?o block))))
                (hold (and (in_motion ?l) (not (agent_holds ?l))))
                (hold (in ?b ?l))
                (once (or (not (in ?b ?l)) (game_over)))
            )
        ))
    ))
))
; TERMINAL: the game ends when a building falls.
(:terminal
    (>= (count-once towerFallsWhileBuilding) 1)
)
; SCORING: for the building with the highest score, 1 point for each block placed in the building and 2 points for each non-block object placed in the building.
(:scoring (external-forall-maximize (+
    (count-overlapping blockPlacedInBuilding)
    (* 2 (count-overlapping nonBlockPlacedInBuilding))
))))


(define (game game-64) (:domain many-objects-room-v1)  ; 64
(:constraints (and
    (forall (?o - (either hexagonal_bin rug wall))
    ; PREFERENCE: count how many times a dodgeball is thrown from next to a hexagonal bin, the rug, or a wall to a bin.
        (preference ballThrownFromObjectToBin (exists (?d - dodgeball ?h - hexagonal_bin)
            (then
                (once (and (agent_holds ?d) (adjacent agent ?o)))
                (hold (and (not (agent_holds ?d)) (in_motion ?d)))
                (once (and (not (in_motion ?d)) (in ?h ?d)))
            )
        ))
    )
))
; SCORING: 1 point for each dodgeball thrown from next to a hexagonal bin, 2 points for each dodgeball thrown from next to the rug, and 3 points for each dodgeball thrown from next to a wall.
(:scoring (+
    (count ballThrownFromObjectToBin:hexagonal_bin)
    (* 2 (count ballThrownFromObjectToBin:rug))
    (* 3 (count ballThrownFromObjectToBin:wall))
)))


(define (game game-65) (:domain many-objects-room-v1)  ; 65
(:constraints (and
    ; PREFERENCE: count how many balls are on the bed at the end of the game
    (preference ballOnBedAtEnd (exists (?b - ball)
        (at-end
            (on bed ?b)
        )
    ))
))
; SCORING: 1 point for each ball on the bed at the end of the game.
(:scoring (count-once-per-objects ballOnBedAtEnd)
))


(define (game game-66) (:domain medium-objects-room-v1)  ; 66
; SCORING: place all bridge blocks and cube blocks near the door for the entire game, and place all flat blocks and pyramid blocks off the shelves for the entire game. To start the game, place all cylindrical blocks and tall cylindrical blocks on the bottom shelf.
(:setup (and
    (forall (?b - (either bridge_block cube_block))
        (game-conserved (< (distance ?b door) 1))
    )
    (forall (?b - (either cylindrical_block tall_cylindrical_block))
        (game-optional (on bottom_shelf ?b))
    )
    (forall (?b - (either flat_block pyramid_block))
        (game-conserved (not (exists (?s - shelf) (on ?s ?b))))
    )
))
(:constraints (and
    (forall (?b - (either cylindrical_block tall_cylindrical_block)) (and
        ; PREFERENCE: count dodgeball throws from the doggie bed that land closest to a bridge block or cube block that is of the same color as the cylndrical or tall cylindrical block on the top shelf.
        (preference blockCorrectlyPicked (exists (?d - dodgeball ?o - doggie_bed ?tb - (either bridge_block cube_block))
            (then
                (once (and
                    (agent_holds ?d)
                    (on agent ?o)
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
        ; PREFERENCE: count dodgeball throws from the doggie bed that land closest to a bridge block or cube block that is of the different color as the cylndrical or tall cylindrical block on the top shelf.
        (preference blockIncorrectlyPicked (exists (?d - dodgeball ?o - doggie_bed ?tb - (either bridge_block cube_block))
            (then
                (once (and
                    (agent_holds ?d)
                    (on agent ?o)
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
; TERMINAL: the game ends when the agent made 4 dodgeball throws that land closest to the block of the color of the block on the top shelf.
(:terminal
    (>= (count-once-per-external-objects blockCorrectlyPicked) 4)
)
; SCORING: 10 points for throwing near the correct block for each block, -1 point for each throw that lands near an incorrect block, and 100 points for throwing near the correct block for at least four different blocks.
(:scoring (+
    (* 10 (count-once-per-external-objects blockCorrectlyPicked))
    (- (count blockIncorrectlyPicked))
    ( * 100 (>= (count-once-per-external-objects blockCorrectlyPicked) 4))
)))


(define (game game-67) (:domain medium-objects-room-v1)  ; 67
; SETUP: Move the chair away from the front of the desk for the entire game. To start the game, take ten of the blocks that are tall cylndrical, bridge blocks, flat blocks, or cylndrical blocks. Place four in a line close to the desk. Place three in a line slightly further from the desk. Place two in a line even further from the desk, and place one even further than that.
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
        ; PREFERENCE: count balls thrown from the rug that knock a block used in the setup
        (preference ballKnocksBlockFromRug (exists (?l - block)
            (then
                (once (and (agent_holds ?b) (on rug agent) (is_setup_object ?l)))
                (hold-while
                    (and (not (agent_holds ?b)) (in_motion ?b))
                    (touch ?b ?l)
                    (in_motion ?l)
                )
            )
        ))
        ; PREFERENCE: count any balls thrown
        (preference throwAttempt
            (then
                (once (and (agent_holds ?b) (on rug agent)))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (not (in_motion ?b))))
            )
        )
    ))
))
; TERMINAL: the game ends when the agent has made 16 throws.
(:terminal
    (>= (count throwAttempt) 16)
)
; SCORING: 1 point for each dodgeball thrown that knocks a block, 0.7 points for each basketball thrown that knocks a block, and 0.5 points for each beachball thrown that knocks a block.
(:scoring (+
    (count-once-per-objects ballKnocksBlockFromRug:dodgeball)
    (* 0.7 (count-once-per-objects ballKnocksBlockFromRug:basketball))
    (* 0.5 (count-once-per-objects ballKnocksBlockFromRug:beachball))
)))

; 68 has subjective scoring -- I could attempt to objectify, but it's hard


(define (game game-69) (:domain many-objects-room-v1)  ; 69
; SETUP: place a curved wooden ramp adjacent to a hexagonal bin for the entire game
(:setup (and
    (exists (?c - curved_wooden_ramp ?h - hexagonal_bin) (game-conserved (adjacent ?c ?h)))
))
(:constraints (and
    ; PREFERENCE: count balls thrown through the ramp that land in the bin
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
; SCORING: 1 point for each ball thrown through the ramp that lands in the bin
(:scoring
    (count ballThrownThroughRampToBin)
))


(define (game game-70) (:domain many-objects-room-v1)  ; 70
; SETUP: place all chairs away from the front of the desk for the entire game. Place a curved wooden ramp next to the front of the desk, and a hexagonal bin with its front to the ramp's back for the entire game.  To start the game, place all golfballs, dodgeballs, triangle blocks, and pyramid blocks near the side table.
(:setup (and
    (forall (?c - chair) (game-conserved (not (adjacent_side desk front ?c))))
    (exists (?h - hexagonal_bin ?c - curved_wooden_ramp )
        (game-conserved (and
            (adjacent_side desk front ?c)
            (adjacent_side ?h front ?c back)
        ))
    )
    (forall (?o - (either golfball dodgeball triangle_block pyramid_block))
        (game-optional (< (distance side_table ?o) 1))
    )
))
(:constraints (and
    (forall (?o - (either golfball dodgeball triangle_block pyramid_block)) (and
        ; PREFERENCE: count all golfballs, dodgeballs, triangle blocks, and pyramid blocks thrown from the agent next ot the bed and that land in the bin.
        (preference objectLandsInBin (exists (?h - hexagonal_bin)
            (then
                (once (and (adjacent agent bed) (agent_holds ?o)))
                (hold (and (in_motion ?o) (not (agent_holds ?o))))
                (once (and (not (in_motion ?o)) (in ?h ?o)))
            )
        ))
        ; PREFERENCE: count all golfballs, dodgeballs, triangle blocks, and pyramid blocks thrown from the agent next to the bed and that hit the desktop or laptop computer
        (preference thrownObjectHitsComputer (exists (?c - (either desktop laptop))
            (then
                (once (and (adjacent agent bed) (agent_holds ?o)))
                (hold (and (in_motion ?o) (not (agent_holds ?o))))
                (once (touch ?o ?c))
            )
        ))
    ))
    ; PREFERENCE: count all golfballs thrown from the agent next to the bed and that land in the bin through the ramp
    (preference golfballLandsInBinThroughRamp (exists (?g - golfball ?c - curved_wooden_ramp ?h - hexagonal_bin)
        (then
            (once (and (adjacent agent bed) (agent_holds ?g)))
            (hold-while
                (and (in_motion ?g) (not (agent_holds ?g)))
                (touch ?c ?g)
            )
            (once (and (not (in_motion ?g)) (in ?h ?g)))
        )
    ))
))
; SCORING: 1 point for each triangle block that lands in the bin, 2 points for each pyramid block that lands in the bin, 2 points for each dodgeball that lands in the bin, 3 points for each golfball that lands in the bin, 6 points for each golfball that lands in the bin through the ramp, and -1 point for each object thrown from the agent next to the bed that hits the computer.
(:scoring (+
    (count objectLandsInBin:triangle_block)
    (* 2 (count objectLandsInBin:pyramid_block))
    (* 2 (count objectLandsInBin:dodgeball))
    (* 3 (count objectLandsInBin:golfball))
    (* 6 (count golfballLandsInBinThroughRamp))
    (- (count thrownObjectHitsComputer))
)))

(define (game game-71) (:domain many-objects-room-v1)  ; 71
; SETUP: place all pillows on the bed for the entire game. Place all bridge blocks on the floor for the entire game. Place all cylndrical blocks near either a pillow or a bridge block for the entire game.
(:setup (and
    (forall (?p - pillow) (game-conserved (on bed ?p)))
    (forall (?b - bridge_block) (game-conserved (on floor ?b)))
    (forall (?c - cylindrical_block) (game-conserved (exists (?o - (either pillow bridge_block)) (< (distance ?c ?o) 1))) )
))
(:constraints (and
    ; PREFERENCE: count all dodgeballs thrown with the agent next to a triangular ramp and near the desk and that touch a pillow without touching a clyndrical block
    (preference dodgeballHitsPillowWithoutTouchingBlock (exists (?d - dodgeball ?p - pillow ?r - triangular_ramp)
        (then
            (once (and (adjacent agent ?r) (< (distance ?r desk) 1) (agent_holds ?d)))
            (hold-while
                (and (in_motion ?d) (not (agent_holds ?d)) (not (exists (?c - cylindrical_block) (touch ?c ?d) )) )
                (touch ?d ?p)
            )
            (once (not (in_motion ?d)))
        )
    ))
    ; PREFERENCE: count all golfballs thrown with the agent next to a triangular ramp and near the desk and that pass under a bridge block without touching a clyndrical block
    (preference golfballUnderBridgeWithoutTouchingBlock (exists (?g - golfball ?b - bridge_block ?r - triangular_ramp)
        (then
            (once (and (adjacent agent ?r) (< (distance ?r desk) 1) (agent_holds ?g)))
            (hold-while
                (and (in_motion ?g) (not (agent_holds ?g)) (not (exists (?c - cylindrical_block) (touch ?c ?g) )) )
                (above ?g ?b)
            )
            (once (not (in_motion ?g)))
        )
    ))
))
; SCORING: 1 point for each dodgeball that hits a pillow without touching a cylindrical block, and 1 point for each golfball that passes under a bridge block without touching a cylindrical block.
(:scoring (+
    (count dodgeballHitsPillowWithoutTouchingBlock)
    (count golfballUnderBridgeWithoutTouchingBlock)
)))


(define (game game-72) (:domain many-objects-room-v1)  ; 72
; To start the game, place all teddy bears on the bed and upright, and place all balls near the desk.
(:setup (and
    (exists (?t - teddy_bear) (game-optional (and (on bed ?t) (object_orientation ?t upright))))
    (forall (?b - ball) (game-optional (< (distance ?b desk) 1)))
))
(:constraints (and
    ; PREFERENCE: count all balls thrown with the agent on a chair next to the desk, that knock over a teddy bear such that it is no longer upright.
    (preference ballKnocksTeddy (exists (?b - ball ?t - teddy_bear ?c - chair)
        (then
            (once (and
                (on ?c agent)
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
; TERMINAL: end the game when the agent knocks over a teddy bear 7 times.
(:terminal
    (>= (count ballKnocksTeddy) 7)
)
; SCORING: 1 point for each time a teddy bear is knocked over.
(:scoring
    (count ballKnocksTeddy)
))

(define (game game-73) (:domain many-objects-room-v1)  ; 73
; SETUP: place a hexagonal bin near the center of the room for the entire game. To start the game, place all dodgeballs on the desk.
(:setup (and
    (exists (?h - hexagonal_bin) (game-conserved (< (distance ?h room_center) 1)))
    (forall (?d - dodgeball) (game-optional (on desk ?d)))
))
(:constraints (and
    ; PREFERENCE: count all dodgeballs thrown with the agent next to the desk, that land in the hexagonal bin.
    (preference dodgeballThrownToBinFromDesk (exists (?d - dodgeball ?h - hexagonal_bin)
        (then
            (once (and (adjacent agent desk) (agent_holds ?d)))
            (hold (and (in_motion ?d) (not (agent_holds ?d))))
            (once (and (not (in_motion ?d)) (in ?h ?d)))
        )
    ))
))
; SCORING: 1 point for each dodgeball throw that lands in the hexagonal bin.
(:scoring
    (count dodgeballThrownToBinFromDesk)
))


(define (game game-74) (:domain many-objects-room-v1)  ; 74
; SETUP: place a hexagonal bin and pillow within 3 units from each other for the entire game.
(:setup (and
    (game-conserved (exists (?h - hexagonal_bin ?p - pillow) (< (distance ?h ?p) 3)))
))
(:constraints (and
    ; PREFERENCE: count all golfballs thrown with the agent next to the setup pillow, that land in the hexagonal bin.
    (preference golfballInBinFromPillow (exists (?g - golfball ?h - hexagonal_bin ?p - pillow)
        (then
            (once (and (adjacent agent ?p) (agent_holds ?g) (is_setup_object ?p) ))
            (hold (and (in_motion ?g) (not (agent_holds ?g))))
            (once (and (not (in_motion ?g)) (in ?h ?g)))
        )
    ))
    ; PREFERENCE: count all golfball throws
    (preference throwAttempt (exists (?g - golfball)
        (then
            (once (agent_holds ?g))
            (hold (and (in_motion ?g) (not (agent_holds ?g))))
            (once (not (in_motion ?g)))
        )
    ))
))
; TERMINAL: the game ends after 10 golfball throws.
(:terminal
    (>= (count throwAttempt) 10)
)
; SCORING: 5 points for each golfball throw that lands in the hexagonal bin.
(:scoring
    (* 5 (count golfballInBinFromPillow))
))


(define (game game-75) (:domain few-objects-room-v1)  ; 75
(:constraints (and
    ; PREFERENCE: count all balls dropped into the hexagonal bin with the agent next to the bin, that land in the bin.
    (preference ballDroppedInBin (exists (?b - ball ?h - hexagonal_bin)
        (then
            (once (and (adjacent agent ?h) (agent_holds ?b)))
            (hold (and (in_motion ?b) (not (agent_holds ?b))))
            (once (and (not (in_motion ?b)) (in ?h ?b)))
        )
    ))
    ; PREFERENCE: count all ball drops with the agent next to the bin.
    (preference dropAttempt (exists (?b - ball ?h - hexagonal_bin)
        (then
            (once (and (adjacent agent ?h) (agent_holds ?b)))
            (hold (and (in_motion ?b) (not (agent_holds ?b))))
            (once (not (in_motion ?b)))
        )
    ))
))
; TERMINAL: the game ends after 5 ball drops or after 1 ball drop into the bin.
(:terminal (or
    (>= (count dropAttempt) 5)
    (>= (count ballDroppedInBin) 1)
))
; SCORING: 5 points for each ball drop into the bin.
(:scoring
    (* 5 (count ballDroppedInBin))
))


; TODO: from here

(define (game game-76) (:domain few-objects-room-v1)  ; 76
(:constraints (and
    (forall (?c - (either pink yellow)) (and
        ; PREFERENCEL count all cube blocks thrown with the agent standing on a rug spot of a particular color that land in the bin or in a building that is in the bin
        (preference blockToBinFromRug (exists (?b - cube_block ?h - hexagonal_bin)
            (then
                (once (and (agent_holds ?b) (rug_color_under agent ?c)))
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
        ; PREFERENCE: count all cube blocks thrown with the agent standing on a rug spot of a particular color
        (preference blockThrowAttempt (exists (?b - cube_block)
            (then
                (once (and (agent_holds ?b) (rug_color_under agent ?c)))
                (hold (and (in_motion ?b) (not (agent_holds ?b))))
                (once (not (in_motion ?b)))
            )
        ))
    ))
    ; PREFERENCE: count all dodgeball throws with the agent standing on a yellow spot in the rug that knock a block from a building in the hexagonal bin
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
    ; PREFERENCE: count all dodgeball throws with the agent standing on a yellow spot in the rug
    (preference ballThrowAttempt (exists (?d - dodgeball)
        (then
            (once (and (agent_holds ?d) (rug_color_under agent yellow)))
            (hold (and (in_motion ?d) (not (agent_holds ?d))))
            (once (not (in_motion ?d)))
        )
    ))
))
; TERMINAL: the game ends after more than 18 cube blocks are thrown and at least 2 dodgeballs are thrown.
(:terminal (and
    (> (count blockThrowAttempt) 18)
    (>= (count ballThrowAttempt) 2)
))
; SCORING: 10 points for each block thrown from a pink spot, 15 points for each block thrown from a yellow spot, 15 points for throwing all 6 blocks from a yellow spot,  15 points for succesfully throwing all 6 blocks in 18 or fewer attempts, and 20 points for each block knocked from a building in the bin with a dodgeball.
(:scoring (+
    (* 10 (count-once-per-objects blockToBinFromRug:pink))
    (* 15 (count-once-per-objects blockToBinFromRug:yellow))
    (* 15 (= (count-once-per-objects blockToBinFromRug:yellow) 6))
    (* 15 (<= (count blockThrowAttempt) 18) (= (count-once-per-objects blockToBinFromRug) 6))
    (* 20 (count-once-per-objects blockKnockedFromBuildingInBin))
)))


(define (game game-77) (:domain many-objects-room-v1)  ; 77
(:constraints (and
    ; PREFERENCE: count all dodgeballs thrown to the hexagonal bin, measuring the distance between the agent and the bin at the time of the throw.
    (preference throwToBinFromDistance (exists (?d - dodgeball ?h - hexagonal_bin)
        (then
            (once-measure (agent_holds ?d) (distance agent ?h))
            (hold (and (not (agent_holds ?d)) (in_motion ?d)))
            (once (and (not (in_motion ?d)) (in ?h ?d)))
        )
    ))
))
; SCORING: score each throw to the bin based on the distance measured.
(:scoring (count-measure throwToBinFromDistance)
))


(define (game game-78) (:domain medium-objects-room-v1)  ; 78
; SETUP: place all hexagonal bins and basketballs near the side table for the entire game. To start the game, place a teddy bear upright near the front left corner of the bed, and place the beachball on the floor near the front left corner of the bed.
(:setup (and
    (exists (?t - teddy_bear) (game-optional (and
        (adjacent_side bed front_left_corner ?t)
        (object_orientation ?t upright)
    )))
    (exists (?b - beachball) (game-optional (and
        (< (distance_side bed front_left_corner ?b) 1)
        (on floor ?b)
    )))
    (forall (?o - (either hexagonal_bin basketball))
        (game-conserved (< (distance ?o side_table) 1))
    )
))
(:constraints (and
    ; PREFERENCE: count all dodgeballs thrown with the agent near the doggie bed that move the beachball without knocking over the teddy bear.
    (preference throwMovesBeachballWithoutKnockingTeddy (exists (?d - dodgeball ?b - beachball ?t - teddy_bear ?db - doggie_bed)
        (then
            (once (and (agent_holds ?d) (< (distance agent ?db) 1) (object_orientation ?t upright)))
            (hold-while
                (and (in_motion ?d) (not (agent_holds ?d)) (not (agent_holds ?t)))
                (touch ?d ?b)
                (in_motion ?b)
            )
            (once (and (not (in_motion ?d)) (not (in_motion ?b)) (object_orientation ?t upright)))
        )
    ))
    ; PREFERENCE: count all dodgeballs thrown with the agent near the doggie bed that knock over the teddy bear.
    (preference throwKnocksOverBear (exists (?d - dodgeball ?b - beachball ?t - teddy_bear ?db - doggie_bed)
        (then
            (once (and (agent_holds ?d) (< (distance agent ?db) 1) (object_orientation ?t upright)))
            (hold (and (in_motion ?d) (not (agent_holds ?d)) (not (agent_holds ?t))))
            (once (and (not (in_motion ?d)) (not (in_motion ?b)) (not (object_orientation ?t upright))))
        )
    ))
))
; SCORING: 3 points for each throw that moves the beachball without knocking over the teddy bear, and -1 point for each throw that knocks over the teddy bear.
(:scoring (+
    (* 3 (count throwMovesBeachballWithoutKnockingTeddy))
    (- (count throwKnocksOverBear))
)))


(define (game game-79) (:domain many-objects-room-v1)  ; 79
(:constraints (and
    ; PREFERENCE: count all golfballs thrown to the hexagonal bin.
    (preference throwGolfballToBin (exists (?g - golfball ?h - hexagonal_bin)
        (then
            (once (agent_holds ?g))
            (hold (and (not (agent_holds ?g)) (in_motion ?g)))
            (once (and (not (in_motion ?g)) (in ?h ?g)))
        )
    ))
))
; SCORING: 1 point for each golfball throw to the bin.
(:scoring (count throwGolfballToBin)
))


(define (game game-80) (:domain few-objects-room-v1)  ; 80
(:constraints (and
    ; PREFERENCE: count all pink objects moved to the room center.
    (preference pinkObjectMovedToRoomCenter (exists (?o - game_object)
        (then
            (once (and (agent_holds ?o) (same_color ?o pink)))
            (hold (and (in_motion ?o) (not (agent_holds ?o))))
            (once (and (not (in_motion ?o)) (< (distance room_center ?o) 1)))
        )
    ))
    ; PREFERENCE: count all blue objects moved to the room center with a pink object already at the room center.
    (preference blueObjectMovedToRoomCenter (exists (?o - game_object)
        (then
            (once (and (agent_holds ?o) (same_color ?o blue)))
            (hold (and (in_motion ?o) (not (agent_holds ?o))))
            (once (and (not (in_motion ?o)) (< (distance room_center ?o) 1)
                (exists (?o1 - game_object) (and
                    (same_color ?o1 pink) (< (distance room_center ?o1) 1)
                ))
            ))
        )
    ))
    ; PREFERENCE: count all brown objects moved to the room center with pink and blue objects already at the room center.
    (preference brownObjectMovedToRoomCenter (exists (?o - game_object)
        (then
            (once (and (agent_holds ?o) (same_color ?o brown)))
            (hold (and (in_motion ?o) (not (agent_holds ?o))))
            (once (and (not (in_motion ?o)) (< (distance room_center ?o) 1)
                (exists (?o1 ?o2 - game_object) (and
                    (same_color ?o1 pink) (< (distance room_center ?o1) 1)
                    (same_color ?o2 blue) (< (distance room_center ?o2) 1)
                ))
            ))
        )
    ))
    ; PREFERENCE: count all pillows moved to the room center with pink, blue, and brown objects already at the room center.
    (preference pillowMovedToRoomCenter (exists (?o - pillow)
        (then
            (once (and (agent_holds ?o)))
            (hold (and (in_motion ?o) (not (agent_holds ?o))))
            (once (and (not (in_motion ?o)) (< (distance room_center ?o) 1)
                (exists (?o1 ?o2 ?o3 - game_object) (and
                    (same_color ?o1 pink) (< (distance room_center ?o1) 1)
                    (same_color ?o2 blue) (< (distance room_center ?o2) 1)
                    (same_color ?o3 brown) (< (distance room_center ?o3) 1)
                ))
            ))
        )
    ))
    ; PREFERENCE: count all green objects moved to the room center with pink, blue, and brown objects, and a pillow already at the room center.
    (preference greenObjectMovedToRoomCenter (exists (?o - game_object)
        (then
            (once (and (agent_holds ?o) (same_color ?o green)))
            (hold (and (in_motion ?o) (not (agent_holds ?o))))
            (once (and (not (in_motion ?o)) (< (distance room_center ?o) 1)
                (exists (?o1 ?o2 ?o3 ?o4 - game_object) (and
                    (same_color ?o1 pink) (< (distance room_center ?o1) 1)
                    (same_color ?o2 blue) (< (distance room_center ?o2) 1)
                    (same_color ?o3 brown) (< (distance room_center ?o3) 1)
                    (same_type ?o4 pillow) (< (distance room_center ?o4) 1)
                ))
            ))
        )
    ))
    ; PREFERENCE: count all tan objects moved to the room center with pink, blue, brown, and green objects, a pillow, already at the room center.
    (preference tanObjectMovedToRoomCenter (exists (?o - game_object)
        (then
            (once (and (agent_holds ?o) (same_color ?o tan)))
            (hold (and (in_motion ?o) (not (agent_holds ?o))))
            (once (and (not (in_motion ?o)) (< (distance room_center ?o) 1)
                (exists (?o1 ?o2 ?o3 ?o4 ?o5 - game_object) (and
                    (same_color ?o1 pink) (< (distance room_center ?o1) 1)
                    (same_color ?o2 blue) (< (distance room_center ?o2) 1)
                    (same_color ?o3 brown) (< (distance room_center ?o3) 1)
                    (same_type ?o4 pillow) (< (distance room_center ?o4) 1)
                    (same_color ?o5 green) (< (distance room_center ?o5) 1)
                ))
            ))
        )
    ))
))
; SCORING: count once for each color or type of object moved correctly to the room center.
(:scoring (+
    (count-once pinkObjectMovedToRoomCenter)
    (count-once blueObjectMovedToRoomCenter)
    (count-once brownObjectMovedToRoomCenter)
    (count-once pillowMovedToRoomCenter)
    (count-once greenObjectMovedToRoomCenter)
    (count-once tanObjectMovedToRoomCenter)
)))


(define (game game-81) (:domain many-objects-room-v1)  ; 81
; SETUP: place the hexagonal bin next to the desk, and the triangular and curvied wooden ramps near the bin for the entire game. Also place no other object next to the front of the bin for the entire game.
(:setup (and
    (exists (?h - hexagonal_bin ?r1 ?r2 - (either triangular_ramp curved_wooden_ramp))
        (game-conserved (and
            (adjacent ?h desk)
            (< (distance ?h ?r1) 1)
            (< (distance ?h ?r2) 1)
            (not (exists (?o - game_object) (adjacent_side ?h front ?o)))
        ))
    )
))
(:constraints (and
    ; PREFERENCE: count all dodgeballs thrown from the rug to the bin.
    (preference dodgeballFromRugToBin (exists (?d - dodgeball ?h - hexagonal_bin)
        (then
            (once (and (agent_holds ?d) (on rug agent)))
            (hold (and (in_motion ?d) (not (agent_holds ?d))))
            (once (and (not (in_motion ?d)) (in ?h ?d)))
        )
    ))
))
; TERMINAL: end the game when 3 or more dodgeballs have been thrown from the rug to the bin.
(:terminal
    (>= (count dodgeballFromRugToBin) 3)
)
; SCORING: 1 point for each dodgeball thrown from the rug to the bin.
(:scoring
    (count dodgeballFromRugToBin)
))


(define (game game-82) (:domain many-objects-room-v1)  ; 82
(:constraints (and
    ; PREFERENCE: count all balls thrown to the bin.
    (preference ballThrownToBin (exists (?b - ball ?h - hexagonal_bin)
        (then
            (once (agent_holds ?b))
            (hold (and (in_motion ?b) (not (agent_holds ?b))))
            (once (and (not (in_motion ?b)) (in ?h ?b)))
        )
    ))
))
; TERMINAL: end the game when 300 seconds have passed.
(:terminal
    (>= (total-time) 300)
)
; SCORING: 1 point for each ball thrown to the bin.
(:scoring
    (count ballThrownToBin)
))


(define (game game-83) (:domain many-objects-room-v1)  ; 83
; SETUP: place the hexagonal bin sideways between two chair for the entire game.
(:setup (and
    (exists (?h - hexagonal_bin ?c1 ?c2 - chair) (game-conserved (and
        (object_orientation ?h sideways)
        (between ?c1 ?h ?c2)
    )))
))
(:constraints (and
    (forall (?b - (either dodgeball golfball))
        ; PREFERENCE: count all dodgeballs and golfballs thrown from the bed to the bin.
        (preference ballToBinFromBed (exists (?h - hexagonal_bin)
            (then
                (once (and (agent_holds ?b) (adjacent bed agent)))
                (hold (and (in_motion ?b) (not (agent_holds ?b))))
                (once (and (not (in_motion ?b)) (in ?h ?b)))
            )
        ))
    )
))
; SCORING: 1 point for each dodgeball thrown from the bed to the bin, and if 3 or more dodgeballs have been thrown from the bed to the bin, 1 point for each golfball thrown from the bed to the bin.
(:scoring (+
    (count-once-per-objects ballToBinFromBed:dodgeball)
    (* (= (count-once-per-objects ballToBinFromBed:dodgeball) 3) (count-once-per-objects ballToBinFromBed:golfball))
)))

; 84 is a hiding game -- invalid


(define (game game-85) (:domain few-objects-room-v1)  ; 85
(:constraints (and
    (forall (?c - color)
        ; PREFERENCE: count all cube blocks thrown with the agent being on a pink spot on the rug with no other cube blocks in the bin, that land in the bin.
        (preference cubeThrownToBin (exists (?h - hexagonal_bin ?b - cube_block)
            (then
                (once (and
                    (agent_holds ?b)
                    (rug_color_under agent pink)
                    (same_color ?b ?c)
                    (not (exists (?ob - cube_block) (in ?h ?ob)))
                ))
                (hold (and (in_motion ?b) (not (agent_holds ?b))))
                (once (and (not (in_motion ?b)) (in ?h ?b)))
            )
        ))
    )
    (forall (?b - cube_block)
        ; PREFERENCE: count all cube blocks thrown with the agent being on a pink spot on the rug
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
; TERMINAL: the game ends when any cube block is thrown more than once, or when 6 or more different cube blocks have been thrown.
(:terminal (or
    (> (external-forall-maximize (count throwAttempt)) 1)
    (>= (count-once-per-objects throwAttempt) 6)
))
; SCORING: 1 point for each yellow cube block thrown in the bin, 2 points for each tan cube block thrown in the bin, 3 points for each blue cube block thrown in the bin, and -1 point for any throw attempt.
(:scoring (+
    (count-once-per-objects cubeThrownToBin:yellow)
    (* 2 (count-once-per-objects cubeThrownToBin:tan))
    (* 3 (count-once-per-objects cubeThrownToBin:blue))
    (- (count-once-per-objects throwAttempt))
)))

; 86 is a dup of 84 -- and is aldo invalid


(define (game game-87) (:domain few-objects-room-v1)  ; 87
; SETUP: place a hexagonal bin on the desk and adjacent to the wall for the entire game.
(:setup (and
    (exists (?h - hexagonal_bin ?w - wall) (game-conserved (and
        (on desk ?h)
        (adjacent ?h ?w)
    )))
))
(:constraints (and
    (forall (?o - (either dodgeball block))
        ; PREFERENCE: count all dodgeballs or blocks thrown from the rug to the bin.
        (preference basketMadeFromRug (exists (?h - hexagonal_bin)
            (then
                (once (and (agent_holds ?o) (on rug agent)))
                (hold (and (in_motion ?o) (not (agent_holds ?o))))
                (once (and (not (in_motion ?o)) (in ?h ?o)))
            )
        ))
    )
))
; SCORING: 1 point for each dodgeball thrown from the rug to the bin, and 2 points for each block thrown from the rug to the bin.
(:scoring (+
    (count basketMadeFromRug:dodgeball)
    (* 2 (count basketMadeFromRug:block))
)))


(define (game game-88) (:domain few-objects-room-v1)  ; 88
; SETUP: place the hexagonal bin on the bed, and a pillow at an angle of 45 degrees for the entire game. Also stack three cube blocks on each side of the hexagonal bin for the entire game.
(:setup (and
    (exists (?h - hexagonal_bin ?p - pillow ?b1 ?b2 ?b3 ?b4 ?b5 ?b6 - cube_block)
        (game-conserved (and
            (on bed ?h)
            (not (object_orientation ?p sideways))
            (not (object_orientation ?p upright))
            (not (object_orientation ?p upside_down))
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
))
(:constraints (and
    ; PREFERENCE: count throws of a dodgeball from the edge of the rug farthest from the bed that land in the bin.
    (preference throwFromEdgeOfRug (exists (?d - dodgeball ?h - hexagonal_bin)
        (then
            (once (and
                (agent_holds ?d)
                (on floor agent)
                (adjacent rug agent)
                (> (distance agent bed) 2)
            ))
            (hold (and (in_motion ?d) (not (agent_holds ?d))))
            (once (and (not (in_motion ?d)) (in ?h ?d)))
        )
    ))
    ; PREFERENCE: count throws of a dodgeball from the edge of the rug farthest from the bed
    (preference throwAttempt (exists (?d - dodgeball)
        (then
            (once (and
                (agent_holds ?d)
                (on floor agent)
                (adjacent rug agent)
                (> (distance agent bed) 2)
            ))
            (hold (and (in_motion ?d) (not (agent_holds ?d))))
            (once (not (in_motion ?d)))
        )
    ))
    ; PREFERENCE: count throws of a dodgeball from the edge of the rug farthest from the bed that knock a cube block
    (preference throwAttemptKnocksBlock (exists (?d - dodgeball ?c - cube_block)
        (then
            (once (and
                (agent_holds ?d)
                (on floor agent)
                (adjacent rug agent)
                (> (distance agent bed) 2)
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
; TERMINAL: the game ends after 10 or more throws, or after at least one throw knocks a cube block, or after the total score is at least 5.
(:terminal (or
    (>= (count throwAttempt) 10)
    (>= (count-once throwAttemptKnocksBlock) 1)
    (>= (total-score) 5)
))
; SCORING: 1 point for each throw that lands in the bin
(:scoring
    (count throwFromEdgeOfRug)
))


(define (game game-89) (:domain medium-objects-room-v1)  ; 89
; SETUP: place a hexagonal bin on the dek and the desktop off the desk for the entire game.
(:setup (and
    (exists (?d - desktop ?h - hexagonal_bin) (game-conserved (and
        (on desk ?h)
        (not (on desk ?d))
    )))
))
(:constraints (and
    (forall (?b - ball)
        ; PREFERENCE: count all balls thrown from the rug to the bin.
        (preference ballThrownFromRug (exists (?h - hexagonal_bin)
            (then
                (once (and (agent_holds ?b) (on rug agent)))
                (hold (and (in_motion ?b) (not (agent_holds ?b))))
                (once (and (not (in_motion ?b)) (in ?h ?b)))
            )
        ))
    )
))
; TERMINAL: the game ends after 180 seconds or after the total score is at least 10.
(:terminal (or
    (>= (total-time) 180)
    (>= (total-score) 10)
))
; SCORING: 1 point for each dodgeball thrown from the rug to the bin, and 2 points for each basketball thrown from the rug to the bin, and 10 points for each beachball thrown from the rug to the bin.
(:scoring (+
    (count ballThrownFromRug:dodgeball)
    (* 2 (count ballThrownFromRug:basketball))
    (* 10 (count ballThrownFromRug:beachball))
)))



(define (game game-90) (:domain many-objects-room-v1)  ; 90
(:constraints (and
    ; PREFERENCE: count dodgeball throws that bounce exactly once on the floor and then land on the doggie bed
    (preference dodgeballBouncesOnceToDoggieBed (exists (?d - dodgeball ?b - doggie_bed)
        (then
            (once (agent_holds ?d))
            (hold (and (in_motion ?d) (not (agent_holds ?d)) (not (touch floor ?d))))
            (once (touch floor ?d))
            (hold (and (in_motion ?d) (not (agent_holds ?d)) (not (touch floor ?d))))
            (once (and (not (in_motion ?d)) (on ?b ?d)))
        )
    ))
))
; SCORING: 1 point for each throw that bounces exactly once on the floor and then lands on the doggie bed
(:scoring
    (count dodgeballBouncesOnceToDoggieBed)
))

; 91 is a dup of 89 with slightly different scoring numbers

; 92 is a hiding game -- invalid


(define (game game-93) (:domain many-objects-room-v1)  ; 93
(:constraints (and
    ; PREFERENCE: count dodgeball throws that land in the bin
    (preference throwBallToBin (exists (?d - dodgeball ?h - hexagonal_bin)
        (then
            (once (agent_holds ?d))
            (hold (and (not (agent_holds ?d)) (in_motion ?d)))
            (once (and (not (in_motion ?d)) (in ?h ?d)))
        )
    ))
))
; SCORING: 1 point for each throw that lands in the bin
(:scoring
    (count throwBallToBin)
))


(define (game game-94) (:domain many-objects-room-v1)  ; 94
(:constraints (and
    (forall (?b - (either dodgeball golfball)) (and
        ; PREFERENCE: count dodgeball or golfball throws with the agent next to the door that land in the bin
        (preference ballThrownFromDoor (exists (?h - hexagonal_bin)
            (then
                (once (and (agent_holds ?b) (adjacent door agent)))
                (hold (and (in_motion ?b) (not (agent_holds ?b))))
                (once (and (not (in_motion ?b)) (in ?h ?b)))
            )
        ))
        ; PREFERENCE: count dodgeball or golfball throws with the agent next to the door
        (preference throwAttemptFromDoor
            (then
                (once (and (agent_holds ?b) (adjacent door agent)))
                (hold (and (in_motion ?b) (not (agent_holds ?b))))
                (once (not (in_motion ?b)))
            )
        )
    ))
))
; TERMINAL: the game ends after 8 throws from the door
(:terminal
    (>= (count throwAttemptFromDoor) 8)
)
; SCORING: 3 points for each dodgeball thrown from the door, and 6 points for each golfball thrown from the door
(:scoring (+
    (* 3 (count ballThrownFromDoor:dodgeball))
    (* 6 (count ballThrownFromDoor:golfball))
)))

; 95 requires counting something that happens during a preference

; 96 requires is underconstrainted -- I'm omitting it for now


(define (game game097-97) (:domain medium-objects-room-v1)  ; 97
(:constraints (and
    ; PREFERENCE: count red dodgeball throws with the agent not on the rug that land on the rug
    (preference ballThrownToRug (exists (?d - red_dodgeball)
        (then
            (once (and (agent_holds ?d) (not (on rug agent))))
            (hold (and (in_motion ?d) (not (agent_holds ?d))))
            (once (and (not (in_motion ?d)) (on rug ?d)))
        )
    ))
))
; TERMINAL: the game ends after 60 seconds
(:terminal
    (>= (total-time) 60)
)
; SCORING: 1 point for each red dodgeball thrown with the agent not on the rug to the rug
(:scoring
    (count ballThrownToRug)
))


(define (game game0-98) (:domain medium-objects-room-v1)  ; 98
; SETUP: place the hexagonal bin such that there are no shelves above it for the entire game. To start the game, place all balls on the bed.
(:setup (and
    (exists (?h - hexagonal_bin) (game-conserved (not (exists (?s - shelf) (above ?h ?s)))))
    (forall (?b - ball) (game-optional (on bed ?b)))
))
(:constraints (and
    (forall (?b - ball)
        ; PREFERENCE: count ball throws with the agent on the bed or adjacent to the bed that land in the bin
        (preference ballThrownToBin (exists (?h - hexagonal_bin)
            (then
                (once (and (agent_holds ?b) (or (on bed agent) (adjacent bed agent))))
                (hold (and (in_motion ?b) (not (agent_holds ?b))))
                (once (and (not (in_motion ?b)) (in ?h ?b)))
            )
        ))
    )
))
; TERMINAL: the game ends once the total score is 6 or more points.
(:terminal
    (>= (total-score) 6)
)
; SCORING: 1 point for each beachball thrown from the bed or adjacent to the bed that lands in the bin, 2 points for each basketball thrown, and 3 points for each dodgeball thrown.
(:scoring (+
    (count-once-per-objects ballThrownToBin:beachball)
    (* 2 (count-once-per-objects ballThrownToBin:basketball))
    (* 3 (count-once-per-objects ballThrownToBin:dodgeball))
)))


(define (game game-99) (:domain few-objects-room-v1)  ; 99
(:constraints (and
    ; PREFERENCE: count cube blocks thrown with the agent next to the bed that land on a shelf
    (preference cubeBlockFromBedToShelf (exists (?c - cube_block ?s - shelf)
        (then
            (once (and (agent_holds ?c) (adjacent bed agent)))
            (hold (and (in_motion ?c) (not (agent_holds ?c))))
            (once (and (not (in_motion ?c)) (on ?s ?c)))
        )
    ))
    ; PREFERENCE: count cube blocks thrown with the agent next to the bed
    (preference cubeBlockThrowAttempt (exists (?c - cube_block)
        (then
            (once (and (agent_holds ?c) (adjacent bed agent)))
            (hold (and (in_motion ?c) (not (agent_holds ?c))))
            (once (not (in_motion ?c)))
        )
    ))
))
; TERMINAL: the game ends after 3 cube block throws
(:terminal
    (>= (count cubeBlockThrowAttempt) 3)
)
; SCORING: 1 point for each cube block thrown with the agent next to the bed that lands on a shelf
(:scoring
    (count cubeBlockFromBedToShelf)
))

(define (game game-100) (:domain medium-objects-room-v1)  ; 100
; SETUP: place the doggie bed on the floor, and the hexagonal bed on the bed, in a line with the same z position.
(:setup (and
    (exists (?h - hexagonal_bin ?d - doggie_bed) (game-conserved (and
        (on floor ?d)
        (on bed ?h)
        (equal_z_position ?h ?d)
    )))
))
(:constraints (and
    (forall (?t - (either hexagonal_bin doggie_bed))
        ; PREFERENCE: count dodgeballs thrown with the agent next to the desk that land on the doggie bed or hexagonal bin;
        (preference dodgeballFromDeskToTarget (exists (?d - dodgeball)
            (then
                (once (and (agent_holds ?d) (adjacent desk agent)))
                (hold (and (in_motion ?d) (not (agent_holds ?d))))
                (once (and (not (in_motion ?d)) (or (in ?t ?d) (on ?t ?d))))
            )
        ))
    )
))
; SCORING: 2 points for each dodgeball thrown with the agent next to the desk that lands on the doggie bed, and 3 points for each dodgeball thrown with the agent next to the desk that lands on the hexagonal bin.
(:scoring (+
    (* 2 (count dodgeballFromDeskToTarget:doggie_bed))
    (* 3 (count dodgeballFromDeskToTarget:hexagonal_bin))
)))


(define (game game-101) (:domain few-objects-room-v1)  ; 101
; SETUP: place the hexagonal bin on the bed, and place the curved wooden ramp adjacent to the bed and facing the desk for the entire game. Place the blue cube blocks 1 unit away from the desk, and the yellow cube blocks 2 units away from the desk. The blue cube blocks must be between the desk and the yellow cube blocks.
(:setup (and
    (exists (?h - hexagonal_bin) (game-conserved (on bed ?h)))
    (exists (?r - curved_wooden_ramp) (game-conserved (and (adjacent bed ?r) (faces ?r desk))))
    (exists (?c1 ?c2 - blue_cube_block ?c3 ?c4 - yellow_cube_block) (game-conserved (and
        (= (distance ?c1 desk) 1)
        (= (distance ?c2 desk) 1)
        (= (distance ?c3 desk) 2)
        (= (distance ?c4 desk) 2)
        (between desk ?c1 ?c3)
        (between desk ?c2 ?c4)
    )))
))
(:constraints (and
    (forall (?c - (either blue_cube_block yellow_cube_block)) (and
        ; PREFERENCE: count balls thrown from behind a blue or a yellow cube block that land in the hexagonal bin
        (preference ballThrownFromBehindBlock (exists (?b - ball ?h - hexagonal_bin)
            (then
                (once (and
                    (agent_holds ?b)
                    (is_setup_object ?c)
                    (>= (distance agent ?h) (distance ?c ?h))
                ))
                (hold (and (in_motion ?b) (not (agent_holds ?b))))
                (once (and (not (in_motion ?b)) (in ?h ?b)))
            )
        ))
        ; PREFERENCE: count balls thrown from behind a blue or a yellow cube block
        (preference throwAttemptFromBehindBlock (exists (?b - ball ?h - hexagonal_bin)
            (then
                (once (and
                    (agent_holds ?b)
                    (is_setup_object ?c)
                    (>= (distance agent ?h) (distance ?c ?h))
                ))
                (hold (and (in_motion ?b) (not (agent_holds ?b))))
                (once (and (not (in_motion ?b))))
            )
        ))
    ))
))
; TERMINAL: the game ends after 2 throws from behind a blue or a yellow cube block, or after 50 or more points are scored.
(:terminal (or
    (>= (count throwAttemptFromBehindBlock) 2)
    (>= (total-score) 50)
))
; SCORING: 10 points for each ball thrown from behind a blue cube block that lands in the hexagonal bin, 5 points for each ball thrown from behind a yellow cube block that lands in the  bin, 30 bonbus points for throwing 2 balls from behind a blue cube block, and 15 bonus points for throwing 2 balls from behind a yellow cube block.
(:scoring (+
    (* 10 (count ballThrownFromBehindBlock:blue_cube_block))
    (* 5 (count ballThrownFromBehindBlock:yellow_cube_block))
    (* 30 (= (count ballThrownFromBehindBlock:blue_cube_block) 2))
    (* 15 (= (count ballThrownFromBehindBlock:yellow_cube_block) 2))
)))


; 102 is almost a copy of 101 and same participant -- omit


(define (game game-103) (:domain few-objects-room-v1)  ; 103
; SETUP: place the hexagonal bin sideways on the bed for the entire game.
(:setup (and
    (exists (?h - hexagonal_bin) (game-conserved (and
        (on bed ?h)
        (object_orientation ?h sideways)
    )))
))
(:constraints (and
    ; PREFERENCE: count dodgeballs thrown that hit the bin
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
    ; PREFERENCE: count dodgeballs thrown that hit the bottom of the bin by being inside it at least for a moment
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
    ; PREFERENCE: count any dodgeball throw attempts
    (preference throwAttempt (exists (?d - dodgeball)
        (then
            (once (agent_holds ?d))
            (hold (and (in_motion ?d) (not (agent_holds ?d))))
            (once (and (not (in_motion ?d))))
        )
    ))
))
; TERMINAL: the game ends after 10 or more throw attempts.
(:terminal
    (>= (count throwAttempt) 10)
)
; SCORING: 1 point for each dodgeball that hits the bin, 2 points for each dodgeball that hits the bottom of the bin.
(:scoring (+
    (count dodgeballHitsBin)
    (* 2 (count dodgeballHitsBinBottom))
)))


(define (game game-104) (:domain few-objects-room-v1)  ; 104
; SETUP: place the hexagonal bin aligned with the east sliding door for the entire game.
(:setup (and
    (exists (?h - hexagonal_bin) (game-conserved (and
        (equal_x_position ?h east_sliding_door)
    )))
))
(:constraints (and
    ; PREFERENCE: count dodgeballs thrown with the agent at the edge of the rug land in the bin or bounce out of it
    (preference throwFromEdgeOfRug (exists (?d - dodgeball ?h - hexagonal_bin)
        (then
            (once (and (agent_holds ?d) (adjacent rug agent)))
            (hold (and (in_motion ?d) (not (agent_holds ?d))))
            (once (in ?h ?d))  ; participant specified that couning out is okay
        )
    ))
))
; TERMINAL: the game ends after 300 seconds.
(:terminal
    (>= (total-time) 300)
)
; SCORING: 1 point for each dodgeball thrown from the edge of the rug
(:scoring
    (count throwFromEdgeOfRug)
))


(define (game game-105) (:domain few-objects-room-v1)  ; 105
; SETUP: remove all dodgeballs from all shelves, and place all chairs on or next to the bed for the entire game. To start the game, place all cube blocks on the rug, and make sure that no objects are below the desk.
(:setup (and
    (forall (?c - chair) (game-conserved (or (on bed ?c) (adjacent bed ?c))))
    (forall (?c - cube_block) (game-optional (on rug ?c)))
    (game-optional (not (exists (?o - game_object) (above ?o desk))))
    (forall (?d - dodgeball) (game-conserved (not (exists (?s - shelf) (on ?s ?d)))))
))
(:constraints (and
    ; PREFERENCE: count all tan cube blocks that are moved from the rug to the desk, by creating a path to the desk using the blue and yellow cube blocks placed close to each other, and then moving the tan cube block to the desk.
    (preference woodenBlockMovedFromRugToDesk (exists (?b - tan_cube_block)
        (then
            (once (and
                (forall (?c - (either blue_cube_block yellow_cube_block)) (on rug ?c))
                (on rug ?b)
            ))
            (hold (forall (?c - (either blue_cube_block yellow_cube_block)) (or
                (on rug ?c)
                (agent_holds ?c)
                (in_motion ?c)
                (exists (?c2 - (either blue_cube_block yellow_cube_block)) (and
                    (not (same_object ?c ?c2))
                    (< (distance ?c ?c2) 0.5)
                    (on floor ?c)
                    (on floor ?c2)
                ))
            )))
            (hold (forall (?c - (either blue_cube_block yellow_cube_block))
                (< (distance desk ?c) 1)
            ))
            (once (above ?b desk))
        )
    ))
))
; SCORING: 1 point for each tan cube block that is moved from the rug to the desk.
(:scoring
    (count-once-per-objects woodenBlockMovedFromRugToDesk)
))


(define (game game-106) (:domain few-objects-room-v1)  ; 106
(:constraints (and
    ; PREFERENCE: count all ball throws that land in the bin
    (preference throwInBin (exists (?b - ball ?h - hexagonal_bin)
        (then
            (once (agent_holds ?b))
            (hold (and (not (agent_holds ?b)) (in_motion ?b)))
            (once (and (not (in_motion ?b)) (in ?h ?b)))
        )
    ))
    ; PREFERENCE: count all ball throws
    (preference throwAttempt (exists (?b - ball)
        (then
            (once (agent_holds ?b))
            (hold (and (not (agent_holds ?b)) (in_motion ?b)))
            (once (not (in_motion ?b)))
        )
    ))
))
; TERMINAL: the game ends after 15 or more throw attempts, or when the agent has scored 6 or more points.
(:terminal (or
    (>= (total-score) 6)
    (>= (count throwAttempt) 15)
))
; SCORING: 1 point for each ball throw that lands in the bin.
(:scoring
    (count throwInBin)
))

; 107 and 109 are by the same participant, and 109 is actually mostly valid


(define (game game-108) (:domain medium-objects-room-v1)  ; 108
; SETUP: place a cylindrical block on the side table. Place both tall cylndrical blocks on the bed, with a pyramid block on each one, on both ends of the bed, one adjacent to the north wall. Place the hexagonal bin on the middle of the bed between the tall cylindrical blocks. To start the game, place all balls on or near the doggie bed.
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
        (< (distance ?d ?b) 0.5)
    ))))
))
(:constraints (and
    ; PREFERENCE: count whether the agent left the vicinity of the doggie bed or ran out of balls near the doggie bed.
    (preference agentLeavesDogbedOrNoMoreBalls (exists (?d - doggie_bed)
        (then
            (hold (<= (distance ?d agent) 1))
            (once (or
                (> (distance ?d agent) 1)
                (forall (?b - ball) (and
                    (not (in_motion ?b))
                    (> (distance agent ?b) 1))
                )
            ))
        )
    ))
    (forall (?c - (either cylindrical_block tall_cylindrical_block pyramid_block))
        ; PREFERENCE: count all ball throws with the agent near the doggie bed that hit a cylindrical block, a tall cylindrical block, or a pyramid block used in the setup.
        (preference throwKnocksBlock (exists (?b - ball ?d - doggie_bed)
            (then
                (once (and
                    (is_setup_object ?c)
                    (agent_holds ?b)
                    (<= (distance ?d agent) 1)
                ))
                (hold-while
                    (and (in_motion ?b) (not (agent_holds ?b)))
                    (touch ?b ?c)
                    (in_motion ?c)
                )
            )
        ))
    )
    (forall (?b - ball)
        ; PREFERENCE: count all balls thrown with the agent near the doggie bed that land on or in the bin.
        (preference ballInOrOnBin (exists (?d - doggie_bed ?h - hexagonal_bin)
            (then
                (once (and
                    (agent_holds ?b)
                    (<= (distance ?d agent) 1)
                ))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (not (in_motion ?b)) (or (in ?h ?b) (on ?h ?b))))
            )
        ))
    )
))
; TERMINAL: the game ends when the agent leaves the vicinity of the doggie bed or runs out of balls near the doggie bed.
(:terminal
    (>= (count-once agentLeavesDogbedOrNoMoreBalls) 1)
)
; SCORING: 3 points for each ball throw that hits a pyramid block, -3 poitns for each ball throw that hits a tall cylindrical block, 1 point for each ball throw that hits a cylindrical block, 2 points for each dodgeball or basketball that lands in the bin, and 4 points for each beachball that lands on the bin.
(:scoring (+
    (* 3 (count-once-per-external-objects throwKnocksBlock:pyramid_block))
    (* (- 3) (count-once-per-external-objects throwKnocksBlock:tall_cylindrical_block))
    (count-once-per-external-objects throwKnocksBlock:cylindrical_block)
    (* 2 (count-once-per-external-objects ballInOrOnBin:dodgeball))
    (* 2 (count-once-per-external-objects ballInOrOnBin:basketball))
    (* 4 (count-once-per-external-objects ballInOrOnBin:beachball))
)))


(define (game game-109) (:domain many-objects-room-v1)  ; 109
(:constraints (and
    ; PREFERENCE: count all balls thrown that land in the bin
    (preference ballThrownToBin (exists (?b - ball ?h - hexagonal_bin)
        (then
            (once (agent_holds ?b))
            (hold (and (not (agent_holds ?b)) (in_motion ?b)))
            (once (and (not (in_motion ?b)) (in ?h ?b)))
        )
    ))
    ; PREFERENCE: count all cube blocks thrown that land on the top shelf
    (preference cubeBlockThrownToTopShelf (exists (?c - cube_block)
        (then
            (once (agent_holds ?c))
            (hold (and (not (agent_holds ?c)) (in_motion ?c)))
            (once (and (not (in_motion ?c)) (on top_shelf ?c)))
        )
    ))
    ; PREFERENCE: count all pillows thrown that land on the doggie bed
    (preference pillowThrownToDoggieBed (exists (?p - pillow ?d - doggie_bed)
        (then
            (once (agent_holds ?p))
            (hold (and (not (agent_holds ?p)) (in_motion ?p)))
            (once (and (not (in_motion ?p)) (on ?d ?p)))
        )
    ))
))
; SCORING: 1 point for each ball thrown that lands in the bin, 1 point for each cube block thrown that lands on the top shelf, and 1 point for each pillow thrown that lands on the doggie bed.
(:scoring (+
    (count-once-per-objects ballThrownToBin)
    (count-once-per-objects cubeBlockThrownToTopShelf)
    (count-once-per-objects pillowThrownToDoggieBed)
)))


(define (game game-110) (:domain few-objects-room-v1)  ; 110
; SETUP: place all chairs aligned with the door for the entire game. Place a hexagonal bin adjacent to and facing the southwest coener of the room for the entire game. To start the game, place all dodgeballs, cube blocks, alarm clocks, and books next to the desk.
(:setup (and
    (forall (?c - chair) (game-conserved (equal_x_position ?c door)))
    (exists (?h - hexagonal_bin) (game-conserved (and
        (adjacent ?h south_west_corner)
        (faces ?h south_west_corner)
    )))
    (forall (?o - (either dodgeball cube_block alarm_clock book)) (game-optional (adjacent ?o desk)))
))
(:constraints (and
    (forall (?o - (either dodgeball cube_block alarm_clock book)) (and
        ; PREFERENCE: count all dodgeball, cube block, alarm clock, or book throws with the agent behind all chairs that land in the bin.
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
        ; PREFERENCE: count all dodgeball, cube block, alarm clock, or book throws with the agent behind all chairs
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
; TERMINAL: the game ends when any dodgeball is thrown more than three times, or when any cube block, alarm clock, or book is thrown more than once.
(:terminal (or
    (> (external-forall-maximize (count throwAttempt:dodgeball)) 3)
    (> (external-forall-maximize (count throwAttempt:cube_block)) 1)
    (> (count throwAttempt:book) 1)
    (> (count throwAttempt:alarm_clock) 1)
))
; SCORING: 8 points for each dodgeball thrown that lands in the bin, 5 points for each cube block thrown that lands in the bin, 20 points for each alarm clock thrown that lands in the bin, and 50 points for each book thrown that lands in the bin.
(:scoring (+
    (* 8 (count throwFromBehindChairsInBin:dodgeball))
    (* 5 (count throwFromBehindChairsInBin:cube_block))
    (* 20 (count throwFromBehindChairsInBin:alarm_clock))
    (* 50 (count throwFromBehindChairsInBin:book))
)))

; 111 requires evaluation that one preference takes place before another preference is evaluated, and it's underconstrained

; 112 is definitely invalid and underdefined

(define (game game-113) (:domain few-objects-room-v1)  ; 113
; SETUP: place a cube block next to the front of the hexagonal bin, and another cube block next to it. Place a cube block on top of each of the first two cube blocks. Place a curved wooden ramp with its back to the further wooden blocks.
(:setup (and
    (exists (?h - hexagonal_bin ?c1 ?c2 ?c3 ?c4 - cube_block ?r - curved_wooden_ramp) (game-conserved (and
        (adjacent_side ?h front ?c1)
        (adjacent ?c1 ?c3)
        (between ?h ?c1 ?c3)
        (on ?c1 ?c2)
        (on ?c3 ?c4)
        (adjacent_side ?r back ?c3)
        (between ?r ?c3 ?c1)
    )))
))
(:constraints (and
    ; PREFERENCE: count all ball throws that land in the bin after passing through the ramp and the two cube blocks.
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
; SCORING: 1 point for each ball thrown that lands in the bin after passing through the ramp and the two cube blocks.
(:scoring
    (count ballThrownThroughRampAndBlocksToBin)
))


(define (game game-114) (:domain medium-objects-room-v1)  ; 114
; SETUP: place a doggie bed in the center of the room.
(:setup (and
    (exists (?d - doggie_bed) (game-conserved (< (distance room_center ?d) 0.5)))
))
(:constraints (and
    ; PREFERENCE: count all game objects that are in a building on the doggie bed, excluding the doggie bed itself, without touching the floor or any walls.
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
; SCORING: 1 point for each game object that is in a building on the doggie bed, excluding the doggie bed itself, without touching the floor or any walls.
(:scoring
    (count-once-per-objects objectInBuilding)
))


(define (game game-115) (:domain medium-objects-room-v1)  ; 115
; SETUP:  place the triangular ramp near the center of the room, with a chair in front of it, and a hexagonal bin on the other side of the chair, such that the chair is between the ramp and the bin, for the entire game. Place all balls in front of the hexagonal bin for the entire game. To start the game, place the teddy bair on the chair.
(:setup (and
    (exists (?c - chair ?r - triangular_ramp ?t - teddy_bear ?h - hexagonal_bin) (and
        (game-conserved (and
            (< (distance room_center ?r) 0.5)
            (adjacent_side ?r front ?c)
            (between ?h ?c ?r)
            (forall (?b - ball) (< (distance ?b ?h) 1))
        ))
        (game-optional (and
            (on ?c ?t)
        ))
    ))
))
(:constraints (and
    ; PREFERENCE: count all throws of a teddy bear, picked up from the chair, that lands in the bin.
    (preference teddyBearLandsInBin (exists (?t - teddy_bear ?h - hexagonal_bin ?c - chair)
        (then
            (once (on ?c ?t))
            (hold (agent_holds ?t))
            (hold (and (not (agent_holds ?t)) (in_motion ?t)))
            (once (and (not (in_motion ?t)) (in ?h ?t)))
        )
    ))
    ; PREFERENCE: count all throws of a teddy bear, picked up from the chair, that touch a ball.
    (preference teddyBearHitsBall (exists (?t - teddy_bear ?b - ball ?c - chair)
        (then
            (once (on ?c ?t))
            (hold (agent_holds ?t))
            (hold (and (not (agent_holds ?t)) (in_motion ?t)))
            (once (touch ?t ?b))
        )
    ))
))
; SCORING: 5 points for each teddy bear throw that lands in the bin, and 1 point for each teddy bear throw that hits a ball.
(:scoring (+
    (* 5 (count teddyBearLandsInBin))
    (count teddyBearHitsBall)
)))

(define (game game-116) (:domain medium-objects-room-v1)  ; 116
; SETUP: place the hexagonal bin on the bed or desk for the entire game.
(:setup (and
    (exists (?h - hexagonal_bin) (game-conserved (or (on bed ?h) (on desk ?h))))
))
(:constraints (and
    (forall (?b - (either basketball dodgeball)) (and
        ; PREFERENCE: count all basketball or dodgeball throws that land in the bin.
        (preference ballThrownToBin (exists (?h - hexagonal_bin)
            (then
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (and (not (in_motion ?b)) (in ?h ?b)))
            )
        ))
        ; PREFERENCE: count all basketball or dodgeball throws
        (preference throwAttempt
            (then
                (once (agent_holds ?b))
                (hold (and (not (agent_holds ?b)) (in_motion ?b)))
                (once (not (in_motion ?b)))
            )
        )
    ))
))
; TERMINAL: the game ends when the agent has thrown any ball more than four times.
(:terminal
    (> (external-forall-maximize (count throwAttempt)) 4)
)
; SCORING: 1 point for each basketball or dodgeball thrown to the bin.
(:scoring
    (count ballThrownToBin)
))


(define (game game-117) (:domain medium-objects-room-v1)  ; 117
; SETUP: place the triangular ramp near close to the hexagonal bin for the entire game.
(:setup (and
    (exists (?h - hexagonal_bin ?r - triangular_ramp) (game-conserved (< (distance ?h ?r) 2)))
))
(:constraints (and
    ; PREFERENCE: count throws of a red dodgeball that land in the bin without touching the floor.
    (preference redDodgeballThrownToBinWithoutTouchingFloor (exists (?h - hexagonal_bin ?r - red_dodgeball)
        (then
            (once (agent_holds ?r))
            (hold (and (not (agent_holds ?r)) (in_motion ?r) (not (touch floor ?r))))
            (once (and (not (in_motion ?r)) (in ?h ?r)))
        )
    ))
    ; PREFERENCE: count throws of a red dodgeball that land in the bin.
    (preference redDodgeballThrownToBin (exists (?h - hexagonal_bin ?r - red_dodgeball)
        (then
            (once (agent_holds ?r))
            (hold (and (not (agent_holds ?r)) (in_motion ?r)))
            (once (and (not (in_motion ?r)) (in ?h ?r)))
        )
    ))
    ; PREFERENCE: count all throws of a red dodgeball.
    (preference throwAttempt (exists (?r - red_dodgeball)
        (then
            (once (agent_holds ?r))
            (hold (and (not (agent_holds ?r)) (in_motion ?r)))
            (once (not (in_motion ?r)))
        )
    ))
))
; TERMINAL: the game ends when the agent has thrown the red dodgeball more than ten times, or when the agent has thrown the red dodgeball at least once and it has landed in the bin.
(:terminal (or
    (>= (count throwAttempt) 10)
    (>= (count-once redDodgeballThrownToBinWithoutTouchingFloor) 1)
    (>= (count-once redDodgeballThrownToBin) 1)
))
; SCORING: 5 points for each red dodgeball throw that lands in the bin, plus 3 bonus points for landing it in the bin without touching the floor on the first throw, and 2 bonus points for landing it in the bin without touching the floor in the first five throws.
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
    (forall (?c - color)
        ; PREFERENCE: count all pairs of objects with matching colors that either on each other, adjacent to each other, or in each other.
        (preference objectWithMatchingColor (exists (?o1 ?o2 - game_object)
            (at-end (and
                (same_color ?o1 ?o2)
                (same_color ?o1 ?c)
                (or
                    (on ?o1 ?o2)
                    (adjacent ?o1 ?o2)
                    (in ?o1 ?o2)
                )
            ))
        ))
    )
    ; PREFERENCE: count if any of the main light switch or lamp are toggled off at the end of the game.
    (preference itemsTurnedOff
        (exists (?o - (either main_light_switch lamp))
            (at-end
                (not (toggled_on ?o))
            )
        )
    )
    ; PREFERENCE: count if any of the objects are broken at the end of the game.
    (preference itemsBroken
        (exists (?o - game_object)
            (at-end
                (broken ?o)
            )
        )
    )
))
; SCORING: 5 point for each pair of objects with matching colors are are together, 5 bonus points for each pair that are green or brown that are together, 15 points for each object that is toggled off, and -10 points for each object that is broken.
(:scoring (+
    (* 5 (count-once-per-objects objectWithMatchingColor))
    (* 5 (count-once-per-objects objectWithMatchingColor:green))
    (* 5 (count-once-per-objects objectWithMatchingColor:brown))
    (* 15 (count-once-per-objects itemsTurnedOff))
    (* -10 (count-once-per-objects itemsBroken))
)))

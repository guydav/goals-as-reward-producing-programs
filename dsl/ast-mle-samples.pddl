(define (game game-id-0) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (and
      (and
        (rug_color_under ?xxx)
        (not
          (in_motion ?xxx bed)
        )
      )
      (agent_holds ?xxx)
    )
  )
)
(:constraints
  (and
    (forall (?f - hexagonal_bin ?f - beachball)
      (and
        (preference preference1
          (then
            (hold-while (touch ?f) (on ?f) )
            (once (in_motion ?f) )
            (hold (and (and (not (and (in_motion ?f ?f) (agent_holds ?f) ) ) (agent_holds ?f) ) (on ?f ?f) ) )
          )
        )
        (preference preference2
          (exists (?q - beachball ?u - hexagonal_bin)
            (then
              (once (not (and (not (and (not (and (on ?f) (> (distance desk ?u ?f) 2) ) ) (and (touch pink ?f) (not (and (and (not (not (agent_holds ?u ?u) ) ) (in_motion side_table floor) ) (not (on ?f) ) ) ) ) ) ) (and (in_motion ?f) (not (< (distance front ?f) (distance ?u ?f)) ) ) ) ) )
              (once (not (in_motion ?u) ) )
              (forall-sequence (?y - shelf)
                (then
                  (hold (not (agent_holds ?f) ) )
                  (once (< (distance agent 0) 4) )
                  (any)
                )
              )
            )
          )
        )
      )
    )
    (preference preference3
      (exists (?u ?q - dodgeball)
        (then
          (hold (and (on ?u ?q ?u) (not (not (not (on ?u) ) ) ) ) )
          (once (adjacent ?q) )
          (once (same_object agent) )
        )
      )
    )
  )
)
(:terminal
  (> (external-forall-maximize (+ (count preference3:cube_block) (+ (+ (* (- (count preference1:basketball) )
              6
            )
            (count preference3:basketball:dodgeball)
            (count preference2:golfball)
            (count preference3:pyramid_block:beachball)
            (* 8 (* (* (count preference2:blue_cube_block) (count-once-per-objects preference2:dodgeball) )
                (total-score)
              )
              6
              (count preference1:rug)
              (count preference2:dodgeball:pink_dodgeball)
              (total-score)
            )
            7
            (count preference3:beachball)
          )
        )
      )
    )
    4
  )
)
(:scoring
  20
)
)


(define (game game-id-1) (:domain medium-objects-room-v1)
(:setup
  (forall (?o - hexagonal_bin)
    (not
      (game-conserved
        (not
          (not
            (and
              (not
                (adjacent ?o)
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
    (forall (?y ?c ?s ?a ?o ?g - dodgeball)
      (and
        (preference preference1
          (exists (?q - ball ?m - wall)
            (then
              (hold-while (and (adjacent bed) (touch green_golfball) (in_motion ?a ?m) ) (and (exists (?l - game_object ?n - ball) (and (or (in ?m ?c) (in_motion ?c) ) (agent_holds rug) ) ) (touch ?s) ) )
              (once (not (on ?y) ) )
              (once (on floor agent) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (> 5 (* 10 3 )
  )
)
(:scoring
  (count preference1)
)
)


(define (game game-id-2) (:domain many-objects-room-v1)
(:setup
  (and
    (and
      (forall (?j - chair ?g - pillow)
        (or
          (game-conserved
            (touch ?g)
          )
        )
      )
      (game-optional
        (not
          (in_motion ?xxx ?xxx)
        )
      )
      (exists (?d - (either yellow_cube_block basketball key_chain))
        (and
          (game-optional
            (in_motion ?d ?d)
          )
          (not
            (game-conserved
              (above ?d ?d)
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?o - ball)
        (then
          (hold-to-end (and (and (and (in_motion ?o ?o) (in_motion ?o ?o) ) (same_color back) ) (and (in_motion ?o ?o) (agent_holds ?o) (agent_holds rug) ) ) )
          (hold (not (in_motion desk ?o) ) )
          (once (not (in_motion ?o) ) )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= 6 (+ (count preference1:purple) 4 )
    )
  )
)
(:scoring
  (* (count preference1:beachball) (count preference1:beachball) )
)
)


(define (game game-id-3) (:domain many-objects-room-v1)
(:setup
  (exists (?f - dodgeball)
    (game-optional
      (agent_holds ?f ?f)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?w - (either tall_cylindrical_block cube_block dodgeball))
        (then
          (once (agent_holds ?w desk) )
          (hold (on ?w top_drawer) )
          (hold (and (and (not (not (and (on ?w ?w) (not (< 7 1) ) ) ) ) (in ?w ?w) ) (on agent agent) (adjacent ?w) ) )
        )
      )
    )
    (preference preference2
      (exists (?f - chair)
        (then
          (hold (and (not (agent_holds ?f pink_dodgeball) ) (not (and (adjacent_side ?f ?f) (not (agent_holds ?f ?f) ) ) ) ) )
          (hold-while (rug_color_under ?f) (and (and (agent_holds bed) (in_motion ?f ?f) ) (agent_holds ?f) ) (and (not (in_motion ?f) ) (and (agent_holds ?f agent) (and (agent_holds ?f) (adjacent ?f ?f) (not (and (in ?f ?f) (in ?f) (rug_color_under blinds ?f) (not (agent_holds ?f agent) ) ) ) ) (agent_holds ?f rug) ) ) )
          (hold-for 10 (in_motion ?f ?f) )
          (once (in_motion bed) )
        )
      )
    )
    (preference preference3
      (exists (?d - bridge_block ?x - dodgeball)
        (at-end
          (and
            (in_motion ?x)
            (and
              (agent_holds ?x rug)
              (exists (?b ?v ?e - hexagonal_bin ?y - dodgeball)
                (on ?x ?x)
              )
              (and
                (in_motion bed)
                (not
                  (on ?x bed right)
                )
              )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 4 20 )
)
(:scoring
  (+ 1 (count-longest preference1:book) 4 (* 4 1 )
  )
)
)


(define (game game-id-4) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (not
      (agent_holds top_shelf)
    )
  )
)
(:constraints
  (and
    (forall (?s - wall)
      (and
        (preference preference1
          (exists (?d - cylindrical_block ?q - hexagonal_bin)
            (then
              (once (and (rug_color_under ?q ?s) (< 1 8) ) )
              (hold (on ?s rug) )
              (hold (not (agent_holds ?q) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (* 5 (count-once preference1:dodgeball:dodgeball) )
      (count-once-per-objects preference1:red_pyramid_block:purple)
    )
    (<= (count-once-per-objects preference1:yellow_cube_block:purple:book) 8 )
    (or
      (>= 4 (+ (count preference1:pink_dodgeball) 6 )
      )
      (>= 6 (- 3 )
      )
    )
    (or
      (>= 10 (count preference1:green) )
      (>= (count-once-per-objects preference1:pink_dodgeball) 3 )
    )
  )
)
(:scoring
  5
)
)


(define (game game-id-5) (:domain medium-objects-room-v1)
(:setup
  (forall (?i - book ?i - pyramid_block ?u - hexagonal_bin)
    (and
      (game-optional
        (in_motion ?u)
      )
      (game-optional
        (in_motion pillow)
      )
    )
  )
)
(:constraints
  (and
    (forall (?y - (either dodgeball red) ?a ?t - dodgeball)
      (and
        (preference preference1
          (exists (?m - building)
            (at-end
              (touch pink)
            )
          )
        )
      )
    )
    (forall (?t - hexagonal_bin)
      (and
        (preference preference2
          (exists (?n - cube_block ?c - block)
            (then
              (hold (same_color rug) )
              (once (in_motion ?c) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (count preference1:rug) (count-once preference2:yellow) )
  )
)
(:scoring
  (* (- 2 )
    (= (count-once preference2:basketball:green) (count-once-per-external-objects preference2:pink) )
  )
)
)


(define (game game-id-6) (:domain many-objects-room-v1)
(:setup
  (forall (?h - teddy_bear)
    (and
      (and
        (forall (?c - hexagonal_bin)
          (or
            (exists (?x - curved_wooden_ramp)
              (game-optional
                (adjacent ?c door)
              )
            )
            (and
              (game-conserved
                (exists (?j - cube_block)
                  (on ?j)
                )
              )
            )
          )
        )
        (game-conserved
          (not
            (agent_holds ?h)
          )
        )
        (game-conserved
          (agent_holds ?h)
        )
      )
      (exists (?r ?t - ball)
        (forall (?e - (either bed cube_block pen) ?e - (either cd yellow_cube_block cube_block) ?j - hexagonal_bin)
          (forall (?c - chair)
            (game-conserved
              (and
                (not
                  (forall (?n - ball)
                    (and
                      (in_motion ?r)
                      (and
                        (< (distance ?c ?t) (distance 0 room_center))
                        (agent_holds ?j ?c)
                      )
                    )
                  )
                )
                (in_motion ?r)
              )
            )
          )
        )
      )
      (exists (?a - dodgeball ?k ?y ?p ?q ?r ?i - tall_cylindrical_block)
        (game-conserved
          (agent_holds ?q)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?g - game_object)
        (then
          (once (and (and (in_motion ?g) (and (not (and (= 1 2) (agent_holds ?g) (in_motion ?g) ) ) (not (> (distance ?g room_center) (distance room_center ?g)) ) ) ) (in_motion ?g) ) )
          (hold (not (< (distance 3 desk) (distance ?g room_center)) ) )
          (once (not (agent_holds ?g ?g) ) )
        )
      )
    )
  )
)
(:terminal
  (> (count-once-per-objects preference1:pink_dodgeball) 3 )
)
(:scoring
  (* (* (count preference1:book) 10 )
    (* (count preference1:basketball) 0.7 (count preference1:brown) 3 )
  )
)
)


(define (game game-id-7) (:domain medium-objects-room-v1)
(:setup
  (and
    (forall (?z - ball)
      (and
        (game-optional
          (in_motion ?z ?z)
        )
        (game-conserved
          (and
            (not
              (not
                (in_motion ?z blue)
              )
            )
            (and
              (agent_holds ?z ?z)
              (in_motion ?z front front_left_corner)
            )
          )
        )
      )
    )
    (game-conserved
      (faces ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?v - hexagonal_bin)
        (at-end
          (and
            (agent_holds ?v)
            (in_motion ?v ?v)
          )
        )
      )
    )
  )
)
(:terminal
  (>= (* (* (count preference1:doggie_bed) (count-once-per-objects preference1:basketball:golfball:yellow) (+ (count preference1:dodgeball:purple) (count-once-per-objects preference1:beachball:basketball) )
      )
      16
    )
    (external-forall-maximize
      (* 0 (count preference1:pink) (* (> (and 30 (count-once-per-objects preference1:alarm_clock:white) (count preference1) ) (* (count preference1:red) (* (count preference1:pink_dodgeball:alarm_clock) (+ (count preference1:red) (+ (count-once-per-objects preference1:dodgeball:basketball) (total-time) )
                  (+ (+ (count-once-per-objects preference1:pink) (* (count-once preference1:pink_dodgeball) (+ (count preference1:pink) (count preference1:basketball:beachball) (count-unique-positions preference1:beachball) (- (count-once preference1:dodgeball:purple) )
                        )
                      )
                    )
                    (/
                      2
                      (* (count preference1:alarm_clock) (+ (+ 0 (external-forall-maximize (count-total preference1:pyramid_block) ) )
                          (and
                            4
                          )
                          (* (= (+ (* 6 (= 1 (count preference1:dodgeball) )
                                )
                                (* 5 (count preference1:yellow) (count preference1:blue_dodgeball) )
                              )
                              (count-total preference1:dodgeball:golfball)
                              (count-once-per-external-objects preference1:dodgeball)
                            )
                            10
                          )
                          5
                          (count-once-per-objects preference1:blue_dodgeball)
                          (+ (* 2 4 )
                            4
                          )
                          (count-once-per-objects preference1:book:cube_block)
                          5
                          (* 6 3 )
                        )
                      )
                    )
                  )
                )
              )
            )
          )
          (+ 3 2 )
          (count preference1:beachball:pink)
        )
      )
    )
  )
)
(:scoring
  (* (external-forall-maximize (count-longest preference1:blue_dodgeball) ) (count-once-per-objects preference1:blue_pyramid_block) )
)
)


(define (game game-id-8) (:domain many-objects-room-v1)
(:setup
  (game-optional
    (on ?xxx ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?t - (either laptop yellow_cube_block))
        (then
          (once (on ?t ?t) )
          (hold (agent_holds bed) )
        )
      )
    )
    (preference preference2
      (exists (?y ?o ?g ?t ?s - hexagonal_bin)
        (at-end
          (agent_holds back ?t)
        )
      )
    )
    (preference preference3
      (exists (?r - wall ?q - hexagonal_bin)
        (at-end
          (on front ?q)
        )
      )
    )
  )
)
(:terminal
  (<= 10 (count-once-per-objects preference2:blue_dodgeball) )
)
(:scoring
  5
)
)


(define (game game-id-9) (:domain medium-objects-room-v1)
(:setup
  (game-optional
    (and
      (agent_holds bed ?xxx)
      (in ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?b - building)
        (then
          (hold (and (in ?b ?b back) (and (and (not (and (in_motion bed) (agent_holds desk) (agent_holds ?b) ) ) (and (and (on ?b) (not (on ?b) ) ) (not (= 1 (x_position ?b ?b)) ) ) ) (not (and (in_motion ?b) (or (agent_holds ?b) (and (agent_holds floor ?b) (agent_holds ?b east_sliding_door) ) ) ) ) ) (exists (?o - game_object) (< (distance room_center 4) (distance ?b desk)) ) ) )
          (hold-while (agent_holds ?b desk) (and (adjacent ?b ?b) (not (in ?b ?b) ) ) (and (adjacent agent) (in_motion ?b) ) )
          (once (touch ?b) )
        )
      )
    )
    (preference preference2
      (exists (?s - teddy_bear ?v - ball)
        (then
          (hold (not (not (agent_holds ?v) ) ) )
          (once (not (exists (?j - (either pyramid_block tall_cylindrical_block)) (adjacent ?j) ) ) )
          (once (in_motion ?v) )
        )
      )
    )
    (preference preference3
      (exists (?o - (either golfball dodgeball) ?i - block)
        (then
          (once (touch ?i) )
          (hold (not (on ?i bed pink) ) )
          (once (and (exists (?b - doggie_bed ?j - hexagonal_bin) (adjacent ?j pink) ) (exists (?x - dodgeball) (on ?i) ) ) )
        )
      )
    )
  )
)
(:terminal
  (>= (- (- (count preference2:basketball) )
    )
    (count-once-per-objects preference2:blue_cube_block)
  )
)
(:scoring
  (* (count-shortest preference3:dodgeball) 1 )
)
)


(define (game game-id-10) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (and
      (not
        (on ?xxx)
      )
      (object_orientation ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?y - desk_shelf)
        (then
          (once (agent_holds ?y ?y) )
          (hold (in_motion ) )
          (once (agent_holds ?y) )
        )
      )
    )
  )
)
(:terminal
  (and
    (or
      (= (count-once-per-objects preference1:purple) (* (+ (* 2 3 )
            (count preference1:red_pyramid_block:dodgeball)
          )
          (* (* 2 (count-once-per-objects preference1) )
            8
          )
        )
      )
    )
    (>= 5 20 )
  )
)
(:scoring
  (count-once-per-objects preference1:pyramid_block)
)
)


(define (game game-id-11) (:domain many-objects-room-v1)
(:setup
  (exists (?c - ball ?r - desk_shelf)
    (game-conserved
      (and
        (and
          (agent_holds ?r)
          (in_motion ?r)
        )
        (exists (?a - game_object)
          (> 1 7)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?d - game_object ?a - chair)
        (then
          (hold (not (above ?a) ) )
          (hold (agent_holds ?a ?a) )
          (once (adjacent ?a blinds) )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference1:basketball) (count-once-per-objects preference1:yellow) )
)
(:scoring
  (+ (count preference1:beachball) (* 2 2 )
  )
)
)


(define (game game-id-12) (:domain many-objects-room-v1)
(:setup
  (forall (?x - ball ?u - building)
    (forall (?p - game_object ?p - (either cube_block book))
      (and
        (forall (?e - chair)
          (exists (?s - shelf ?w - dodgeball ?a - (either cellphone bridge_block cube_block) ?c - hexagonal_bin ?k ?f ?z ?t ?i - (either side_table triangle_block teddy_bear))
            (forall (?r - (either pyramid_block cube_block) ?v - game_object ?d - block ?v - pillow)
              (and
                (game-optional
                  (exists (?m - wall ?y - hexagonal_bin)
                    (agent_holds bed ?v)
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
(:constraints
  (and
    (preference preference1
      (then
        (once (and (not (agent_holds ?xxx ?xxx) ) (agent_crouches ?xxx) ) )
        (hold (not (adjacent ?xxx) ) )
        (once (< (x_position ?xxx ?xxx) 1) )
      )
    )
  )
)
(:terminal
  (>= (* (= 3 (+ (count preference1:hexagonal_bin) (+ (count-once preference1:red:basketball) (and 4 1 20 ) )
        )
      )
      (count-once preference1:hexagonal_bin)
    )
    (count preference1:red)
  )
)
(:scoring
  (- (+ (count preference1:blue_dodgeball) (external-forall-minimize (* (count preference1:golfball) (count preference1:purple:book) )
      )
    )
  )
)
)


(define (game game-id-13) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (in ?xxx ?xxx ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?g - dodgeball ?q - blue_cube_block)
        (at-end
          (same_object ?q)
        )
      )
    )
  )
)
(:terminal
  (and
    (> (* (= (count preference1:hexagonal_bin) 3 (count-once preference1:alarm_clock) )
        (count preference1:blue_dodgeball)
      )
      (* (+ (* 10 (or (count preference1:green) (count preference1:dodgeball:basketball) ) )
          (count preference1:dodgeball:rug)
        )
        4
      )
    )
    (>= (count-once preference1:beachball:dodgeball) (count preference1:red) )
  )
)
(:scoring
  (count-once-per-objects preference1:blue_cube_block)
)
)


(define (game game-id-14) (:domain medium-objects-room-v1)
(:setup
  (forall (?t - hexagonal_bin)
    (exists (?u - ball)
      (and
        (game-conserved
          (not
            (agent_holds ?t)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?r - hexagonal_bin)
        (then
          (hold (on ?r ?r) )
          (once (not (exists (?p - hexagonal_bin) (< 1 (distance ?p room_center)) ) ) )
          (hold (game_start ?r ?r) )
        )
      )
    )
  )
)
(:terminal
  (or
    (not
      (>= 10 (>= (not (count-overlapping preference1:dodgeball) ) 8 )
      )
    )
    (or
      (or
        (and
          (>= 10 3 )
          (or
            (>= (count preference1:orange:red) 2 )
          )
        )
        (and
          (= (* (external-forall-minimize (count preference1:basketball) ) (count-once preference1:beachball) )
            8
          )
        )
        (> 5 (* (count-increasing-measure preference1:dodgeball) (count preference1:blue_dodgeball:dodgeball) )
        )
      )
      (>= (count-once-per-objects preference1:pink) (count preference1:pink_dodgeball:blue_dodgeball:pink) )
      (>= (count-once-per-objects preference1:dodgeball:beachball) (* (- (+ 2 (total-time) )
          )
          2
        )
      )
    )
  )
)
(:scoring
  (* 2 (* (count-same-positions preference1:golfball) (count preference1:blue_dodgeball) )
  )
)
)


(define (game game-id-15) (:domain few-objects-room-v1)
(:setup
  (not
    (game-conserved
      (and
        (agent_holds ?xxx ?xxx)
        (not
          (not
            (and
              (in_motion ?xxx)
              (rug_color_under ?xxx ?xxx)
            )
          )
        )
        (agent_holds ?xxx)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?v - cube_block)
        (then
          (once (adjacent_side ?v) )
          (once (in_motion ?v) )
          (hold (on ?v) )
        )
      )
    )
    (preference preference2
      (exists (?m - pillow ?z - (either alarm_clock laptop dodgeball golfball golfball laptop golfball))
        (then
          (hold (in_motion ?z ?z) )
          (hold (not (in_motion desk) ) )
          (once (object_orientation ?z) )
        )
      )
    )
    (preference preference3
      (exists (?s ?i - (either bridge_block teddy_bear))
        (then
          (once (in ?i) )
          (once (and (and (in_motion ?s ?s) (object_orientation ?i ?i) ) (and (not (agent_holds ?i) ) (and (and (adjacent ?s ?i) (and (in_motion ?i ?s) (in_motion ?i) ) (in agent ?s) ) (not (agent_holds ?i) ) ) (in ?s) ) ) )
          (hold-while (not (in_motion ?i) ) (agent_holds green_golfball) (not (adjacent ?i) ) (adjacent ?i) )
        )
      )
    )
  )
)
(:terminal
  (= (>= (* (count preference1:beachball:blue:beachball) (external-forall-maximize 4 ) )
      (+ (* (count-once-per-objects preference3:hexagonal_bin) (count preference2:yellow) )
        (* (+ (count-once-per-objects preference3:dodgeball) 60 )
          3
        )
      )
    )
    (* (count preference3:beachball:dodgeball) 7 )
  )
)
(:scoring
  8
)
)


(define (game game-id-16) (:domain many-objects-room-v1)
(:setup
  (forall (?c - ball)
    (and
      (game-conserved
        (and
          (exists (?z ?j ?u - building)
            (in ?j)
          )
          (and
            (on ?c pink)
            (agent_holds ?c)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?a - ball ?e - (either dodgeball))
        (then
          (hold (and (on ?e) (in_motion bed ?e) ) )
          (once (and (on ?e ?e) (and (agent_holds agent) (and (not (not (opposite blinds) ) ) (agent_holds ?e) ) ) (and (agent_holds ?e) (is_setup_object ?e) ) ) )
          (hold (exists (?m - game_object ?x - ball) (not (agent_holds front) ) ) )
        )
      )
    )
  )
)
(:terminal
  (<= (* 3 (+ (* (count preference1:top_drawer:yellow:hexagonal_bin) (* 2 )
        )
        4
      )
    )
    3
  )
)
(:scoring
  (* (count preference1:blue_dodgeball:hexagonal_bin) 30 5 )
)
)


(define (game game-id-17) (:domain few-objects-room-v1)
(:setup
  (exists (?i - triangular_ramp)
    (forall (?o - (either teddy_bear cd))
      (game-conserved
        (not
          (and
            (on ?o ?i)
            (not
              (agent_holds floor)
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?q - hexagonal_bin)
        (then
          (hold (forall (?w - wall) (in_motion ?q) ) )
          (hold (not (not (in_motion ?q) ) ) )
        )
      )
    )
    (preference preference2
      (exists (?e - (either cube_block laptop) ?d - building)
        (then
          (once (rug_color_under ?d ?d) )
          (hold (and (agent_holds ?d agent) (in_motion ?d) ) )
          (once (in_motion ?d) )
        )
      )
    )
  )
)
(:terminal
  (>= (total-time) 2 )
)
(:scoring
  (count preference1:yellow_cube_block)
)
)


(define (game game-id-18) (:domain many-objects-room-v1)
(:setup
  (game-optional
    (or
      (in upright)
      (not
        (not
          (agent_holds ?xxx)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?q - ball)
      (and
        (preference preference1
          (exists (?v ?f ?m ?o - hexagonal_bin ?n - ball ?c - ball)
            (then
              (once (agent_holds ?c) )
              (once (agent_holds ?q ?q) )
              (once (in agent) )
            )
          )
        )
        (preference preference2
          (exists (?x - doggie_bed)
            (then
              (hold-while (or (in_motion ?q) (in_motion ?q ?q) (and (and (on bed) (not (adjacent ?x ?x) ) ) (object_orientation ?x agent) ) (adjacent_side ?x) ) (in_motion ?x bed) )
              (once (agent_holds ?x ?q agent) )
              (once (agent_holds ?x) )
              (once (in ?q ?q) )
            )
          )
        )
        (preference preference3
          (exists (?j - dodgeball ?n - hexagonal_bin ?c - book)
            (then
              (hold (agent_holds ?q) )
            )
          )
        )
      )
    )
    (preference preference4
      (exists (?m - hexagonal_bin)
        (then
          (hold (not (adjacent ?m) ) )
          (once (agent_holds ?m ?m) )
          (once (and (in_motion ?m ?m) ) )
        )
      )
    )
    (preference preference5
      (exists (?c - dodgeball ?t - hexagonal_bin)
        (then
          (hold (adjacent ?t) )
          (hold (not (not (in ?t) ) ) )
          (once (in ?t) )
        )
      )
    )
  )
)
(:terminal
  (>= (* (* (count-overlapping preference2:dodgeball) 3 )
      (= (* (+ 15 5 )
          100
        )
        6
      )
    )
    (* 1 60 3 (count preference5:beachball) (count-once preference3:red) )
  )
)
(:scoring
  (external-forall-maximize
    (count-once-per-objects preference2:dodgeball)
  )
)
)


(define (game game-id-19) (:domain medium-objects-room-v1)
(:setup
  (and
    (game-conserved
      (on ?xxx ?xxx)
    )
    (game-optional
      (and
        (forall (?a - dodgeball)
          (in ?a ?a)
        )
        (not
          (on ?xxx)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?d - ball)
        (then
          (hold (adjacent ?d) )
          (hold (and (adjacent bed) (not (and (in agent) (not (and (not (in_motion ?d) ) (in_motion ?d ?d) ) ) ) ) ) )
        )
      )
    )
  )
)
(:terminal
  (or
    (> (count preference1:green:dodgeball) 9 )
    (>= 6 (count-measure preference1:pink_dodgeball) )
  )
)
(:scoring
  (count preference1:blue_cube_block:purple)
)
)


(define (game game-id-20) (:domain many-objects-room-v1)
(:setup
  (game-optional
    (in ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?a - ball ?i - ball)
        (then
          (once (in_motion ?i) )
          (once (not (in ?i) ) )
          (hold-for 7 (agent_holds bed) )
        )
      )
    )
    (preference preference2
      (exists (?c - hexagonal_bin)
        (then
          (hold (or (between ?c) ) )
          (once (adjacent_side ?c) )
          (hold (and (in_motion ?c) (and (and (in_motion agent ?c) (= 4 1 (distance 0 side_table)) ) (equal_x_position ?c ?c) (in_motion ?c ?c) (in_motion ?c ?c) ) ) )
        )
      )
    )
  )
)
(:terminal
  (>= 300 (count preference2:basketball) )
)
(:scoring
  (count preference2:book)
)
)


(define (game game-id-21) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (in_motion ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?q - wall ?w - game_object)
        (then
          (hold (touch bed pillow) )
          (once (in_motion ?w) )
          (hold (agent_holds ?w) )
        )
      )
    )
  )
)
(:terminal
  (or
    (and
      (>= (or (- (* 4 (= (count preference1:beachball) 18 )
              (count preference1:pink)
              (count-once preference1:green)
              6
              4
            )
          )
          (- (count-once preference1:wall) )
          (+ (count preference1:dodgeball) (count-once-per-objects preference1:red) )
        )
        10
      )
      (>= (count-once preference1:orange:pink) (+ (count preference1:basketball) (* (+ (+ 7 (count-once-per-objects preference1:yellow_cube_block:rug) )
              0
            )
            (* (external-forall-maximize 10 ) (count preference1:dodgeball) )
            (* (count-once-per-objects preference1:blue_pyramid_block:blue_cube_block) )
          )
          (count-same-positions preference1:top_drawer)
        )
      )
      (not
        (or
          (>= 4 (* 3 (- (count preference1:wall) )
            )
          )
          (>= (count preference1:doggie_bed:basketball) (+ (* (count preference1:pink) (count preference1:blue_cube_block) )
              (+ 3 (count preference1:green) (+ 7 (* (* 1 (count-total preference1:pink) )
                    (* (or (count preference1) ) (count preference1:red) )
                    10
                  )
                  (* (count-measure preference1:dodgeball) (count-once-per-objects preference1:top_drawer) )
                  (< (* 2 (count preference1:dodgeball:tan) )
                    2
                  )
                  (count preference1:purple:dodgeball)
                  1
                )
              )
              8
            )
          )
        )
      )
    )
    (>= 10 (= 1 (count preference1:dodgeball) )
    )
  )
)
(:scoring
  (* 1 15 (* (count-once-per-external-objects preference1:beachball) )
  )
)
)


(define (game game-id-22) (:domain few-objects-room-v1)
(:setup
  (exists (?b - (either dodgeball dodgeball))
    (game-optional
      (and
        (and
          (< 1 (distance ?b agent))
          (< (distance ?b ?b) (distance ?b ?b))
          (not
            (faces ?b)
          )
          (object_orientation ?b)
        )
        (on ?b)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?v - hexagonal_bin)
        (then
          (once (and (on floor) (on back agent) ) )
          (hold (exists (?y - green_triangular_ramp) (not (< 3 (distance desk ?v)) ) ) )
          (hold (on ?v) )
        )
      )
    )
  )
)
(:terminal
  (or
    (and
      (>= 0 3 )
      (>= (> 3 (count preference1:blue_dodgeball) )
        (count preference1:dodgeball)
      )
    )
    (>= (- (count-shortest preference1) )
      (* 3 )
    )
  )
)
(:scoring
  1
)
)


(define (game game-id-23) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (same_type ?xxx)
  )
)
(:constraints
  (and
    (forall (?t - (either main_light_switch alarm_clock desktop))
      (and
        (preference preference1
          (exists (?r - dodgeball ?q - (either dodgeball rug basketball))
            (then
              (hold (in_motion desk) )
              (once (in tan) )
              (once (in_motion ?q) )
              (once (agent_holds ?t) )
              (once (on ?q ?q) )
            )
          )
        )
      )
    )
    (preference preference2
      (exists (?n - cube_block ?p - (either doggie_bed chair) ?t - cube_block)
        (then
          (once (agent_holds ?t) )
          (hold (and (exists (?k - hexagonal_bin) (not (in_motion ?k) ) ) (not (agent_holds ?t ?t) ) ) )
          (once (same_type ?t ?t) )
        )
      )
    )
  )
)
(:terminal
  (>= 2 (count preference1:rug) )
)
(:scoring
  (count-unique-positions preference1:dodgeball)
)
)


(define (game game-id-24) (:domain few-objects-room-v1)
(:setup
  (and
    (and
      (and
        (exists (?r ?j ?c - pyramid_block)
          (and
            (game-optional
              (agent_holds ?c)
            )
            (game-optional
              (< (distance front_left_corner ?j) 7)
            )
            (forall (?u - block)
              (game-optional
                (agent_holds ?j)
              )
            )
          )
        )
        (and
          (game-conserved
            (agent_holds ?xxx floor)
          )
          (exists (?i - game_object)
            (and
              (game-conserved
                (and
                  (not
                    (not
                      (and
                        (and
                          (and
                            (in_motion ?i top_drawer)
                            (in ?i ?i)
                            (same_object ?i front)
                            (not
                              (adjacent ?i ?i)
                            )
                          )
                          (agent_holds ?i)
                          (adjacent ?i)
                        )
                        (in_motion bed ?i agent)
                        (and
                          (< 1 (distance_side ?i back))
                          (on ?i)
                        )
                      )
                    )
                  )
                  (agent_holds ?i)
                )
              )
            )
          )
          (game-conserved
            (>= 4 1)
          )
        )
      )
    )
    (exists (?i - hexagonal_bin)
      (game-optional
        (not
          (and
            (in ?i ?i)
            (and
              (not
                (on agent ?i)
              )
              (not
                (in ?i)
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
    (preference preference1
      (exists (?g - hexagonal_bin ?e - (either book))
        (then
          (once (> (distance desk 1) 1) )
          (hold (not (agent_holds ?e ?e) ) )
          (forall-sequence (?a - chair)
            (then
              (once (agent_holds agent) )
              (hold (in front_left_corner) )
              (hold (> (distance agent ?a) 2) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (external-forall-maximize (count-overlapping preference1:pink) ) (count preference1:basketball:yellow) )
  )
)
(:scoring
  (count-same-positions preference1:cube_block)
)
)


(define (game game-id-25) (:domain many-objects-room-v1)
(:setup
  (exists (?p ?g ?k ?m ?t ?z ?q ?j ?c ?n - ball)
    (game-optional
      (not
        (same_type ?m)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?y - building ?s - (either pink teddy_bear basketball alarm_clock cylindrical_block cylindrical_block yellow_cube_block))
        (then
          (forall-sequence (?w - hexagonal_bin)
            (then
              (hold (not (in_motion ?w) ) )
              (once (in ?s ?w) )
              (once (or (> 4 2) (and (exists (?q - hexagonal_bin ?y - cube_block) (touch desk) ) (in_motion ?s) ) ) )
            )
          )
          (hold (adjacent rug) )
          (once (same_color ?s) )
        )
      )
    )
  )
)
(:terminal
  (>= (count-longest preference1:golfball) (count-once-per-objects preference1:blue) )
)
(:scoring
  (count preference1:beachball:pink:pink)
)
)


(define (game game-id-26) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (agent_holds back ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?d ?a - dodgeball ?a - ball)
        (at-end
          (not
            (not
              (and
                (adjacent agent)
                (not
                  (not
                    (and
                      (on ?a ?a)
                      (not
                        (not
                          (and
                            (agent_holds ?a ?a)
                            (not
                              (and
                                (on ?a)
                                (agent_holds agent ?a)
                              )
                            )
                          )
                        )
                      )
                      (agent_holds ?a ?a)
                      (not
                        (faces ?a door)
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
  )
)
(:terminal
  (and
    (<= (count preference1:pink_dodgeball:doggie_bed:yellow) (* 2 3 )
    )
  )
)
(:scoring
  (count-once preference1:beachball)
)
)


(define (game game-id-27) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (not
      (rug_color_under ?xxx ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?n - teddy_bear)
        (then
          (once (agent_holds ?n ?n) )
          (hold (equal_z_position ?n ?n) )
          (once (not (touch ?n ?n) ) )
        )
      )
    )
    (preference preference2
      (exists (?f - block ?l - building)
        (then
          (once (not (adjacent_side desk ?l) ) )
          (once (in_motion ?l) )
          (hold (adjacent ?l ?l) )
        )
      )
    )
    (preference preference3
      (then
        (hold (< (distance_side ?xxx 6) 2) )
      )
    )
    (preference preference4
      (exists (?q - hexagonal_bin ?w - color)
        (then
          (hold-while (not (adjacent desk pink_dodgeball ?w) ) (agent_holds ?w) )
          (once (not (and (agent_holds ?w) (on ?w ?w) ) ) )
          (once (not (forall (?r ?x ?a - doggie_bed) (touch ?x) ) ) )
        )
      )
    )
    (preference preference5
      (exists (?g - game_object)
        (then
          (once (not (< (distance agent) 7) ) )
          (once (not (in_motion ?g) ) )
          (once (agent_holds ?g) )
        )
      )
    )
    (preference preference6
      (exists (?g - triangular_ramp)
        (at-end
          (agent_holds ?g agent)
        )
      )
    )
  )
)
(:terminal
  (not
    (>= (- (+ (count-same-positions preference5:alarm_clock:red) 5 )
      )
      (count preference6:golfball)
    )
  )
)
(:scoring
  (count-once-per-objects preference4:yellow)
)
)


(define (game game-id-28) (:domain many-objects-room-v1)
(:setup
  (game-optional
    (in_motion ?xxx ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?r - ball)
        (then
          (hold (on top_drawer ?r) )
          (hold-to-end (agent_holds ?r) )
          (once (in floor) )
        )
      )
    )
    (preference preference2
      (exists (?x - red_dodgeball ?x - dodgeball ?f - pyramid_block)
        (then
          (once (not (agent_holds ?f ?f) ) )
          (hold (< (x_position ?f side_table) (distance ?f ?f)) )
          (once (in_motion ?f) )
        )
      )
    )
    (preference preference3
      (exists (?x - hexagonal_bin ?p - hexagonal_bin ?g - hexagonal_bin)
        (then
          (once (forall (?l - beachball) (adjacent_side ?g ?g) ) )
          (hold (and (not (touch ?g ?g) ) (agent_holds ?g ?g) ) )
          (once (not (and (not (in ?g ?g) ) (not (< 1 (distance 9 desk)) ) (and (in_motion ?g ?g) (agent_holds agent ?g) ) (not (agent_holds ?g ?g) ) (rug_color_under ?g) (in_motion ?g ?g) (not (in ?g ?g) ) (and (not (open ?g) ) (and (on ?g ?g) (adjacent ?g ?g) ) ) ) ) )
          (once (< (distance agent ?g) (distance ?g door)) )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference1:dodgeball) 100 )
)
(:scoring
  (/
    (* (* 4 (count preference2:hexagonal_bin) )
      (- (total-score) )
    )
    (count preference2:blue_dodgeball)
  )
)
)


(define (game game-id-29) (:domain medium-objects-room-v1)
(:setup
  (forall (?s - ball)
    (forall (?j - dodgeball ?y - drawer ?e - cube_block)
      (and
        (forall (?h - tall_cylindrical_block ?k - curved_wooden_ramp)
          (exists (?u - hexagonal_bin)
            (exists (?r - doggie_bed)
              (game-optional
                (and
                  (on ?s)
                  (< 7 (distance ?s agent))
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
    (preference preference1
      (exists (?g - wall)
        (then
          (hold (not (in_motion ?g ?g) ) )
          (hold (not (touch ?g ?g) ) )
          (hold-while (touch ?g ?g) (agent_holds bed) (not (in_motion ?g agent) ) )
        )
      )
    )
    (preference preference2
      (exists (?l - dodgeball)
        (then
          (hold-while (not (touch ?l back) ) (adjacent ?l ?l) )
          (once (and (and (not (in_motion ?l) ) (touch back ?l) (agent_holds ?l brown) (not (agent_holds ?l) ) (not (in_motion ?l ?l) ) (on ?l ?l) (and (and (exists (?w - hexagonal_bin) (agent_holds ?w pink_dodgeball) ) (in_motion bed) ) ) ) (and (and (> 5 1) (and (and (adjacent ?l ?l) (and (on ?l) (not (in_motion ?l) ) ) ) (faces ?l ?l) (in_motion ?l agent) ) (and (not (adjacent ?l ?l) ) (agent_holds ?l ?l) ) ) (and (exists (?o - doggie_bed ?b - doggie_bed) (agent_holds ?b) ) (not (and (in_motion ?l ?l) (in floor) ) ) ) ) ) )
          (once (in_motion ?l) )
        )
      )
    )
  )
)
(:terminal
  (< (* (* (count-once-per-objects preference2:hexagonal_bin) (* (+ (* (* (count preference1:dodgeball) 10 40 )
              0.5
              6
            )
            5
          )
          (count preference2:golfball)
        )
      )
      7
    )
    (count preference2:cube_block:golfball)
  )
)
(:scoring
  (* (* 50 10 )
    (* 8 (+ 2 (* (* (* (- 2 )
              5
              15
            )
            (count-once-per-objects preference2:yellow:basketball)
          )
          10
          (* 6 (+ (count-unique-positions preference1:cube_block:blue_dodgeball) (count preference2:beachball) )
          )
          (+ (count preference2:doggie_bed) (* (count preference1:yellow) (* (* 20 (count-once-per-external-objects preference1:dodgeball) 6 )
                (count preference2:blue_dodgeball)
              )
            )
          )
        )
      )
      (total-score)
    )
    3
  )
)
)


(define (game game-id-30) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (in bed ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?i - doggie_bed)
        (then
          (once (equal_z_position ?i ?i) )
          (once (in_motion ?i ?i) )
          (hold (in_motion ?i ?i) )
          (hold-while (on ?i ?i) (in ?i) (not (exists (?a - block) (in ?a) ) ) )
          (hold (not (not (in_motion ?i) ) ) )
          (hold-while (not (in_motion ?i) ) (agent_holds ?i ?i) )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (count-once-per-external-objects preference1:brown) 1 )
    (not
      (>= 2 (not (* 5 (count preference1:beachball:blue_dodgeball) )
        )
      )
    )
    (> (+ 2 (count preference1:beachball) )
      (count-once-per-objects preference1:hexagonal_bin:dodgeball)
    )
  )
)
(:scoring
  (count-increasing-measure preference1:dodgeball)
)
)


(define (game game-id-31) (:domain medium-objects-room-v1)
(:setup
  (and
    (or
      (game-conserved
        (not
          (in_motion west_wall)
        )
      )
      (game-conserved
        (not
          (in_motion ?xxx)
        )
      )
    )
    (forall (?u - game_object)
      (forall (?i - hexagonal_bin)
        (game-conserved
          (not
            (same_type rug)
          )
        )
      )
    )
    (and
      (and
        (game-conserved
          (agent_holds ?xxx)
        )
        (game-conserved
          (agent_holds ?xxx ?xxx)
        )
        (exists (?v - hexagonal_bin ?q ?h - ball)
          (forall (?t - building ?w - ball)
            (and
              (game-optional
                (agent_holds ?q ?w)
              )
            )
          )
        )
        (game-conserved
          (not
            (in ?xxx)
          )
        )
        (forall (?o - ball)
          (and
            (and
              (exists (?e - chair ?w - dodgeball)
                (exists (?k - ball ?t - hexagonal_bin)
                  (game-conserved
                    (in_motion ?w ?o)
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
(:constraints
  (and
    (preference preference1
      (exists (?j - pillow)
        (then
          (once (in_motion ?j) )
          (once (and (agent_holds ?j ?j) ) )
          (hold (in_motion ?j) )
        )
      )
    )
  )
)
(:terminal
  (and
    (> (< 3 (count preference1:yellow:basketball) )
      (count-once preference1:red:beachball)
    )
    (and
      (and
        (or
          (>= 3 (+ 6 (+ 6 (* 3 (count preference1:basketball) )
              )
            )
          )
          (> (* (count preference1:dodgeball:hexagonal_bin:hexagonal_bin) 2 )
            (* (external-forall-minimize (count-overlapping preference1:pink) ) (total-score) )
          )
        )
      )
    )
    (or
      (>= 6 (+ (count preference1:dodgeball) (* (count preference1:yellow:dodgeball) (+ 10 (count preference1:dodgeball) (* 3 (count-once-per-objects preference1:blue_cube_block) )
              10
              9
              8
              (* (+ (* (count preference1:golfball) (count-once-per-objects preference1:blue_pyramid_block) )
                  (count-once preference1:basketball:dodgeball)
                )
                180
              )
              (count preference1:dodgeball)
              (count preference1:basketball)
              5
              5
              (count-same-positions preference1:dodgeball:dodgeball)
            )
          )
        )
      )
    )
  )
)
(:scoring
  3
)
)


(define (game game-id-32) (:domain many-objects-room-v1)
(:setup
  (not
    (game-conserved
      (= 8 (distance ?xxx agent desk))
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?y - dodgeball ?h - game_object)
        (then
          (hold (not (not (agent_holds ?h) ) ) )
          (hold-for 3 (adjacent agent bed) )
          (once (agent_holds desk) )
        )
      )
    )
    (forall (?g - ball)
      (and
        (preference preference2
          (exists (?h - (either doggie_bed desktop cube_block))
            (then
              (hold (agent_holds ?g ?g) )
            )
          )
        )
        (preference preference3
          (exists (?s - doggie_bed ?i - pillow ?t - hexagonal_bin)
            (then
              (hold (adjacent ?g ?t) )
              (once (and (agent_holds floor drawer) (not (on ?t) ) ) )
              (once (same_type ?t ?g) )
              (hold-while (in_motion ?t ?t) (not (in_motion ?t) ) (agent_holds ?g ?t) (and (not (on ?g ?g) ) (agent_holds agent) ) )
            )
          )
        )
        (preference preference4
          (exists (?c - doggie_bed ?e - beachball ?j - hexagonal_bin ?s - hexagonal_bin)
            (then
              (once (not (in ?s agent) ) )
              (once (not (adjacent ?g) ) )
              (hold (and (in ?s) (is_setup_object ?g pink_dodgeball) (agent_holds ?g) (in_motion ?s) (and (agent_holds ?g bed) (not (not (agent_holds ?g) ) ) (agent_holds ?g) ) (in_motion ?g) ) )
            )
          )
        )
      )
    )
    (preference preference5
      (at-end
        (and
          (in_motion ?xxx ?xxx)
          (adjacent ?xxx)
        )
      )
    )
  )
)
(:terminal
  (>= (count-once-per-objects preference1:basketball) (total-time) )
)
(:scoring
  (count preference2:dodgeball:blue_dodgeball)
)
)


(define (game game-id-33) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (and
      (open ?xxx ?xxx)
      (game_over ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?e - dodgeball)
        (then
          (once (touch ?e ?e) )
          (hold (exists (?w - hexagonal_bin) (in ?w ?w) ) )
          (hold (and (agent_holds ?e ?e) (game_over ?e rug) (and (agent_holds ?e) (and (not (on bed) ) (not (not (is_setup_object ?e) ) ) (touch agent) ) ) ) )
        )
      )
    )
    (preference preference2
      (exists (?d - (either basketball pink wall) ?k - hexagonal_bin)
        (then
          (once (< (distance bed room_center) (distance side_table agent)) )
          (hold (in_motion ?k) )
          (once (and (on ?k) (not (and (in ?k) (and (on desk floor) (in ?k) ) ) ) ) )
          (hold-for 5 (and (agent_holds pink ?k) (agent_holds ?k ?k) (not (on ?k) ) ) )
        )
      )
    )
    (preference preference3
      (exists (?w - cube_block)
        (then
          (once (agent_holds ?w) )
          (once (forall (?q - chair) (in_motion ?q ?w) ) )
          (hold (in_motion ?w) )
        )
      )
    )
  )
)
(:terminal
  (<= (+ 18 (count preference3:beachball) )
    (count-once-per-objects preference3:red)
  )
)
(:scoring
  (count preference3:basketball:dodgeball)
)
)


(define (game game-id-34) (:domain medium-objects-room-v1)
(:setup
  (and
    (game-conserved
      (exists (?d - hexagonal_bin)
        (adjacent ?d)
      )
    )
  )
)
(:constraints
  (and
    (forall (?p - ball)
      (and
        (preference preference1
          (exists (?y - hexagonal_bin)
            (at-end
              (in_motion ?p ?y)
            )
          )
        )
        (preference preference2
          (exists (?s - cube_block)
            (then
              (once (and (on ?p) (adjacent rug) (touch ?s) ) )
              (hold (on agent floor) )
              (hold (in ?p) )
            )
          )
        )
      )
    )
    (preference preference3
      (exists (?r - hexagonal_bin)
        (then
          (hold (or (not (not (in_motion ?r) ) ) (and (in_motion upright ?r) (in_motion bed) ) (and (and (in ?r ?r) (not (and (agent_holds ?r) (agent_holds agent ?r) (in_motion ?r) ) ) (in_motion ?r) ) (< (x_position 5) (distance ?r 6)) ) ) )
          (hold (agent_holds ?r) )
          (once (equal_z_position ?r ?r) )
        )
      )
    )
  )
)
(:terminal
  (>= 6 (total-score) )
)
(:scoring
  (= (* 15 (+ (count preference2:yellow) (* (+ (count preference3:dodgeball) (* (count preference2:orange) 9 )
          )
          (* 5 (* (* (count-once preference1) 7 )
            )
          )
        )
      )
    )
    (count preference3:dodgeball)
  )
)
)


(define (game game-id-35) (:domain medium-objects-room-v1)
(:setup
  (and
    (game-optional
      (or
        (in_motion ?xxx ?xxx)
        (adjacent ?xxx agent)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?g ?j ?z ?m ?u ?n - hexagonal_bin)
        (then
          (once (in_motion upright) )
          (hold (adjacent ?j ?g) )
          (once (touch ?u ?j) )
        )
      )
    )
    (preference preference2
      (exists (?q - hexagonal_bin ?f - dodgeball)
        (then
          (hold (and (not (not (in_motion ?f) ) ) (rug_color_under agent) (in_motion agent) ) )
        )
      )
    )
    (preference preference3
      (exists (?f ?u - dodgeball)
        (at-end
          (not
            (in_motion rug ?u agent)
          )
        )
      )
    )
  )
)
(:terminal
  (< (count preference3:dodgeball) 3 )
)
(:scoring
  2
)
)


(define (game game-id-36) (:domain medium-objects-room-v1)
(:setup
  (and
    (forall (?a - dodgeball ?w - cube_block)
      (exists (?v - hexagonal_bin)
        (forall (?a - golfball ?i - block ?p - tall_cylindrical_block)
          (exists (?o - (either game_object desktop))
            (and
              (game-optional
                (between ?p)
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
    (forall (?z - teddy_bear)
      (and
        (preference preference1
          (exists (?m - shelf)
            (then
              (hold-while (touch ?m) (not (and (agent_holds ?m ?z) (in_motion ?z) (not (on ?m) ) ) ) (<= (distance 4 ?m) (distance agent desk ?m)) )
              (once (in_motion ?m) )
              (hold-while (rug_color_under ?z) (in_motion ?m) )
            )
          )
        )
      )
    )
    (forall (?y - hexagonal_bin ?q - (either pyramid_block dodgeball))
      (and
        (preference preference2
          (exists (?m - pillow)
            (then
              (once (game_start ?m) )
              (once (adjacent ?q) )
              (once (in ?m) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 3 7 )
)
(:scoring
  (count preference1:red)
)
)


(define (game game-id-37) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (rug_color_under ?xxx ?xxx desk)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?v - doggie_bed)
        (then
          (once (and (and (agent_holds ?v) (on ?v) (on ?v ?v) ) (not (not (touch ?v) ) ) ) )
          (hold (agent_holds ?v) )
          (once (in_motion door) )
        )
      )
    )
    (preference preference2
      (exists (?e - triangular_ramp)
        (then
          (any)
          (once (exists (?z - drawer) (same_type ?e) ) )
          (hold (in_motion ?e desk) )
        )
      )
    )
  )
)
(:terminal
  (>= 10 (+ (* (+ (count-shortest preference1:beachball) 2 )
        8
        (count preference2:yellow:dodgeball)
      )
      (count preference1:brown)
      (* (count-once-per-objects preference2:blue_dodgeball) (count-unique-positions preference2:wall:pink) )
    )
  )
)
(:scoring
  (count-once-per-objects preference2:wall)
)
)


(define (game game-id-38) (:domain many-objects-room-v1)
(:setup
  (and
    (forall (?w - hexagonal_bin)
      (and
        (exists (?q - hexagonal_bin)
          (game-conserved
            (on ?q)
          )
        )
        (and
          (game-conserved
            (agent_holds agent ?w)
          )
        )
        (game-conserved
          (and
            (in_motion ?w)
            (in_motion ?w)
          )
        )
        (exists (?u - pyramid_block ?d - dodgeball)
          (and
            (exists (?e ?b ?j - (either basketball pink) ?z - red_pyramid_block)
              (forall (?s - chair ?b - building)
                (and
                  (forall (?u - cube_block)
                    (and
                      (exists (?i - hexagonal_bin)
                        (game-conserved
                          (above ?w desk)
                        )
                      )
                    )
                  )
                )
              )
            )
            (or
              (game-conserved
                (agent_holds ?d)
              )
              (game-conserved
                (not
                  (and
                    (agent_holds agent)
                    (in_motion ?d)
                    (agent_holds ?d ?w)
                  )
                )
              )
            )
          )
        )
        (game-optional
          (toggled_on ?w)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?y - block)
        (then
          (any)
          (hold (agent_holds ?y ?y) )
          (once (in_motion ?y) )
        )
      )
    )
  )
)
(:terminal
  (>= (* 2 (count preference1:hexagonal_bin) (count preference1:basketball) 10 (count preference1:beachball) (count-once-per-objects preference1:red:blue_dodgeball) )
    (count preference1:yellow:basketball:beachball)
  )
)
(:scoring
  (* (* (count preference1:pink) 6 3 )
    (count preference1:golfball)
  )
)
)


(define (game game-id-39) (:domain medium-objects-room-v1)
(:setup
  (exists (?g - triangular_ramp)
    (forall (?w - doggie_bed)
      (exists (?l - building ?i - dodgeball)
        (and
          (game-conserved
            (and
              (agent_holds ?g ?w)
              (and
                (not
                  (on agent)
                )
                (agent_holds ?w)
                (not
                  (and
                    (in_motion ?i ?w)
                    (agent_holds ?g)
                  )
                )
              )
            )
          )
          (and
            (game-optional
              (in_motion ?w)
            )
            (game-conserved
              (not
                (in ?i ?i)
              )
            )
            (game-conserved
              (and
                (not
                  (not
                    (agent_holds ?w ?i)
                  )
                )
                (or
                  (and
                    (agent_holds ?g)
                    (on desk)
                  )
                  (in_motion ?w)
                  (agent_holds ?i)
                  (in_motion )
                )
              )
            )
          )
          (not
            (and
              (game-conserved
                (not
                  (in ?i)
                )
              )
              (game-conserved
                (agent_holds )
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
    (forall (?m - pillow)
      (and
        (preference preference1
          (exists (?k - block)
            (then
              (hold (and (= (distance 10) 1) (in_motion ?k) (on ?k) ) )
              (hold (and (agent_holds front) (in_motion ?k) ) )
              (hold (and (agent_holds bed ?m) (equal_x_position ?m) (not (on ?k) ) ) )
            )
          )
        )
      )
    )
    (forall (?g - chair)
      (and
        (preference preference2
          (then
            (hold-while (agent_holds ?g ?g) (same_type top_shelf ?g) (not (not (and (on ?g ?g) (not (and (and (on ?g ?g) (in_motion ?g) (rug_color_under left) ) ) ) (same_color ?g ?g) (on ?g ?g) ) ) ) )
            (once (and (not (not (adjacent rug) ) ) (between bed) ) )
            (hold (agent_holds desk) )
          )
        )
        (preference preference3
          (exists (?x - ball ?s - teddy_bear)
            (then
              (hold (agent_holds ?g pink) )
              (hold (in_motion ?s) )
              (hold (in ?s ?s) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (and
    (>= 2 9 )
    (not
      (or
        (>= (* (* 1 10 )
            (+ 10 (total-time) (+ (+ (* 15 2 (* 8 (count preference3:golfball) 0.7 )
                    (+ 5 (+ (* (count preference2:dodgeball) (+ 2 (count preference3:golfball) )
                        )
                        (<= (count preference1:dodgeball) (external-forall-maximize (count-once preference3:doggie_bed) ) )
                      )
                    )
                    (count preference1:beachball)
                    7
                  )
                  (count preference3:dodgeball)
                )
                300
              )
              3
            )
          )
          8
        )
      )
    )
    (or
      (> 2 (not (count preference3:pink_dodgeball) ) )
      (>= 15 (count preference1:basketball) )
    )
  )
)
(:scoring
  (count preference2:pink_dodgeball)
)
)


(define (game game-id-40) (:domain medium-objects-room-v1)
(:setup
  (exists (?b - hexagonal_bin)
    (forall (?r - ball)
      (game-conserved
        (same_color ?b)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?m - hexagonal_bin ?u - color)
        (then
          (hold (forall (?z - hexagonal_bin ?a - chair) (agent_holds east_sliding_door) ) )
          (once (not (< 1 (distance desk 7)) ) )
          (once (not (exists (?z - cube_block) (or (agent_holds ?u rug) (not (agent_holds south_west_corner) ) ) ) ) )
        )
      )
    )
  )
)
(:terminal
  (> (count preference1:cylindrical_block) 1 )
)
(:scoring
  (- 10 )
)
)


(define (game game-id-41) (:domain many-objects-room-v1)
(:setup
  (and
    (forall (?o - wall)
      (game-conserved
        (adjacent_side bed ?o)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?t - chair ?z - chair)
        (then
          (once (not (not (on ?z) ) ) )
          (once (= (distance ?z 9) 5) )
          (once (game_start ?z) )
        )
      )
    )
  )
)
(:terminal
  (or
    (= 5 (count preference1:pink_dodgeball:cube_block) )
    (>= (count preference1:dodgeball) (* 7 (count-measure preference1:dodgeball) )
    )
  )
)
(:scoring
  (and
    (count preference1:basketball)
  )
)
)


(define (game game-id-42) (:domain few-objects-room-v1)
(:setup
  (forall (?h - sliding_door)
    (game-conserved
      (and
        (in ?h)
        (not
          (between agent)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?a - (either pyramid_block hexagonal_bin laptop))
        (then
          (once (< 0 4) )
          (once (or (and (and (agent_holds ?a ?a) (agent_holds front) ) (or (not (not (and (on ?a ?a) (and (agent_holds ?a ?a) (in_motion ?a ?a) (on ?a) ) ) ) ) (in_motion ?a) ) (on door ?a) ) (not (is_setup_object ?a) ) ) )
          (once (not (on pink ?a) ) )
        )
      )
    )
    (preference preference2
      (exists (?c - (either ball cylindrical_block) ?n - dodgeball ?y - (either alarm_clock book key_chain))
        (at-end
          (agent_holds ?y)
        )
      )
    )
    (preference preference3
      (exists (?q - (either flat_block key_chain golfball))
        (then
          (once (in_motion ?q ?q) )
        )
      )
    )
  )
)
(:terminal
  (>= 7 3 )
)
(:scoring
  (count-once-per-objects preference1:dodgeball:dodgeball)
)
)


(define (game game-id-43) (:domain many-objects-room-v1)
(:setup
  (exists (?a - flat_block)
    (forall (?q - (either blue_cube_block main_light_switch))
      (and
        (forall (?h - red_dodgeball)
          (game-conserved
            (agent_holds ?q ?q)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?n - ball ?v - wall ?a - ball ?k ?z - cube_block ?c - block)
      (and
        (preference preference1
          (exists (?h - (either golfball golfball))
            (then
              (hold (not (agent_holds ) ) )
              (hold (is_setup_object ?c ?h) )
              (once (agent_holds ?c ?h) )
            )
          )
        )
      )
    )
    (preference preference2
      (exists (?j - triangular_ramp ?p - wall)
        (then
          (once-measure (on ?p ?p) (distance 5 desk) )
          (once (agent_holds ?p) )
          (hold (in_motion front) )
        )
      )
    )
  )
)
(:terminal
  (>= (* 5 2 )
    (count-once preference1:basketball)
  )
)
(:scoring
  (count preference1:dodgeball)
)
)


(define (game game-id-44) (:domain few-objects-room-v1)
(:setup
  (forall (?e - (either basketball dodgeball) ?n ?h - yellow_cube_block)
    (exists (?u - ball)
      (and
        (forall (?p - red_dodgeball ?w - (either pillow dodgeball) ?b - hexagonal_bin)
          (game-conserved
            (not
              (agent_holds desk)
            )
          )
        )
        (and
          (exists (?v - doggie_bed)
            (forall (?l - watch)
              (exists (?s - dodgeball)
                (game-conserved
                  (not
                    (and
                      (is_setup_object ?l front)
                      (and
                        (in_motion ?v ?s)
                        (not
                          (and
                            (agent_holds ?n ?s)
                            (in desk)
                            (< 1 (x_position ?h 3))
                            (or
                              (in_motion ?v)
                              (and
                                (in_motion ?h ?n ?u)
                                (and
                                  (exists (?p - dodgeball ?z ?q ?m - dodgeball ?e ?z - curved_wooden_ramp ?r - building)
                                    (<= 2 5)
                                  )
                                  (< 2 (distance ?v agent))
                                )
                              )
                            )
                            (on ?v)
                            (not
                              (not
                                (in ?n)
                              )
                            )
                            (in_motion ?s)
                            (exists (?x - (either hexagonal_bin bridge_block))
                              (on ?n)
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
          (game-optional
            (in_motion ?u)
          )
        )
        (forall (?d - beachball ?v - block)
          (forall (?a - shelf ?y - dodgeball ?f - doggie_bed)
            (exists (?q - pyramid_block ?c - dodgeball ?c - hexagonal_bin)
              (game-conserved
                (touch ?n)
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
    (preference preference1
      (exists (?i - hexagonal_bin)
        (at-end
          (not
            (agent_holds ?i agent)
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (or
      (< 3 (count-overlapping preference1:basketball:tall_cylindrical_block:alarm_clock) )
      (>= (* 1 (- (count preference1:basketball) )
          (count-once-per-external-objects preference1:hexagonal_bin)
        )
        (not
          30
        )
      )
    )
  )
)
(:scoring
  (count preference1:pink)
)
)


(define (game game-id-45) (:domain many-objects-room-v1)
(:setup
  (and
    (forall (?b - (either dodgeball laptop))
      (game-conserved
        (<= (distance ?b ?b) (distance ?b ?b back))
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?e - block)
        (at-end
          (= (distance ) 7)
        )
      )
    )
    (preference preference2
      (exists (?u - dodgeball)
        (then
          (once (touch ?u) )
          (hold-to-end (same_type ?u ?u) )
          (once (= 1 4 1) )
        )
      )
    )
  )
)
(:terminal
  (> (* 2 (count preference2:red) )
    3
  )
)
(:scoring
  (total-time)
)
)


(define (game game-id-46) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (adjacent ?xxx ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?j - curved_wooden_ramp ?l - (either alarm_clock dodgeball))
        (then
          (once (not (adjacent ?l ?l) ) )
          (once-measure (adjacent_side ?l bed) (distance 4) )
          (hold (in_motion ?l) )
        )
      )
    )
    (forall (?n ?m - hexagonal_bin)
      (and
        (preference preference2
          (exists (?k - building)
            (then
              (once (in ?m ?k) )
              (hold (agent_holds ?m) )
              (hold-while (exists (?v - dodgeball) (agent_holds ?m) ) (and (exists (?q - red_dodgeball ?l - block) (not (agent_holds desk ?m) ) ) (not (= 0.5 2) ) ) (same_color ?k) )
            )
          )
        )
        (preference preference3
          (exists (?g - desk_shelf)
            (then
              (once (not (agent_holds agent green) ) )
              (hold-while (or (and (in ?g ?n ?g) (adjacent ?g) ) (on ?g ?m) ) (on ?m) )
              (once (and (on ?m) (and (and (above ?m ?m) (in_motion ?n) ) (in_motion ?g ?g) (agent_holds ?n) (agent_holds ?g) (rug_color_under ?n) (in_motion ?n) (on ?m) (not (in_motion ?m) ) (touch ?m) (not (not (agent_holds ?g ?n) ) ) (and (agent_holds ?n desk) (in_motion ?g) ) (and (same_type ?n) (and (and (not (agent_holds ?n) ) (and (touch ?g ?g) (touch ?g) (in_motion bed ?g) ) ) (< (distance agent desk) 1) ) (and (in ?n ?m) (agent_holds floor) ) ) ) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (or
      (>= 4 (count preference2:white) )
      (not
        (or
          (<= 6 (count-longest preference3:dodgeball) )
          (>= 5 (total-time) )
        )
      )
      (not
        (>= (total-score) (count-once-per-objects preference2:red:dodgeball) )
      )
    )
    (>= (+ 5 (* (count-once-per-objects preference2:basketball) (* (external-forall-maximize (count preference1:dodgeball:basketball) ) (* (count-longest preference1:golfball) 10 4 )
          )
          (count-once-per-objects preference2)
          (* (count-once-per-objects preference2:pink:basketball) (or 3 ) )
        )
      )
      2
    )
    (<= (* (+ (count-once-per-objects preference1:dodgeball) (count-once-per-objects preference2:dodgeball) (+ (= 9 (count preference3:basketball:pink_dodgeball) )
            (count preference3:pink)
          )
        )
        (count-shortest preference3:dodgeball)
      )
      (count-overlapping preference2:hexagonal_bin:yellow)
    )
  )
)
(:scoring
  2
)
)


(define (game game-id-47) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (not
      (agent_holds door)
    )
  )
)
(:constraints
  (and
    (forall (?y - wall)
      (and
        (preference preference1
          (exists (?w ?j - doggie_bed)
            (then
              (once (touch ?j) )
              (hold-while (on ?w ?j) (not (in_motion ?w ?y) ) (agent_holds ?j) (not (not (in desk) ) ) )
              (once (or (equal_z_position ?w ?w) (and (in ?y ?j) (touch ?y) (agent_holds ?w ?w) ) ) )
            )
          )
        )
        (preference preference2
          (exists (?u - (either flat_block blue_cube_block cylindrical_block))
            (then
              (once (in_motion ?u) )
              (once (in_motion ?y) )
              (once (not (and (in_motion ?u ?y) (and (not (and (forall (?n - teddy_bear) (not (and (not (and (in_motion right) (and (not (agent_holds ?n) ) (in ?n) ) (not (and (not (and (in_motion ?n) (not (in ?y ?y) ) ) ) (touch ?n agent) (agent_holds pink_dodgeball ?u) ) ) ) ) (in_motion ?y) ) ) ) (in ?u) ) ) (in_motion bed ?u) (and (in ?y) (touch ?y) (agent_holds ?y) ) ) (agent_holds ?u ?y) ) ) )
            )
          )
        )
        (preference preference3
          (exists (?g - dodgeball ?g - ball)
            (then
              (hold (agent_holds ?g ?g) )
              (hold (and (not (in_motion ?g) ) ) )
              (hold (not (and (and (= (distance ?y)) (agent_holds ?g ?g) ) (and (in_motion ?g ?g) (and (< (distance room_center ?y) 1) (and (and (agent_holds agent ?g) (not (not (< (distance 9 3) 1) ) ) ) (not (in_motion agent) ) ) ) ) ) ) )
            )
          )
        )
      )
    )
    (preference preference4
      (exists (?d - cylindrical_block)
        (then
          (once (in_motion pink_dodgeball) )
          (hold (in_motion floor agent) )
          (hold-while (on ?d) (not (in_motion agent ?d) ) (in_motion ?d) )
          (once (agent_holds ?d) )
          (once (not (not (agent_holds ?d) ) ) )
          (once (not (on ?d) ) )
        )
      )
    )
  )
)
(:terminal
  (>= 20 (total-score) )
)
(:scoring
  3
)
)


(define (game game-id-48) (:domain few-objects-room-v1)
(:setup
  (and
    (and
      (and
        (forall (?j ?y - hexagonal_bin)
          (game-conserved
            (in_motion agent)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?s - dodgeball ?p - (either dodgeball golfball) ?l - curved_wooden_ramp ?n - (either laptop dodgeball triangle_block))
      (and
        (preference preference1
          (exists (?w - curved_wooden_ramp)
            (at-end
              (not
                (not
                  (in_motion ?w)
                )
              )
            )
          )
        )
        (preference preference2
          (exists (?c - building)
            (then
              (hold (not (> (distance desk ?n) (distance 8 desk)) ) )
              (once (above ?n) )
              (once (in_motion ?c) )
            )
          )
        )
      )
    )
    (forall (?d - game_object ?k - pillow)
      (and
        (preference preference3
          (exists (?m - hexagonal_bin ?d - dodgeball)
            (then
              (once (in_motion ?k) )
              (once (not (not (agent_holds bed agent) ) ) )
              (once (>= 1 (distance agent ?d)) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (external-forall-maximize (* (external-forall-maximize (* (+ 4 4 2 )
            (count preference1:dodgeball)
          )
        )
        (external-forall-maximize
          (* (count-once-per-objects preference1:purple) (count preference2:pink:golfball) )
        )
        5
      )
    )
    2
  )
)
(:scoring
  (count preference2:dodgeball)
)
)


(define (game game-id-49) (:domain many-objects-room-v1)
(:setup
  (and
    (and
      (and
        (forall (?e - hexagonal_bin ?c - hexagonal_bin)
          (exists (?f - doggie_bed)
            (and
              (not
                (and
                  (and
                    (exists (?v ?a ?s ?w ?u ?d ?t ?i ?j ?r - dodgeball)
                      (exists (?e ?p - (either golfball))
                        (game-conserved
                          (and
                            (or
                              (adjacent ?w ?w)
                              (in_motion agent ?w)
                              (in ?a green)
                            )
                            (< 1 (distance bed 5))
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
      )
    )
  )
)
(:constraints
  (and
    (forall (?f - dodgeball)
      (and
        (preference preference1
          (exists (?n - desktop ?x - building)
            (then
              (hold (on ?x ?f) )
              (hold (in_motion ?x) )
              (once (touch ?f ?f) )
            )
          )
        )
      )
    )
    (preference preference2
      (exists (?e - curved_wooden_ramp)
        (then
          (hold (and (same_object ?e ?e) (in_motion ?e) ) )
          (hold-while (in_motion door ?e) (agent_holds ?e ?e) (in_motion ?e) )
          (hold-while (not (agent_holds ?e) ) (and (agent_holds ?e) (not (not (in_motion ?e bridge_block) ) ) ) (in ?e ?e) )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (count preference1:pink_dodgeball) (* (total-time) )
    )
    (= 10 (- (* (external-forall-maximize 30 ) 6 )
      )
    )
  )
)
(:scoring
  8
)
)


(define (game game-id-50) (:domain many-objects-room-v1)
(:setup
  (forall (?t - ball ?v - chair)
    (exists (?f - hexagonal_bin)
      (game-conserved
        (agent_holds agent)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?z - teddy_bear ?f - dodgeball)
        (then
          (hold (agent_holds ?f) )
          (once (and (not (agent_holds ?f) ) ) )
          (hold (and (in_motion agent) (not (and (and (agent_holds ?f) (in_motion ?f) ) (toggled_on ) ) ) ) )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (external-forall-maximize (* (total-time) (total-time) )
      )
      4
    )
    (>= 10 5 )
  )
)
(:scoring
  (count preference1:red:blue_pyramid_block)
)
)


(define (game game-id-51) (:domain many-objects-room-v1)
(:setup
  (and
    (and
      (game-optional
        (and
          (< 1 1)
          (on ?xxx bed ?xxx)
          (adjacent ?xxx)
        )
      )
    )
    (forall (?q - game_object)
      (not
        (game-conserved
          (in_motion agent sideways)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?t - building ?j - block)
      (and
        (preference preference1
          (exists (?s - hexagonal_bin)
            (at-end
              (not
                (and
                  (in_motion ?s)
                  (not
                    (in_motion ?j)
                  )
                  (agent_holds ?s ?j)
                  (above ?j rug)
                )
              )
            )
          )
        )
      )
    )
    (preference preference2
      (exists (?m - wall ?v - building)
        (then
          (once (forall (?j - game_object) (in_motion agent ?j) ) )
          (once (not (agent_holds ?v) ) )
          (once (and (not (agent_holds ?v ?v) ) (in ?v ?v) ) )
        )
      )
    )
  )
)
(:terminal
  (<= (count preference2:cube_block) (+ 0 (and (count-once-per-objects preference2:yellow) (count-once-per-objects preference1:blue_cube_block:dodgeball) ) )
  )
)
(:scoring
  (count-overlapping preference1:bed)
)
)


(define (game game-id-52) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (in_motion ?xxx rug ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (same_color ?xxx) )
        (hold (and (and (not (in_motion ?xxx rug) ) (in ?xxx ?xxx) ) (and (object_orientation ?xxx agent) (and (touch ?xxx) (adjacent ?xxx agent) ) ) ) )
        (hold (in ?xxx) )
      )
    )
    (preference preference2
      (exists (?p - block)
        (then
          (hold-while (and (object_orientation ?p ?p) (or (agent_holds rug ?p) (on ?p) ) (and (on ?p) (not (in_motion ) ) ) ) (and (on ?p) (and (agent_holds floor ?p) (is_setup_object ?p ?p) ) ) (touch ?p right) )
          (once (in_motion ?p ?p) )
          (once (in door) )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference1:dodgeball:pink_dodgeball) 5 )
)
(:scoring
  (- (count preference1:triangle_block) )
)
)


(define (game game-id-53) (:domain few-objects-room-v1)
(:setup
  (exists (?r - game_object ?v - dodgeball)
    (exists (?y - (either laptop cube_block) ?m - building ?o - dodgeball)
      (forall (?h - (either alarm_clock golfball))
        (and
          (and
            (game-conserved
              (on brown desk)
            )
            (and
              (game-conserved
                (on ?h)
              )
            )
            (forall (?c - game_object ?j - cube_block)
              (exists (?f - game_object)
                (game-conserved
                  (on ?h ?j)
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
    (preference preference1
      (exists (?k - hexagonal_bin ?i - dodgeball ?k - dodgeball)
        (then
          (hold (and (not (in_motion ?k ?k) ) (in_motion ?k ?k) (touch brown) ) )
          (once (agent_holds ?k) )
          (hold (not (<= 9 (distance ?k)) ) )
        )
      )
    )
  )
)
(:terminal
  (> (count preference1:dodgeball) 10 )
)
(:scoring
  (count preference1:red)
)
)


(define (game game-id-54) (:domain medium-objects-room-v1)
(:setup
  (forall (?b - hexagonal_bin)
    (game-conserved
      (in ?b)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?x ?u - (either yellow lamp hexagonal_bin pink basketball) ?z - chair)
        (then
          (once (< (distance ?z ?z) 5) )
          (once (in_motion ?z) )
          (hold-while (on desk) (not (agent_holds ?z ?z) ) )
        )
      )
    )
  )
)
(:terminal
  (>= (count-overlapping preference1:dodgeball:yellow) 4 )
)
(:scoring
  (count preference1:beachball:book)
)
)


(define (game game-id-55) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (and
      (not
        (and
          (object_orientation ?xxx ?xxx)
          (on ?xxx)
        )
      )
      (and
        (< (distance room_center ?xxx) 4)
        (not
          (in agent ?xxx)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?t - (either key_chain wall alarm_clock) ?i ?c - hexagonal_bin)
        (then
          (hold (touch ?c) )
          (hold (game_over ?c) )
          (any)
        )
      )
    )
    (preference preference2
      (exists (?h - hexagonal_bin ?x - building ?k - chair)
        (at-end
          (on ?k ?k)
        )
      )
    )
  )
)
(:terminal
  (> (* 5 (count preference1:dodgeball) )
    (count preference2:golfball:basketball)
  )
)
(:scoring
  3
)
)


(define (game game-id-56) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (in_motion ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?f - cylindrical_block)
        (then
          (once (not (on ?f ?f ?f) ) )
          (once (and (agent_holds agent) (= 3 10) ) )
          (once (adjacent ?f ?f ?f) )
        )
      )
    )
  )
)
(:terminal
  (> 5 (count-once-per-external-objects preference1:golfball) )
)
(:scoring
  2
)
)


(define (game game-id-57) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (agent_holds ?xxx ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?q - dodgeball ?g - block)
        (then
          (hold-while (agent_holds ?g) (in ?g ?g) (not (in ?g) ) )
          (once (on ?g) )
          (hold (agent_holds ?g) )
        )
      )
    )
    (preference preference2
      (exists (?q - dodgeball)
        (then
          (once (agent_holds agent) )
          (hold (not (agent_holds agent ?q) ) )
          (hold-while (not (not (in_motion ?q) ) ) (in ?q) )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (* (count-once-per-objects preference2:dodgeball) 3 )
      5
    )
    (and
      (or
        (>= (* (* (count-once-per-objects preference2:dodgeball) 8 )
            7
          )
          5
        )
        (or
          (>= (* (+ 9 (* 4 (count preference2:hexagonal_bin) )
              )
              (count preference2:dodgeball:red_pyramid_block)
            )
            (/
              (count-same-positions preference1:dodgeball:pink:yellow)
              1
            )
          )
          (= (+ (count preference1:top_drawer) (= (external-forall-minimize (* (* 10 (- (count preference2:golfball:doggie_bed:hexagonal_bin) )
                    )
                    (count preference2:pink)
                  )
                )
                20
              )
              (count-once-per-objects preference1:dodgeball:dodgeball)
            )
            50
          )
          (>= (count-once-per-objects preference1:golfball:dodgeball) (count preference1:purple:dodgeball) )
        )
      )
    )
  )
)
(:scoring
  (count preference2:golfball)
)
)


(define (game game-id-58) (:domain few-objects-room-v1)
(:setup
  (forall (?h - ball ?l - chair ?x - tall_cylindrical_block)
    (and
      (game-optional
        (not
          (or
            (and
              (or
                (on ?x)
                (not
                  (agent_holds ?x)
                )
              )
              (in_motion ?x ?x)
            )
            (in ?x)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?i - blue_pyramid_block)
      (and
        (preference preference1
          (exists (?o - (either dodgeball cube_block) ?f - ball)
            (then
              (once (not (open ?f) ) )
              (hold-while (is_setup_object ?i) (in_motion ?i) )
              (once (agent_holds ?i) )
            )
          )
        )
      )
    )
    (preference preference2
      (exists (?n ?z - dodgeball ?e - golfball ?t - yellow_cube_block ?s - pyramid_block)
        (then
          (hold (<= 10 1) )
          (once (agent_holds ?s ?s) )
          (hold (on ?s) )
        )
      )
    )
  )
)
(:terminal
  (< (external-forall-maximize (count-once-per-objects preference1:yellow) ) (+ (count preference1:blue_cube_block:doggie_bed) 7 )
  )
)
(:scoring
  (+ (* (- 2 )
      (* 10 2 )
    )
    (count-once-per-objects preference1:triangle_block)
  )
)
)


(define (game game-id-59) (:domain medium-objects-room-v1)
(:setup
  (and
    (game-conserved
      (not
        (in ?xxx)
      )
    )
    (and
      (forall (?e - doggie_bed ?f - building)
        (game-optional
          (not
            (agent_holds ?f ?f)
          )
        )
      )
      (and
        (exists (?j - ball)
          (and
            (game-optional
              (not
                (and
                  (and
                    (not
                      (in ?j ?j)
                    )
                    (same_color ?j front)
                  )
                  (agent_holds desk)
                  (agent_holds ?j)
                )
              )
            )
            (exists (?a - ball)
              (game-optional
                (in_motion ?a agent)
              )
            )
            (exists (?u - dodgeball ?v - game_object)
              (exists (?p - dodgeball)
                (and
                  (forall (?u - game_object ?w - hexagonal_bin)
                    (and
                      (forall (?a - hexagonal_bin ?s - (either basketball cd blue_cube_block))
                        (game-optional
                          (not
                            (on ?v)
                          )
                        )
                      )
                      (and
                        (game-optional
                          (agent_holds ?w)
                        )
                      )
                    )
                  )
                  (game-optional
                    (not
                      (adjacent ?v)
                    )
                  )
                  (and
                    (exists (?q - dodgeball)
                      (exists (?y - (either alarm_clock pyramid_block dodgeball) ?o - curved_wooden_ramp)
                        (game-optional
                          (< 0.5 7)
                        )
                      )
                    )
                  )
                  (exists (?d - dodgeball)
                    (forall (?a - building)
                      (exists (?q ?o - hexagonal_bin)
                        (exists (?s - building)
                          (game-conserved
                            (and
                              (and
                                (or
                                  (not
                                    (exists (?z - (either dodgeball chair))
                                      (equal_z_position ?d)
                                    )
                                  )
                                )
                                (and
                                  (not
                                    (not
                                      (not
                                        (adjacent desk ?a)
                                      )
                                    )
                                  )
                                  (agent_holds ?q)
                                )
                                (> 1 0)
                                (and
                                  (same_object ?o ?a)
                                  (adjacent ?a ?p)
                                )
                                (in ?a)
                                (touch ?s)
                                (not
                                  (and
                                    (not
                                      (and
                                        (and
                                          (forall (?x - shelf ?e - hexagonal_bin ?m - game_object ?r - building)
                                            (in_motion agent ?v)
                                          )
                                          (not
                                            (not
                                              (in_motion ?q)
                                            )
                                          )
                                        )
                                        (not
                                          (not
                                            (not
                                              (< (building_size room_center 5) 1)
                                            )
                                          )
                                        )
                                        (not
                                          (in ?p)
                                        )
                                      )
                                    )
                                    (agent_holds ?a)
                                  )
                                )
                                (adjacent ?o ?d)
                              )
                              (touch ?s)
                            )
                          )
                        )
                      )
                    )
                  )
                  (game-optional
                    (not
                      (on ?v ?p)
                    )
                  )
                )
              )
            )
          )
        )
        (and
          (game-conserved
            (in_motion agent ?xxx)
          )
        )
      )
      (forall (?t - (either basketball golfball pyramid_block))
        (game-conserved
          (adjacent ?t)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?c - (either pyramid_block))
        (then
          (once (not (< 8 1) ) )
          (once (in_motion ?c) )
          (once (not (in_motion ?c) ) )
        )
      )
    )
  )
)
(:terminal
  (>= 1 5 )
)
(:scoring
  (count preference1:golfball)
)
)


(define (game game-id-60) (:domain few-objects-room-v1)
(:setup
  (exists (?z - wall ?w - cube_block)
    (or
      (game-optional
        (or
          (not
            (in_motion bed ?w)
          )
          (< 1 1)
        )
      )
      (game-optional
        (agent_holds ?w ?w)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?e - hexagonal_bin ?q - dodgeball ?c - (either ball watch))
        (then
          (once (and (touch ?c) (in_motion blue) ) )
          (once (same_color ?c ?c) )
          (once (and (not (and (not (in_motion agent ?c) ) (and (and (not (agent_holds ?c) ) (in_motion ?c agent) ) (not (touch front bed) ) ) ) ) (in ?c ?c) (and (agent_holds ?c ?c) (agent_holds ?c ?c) (in_motion floor) ) ) )
        )
      )
    )
    (preference preference2
      (exists (?i ?k - color)
        (at-end
          (agent_holds ?i)
        )
      )
    )
    (forall (?f - dodgeball)
      (and
        (preference preference3
          (exists (?u - hexagonal_bin ?k - hexagonal_bin ?b - game_object ?l - dodgeball)
            (then
              (once (exists (?k ?e - (either lamp hexagonal_bin) ?q - hexagonal_bin ?v - cube_block) (in_motion ?f ?v) ) )
              (once (> (distance 0) 1) )
              (once (same_color ?l ?f) )
            )
          )
        )
        (preference preference4
          (exists (?p - triangular_ramp)
            (then
              (hold (agent_holds ?p) )
              (hold (in_motion pink) )
              (hold (not (in_motion ?f) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count-once-per-objects preference2:pink_dodgeball) (count-once preference4:book) )
)
(:scoring
  (count preference4:red)
)
)


(define (game game-id-61) (:domain few-objects-room-v1)
(:setup
  (and
    (exists (?l - (either dodgeball chair))
      (and
        (game-conserved
          (and
            (and
              (between ?l)
              (on bed ?l)
              (agent_holds ?l ?l)
            )
            (and
              (agent_holds ?l)
              (agent_holds ?l ?l)
            )
          )
        )
        (exists (?c - block ?h - hexagonal_bin)
          (game-optional
            (adjacent ?h)
          )
        )
        (game-conserved
          (in_motion ?l)
        )
      )
    )
    (and
      (exists (?d - building)
        (forall (?b - ball)
          (exists (?o - (either laptop tall_cylindrical_block dodgeball top_drawer key_chain book golfball) ?z - (either alarm_clock pencil basketball) ?e - triangular_ramp)
            (exists (?w - block ?k - ball ?p - block)
              (and
                (forall (?t ?q - (either cd))
                  (game-conserved
                    (agent_holds ?q)
                  )
                )
              )
            )
          )
        )
      )
      (game-optional
        (on ?xxx front)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?g - dodgeball ?u - bridge_block ?z ?i - dodgeball ?i - cube_block ?x - ball)
        (then
          (once (equal_x_position sideways) )
          (once (agent_holds ?x) )
          (once (and (and (not (not (in_motion ?x) ) ) (not (in ?x) ) (in ?x) (not (in_motion ?x) ) (and (not (not (touch ?x front) ) ) (in_motion ?x ?x) ) (in ?x ?x) ) ) )
        )
      )
    )
  )
)
(:terminal
  (>= (= (count preference1:basketball) (* 2 (count-once preference1:dodgeball) )
      3
    )
    (<= (* (* (count-once-per-objects preference1:golfball) (total-score) )
        2
      )
      30
    )
  )
)
(:scoring
  (* (count preference1:basketball) (count preference1:green) )
)
)


(define (game game-id-62) (:domain medium-objects-room-v1)
(:setup
  (and
    (game-optional
      (adjacent ?xxx ?xxx)
    )
    (and
      (game-conserved
        (agent_holds ?xxx ?xxx)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (at-end
        (on ?xxx)
      )
    )
  )
)
(:terminal
  (>= (count-once-per-objects preference1:green) (* 4 (external-forall-maximize (count-same-positions preference1:yellow:tan) ) )
  )
)
(:scoring
  (* (count preference1:purple:yellow) (- (- (- (total-time) )
      )
      1
    )
    (* 7 10 (count-once-per-objects preference1:beachball:red) (count-once-per-objects preference1:basketball:green) )
  )
)
)


(define (game game-id-63) (:domain medium-objects-room-v1)
(:setup
  (and
    (game-optional
      (not
        (agent_holds ?xxx yellow)
      )
    )
    (and
      (exists (?y - (either dodgeball hexagonal_bin cylindrical_block))
        (and
          (and
            (game-optional
              (not
                (not
                  (agent_holds ?y)
                )
              )
            )
            (and
              (and
                (and
                  (game-conserved
                    (not
                      (on rug upright)
                    )
                  )
                )
              )
            )
            (exists (?h - dodgeball)
              (exists (?c - (either dodgeball laptop))
                (and
                  (and
                    (and
                      (game-conserved
                        (or
                          (not
                            (< (distance 4 ?y ?h) 2)
                          )
                          (on ?y)
                        )
                      )
                    )
                    (game-optional
                      (touch rug)
                    )
                    (forall (?x - shelf)
                      (game-conserved
                        (agent_holds desk desk)
                      )
                    )
                  )
                )
              )
            )
          )
          (and
            (and
              (game-conserved
                (agent_holds ?y ?y)
              )
              (exists (?q - shelf ?r - game_object)
                (game-optional
                  (and
                    (= (distance desk))
                    (adjacent ?y)
                  )
                )
              )
              (game-conserved
                (not
                  (in_motion ?y)
                )
              )
              (game-optional
                (not
                  (exists (?f - block)
                    (in ?f)
                  )
                )
              )
              (game-conserved
                (in_motion agent)
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
    (preference preference1
      (then
        (hold (adjacent ?xxx ?xxx) )
        (hold (game_over ?xxx) )
        (hold-to-end (and (not (adjacent ?xxx) ) (not (rug_color_under ?xxx bed) ) (and (agent_holds ?xxx pink) (agent_holds ?xxx ?xxx) ) ) )
      )
    )
  )
)
(:terminal
  (and
    (>= 10 (count preference1:basketball) )
    (>= (count preference1:book) (* (total-score) (* 2 10 )
        (count-once-per-external-objects preference1:blue_dodgeball:red)
        0
        (* (+ (count preference1:green) (external-forall-minimize (count-once-per-objects preference1:dodgeball:golfball) ) )
          (= (+ 1 3 )
            (external-forall-maximize
              10
            )
          )
        )
        (count preference1:basketball:hexagonal_bin)
      )
    )
    (>= (* (count-once-per-objects preference1:hexagonal_bin:orange) (count preference1:blue_dodgeball) )
      (count preference1:red)
    )
  )
)
(:scoring
  (count-once-per-objects preference1:blue_dodgeball:yellow)
)
)


(define (game game-id-64) (:domain many-objects-room-v1)
(:setup
  (exists (?e - beachball)
    (and
      (or
        (forall (?f - building)
          (exists (?p - hexagonal_bin)
            (forall (?i ?a ?z ?j ?v ?h - ball ?y - (either flat_block cube_block) ?i ?h - hexagonal_bin)
              (and
                (game-optional
                  (not
                    (on ?i)
                  )
                )
                (exists (?n - game_object)
                  (game-optional
                    (on ?i ?e)
                  )
                )
              )
            )
          )
        )
        (game-conserved
          (in green ?e)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?w - game_object)
      (and
        (preference preference1
          (exists (?r - building)
            (then
              (once (in_motion ?w ?r) )
              (hold-while (in_motion ?w) (in ?w) )
              (once (in_motion ?r) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (> (count-once-per-external-objects preference1:basketball:dodgeball) (count preference1:yellow:dodgeball) )
)
(:scoring
  2
)
)


(define (game game-id-65) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (agent_holds ?xxx ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (at-end
        (agent_holds floor)
      )
    )
  )
)
(:terminal
  (>= 2 2 )
)
(:scoring
  (+ 3 (* (count preference1:yellow) 2 (count preference1:golfball) )
  )
)
)


(define (game game-id-66) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (not
      (in_motion ?xxx)
    )
  )
)
(:constraints
  (and
    (forall (?n - desk_shelf)
      (and
        (preference preference1
          (then
            (once (and (in_motion ?n ?n) (in ?n ?n) ) )
            (hold (adjacent ?n sideways) )
            (hold (and (and (in_motion ?n) (in ?n) ) (not (adjacent ?n ?n) ) ) )
          )
        )
        (preference preference2
          (exists (?x - ball)
            (then
              (hold-while (agent_holds side_table) (< 1 1) )
              (once (agent_holds ?x ?n) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (> (/ (+ 5 (count-once-per-objects preference2:beachball:cylindrical_block) )
      (count preference2:dodgeball)
    )
    (total-score)
  )
)
(:scoring
  (count preference2:pink_dodgeball)
)
)


(define (game game-id-67) (:domain few-objects-room-v1)
(:setup
  (exists (?i - hexagonal_bin)
    (and
      (game-conserved
        (not
          (agent_holds ?i)
        )
      )
      (game-conserved
        (agent_holds ?i)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?h - cube_block)
        (then
          (hold (or (agent_holds ?h ?h) (agent_holds ?h) (in_motion ?h) ) )
          (once (agent_holds ?h pink_dodgeball ?h) )
          (hold-while (in_motion ?h) (agent_holds ?h) )
          (once (forall (?v - red_dodgeball ?q - hexagonal_bin) (agent_holds ?q ?q) ) )
        )
      )
    )
    (preference preference2
      (exists (?j - teddy_bear)
        (at-end
          (not
            (not
              (and
                (adjacent ?j)
                (touch ?j)
                (and
                  (and
                    (< (distance ?j 2) 4)
                    (agent_holds ?j)
                  )
                  (not
                    (in_motion ?j ?j)
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
(:terminal
  (>= (* 10 3 )
    (count preference2:bed)
  )
)
(:scoring
  (* (count-once-per-external-objects preference2:red:pink) (count-once-per-objects preference2:golfball) )
)
)


(define (game game-id-68) (:domain many-objects-room-v1)
(:setup
  (and
    (exists (?g - (either hexagonal_bin dodgeball))
      (game-conserved
        (in_motion ?g)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?q - (either dodgeball dodgeball) ?x - game_object ?n - game_object ?k - dodgeball ?p - hexagonal_bin)
        (then
          (hold (not (on rug) ) )
          (hold (on desk) )
          (once (in_motion ?p ?p) )
        )
      )
    )
    (preference preference2
      (exists (?x - hexagonal_bin ?p - block)
        (then
          (once (or (in_motion desk ?p) (agent_holds east_sliding_door) (on ?p) ) )
          (once (and (in_motion ?p) (agent_holds ?p ?p) ) )
          (once (agent_holds desk) )
        )
      )
    )
    (forall (?h - shelf ?k - dodgeball)
      (and
        (preference preference3
          (exists (?w - game_object ?e - ball)
            (at-end
              (and
                (in_motion ?e)
              )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (> 7 20 )
)
(:scoring
  (count preference3:red)
)
)


(define (game game-id-69) (:domain many-objects-room-v1)
(:setup
  (and
    (exists (?o - dodgeball)
      (exists (?p - beachball)
        (game-conserved
          (and
            (exists (?j - cube_block ?j - hexagonal_bin ?q - game_object)
              (not
                (not
                  (agent_holds desk)
                )
              )
            )
            (agent_holds ?p)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?y - bridge_block ?q ?x - teddy_bear)
        (at-end
          (agent_holds ?q)
        )
      )
    )
  )
)
(:terminal
  (= (- (+ (count-once-per-external-objects preference1:blue_dodgeball) (count preference1:blue_dodgeball) (external-forall-maximize (* (count-once-per-objects preference1:basketball) )
        )
      )
    )
    (* (count-once-per-objects preference1:golfball) (count preference1:blue_dodgeball) )
  )
)
(:scoring
  3
)
)


(define (game game-id-70) (:domain medium-objects-room-v1)
(:setup
  (game-optional
    (adjacent ?xxx)
  )
)
(:constraints
  (and
    (forall (?v - game_object)
      (and
        (preference preference1
          (exists (?q - (either cd main_light_switch) ?m - hexagonal_bin)
            (then
              (once (adjacent ?v) )
              (hold (in_motion agent) )
              (hold-to-end (in_motion ?v) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (+ (* 5 2 )
      (* (count preference1:blue_dodgeball) (count preference1:purple) )
      (count-total preference1:blue_dodgeball:dodgeball)
      (- 12 )
      2
      (count preference1:dodgeball)
    )
    5
  )
)
(:scoring
  (total-time)
)
)


(define (game game-id-71) (:domain few-objects-room-v1)
(:setup
  (and
    (exists (?f - (either dodgeball yellow_cube_block))
      (game-conserved
        (in ?f)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?z - dodgeball)
        (then
          (hold (agent_holds ?z ?z) )
          (hold-while (on ball) (in_motion ?z ?z) )
          (once (between ?z) )
        )
      )
    )
    (preference preference2
      (exists (?l - (either laptop bridge_block))
        (then
          (once (in_motion ?l ?l) )
          (once (> 3 1) )
          (once (agent_holds ?l ?l) )
        )
      )
    )
  )
)
(:terminal
  (<= (total-score) (count preference1:side_table) )
)
(:scoring
  (not
    (count-measure preference1:beachball)
  )
)
)


(define (game game-id-72) (:domain medium-objects-room-v1)
(:setup
  (and
    (and
      (and
        (game-conserved
          (agent_holds ?xxx)
        )
        (game-optional
          (touch pink)
        )
      )
    )
    (forall (?n ?c ?o - cube_block)
      (and
        (game-conserved
          (same_type rug ?n)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?a - shelf)
        (then
          (hold-while (in_motion ?a) (on desk) )
          (hold-while (and (not (in ?a) ) (in ?a) ) (on ?a) (in ?a floor) )
          (once-measure (agent_holds ?a) (distance ) )
        )
      )
    )
  )
)
(:terminal
  (>= 5 (count-once-per-objects preference1:green) )
)
(:scoring
  10
)
)


(define (game game-id-73) (:domain many-objects-room-v1)
(:setup
  (exists (?w - teddy_bear ?a - (either pyramid_block cylindrical_block))
    (game-conserved
      (forall (?y - hexagonal_bin)
        (in floor ?y)
      )
    )
  )
)
(:constraints
  (and
    (forall (?k ?o ?g ?f ?v ?a - hexagonal_bin ?w - (either key_chain book))
      (and
        (preference preference1
          (exists (?m - curved_wooden_ramp)
            (then
              (once (in_motion ?w ?m) )
              (hold (not (exists (?k - ball ?u - wall) (in brown) ) ) )
              (hold-to-end (touch ?w rug) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (external-forall-minimize 5 ) (* (count-overlapping preference1:dodgeball) (count preference1:basketball:blue_pyramid_block) )
  )
)
(:scoring
  (count-measure preference1:dodgeball:orange)
)
)


(define (game game-id-74) (:domain many-objects-room-v1)
(:setup
  (and
    (and
      (game-conserved
        (and
          (and
            (not
              (not
                (not
                  (not
                    (exists (?s - hexagonal_bin ?e - sliding_door)
                      (not
                        (in_motion ?e)
                      )
                    )
                  )
                )
              )
            )
            (agent_holds ?xxx ?xxx)
          )
          (not
            (and
              (same_color front)
              (in_motion ?xxx ?xxx)
            )
          )
          (and
            (agent_holds rug desktop)
            (not
              (in_motion ?xxx ?xxx)
            )
            (same_type ?xxx)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?g - hexagonal_bin)
        (then
          (once (in_motion ?g ?g) )
          (once (and (not (and (on agent) (in_motion ?g ?g) (and (not (in_motion ?g ?g) ) (agent_holds ?g) (equal_z_position upright) ) (same_type ?g agent ?g) ) ) (and (agent_holds ?g bed) (and (on ?g ?g) (= 1 8 1 0) ) (not (in ?g) ) ) ) )
          (once (in_motion ?g) )
        )
      )
    )
  )
)
(:terminal
  (> 3 (+ (external-forall-minimize (external-forall-maximize (count-once-per-objects preference1:beachball) ) ) )
  )
)
(:scoring
  (* (* (/ (* (* (count-once preference1:beachball) 10 20 (* (* (count preference1:red) (count preference1:red) 10 (count preference1:pink_dodgeball) (count preference1:top_drawer:pink) (count preference1:orange) )
              6
            )
          )
          3
        )
        (count-once-per-objects preference1:beachball)
      )
      (count preference1:yellow)
    )
    6
  )
)
)


(define (game game-id-75) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (agent_holds ?xxx ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?l - hexagonal_bin ?j - dodgeball)
        (then
          (once (in_motion floor) )
          (hold-for 0 (in agent ?j) )
          (once (and (and (and (in_motion ?j) (and (on floor ?j) (not (and (not (and (not (< (distance side_table ?j) 6) ) (not (not (and (not (on ?j ?j) ) (not (and (on floor ?j) (in_motion ?j) (and (and (forall (?m - doggie_bed) (in_motion ?j ?j) ) (exists (?x - ball) (< 1 (x_position desk ?x)) ) (on ?j ?j) ) (in_motion ?j ?j) ) (agent_holds ?j) ) ) (in_motion ?j agent) ) ) ) ) ) (not (exists (?z - hexagonal_bin ?l - (either blue_cube_block ball)) (in_motion ?l ?l) ) ) ) ) (in ?j south_west_corner) (agent_holds ?j) (not (exists (?k - bridge_block ?x - watch) (in_motion agent ?x) ) ) (not (is_setup_object ?j ?j) ) ) (not (and (not (rug_color_under ?j) ) (and (in ?j ?j) (in_motion ?j) (agent_holds agent) ) ) ) (adjacent ?j) ) (on ?j ?j) ) (in_motion ?j) ) )
        )
      )
    )
    (forall (?w - block ?e - teddy_bear)
      (and
        (preference preference2
          (exists (?x - ball ?v - game_object)
            (at-end
              (not
                (toggled_on ?e)
              )
            )
          )
        )
        (preference preference3
          (exists (?w - hexagonal_bin)
            (then
              (once (in_motion ?w) )
              (hold (same_type ?e) )
              (hold-while (< 1 1) (not (touch agent ?w) ) )
            )
          )
        )
      )
    )
    (preference preference4
      (exists (?g - hexagonal_bin ?y ?n - dodgeball)
        (then
          (once-measure (not (or (and (in_motion ?n ?y) (adjacent ?n) ) (and (agent_holds ?n door ?n) (and (not (forall (?v - dodgeball) (on ?v) ) ) (and (on ?n ?y) (exists (?t - wall) (on ?y) ) ) ) (in_motion ?n) ) ) ) (distance desk room_center) )
          (once (agent_holds ?n) )
          (once (between ?y) )
        )
      )
    )
  )
)
(:terminal
  (>= 5 (total-time) )
)
(:scoring
  (count preference2:pyramid_block)
)
)


(define (game game-id-76) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (on ?xxx ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?k - doggie_bed ?q - hexagonal_bin)
        (then
          (once-measure (in_motion ?q) (distance 7 ?q) )
          (once (agent_holds ?q) )
          (once (not (not (on ?q ?q) ) ) )
        )
      )
    )
  )
)
(:terminal
  (or
    (> 180 5 )
    (>= 9 100 )
  )
)
(:scoring
  (count preference1:alarm_clock)
)
)


(define (game game-id-77) (:domain medium-objects-room-v1)
(:setup
  (game-optional
    (not
      (or
        (on ?xxx)
        (adjacent ?xxx ?xxx)
      )
    )
  )
)
(:constraints
  (and
    (forall (?s - hexagonal_bin ?h - beachball)
      (and
        (preference preference1
          (exists (?x - block ?b - dodgeball ?y ?i - hexagonal_bin)
            (then
              (hold (not (not (in_motion ?h ?h) ) ) )
              (once (not (and (and (not (in_motion ?i) ) (not (on ?h) ) ) (not (not (on green ?i) ) ) ) ) )
              (once (and (not (not (and (not (agent_holds ?h) ) (on right ?i) ) ) ) (agent_holds ?i) (and (not (not (not (and (not (and (agent_holds ?h) (and (is_setup_object ?h) (and (agent_holds ?i ?i) (agent_holds ?h) ) ) (and (in_motion ?i) (in_motion upright rug) (adjacent ?h) ) ) ) (exists (?o - ball ?m - dodgeball ?z - hexagonal_bin) (< 7 1) ) ) ) ) ) (forall (?x - cube_block) (and (and (in_motion ?x) (not (on ?h) ) ) (in ?h) (not (in ?y) ) ) ) ) (in_motion desk ?i) (agent_holds ?i) (agent_holds ?y ?y) (exists (?a - hexagonal_bin ?c - desk_shelf) (and (toggled_on ?c) (same_color ?i) ) ) (adjacent ?h ?i) (in ?y ?h) (exists (?p - hexagonal_bin ?w ?j - hexagonal_bin) (not (agent_holds ?j) ) ) ) )
            )
          )
        )
        (preference preference2
          (exists (?z - hexagonal_bin)
            (at-end
              (and
                (agent_holds ?h)
                (or
                  (not
                    (not
                      (not
                        (agent_holds ?z)
                      )
                    )
                  )
                  (not
                    (not
                      (exists (?t - cube_block ?n - tall_cylindrical_block)
                        (not
                          (in ?h ?z)
                        )
                      )
                    )
                  )
                )
              )
            )
          )
        )
        (preference preference3
          (then
            (once (on ?h) )
            (once (on ?h) )
            (once (on ?h) )
          )
        )
      )
    )
  )
)
(:terminal
  (<= 0 (count-once-per-objects preference3:hexagonal_bin) )
)
(:scoring
  (count-shortest preference2:dodgeball)
)
)


(define (game game-id-78) (:domain medium-objects-room-v1)
(:setup
  (exists (?r - dodgeball)
    (and
      (not
        (game-optional
          (and
            (agent_holds front desk)
            (not
              (in_motion ?r ?r)
            )
          )
        )
      )
      (game-conserved
        (and
          (and
            (agent_holds ?r)
            (same_type ?r ?r)
          )
          (and
            (in ?r)
            (and
              (and
                (touch ?r ?r)
                (agent_holds ?r ?r)
              )
              (in ?r ?r)
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?x - hexagonal_bin)
        (then
          (hold (agent_holds ?x ?x) )
          (hold (and (and (and (on ?x ?x) (in_motion ?x rug) ) (and (agent_holds ?x) (< 8 10) ) ) (touch ?x desk) (in_motion ?x ?x) ) )
          (once (not (not (not (agent_holds ?x ?x) ) ) ) )
        )
      )
    )
    (preference preference2
      (exists (?c - ball)
        (then
          (once (and (not (agent_holds ?c ?c) ) (and (not (and (on ?c) (above ?c) ) ) (and (forall (?f - building ?h - (either cellphone triangular_ramp)) (or (in_motion ?h) (touch ?h ?c) (on ?h) ) ) (and (agent_holds ?c ?c) (in_motion ?c) ) ) ) (not (adjacent_side ?c ?c) ) ) )
          (once (and (and (> (distance ?c ?c) (distance )) (in_motion floor ?c) ) (agent_holds top_drawer) ) )
          (hold (and (not (and (on ?c ?c ?c) (forall (?f - red_dodgeball) (< (distance ?c ?c) 10) ) ) ) ) )
        )
      )
    )
  )
)
(:terminal
  (> (+ 2 15 )
    10
  )
)
(:scoring
  (count preference1:wall:red)
)
)


(define (game game-id-79) (:domain few-objects-room-v1)
(:setup
  (and
    (forall (?w - hexagonal_bin)
      (exists (?f - desk_shelf)
        (exists (?j ?k ?x - wall)
          (game-conserved
            (touch ?j)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?w - (either tall_cylindrical_block cube_block))
      (and
        (preference preference1
          (exists (?l - hexagonal_bin ?q - golfball ?b - ball ?m - curved_wooden_ramp)
            (then
              (hold (agent_holds ?w ?m) )
              (once (is_setup_object ?w) )
              (once (agent_holds ?m front) )
            )
          )
        )
      )
    )
    (forall (?x ?f - dodgeball)
      (and
        (preference preference2
          (exists (?b - building)
            (at-end
              (agent_holds ?b)
            )
          )
        )
        (preference preference3
          (exists (?h - wall ?h - hexagonal_bin)
            (then
              (hold-while (agent_holds ?h) (in ?f ?h) )
              (hold (and (agent_holds ?h) (or (and (opposite ?f) (not (and (>= 1 (distance 6)) (not (in_motion ?x) ) (agent_holds ?x) ) ) ) (in_motion ?f ?f) ) ) )
              (once (adjacent ?f ?f) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (- (count preference3:basketball:beachball) )
    (* 5 (* (count-once-per-objects preference3:pink) (count-measure preference2:blue_dodgeball) )
      (* (* (count preference3) (count-longest preference3:hexagonal_bin) )
        (* (* (count-once preference1:pink:dodgeball:beachball) (and (* (count-once-per-objects preference1:hexagonal_bin) 2 )
              (* (and 3 ) (count-once-per-objects preference3:golfball:yellow_cube_block) (count-once-per-objects preference1:hexagonal_bin) )
            )
            (count preference1:red)
          )
          (and
            (+ (* (count-shortest preference2:dodgeball:doggie_bed) 10 1 )
              10
            )
            (count preference3:pink)
          )
          (count-once preference3:purple)
          (count preference3:yellow_cube_block:golfball)
        )
      )
    )
  )
)
(:scoring
  (* (* (- (external-forall-maximize 2 ) )
      2
    )
    2
  )
)
)


(define (game game-id-80) (:domain few-objects-room-v1)
(:setup
  (and
    (forall (?h - game_object)
      (not
        (game-optional
          (on ?h ?h)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?d - block ?w - building)
        (then
          (once (on ?w ?w) )
          (hold (not (= (distance ?w room_center)) ) )
          (once (in_motion agent) )
          (hold (in_motion ?w ?w) )
        )
      )
    )
    (preference preference2
      (exists (?o ?e - (either doggie_bed main_light_switch pink))
        (then
          (hold (and (and (same_type ?e) (> 3 1) (< (distance door door) (distance )) ) (>= 1 8) (same_object ?o ?o) ) )
          (hold-while (and (in ?o) (not (and (in_motion ?o) (not (not (agent_holds ?e) ) ) ) ) ) (and (agent_holds ?o) ) )
          (once (and (and (in_motion ?o) (in_motion ?o ?e) ) (agent_holds ?e ?e) ) )
        )
      )
    )
    (forall (?j - block)
      (and
        (preference preference3
          (exists (?g - color)
            (then
              (once (agent_holds ?g agent) )
              (once (and (and (and (not (< (distance room_center 1) (distance ?j room_center)) ) (in_motion ?j) ) (on ?j) ) (agent_holds ?g) ) )
              (once (agent_holds ?g ?g) )
            )
          )
        )
        (preference preference4
          (exists (?v - hexagonal_bin)
            (then
              (hold-to-end (and (in_motion ?v) (and (on ?j ?j) (in ?v) (touch ?j) ) ) )
              (hold (agent_holds ?v) )
              (hold (in ?v) )
            )
          )
        )
        (preference preference5
          (exists (?t - golfball)
            (then
              (hold (agent_holds blue ?t) )
              (once (faces bed ?t) )
              (once (not (agent_holds desk desk) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (and
    (>= (count preference5:basketball) (total-score) )
    (or
      (>= (total-score) 10 )
    )
    (>= (count preference4:basketball) 10 )
  )
)
(:scoring
  (<= (count preference2:green:red) (count preference1:pink) )
)
)


(define (game game-id-81) (:domain many-objects-room-v1)
(:setup
  (exists (?p - chair ?c - (either yellow_cube_block yellow_cube_block) ?j - triangular_ramp)
    (game-conserved
      (agent_holds ?j)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?z - (either cylindrical_block tall_cylindrical_block golfball))
        (at-end
          (exists (?o - ball ?u - ball)
            (and
              (not
                (not
                  (rug_color_under ?z)
                )
              )
            )
          )
        )
      )
    )
    (forall (?o - chair)
      (and
        (preference preference2
          (exists (?i - wall)
            (then
              (once (in_motion ?o color) )
              (once (agent_holds ?i ?o) )
              (hold-while (in_motion ?i) (< 1 1) )
            )
          )
        )
      )
    )
    (preference preference3
      (exists (?v - chair ?m - dodgeball)
        (then
          (hold (< (distance ?m room_center) 9) )
          (once (agent_holds ?m ?m) )
          (hold (or (and (agent_holds ?m) (and (not (agent_holds right) ) (and (touch ?m ?m) (in_motion ?m) ) (and (and (not (in_motion ?m) ) (and (opposite ?m) (agent_holds ?m) (on ?m) (not (not (and (agent_holds pink_dodgeball) (< 1 (distance agent ?m)) (in ?m) ) ) ) ) ) (not (in_motion ?m) ) ) (and (agent_holds ?m ?m) (not (and (or (agent_holds ?m) (and (adjacent ?m rug) (not (in_motion ?m ?m) ) (on ?m ?m) ) ) (and (and (agent_holds ?m bed) (and (in_motion drawer ?m) (and (not (in_motion ?m) ) (in_motion ?m) ) ) ) (agent_holds ?m upside_down) (and (and (adjacent ?m) (on ?m) (on ?m) ) (on agent ?m) ) (in_motion ?m) ) ) ) ) ) ) (exists (?u - teddy_bear) (and (not (in ?m) ) (same_type agent ?m ?u) ) ) ) )
        )
      )
    )
  )
)
(:terminal
  (> (count-overlapping preference2:dodgeball:yellow) (- (>= (count preference1:yellow_cube_block) 2 )
    )
  )
)
(:scoring
  (count-same-positions preference2:basketball)
)
)


(define (game game-id-82) (:domain many-objects-room-v1)
(:setup
  (and
    (game-conserved
      (agent_holds ?xxx ?xxx)
    )
    (forall (?x - triangular_ramp)
      (exists (?p ?f - dodgeball ?y - curved_wooden_ramp)
        (and
          (exists (?a - block ?u ?w - bridge_block ?v - dodgeball)
            (forall (?e - block ?u - hexagonal_bin ?t - (either pencil doggie_bed cd) ?a - sliding_door ?l - doggie_bed)
              (and
                (game-conserved
                  (not
                    (and
                      (adjacent_side ?y ?x)
                      (not
                        (and
                          (not
                            (is_setup_object ?x ?y)
                          )
                          (< 2 (distance room_center ?y))
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
    )
    (and
      (game-conserved
        (not
          (and
            (agent_holds desk)
            (agent_holds agent)
            (in ?xxx)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?a - (either wall cellphone cellphone wall golfball dodgeball dodgeball) ?g - bridge_block)
        (then
          (once (in_motion ?g) )
          (hold (in_motion ?g ?g) )
          (once (not (in_motion ?g ?g) ) )
        )
      )
    )
  )
)
(:terminal
  (>= (* 5 (* 10 )
    )
    (count-once preference1:beachball:pyramid_block)
  )
)
(:scoring
  (count-once-per-objects preference1)
)
)


(define (game game-id-83) (:domain many-objects-room-v1)
(:setup
  (exists (?w - (either alarm_clock blue_cube_block))
    (exists (?a - wall ?g - building)
      (game-conserved
        (not
          (and
            (between ?g ?w)
            (not
              (exists (?b - hexagonal_bin)
                (in ?w)
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
    (preference preference1
      (then
        (once (and (and (not (adjacent pink_dodgeball) ) (and (not (object_orientation agent ?xxx) ) (in ?xxx ?xxx) ) (and (in_motion ?xxx) (< 7 1) ) ) (not (and (not (and (not (in_motion ?xxx floor) ) (not (in_motion ?xxx ?xxx) ) ) ) (and (on ?xxx) (and (on ?xxx) (or (not (not (and (not (and (same_type ?xxx) (> 1 3) ) ) (agent_holds ?xxx ?xxx) (and (adjacent ?xxx) (not (on top_drawer ?xxx) ) ) ) ) ) (in_motion ?xxx ?xxx) ) (touch ?xxx ?xxx) ) ) (agent_holds ?xxx ?xxx) ) ) ) )
        (once (agent_holds ?xxx) )
        (once (not (on pillow) ) )
      )
    )
  )
)
(:terminal
  (or
    (> (count-once-per-objects preference1:dodgeball) (or (* (+ (count-once-per-objects preference1:orange:beachball) (external-forall-maximize (count preference1:pink:basketball) ) (> 2 (count-once-per-objects preference1:hexagonal_bin:pink) )
          )
          (count-once-per-objects preference1:dodgeball)
        )
        (count-once-per-objects preference1:blue_pyramid_block:beachball)
      )
    )
    (>= (= 3 (count-once-per-objects preference1:pink) )
      (count preference1:basketball)
    )
  )
)
(:scoring
  (* (* (- (count-once-per-objects preference1:beachball) )
      10
    )
    (+ (total-score) (count-once preference1:top_drawer:hexagonal_bin) )
  )
)
)


(define (game game-id-84) (:domain many-objects-room-v1)
(:setup
  (forall (?k - curved_wooden_ramp)
    (exists (?o - shelf)
      (forall (?g - block)
        (exists (?j - (either dodgeball pyramid_block))
          (game-conserved
            (in_motion ?g)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?b - chair)
        (then
          (once (on ?b ?b) )
          (hold (and (and (open east_sliding_door) (in_motion ?b) ) (not (on floor) ) ) )
          (once (or (agent_holds ?b) (not (agent_holds ?b floor) ) ) )
        )
      )
    )
    (forall (?b - hexagonal_bin ?b - wall)
      (and
        (preference preference2
          (exists (?w - triangular_ramp)
            (at-end
              (and
                (agent_holds agent)
                (and
                  (not
                    (and
                      (not
                        (agent_holds ?w)
                      )
                      (in_motion ?b)
                    )
                  )
                )
              )
            )
          )
        )
      )
    )
    (preference preference3
      (then
        (once (same_object pink_dodgeball) )
        (once (and (and (on ?xxx ?xxx ?xxx) (exists (?i - shelf) (not (in_motion ?i) ) ) ) (between ?xxx) ) )
        (hold-while (agent_holds desk ?xxx) (in_motion ?xxx ?xxx) )
      )
    )
    (preference preference4
      (exists (?u - ball)
        (then
          (hold (agent_holds ?u) )
          (once (not (on ?u bed) ) )
          (hold (in_motion ?u ?u) )
        )
      )
    )
    (preference preference5
      (exists (?z - flat_block)
        (then
          (once (not (adjacent ?z) ) )
          (once (agent_holds desk ?z) )
          (any)
        )
      )
    )
  )
)
(:terminal
  (>= 5 6 )
)
(:scoring
  (count-once-per-objects preference3:brown:dodgeball)
)
)


(define (game game-id-85) (:domain few-objects-room-v1)
(:setup
  (and
    (not
      (forall (?w - curved_wooden_ramp ?i - ball ?y - hexagonal_bin)
        (and
          (game-optional
            (and
              (not
                (in_motion ?y ?y)
              )
              (object_orientation ?y)
              (not
                (agent_holds floor ?y)
              )
            )
          )
          (and
            (forall (?r ?o - building)
              (game-optional
                (on ?y desk)
              )
            )
            (exists (?h - book)
              (and
                (forall (?z - dodgeball)
                  (and
                    (exists (?v - teddy_bear)
                      (and
                        (game-conserved
                          (agent_holds front)
                        )
                        (and
                          (game-conserved
                            (in_motion ?z ?v)
                          )
                          (game-optional
                            (and
                              (in_motion ?h)
                              (in_motion ?h ?z)
                            )
                          )
                          (exists (?j - teddy_bear ?g - dodgeball)
                            (and
                              (forall (?m ?t ?o ?s ?l ?a - (either cube_block cube_block))
                                (forall (?f - chair)
                                  (game-optional
                                    (on ?g ?g)
                                  )
                                )
                              )
                            )
                          )
                        )
                        (and
                          (and
                            (forall (?s - drawer)
                              (game-optional
                                (and
                                  (and
                                    (not
                                      (is_setup_object ?v)
                                    )
                                    (in ?y ?z)
                                  )
                                  (and
                                    (adjacent upright)
                                    (agent_holds ?h)
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
                (exists (?l - dodgeball ?s - rug)
                  (and
                    (game-conserved
                      (and
                        (and
                          (in ?h)
                          (not
                            (in_motion ?s)
                          )
                        )
                        (in_motion ?s ?h)
                      )
                    )
                  )
                )
                (and
                  (exists (?f - pillow)
                    (forall (?n ?r ?t ?a ?d ?m - game_object ?q - hexagonal_bin)
                      (or
                        (game-conserved
                          (not
                            (on ?h ?h)
                          )
                        )
                      )
                    )
                  )
                  (exists (?l - ball ?k - hexagonal_bin)
                    (exists (?d - triangular_ramp)
                      (game-optional
                        (in floor ?d)
                      )
                    )
                  )
                  (exists (?s - doggie_bed ?f - (either cylindrical_block main_light_switch))
                    (forall (?g - game_object ?s - dodgeball ?x - game_object ?s - pillow)
                      (and
                        (game-conserved
                          (and
                            (not
                              (in_motion ?h ?y agent)
                            )
                            (and
                              (not
                                (adjacent ?f)
                              )
                              (exists (?k - doggie_bed)
                                (in_motion ?f)
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
            (forall (?b - (either golfball cube_block golfball golfball))
              (and
                (forall (?l - (either dodgeball) ?w - dodgeball)
                  (game-conserved
                    (not
                      (and
                        (not
                          (in_motion ?b ?y)
                        )
                        (agent_holds ?y ?w)
                        (in_motion ?y ?w)
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
  )
)
(:constraints
  (and
    (forall (?k - (either cellphone triangular_ramp basketball) ?i - hexagonal_bin ?d ?g - (either golfball triangle_block))
      (and
        (preference preference1
          (exists (?m - doggie_bed ?h - dodgeball ?b - (either laptop cellphone) ?o - hexagonal_bin)
            (then
              (hold (not (not (in_motion ?g ?o) ) ) )
              (once (not (not (not (broken ?o) ) ) ) )
              (once (and (in_motion ?d) (touch ?d) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (+ (count-same-positions preference1:dodgeball) 1 )
      (count preference1:hexagonal_bin)
    )
    (> (count preference1) (count preference1:rug) )
  )
)
(:scoring
  (+ (* 3 (count-once preference1:alarm_clock:dodgeball) )
    (total-score)
    (external-forall-maximize
      (count-once-per-objects preference1:dodgeball)
    )
    8
    4
    0.7
  )
)
)


(define (game game-id-86) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (agent_holds ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?n - (either doggie_bed yellow_cube_block))
        (at-end
          (and
            (< (distance ?n ?n ?n) 1)
            (not
              (agent_holds ?n ?n)
            )
          )
        )
      )
    )
    (forall (?i - building)
      (and
        (preference preference2
          (exists (?w - building)
            (then
              (once (< 2 (distance room_center room_center)) )
            )
          )
        )
      )
    )
    (preference preference3
      (exists (?e - green_triangular_ramp)
        (then
          (hold (not (not (not (adjacent ?e) ) ) ) )
          (hold (in_motion bed) )
          (once (or (and (adjacent ?e ?e) (agent_holds ?e) ) (agent_holds ?e ?e) ) )
        )
      )
    )
    (preference preference4
      (exists (?a - (either desktop cellphone))
        (then
          (once (agent_holds desk ?a) )
          (hold-while (and (not (agent_holds ?a) ) (and (agent_holds ?a ?a) (not (>= (distance_side ?a ?a) (distance ?a ?a)) ) ) (and (on ?a) (not (and (agent_holds ?a) (and (agent_holds ?a) (not (not (on top_drawer ?a) ) ) (agent_holds ?a ?a) ) ) ) ) ) (not (in_motion ?a) ) (adjacent_side ?a ?a ?a) )
          (once (and (in_motion ?a ?a) (and (not (and (agent_holds ?a) (touch ?a rug) (in_motion ?a) (agent_holds ?a ?a) ) ) (and (not (on ?a) ) (not (in_motion ?a ?a desktop) ) (not (and (and (agent_holds ?a floor) (not (agent_holds ?a) ) ) (not (not (in_motion agent ?a) ) ) ) ) ) ) ) )
        )
      )
    )
    (forall (?g - teddy_bear)
      (and
        (preference preference5
          (then
            (once (on agent) )
            (once (and (not (agent_holds ?g) ) (agent_holds ?g) ) )
            (once (agent_holds ?g) )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count-overlapping preference1:green) (* 1 2 3 )
  )
)
(:scoring
  (/
    3
    (count-total preference1:red)
  )
)
)


(define (game game-id-87) (:domain many-objects-room-v1)
(:setup
  (exists (?o - dodgeball)
    (exists (?l ?b - hexagonal_bin ?l - game_object ?s - block)
      (and
        (forall (?p - hexagonal_bin ?k - (either cellphone golfball pyramid_block))
          (game-conserved
            (in_motion ?o)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?a - cube_block)
      (and
        (preference preference1
          (exists (?r - ball)
            (then
              (once (agent_holds ?a) )
              (once (on ?r) )
              (once (on bed ?a) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 5 (* (* (count-once-per-objects preference1:dodgeball) (count-total preference1:beachball) 2 )
      2
      (count-measure preference1:brown:basketball:dodgeball)
    )
  )
)
(:scoring
  (count preference1:pink)
)
)


(define (game game-id-88) (:domain few-objects-room-v1)
(:setup
  (exists (?r - bridge_block)
    (and
      (game-conserved
        (on ?r ?r)
      )
      (and
        (forall (?v - doggie_bed)
          (exists (?h - block)
            (and
              (game-optional
                (is_setup_object ?h)
              )
              (game-conserved
                (not
                  (< 1 3)
                )
              )
            )
          )
        )
        (game-conserved
          (not
            (not
              (and
                (game_start bed)
                (and
                  (on ?r ?r)
                  (and
                    (in_motion ?r)
                    (and
                      (is_setup_object ?r)
                      (and
                        (agent_holds ?r ?r)
                        (touch ?r)
                      )
                    )
                  )
                )
                (in ?r ?r)
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
    (preference preference1
      (exists (?v - color)
        (at-end
          (agent_holds ?v)
        )
      )
    )
  )
)
(:terminal
  (and
    (>= (- (+ (* (count-once preference1:beachball) 2 (count preference1:beachball) )
          (+ (- (count-once-per-objects preference1:dodgeball:yellow) )
            2
            3
          )
          (count-once preference1:basketball)
          (count-overlapping preference1:dodgeball)
          3
          3
          (count preference1:dodgeball)
          (count preference1:hexagonal_bin)
          2
        )
      )
      (- (* 2 15 )
      )
    )
    (or
      (> (count-once-per-objects preference1:basketball) (count preference1:pink:yellow) )
      (or
        (> 5 (count-once preference1:doggie_bed:beachball) )
        (not
          (or
            (or
              (>= 50 (< (* 5 (count-once-per-objects preference1:blue_dodgeball) (total-time) 6 2 30 )
                  2
                )
              )
              (>= (count-once preference1:blue_dodgeball:golfball) (+ 1 (* (+ 5 3 )
                    (>= 4 (count preference1:red) )
                  )
                )
              )
            )
          )
        )
      )
      (< 8 3 )
    )
  )
)
(:scoring
  6
)
)


(define (game game-id-89) (:domain medium-objects-room-v1)
(:setup
  (exists (?q - dodgeball)
    (game-conserved
      (not
        (not
          (not
            (not
              (agent_holds ?q)
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (hold (rug_color_under agent ?xxx) )
        (once (agent_holds ?xxx) )
        (once (and (agent_holds ?xxx ?xxx) (on ) (not (and (not (in_motion ?xxx floor) ) (agent_holds bed) ) ) ) )
      )
    )
  )
)
(:terminal
  (<= (count-once-per-objects preference1:orange) (count preference1:dodgeball) )
)
(:scoring
  2
)
)


(define (game game-id-90) (:domain medium-objects-room-v1)
(:setup
  (and
    (and
      (and
        (game-conserved
          (in_motion ?xxx ?xxx)
        )
        (exists (?b - ball ?t - dodgeball ?s - (either blue_cube_block cube_block))
          (exists (?d - hexagonal_bin)
            (forall (?m ?r ?x - hexagonal_bin ?y - block)
              (game-optional
                (touch ?d)
              )
            )
          )
        )
        (game-conserved
          (adjacent ?xxx)
        )
        (exists (?l ?n - (either cube_block hexagonal_bin dodgeball laptop laptop) ?v - cube_block)
          (game-conserved
            (in_motion ?v ?v)
          )
        )
        (game-optional
          (in_motion rug ?xxx)
        )
      )
      (game-conserved
        (between ?xxx)
      )
      (exists (?g - ball)
        (or
          (game-conserved
            (on ?g)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?t - pillow ?i - cube_block ?l - cube_block ?j - doggie_bed ?m - (either cellphone key_chain cylindrical_block doggie_bed))
        (then
          (once (on ?m) )
          (once (in ?m) )
          (once (and (and (not (agent_holds ?m) ) (in_motion ?m ?m) ) (in ?m agent) ) )
          (once (opposite ?m) )
          (once (on ?m ?m) )
        )
      )
    )
    (preference preference2
      (exists (?z - hexagonal_bin)
        (then
          (hold (and (on ?z ?z) (agent_holds ?z ?z) ) )
          (once (or (and (on ?z) (not (in_motion agent) ) (agent_holds ?z sideways) ) (in_motion door) ) )
          (once (and (opposite floor) (in_motion ?z) (not (not (and (adjacent_side ?z ?z) (not (touch ?z) ) ) ) ) ) )
        )
      )
    )
  )
)
(:terminal
  (= (+ (count preference1:hexagonal_bin:hexagonal_bin) (= (count preference1:pink_dodgeball:dodgeball) 3 )
    )
    (* (* (count preference1:blue_dodgeball:golfball:hexagonal_bin) 4 )
    )
  )
)
(:scoring
  5
)
)


(define (game game-id-91) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (or
      (agent_holds ?xxx ?xxx)
      (in ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?m - hexagonal_bin ?p - hexagonal_bin)
        (then
          (hold (agent_holds ?p ?p) )
          (hold (in_motion desk) )
          (hold (and (not (touch ?p) ) (adjacent ?p) ) )
          (once (and (not (and (agent_holds ?p ?p) (and (in_motion ?p) (agent_holds ?p) ) ) ) (in ?p ?p) ) )
        )
      )
    )
  )
)
(:terminal
  (>= 10 (count-once preference1:blue_dodgeball) )
)
(:scoring
  (+ (* 2 1 )
    3
  )
)
)


(define (game game-id-92) (:domain many-objects-room-v1)
(:setup
  (and
    (forall (?b - bridge_block)
      (or
        (exists (?m - game_object ?t - dodgeball ?w - color)
          (game-conserved
            (not
              (not
                (agent_holds ?b)
              )
            )
          )
        )
      )
    )
    (and
      (and
        (and
          (game-conserved
            (adjacent_side agent)
          )
          (exists (?y - hexagonal_bin)
            (forall (?h - (either golfball dodgeball cube_block) ?l - triangular_ramp)
              (exists (?x - dodgeball)
                (and
                  (exists (?q - block)
                    (game-optional
                      (not
                        (agent_holds ?x ?x)
                      )
                    )
                  )
                )
              )
            )
          )
        )
        (game-optional
          (>= (distance front) 2)
        )
      )
      (game-optional
        (and
          (not
            (not
              (in_motion ?xxx ?xxx)
            )
          )
          (opposite ?xxx)
        )
      )
      (not
        (and
          (exists (?y - cube_block)
            (game-optional
              (on ?y ?y)
            )
          )
          (and
            (or
              (and
                (and
                  (game-optional
                    (in_motion ?xxx ?xxx)
                  )
                  (game-conserved
                    (in_motion ?xxx ?xxx)
                  )
                )
              )
            )
          )
        )
      )
    )
    (exists (?b - block ?i - cube_block)
      (game-optional
        (and
          (is_setup_object ?i ?i)
          (not
            (in_motion ?i)
          )
          (< (distance room_center ?i 10) (building_size 8 10))
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?t - cylindrical_block)
      (and
        (preference preference1
          (then
            (once (in_motion blue) )
            (once (not (and (in agent ?t) (and (in_motion ?t) (not (agent_holds ?t front) ) ) ) ) )
            (once (in_motion left) )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 4 (count preference1:brown) )
)
(:scoring
  (+ (* (count-once-per-objects preference1:dodgeball) (count preference1:red) )
    (count-unique-positions preference1:dodgeball)
  )
)
)


(define (game game-id-93) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (not
      (in_motion ?xxx ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?z - chair ?g - ball)
        (then
          (once (and (in_motion ?g floor) (in agent) ) )
          (once (in_motion blinds desk) )
          (hold (< (distance 8 room_center) (distance ?g ?g)) )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (+ (count-once-per-objects preference1:basketball:orange) (- (count-once-per-objects preference1:pink) )
      )
      (count preference1:red)
    )
    (< (count preference1:dodgeball) (count-once-per-external-objects preference1:yellow) )
  )
)
(:scoring
  (count-once-per-objects preference1:beachball)
)
)


(define (game game-id-94) (:domain many-objects-room-v1)
(:setup
  (forall (?x ?t - chair)
    (and
      (forall (?l - hexagonal_bin)
        (game-optional
          (and
            (< (x_position ?t 4) 6)
            (in_motion north_wall)
            (between ?t)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?t - hexagonal_bin ?y - ball ?s ?g ?b ?w ?z ?a ?u ?q ?h ?y - hexagonal_bin ?o - block)
        (then
          (once (in_motion ?o) )
          (once (not (in_motion ?o) ) )
          (hold (agent_holds bed) )
        )
      )
    )
    (preference preference2
      (exists (?e - ball ?w - hexagonal_bin)
        (then
          (hold (not (same_color ?w floor) ) )
          (once (object_orientation ?w) )
          (once (not (not (in_motion brown ?w) ) ) )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference2:yellow) (count-once-per-objects preference2:pink_dodgeball) )
)
(:scoring
  (count-measure preference2:pink)
)
)


(define (game game-id-95) (:domain few-objects-room-v1)
(:setup
  (and
    (forall (?i - pyramid_block ?w - green_triangular_ramp)
      (game-optional
        (in ?w)
      )
    )
    (game-conserved
      (in_motion ?xxx)
    )
    (and
      (game-optional
        (on east_sliding_door)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?i - triangular_ramp ?j - dodgeball)
        (then
          (once (not (not (on ?j) ) ) )
          (once (in_motion agent) )
          (hold-while (and (not (< 6 (distance 6 ?j)) ) (not (same_object bed) ) ) (and (in ?j) (not (in ?j) ) (in_motion ?j) ) (agent_holds ?j ?j) )
        )
      )
    )
    (forall (?v - dodgeball)
      (and
        (preference preference2
          (exists (?r - game_object ?b - game_object)
            (then
              (once (in_motion ?b bed) )
              (once (in_motion ?b desk) )
              (once (faces ?b) )
            )
          )
        )
        (preference preference3
          (exists (?g - hexagonal_bin ?h - beachball)
            (then
              (once (not (and (touch ?v) (touch ?v ?v) ) ) )
              (hold (and (not (and (not (in ?v ?h) ) (on ?h) ) ) (in_motion ?h) ) )
              (once (not (or (< 1 (distance room_center 1)) (agent_holds ?h ?h) ) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (- (count preference3:triangle_block) )
    40
  )
)
(:scoring
  (* (count-overlapping preference2:dodgeball) (external-forall-maximize (count-once-per-objects preference3:pink_dodgeball) ) )
)
)


(define (game game-id-96) (:domain many-objects-room-v1)
(:setup
  (and
    (exists (?a - ball ?c - (either pink cylindrical_block tall_cylindrical_block) ?z - chair)
      (game-conserved
        (agent_holds ?z pink ?z)
      )
    )
    (game-optional
      (= 1 (distance ?xxx ?xxx))
    )
    (game-conserved
      (and
        (rug_color_under ?xxx ?xxx)
        (not
          (on ?xxx ?xxx)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?w - (either key_chain laptop))
        (then
          (hold (on ?w) )
          (once (touch ?w) )
          (once (not (and (< 5 1) ) ) )
        )
      )
    )
    (preference preference2
      (exists (?n - game_object)
        (then
          (once (in_motion ?n) )
          (once (not (and (and (and (in_motion ?n ?n) (broken ?n ?n) ) (> 1 10) (on ?n) ) (>= 8 7) ) ) )
          (hold (< 1 (distance_side agent 3)) )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (count-once preference2:pink) 5 )
    (>= 8 9 )
  )
)
(:scoring
  (count preference2:blue_pyramid_block)
)
)


(define (game game-id-97) (:domain few-objects-room-v1)
(:setup
  (forall (?v - hexagonal_bin)
    (exists (?j - dodgeball)
      (forall (?y - red_dodgeball ?w - hexagonal_bin ?a - doggie_bed ?w - hexagonal_bin)
        (not
          (and
            (game-conserved
              (adjacent ?v ?j)
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?z ?j - hexagonal_bin)
      (and
        (preference preference1
          (exists (?q - block ?r - drawer)
            (then
              (once (not (and (agent_holds ?r agent) (in_motion ?j ?z) ) ) )
              (once (agent_holds ?r) )
              (once (same_color ?z bed ?z) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (total-score) (* (external-forall-maximize (+ (* (* (* (* (count-once-per-objects preference1:dodgeball) (+ (count preference1:dodgeball) (* 3 10 )
                  )
                )
                (count-once preference1:basketball)
              )
              (count-once-per-objects preference1:cube_block)
            )
            (count preference1:green)
          )
          (* (count preference1:pink:basketball) (count-once-per-objects preference1:dodgeball:hexagonal_bin) )
        )
      )
      2
    )
  )
)
(:scoring
  (count preference1:pink)
)
)


(define (game game-id-98) (:domain few-objects-room-v1)
(:setup
  (and
    (exists (?n - wall)
      (forall (?a ?y - wall)
        (game-optional
          (in ?n bed)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?r ?z ?l ?b - golfball ?s - (either alarm_clock dodgeball doggie_bed blue_cube_block))
        (then
          (hold (on ?s) )
          (hold (agent_holds ?s pink_dodgeball ?s) )
          (once (not (and (in ?s ?s) (agent_holds ?s) ) ) )
        )
      )
    )
    (preference preference2
      (exists (?w - teddy_bear)
        (then
          (hold (on ?w) )
          (hold (agent_holds ?w ?w ?w) )
          (once (in_motion ?w ?w) )
        )
      )
    )
  )
)
(:terminal
  (>= 7 (- 1 )
  )
)
(:scoring
  10
)
)


(define (game game-id-99) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (not
      (and
        (not
          (agent_holds ?xxx)
        )
        (in_motion ?xxx)
      )
    )
  )
)
(:constraints
  (and
    (forall (?b - (either laptop cylindrical_block))
      (and
        (preference preference1
          (exists (?n - hexagonal_bin ?o - curved_wooden_ramp ?a - game_object)
            (then
              (hold (not (on ?b ?a) ) )
              (hold (agent_holds ?a ?a) )
              (once (and (and (agent_holds ?b) (in_motion ?a) (not (and (agent_holds ?b) (agent_holds ?b) ) ) (rug_color_under ) ) (touch ?a) ) )
            )
          )
        )
        (preference preference2
          (exists (?w - game_object ?r - (either pen dodgeball lamp))
            (then
              (once-measure (< 1 (distance ?b ?r)) (distance room_center room_center) )
              (once (in ?r) )
              (hold (and (on desk ?b) (not (agent_holds ?r) ) ) )
            )
          )
        )
      )
    )
    (preference preference3
      (exists (?s ?t - teddy_bear)
        (then
          (once (and (same_type rug) (agent_holds agent ?t) (not (and (in_motion ?t ?t) (agent_holds desk) (in_motion ?t ?t ?s) (not (in_motion ?s) ) (agent_holds ?t) (and (not (above agent ?t) ) (in_motion upright) ) ) ) ) )
          (once (equal_z_position ?s ?t) )
          (hold (and (not (same_object ?s ?t) ) (agent_holds ?t green) (in ?t) (in_motion pillow) ) )
        )
      )
    )
  )
)
(:terminal
  (or
    (> (- (+ (* (* (count-once-per-objects preference1:basketball:dodgeball) (count-once-per-objects preference3:top_drawer) )
            (+ (count preference2:dodgeball) (+ 8 (count preference3:red) )
            )
          )
          (+ (count-once preference2:dodgeball) (count-once-per-external-objects preference1:green:white) )
          (+ 7 5 )
        )
        3
      )
      (= (count preference2:green) (count preference1:dodgeball) )
    )
    (>= 3 (- (count-once-per-objects preference3:blue_dodgeball) )
    )
  )
)
(:scoring
  (* (* (- (count-once-per-objects preference2:purple:red) )
      (count-longest preference1:cube_block)
      (- 20 )
    )
    (count-total preference2:pink_dodgeball)
  )
)
)


(define (game game-id-100) (:domain medium-objects-room-v1)
(:setup
  (or
    (game-conserved
      (not
        (on ?xxx)
      )
    )
  )
)
(:constraints
  (and
    (forall (?w - desk_shelf ?c - triangular_ramp)
      (and
        (preference preference1
          (at-end
            (not
              (in ?c upside_down)
            )
          )
        )
      )
    )
    (preference preference2
      (exists (?z - game_object)
        (then
          (once (toggled_on ?z) )
          (hold-while (agent_holds ?z ?z) (on ?z) (not (and (agent_holds ?z) (in ?z ?z) ) ) (on rug) )
          (hold (not (not (game_start ?z) ) ) )
        )
      )
    )
  )
)
(:terminal
  (>= 18 (* 3 (* (count-once-per-external-objects preference1:blue_dodgeball) (+ (+ (external-forall-minimize 8 ) (* (+ (count preference1:blue_pyramid_block) (- (= (* 5 (* 3 40 )
                    )
                    300
                    (count preference1:beachball)
                  )
                )
              )
              (count preference2:pink)
            )
            (count-once-per-objects preference2:green)
            (- 5 )
            (count-once-per-objects preference1:hexagonal_bin)
            (count preference2:dodgeball)
          )
          (count preference2:beachball:pink)
          300
          (count preference1:pyramid_block:dodgeball)
          (+ (total-time) 5 )
        )
      )
    )
  )
)
(:scoring
  (count-once-per-objects preference1:hexagonal_bin)
)
)


(define (game game-id-101) (:domain many-objects-room-v1)
(:setup
  (exists (?z - block)
    (exists (?s - hexagonal_bin ?e - game_object ?k - dodgeball)
      (game-conserved
        (on ?z)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?z - (either laptop cube_block key_chain))
        (at-end
          (touch top_shelf)
        )
      )
    )
    (preference preference2
      (then
        (hold (in_motion ?xxx) )
        (hold (agent_holds ?xxx) )
        (once (in_motion ?xxx ?xxx) )
      )
    )
  )
)
(:terminal
  (not
    (< (* (+ (count preference2:basketball) (* (- (= (* (* (- 40 )
                    (count preference1:green)
                  )
                  (* (count preference2:yellow_pyramid_block) (count-once preference1:orange:purple) (>= (+ (count preference1:golfball) (or 5 ) (- (count-once-per-external-objects preference1:alarm_clock) )
                        2
                        2
                        (* 10 10 )
                      )
                      10
                    )
                  )
                )
                (count preference2:golfball:dodgeball)
              )
            )
            (+ (* (* (count-once-per-objects preference2:pink_dodgeball) 3 )
                (count preference1:pink)
                (* 0 (count preference1:yellow:golfball) )
              )
              6
            )
            (count preference1:dodgeball:basketball)
          )
        )
        (count-once preference2:cube_block:golfball)
      )
      300
    )
  )
)
(:scoring
  (count-shortest preference2:dodgeball)
)
)


(define (game game-id-102) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (not
      (not
        (in_motion ?xxx ?xxx)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?r - color)
        (then
          (hold (or (object_orientation ?r) (in ?r) (on ?r ?r) ) )
          (hold (and (and (not (exists (?i - block) (and (on ?i) (not (agent_holds ?i ?i) ) ) ) ) (on agent) ) (on ?r ?r) ) )
          (once (not (on ?r ?r) ) )
        )
      )
    )
  )
)
(:terminal
  (>= (* (count-once preference1:tan) (* (>= 4 (count-measure preference1:doggie_bed:golfball) )
      )
    )
    2
  )
)
(:scoring
  40
)
)


(define (game game-id-103) (:domain medium-objects-room-v1)
(:setup
  (forall (?a - hexagonal_bin ?z - dodgeball)
    (game-conserved
      (on ?z)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?t - game_object ?x - (either yellow alarm_clock cellphone))
        (then
          (hold (same_object ?x tan) )
          (hold (agent_holds ?x agent) )
          (once (not (< 6 (distance ?x room_center)) ) )
        )
      )
    )
  )
)
(:terminal
  (>= 10 (* (count preference1:yellow) (+ (count-once-per-external-objects preference1:basketball) 1 )
    )
  )
)
(:scoring
  (* (+ (+ (count preference1:basketball:hexagonal_bin) (* (count preference1:blue_dodgeball:pink_dodgeball) 9 )
      )
      (+ (count-overlapping preference1:yellow:red) 2 )
    )
    (+ (count-once preference1:blue_dodgeball) (>= (count-once-per-objects preference1:dodgeball) (count-once preference1:yellow) )
    )
    (count-once-per-objects preference1:pink)
    7
    (count-once preference1:dodgeball)
    (+ (not 8 ) (* (* (count preference1:beachball) (+ (count preference1:basketball) (* (+ (count-same-positions preference1:blue_pyramid_block) (count preference1:yellow) )
              (count preference1:pink_dodgeball)
            )
            (count-longest preference1:side_table:hexagonal_bin)
          )
        )
        (* (count preference1) (- (count-once-per-objects preference1:basketball:orange) )
        )
        (count-once-per-objects preference1:beachball:basketball)
        (/
          (count preference1:alarm_clock:orange)
          (* 5 (count preference1:book) )
        )
        3
        (+ (= (+ 10 (* (* (external-forall-minimize (count-once-per-objects preference1:alarm_clock) ) (- (+ 5 (* (count preference1:orange:doggie_bed) (count-once-per-objects preference1:dodgeball) )
                    )
                  )
                )
                (count-measure preference1:beachball)
              )
            )
            (* (= (count preference1:green) (count preference1:beachball) )
              (* (count-shortest preference1:top_drawer) (count preference1:tall_cylindrical_block) )
            )
            2
          )
          (+ (= (count preference1:pink) (count-once-per-objects preference1:pink) )
          )
        )
      )
    )
  )
)
)


(define (game game-id-104) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (in_motion ?xxx south_wall)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?i - curved_wooden_ramp ?u ?f - game_object)
        (then
          (once (agent_holds ?f) )
          (hold (in ball) )
          (once (touch floor) )
        )
      )
    )
    (preference preference2
      (exists (?r - dodgeball ?p - ball)
        (then
          (once (and (in ?p ?p) ) )
          (hold (on bed) )
          (hold (and (in_motion rug) (> (distance_side ?p room_center) (distance 7)) ) )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (count preference1:pyramid_block) 0.7 )
    (> (+ (count-once-per-objects preference1) (count preference1:green:blue_dodgeball) )
      7
    )
  )
)
(:scoring
  (count preference1:dodgeball)
)
)


(define (game game-id-105) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (in_motion ?xxx)
  )
)
(:constraints
  (and
    (forall (?z - chair)
      (and
        (preference preference1
          (exists (?c - (either flat_block))
            (then
              (hold (agent_holds rug floor) )
              (once (game_over ?c) )
              (hold-while (< 10 (x_position ?c)) (and (in_motion ?c ?z) (and (not (same_color ?z) ) ) ) )
              (hold-while (and (on ?c ?c) (not (in ?z rug floor) ) ) (on ?z ?z) (on ?z ?c) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (<= (* 20 2 5 )
      (+ (* (- 4 )
          (+ (* (count preference1:dodgeball:beachball:beachball) 10 )
            20
            (count preference1:triangle_block:beachball)
          )
        )
        5
      )
    )
    (>= (* (count preference1:doggie_bed:golfball:blue_dodgeball) (count preference1:blue_dodgeball:dodgeball:pink_dodgeball) )
      10
    )
  )
)
(:scoring
  (count preference1:pyramid_block:red)
)
)


(define (game game-id-106) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (exists (?m - cylindrical_block)
      (< 8 2)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?n - chair)
        (then
          (once (and (is_setup_object ?n) (on ?n) ) )
          (once (in ?n) )
          (hold-to-end (> (distance 7 ?n) 4) )
        )
      )
    )
  )
)
(:terminal
  (> (* (* (count-unique-positions preference1:beachball) 1 (* (count-increasing-measure preference1:triangle_block) 7 )
        (* 3 10 )
      )
      (count preference1:beachball)
    )
    (* (* 4 7 )
      5
    )
  )
)
(:scoring
  (not
    3
  )
)
)


(define (game game-id-107) (:domain medium-objects-room-v1)
(:setup
  (exists (?h - hexagonal_bin ?m - dodgeball ?d - building)
    (exists (?s - hexagonal_bin ?e - ball)
      (game-optional
        (> 1 (distance room_center door))
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (at-end
        (on ?xxx ?xxx)
      )
    )
  )
)
(:terminal
  (< (count-once-per-objects preference1:red:golfball) (= (count preference1:basketball) )
  )
)
(:scoring
  10
)
)


(define (game game-id-108) (:domain few-objects-room-v1)
(:setup
  (exists (?n - chair ?z ?s - hexagonal_bin)
    (forall (?v - block ?v - hexagonal_bin)
      (or
        (exists (?j - (either chair desktop))
          (forall (?m ?q - (either dodgeball book cellphone) ?u - hexagonal_bin)
            (game-conserved
              (same_type ?v ?u)
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?c - block)
        (then
          (once (not (agent_holds ?c) ) )
          (once (agent_holds floor ?c) )
          (once (same_color ?c) )
        )
      )
    )
  )
)
(:terminal
  (>= 300 (* 8 (* 30 (external-forall-maximize 3 ) )
      (* (count preference1:blue_dodgeball) (/ 3 9 ) 8 )
    )
  )
)
(:scoring
  (count preference1:beachball)
)
)


(define (game game-id-109) (:domain few-objects-room-v1)
(:setup
  (game-optional
    (and
      (agent_holds ?xxx)
      (and
        (touch agent agent)
        (not
          (agent_holds ?xxx)
        )
        (adjacent ?xxx)
        (and
          (on ?xxx)
          (on floor ?xxx)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?p - shelf)
        (then
          (once (not (in_motion agent) ) )
          (once (exists (?l - cube_block) (and (in_motion ?p) (and (and (not (and (and (or (not (and (same_object ?p ?p rug) (and (not (and (not (in_motion ?p) ) (in_motion ?l) ) ) ) ) ) (and (not (in_motion agent ?l) ) (in_motion ?l) (on ?l) ) ) (agent_holds ?p ?l) ) (in_motion ?p agent) ) ) (not (and (not (agent_holds ?l) ) (not (not (in_motion ?p) ) ) ) ) ) (on ?l ?p) (and (not (and (and (and (not (in_motion ?l ?l) ) (on ?l ?p) (in ?l ?p) (in_motion agent ?p) (not (agent_holds ?p) ) (in_motion floor) (touch ?p) (agent_holds desk) (not (in_motion ?l) ) (agent_holds ?p ?p) (on ?l) (in_motion ?l) ) (agent_holds ?p ?p) ) (touch ?l ?l) ) ) (agent_holds ?p ?p) ) (not (agent_holds ?l ?l) ) (not (agent_holds ?p) ) (not (in agent) ) ) ) ) )
          (once (in_motion ?p rug) )
        )
      )
    )
    (preference preference2
      (exists (?f ?c ?s - pillow)
        (then
          (hold-while (and (in_motion ?s) (in ?f ?f) ) (< 7 (distance_side room_center front_left_corner)) )
          (once (on ?f) )
          (once (and (agent_holds upright) (in_motion ?f floor) ) )
        )
      )
    )
  )
)
(:terminal
  (>= (+ (count-once-per-external-objects preference2:book) (count-once-per-objects preference1:hexagonal_bin) )
    30
  )
)
(:scoring
  (+ (count-once-per-objects preference1:dodgeball) (count preference2:tall_cylindrical_block) )
)
)


(define (game game-id-110) (:domain few-objects-room-v1)
(:setup
  (forall (?w - hexagonal_bin)
    (and
      (game-conserved
        (agent_holds ?w ?w)
      )
    )
  )
)
(:constraints
  (and
    (forall (?d - blue_cube_block)
      (and
        (preference preference1
          (exists (?o - hexagonal_bin)
            (then
              (once (not (< (distance ?d ?o) 0.5) ) )
              (hold (adjacent ?d) )
              (hold (and (adjacent ?d) (adjacent ?o agent) ) )
              (hold-while (and (in ?d) (not (not (agent_holds pink ?o) ) ) ) (in_motion ) )
            )
          )
        )
      )
    )
    (preference preference2
      (exists (?k - game_object)
        (at-end
          (and
            (and
              (and
                (agent_holds ?k)
                (in_motion ?k)
              )
              (and
                (not
                  (in_motion upright)
                )
                (adjacent_side ?k)
                (in_motion ?k)
              )
            )
            (same_color ?k ?k)
          )
        )
      )
    )
    (forall (?k - ball)
      (and
        (preference preference3
          (exists (?p ?f - chair)
            (then
              (once (agent_holds top_drawer) )
              (once (and (in_motion ?p) (in_motion ?p ?k) (and (on ?f) (on ?p ?k) ) ) )
              (hold (agent_holds ?f) )
            )
          )
        )
        (preference preference4
          (exists (?g - green_triangular_ramp)
            (then
              (once (or (not (agent_holds ?g ?g) ) (agent_holds ?k) (and (and (and (not (in upright) ) (in_motion ?k) (= 10 0) (in_motion ?g ?g) ) (on pillow) ) (on desk drawer) ) ) )
              (hold (agent_holds ?k ?k) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (* 6 (count preference4:green) )
    (* 2 18 )
  )
)
(:scoring
  (- (count preference2:yellow) )
)
)


(define (game game-id-111) (:domain many-objects-room-v1)
(:setup
  (game-optional
    (not
      (and
        (exists (?e - chair)
          (< 3 1)
        )
        (agent_holds floor ?xxx)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (hold (and (in_motion ?xxx ?xxx) (agent_holds south_west_corner) ) )
        (once (agent_holds ?xxx) )
        (once (or (on ?xxx) (adjacent_side ?xxx) (on color) ) )
      )
    )
  )
)
(:terminal
  (= (+ (count preference1:yellow:white) (count-once preference1:golfball) )
    3
  )
)
(:scoring
  (count-once-per-objects preference1:basketball)
)
)


(define (game game-id-112) (:domain many-objects-room-v1)
(:setup
  (exists (?r - rug)
    (exists (?p - cube_block)
      (game-conserved
        (in_motion bed)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?h - (either dodgeball hexagonal_bin pyramid_block))
        (at-end
          (adjacent ?h)
        )
      )
    )
  )
)
(:terminal
  (>= (* (count-once-per-objects preference1:side_table) 3 )
    3
  )
)
(:scoring
  7
)
)


(define (game game-id-113) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (in ?xxx)
  )
)
(:constraints
  (and
    (forall (?l - block)
      (and
        (preference preference1
          (exists (?j - hexagonal_bin)
            (then
              (hold (not (touch ?l ?j) ) )
              (hold (and (adjacent ?j ?j) (> 7 (distance_side agent back)) ) )
              (hold (agent_holds agent) )
              (hold (agent_holds ?l ?l) )
              (once (adjacent_side ?l ?l ?j) )
            )
          )
        )
      )
    )
    (forall (?z - game_object)
      (and
        (preference preference2
          (at-end
            (open ?z bridge_block)
          )
        )
      )
    )
  )
)
(:terminal
  (>= 4 (count-once-per-objects preference1:blue_cube_block:pink) )
)
(:scoring
  (count preference2:alarm_clock)
)
)


(define (game game-id-114) (:domain few-objects-room-v1)
(:setup
  (not
    (and
      (and
        (exists (?q - hexagonal_bin)
          (game-conserved
            (and
              (in_motion ?q)
              (touch ?q ?q)
            )
          )
        )
        (and
          (game-conserved
            (and
              (and
                (on blue)
                (touch ?xxx)
              )
              (not
                (touch ?xxx ?xxx)
              )
            )
          )
        )
      )
      (game-conserved
        (object_orientation ?xxx)
      )
      (exists (?g - triangular_ramp ?d - block)
        (game-conserved
          (in_motion ?d ?d)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?s - golfball ?k - dodgeball ?v - pillow)
        (then
          (hold (in ?v ?v) )
          (hold-while (and (not (and (in_motion ?v ?v) (in_motion ?v) ) ) (in_motion ?v) (and (touch ?v ?v) (and (>= 1 (distance 10 ?v)) (and (is_setup_object pink_dodgeball) (in_motion blue) ) ) ) ) (in ?v ?v) )
          (once (agent_holds green) )
        )
      )
    )
  )
)
(:terminal
  (or
    (or
      (>= (external-forall-maximize (count preference1:hexagonal_bin) ) 7 )
      (>= (count preference1:red) (count-once preference1:blue_dodgeball) )
    )
    (or
      (> 2 (total-score) )
      (or
        (>= (* (count preference1:beachball) (count preference1:dodgeball) )
          10
        )
      )
    )
    (>= (* 4 0 )
      15
    )
    (>= (total-score) (count preference1:pink:blue_dodgeball:yellow) )
  )
)
(:scoring
  15
)
)


(define (game game-id-115) (:domain medium-objects-room-v1)
(:setup
  (exists (?g - beachball)
    (exists (?x - curved_wooden_ramp)
      (and
        (exists (?e - cube_block ?a - block)
          (game-conserved
            (and
              (not
                (and
                  (not
                    (on ?a)
                  )
                  (exists (?k - dodgeball ?r - dodgeball)
                    (not
                      (and
                        (and
                          (on ?r ?x)
                        )
                        (in_motion rug)
                      )
                    )
                  )
                  (or
                    (and
                      (on ?a ?x)
                      (on desk ?g ?g)
                    )
                    (and
                      (agent_holds ?x)
                      (and
                        (is_setup_object bed)
                      )
                    )
                  )
                )
              )
              (in_motion ?x)
            )
          )
        )
        (and
          (game-optional
            (object_orientation ?x)
          )
          (game-conserved
            (exists (?m - building)
              (in ?x ?m)
            )
          )
        )
        (and
          (and
            (exists (?c - ball ?v - hexagonal_bin)
              (game-conserved
                (not
                  (not
                    (and
                      (agent_holds ?g)
                      (in_motion ?v ?v)
                    )
                  )
                )
              )
            )
            (forall (?m - hexagonal_bin ?m - dodgeball)
              (and
                (game-conserved
                  (on ?g ?g)
                )
              )
            )
            (game-conserved
              (in_motion ?x)
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?e - ball)
        (then
          (hold (and (exists (?w - hexagonal_bin) (in ?e) ) (agent_holds ?e) (on ?e) ) )
        )
      )
    )
  )
)
(:terminal
  (> 10 1 )
)
(:scoring
  3
)
)


(define (game game-id-116) (:domain many-objects-room-v1)
(:setup
  (or
    (and
      (forall (?c - desktop)
        (and
          (game-conserved
            (exists (?r - curved_wooden_ramp)
              (in_motion ?r ?c)
            )
          )
          (game-optional
            (adjacent_side ?c)
          )
        )
      )
      (and
        (game-conserved
          (on ?xxx)
        )
      )
      (game-conserved
        (exists (?o - dodgeball)
          (and
            (agent_holds ?o agent)
            (and
              (touch ?o)
              (exists (?u - hexagonal_bin)
                (not
                  (agent_holds ?o)
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
    (forall (?r - hexagonal_bin ?m - game_object)
      (and
        (preference preference1
          (exists (?n - cube_block ?a - drawer ?c - chair)
            (then
              (hold (agent_holds ?m) )
              (hold (rug_color_under ?c) )
              (hold (in_motion ?c) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (not
    (>= (count preference1:pink) 9 )
  )
)
(:scoring
  10
)
)


(define (game game-id-117) (:domain few-objects-room-v1)
(:setup
  (exists (?k - curved_wooden_ramp ?w - (either dodgeball dodgeball) ?y - chair)
    (game-conserved
      (in_motion ?y ?y)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?k - golfball ?y - doggie_bed ?n - (either golfball game_object))
        (then
          (hold (agent_holds ?n bed) )
          (once (not (not (adjacent ?n) ) ) )
          (hold (and (or (not (not (and (in_motion ?n ?n) (touch ?n) ) ) ) (in_motion ?n ?n) ) (in_motion ?n) ) )
        )
      )
    )
    (forall (?y - hexagonal_bin ?a - dodgeball)
      (and
        (preference preference2
          (exists (?w - dodgeball ?m - (either alarm_clock lamp game_object))
            (at-end
              (in_motion agent)
            )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (not
      (>= (> 10 1 )
        (+ (count-once-per-external-objects preference1:green) (+ (count-once preference1:alarm_clock) 6 (* (* (count preference1:yellow:dodgeball) 1 )
              6
            )
            2
            (count preference2:block:triangle_block)
            (count preference1:basketball)
          )
        )
      )
    )
    (and
      (not
        (or
          (and
            (or
              (or
                (>= 10 (count preference1:red) )
                (>= 3 (count-once preference2:hexagonal_bin) )
                (>= 1 1 )
                (>= (count-once-per-objects preference2:basketball) (count preference2:dodgeball) )
              )
              (>= (- (>= (- 9 )
                    10
                  )
                )
                30
              )
              (>= (count-once-per-objects preference1:pink_dodgeball) (count preference2:blue_pyramid_block) )
            )
            (>= 2 3 )
          )
          (>= (count-once-per-objects preference1:hexagonal_bin:blue_pyramid_block) 3 )
        )
      )
      (or
        (or
          (> (* 4 (count-once preference2:yellow) )
            (count preference1:orange:block)
          )
          (<= 5 (count preference2:golfball:basketball) )
        )
        (>= (count-once-per-objects preference2:green) (<= (count preference2:pink) (count-once-per-objects preference2:basketball) )
        )
      )
    )
    (or
      (>= 2 (count-measure preference2:dodgeball) )
      (or
        (< (total-score) (count-overlapping preference1:green:yellow) )
        (or
          (> (count preference1:basketball:dodgeball) 5 )
          (>= (count preference1:red) (* (count preference1:pink:orange) (count preference2:dodgeball:dodgeball) (count preference1:dodgeball) (count preference1:orange) 5 10 )
          )
          (<= (count preference1:pink) 4 )
        )
      )
    )
  )
)
(:scoring
  (count-once preference2:beachball)
)
)


(define (game game-id-118) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (in_motion agent pink)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?e - building)
        (then
          (hold (exists (?m - (either alarm_clock blue_cube_block blue_cube_block) ?r - dodgeball) (not (between ?e ?e) ) ) )
          (hold-while (touch ?e ?e) (agent_holds ?e ?e) (not (exists (?h - (either yellow triangle_block) ?w - cube_block) (and (in_motion ?w) (in desk) ) ) ) )
          (once (and (exists (?t - bridge_block ?h - dodgeball ?k - pillow) (on ?e) ) (in ?e ?e) ) )
        )
      )
    )
    (preference preference2
      (at-end
        (agent_holds ?xxx ?xxx)
      )
    )
  )
)
(:terminal
  (or
    (or
      (or
        (>= (count-longest preference2:beachball:dodgeball) (* (count preference1:purple) (+ 100 20 )
          )
        )
      )
      (= 3 2 )
    )
  )
)
(:scoring
  (- (count preference1:beachball:dodgeball:beachball) )
)
)


(define (game game-id-119) (:domain many-objects-room-v1)
(:setup
  (not
    (exists (?o - hexagonal_bin)
      (and
        (and
          (forall (?d - cube_block)
            (or
              (and
                (and
                  (and
                    (and
                      (game-optional
                        (on ?d rug)
                      )
                      (and
                        (exists (?m - (either blue_cube_block basketball) ?n - wall ?e - block ?p - beachball ?s - shelf)
                          (game-conserved
                            (in ?s)
                          )
                        )
                      )
                    )
                    (game-conserved
                      (agent_holds ?d ?o)
                    )
                  )
                  (or
                    (game-conserved
                      (not
                        (not
                          (agent_holds ?o ?o)
                        )
                      )
                    )
                    (forall (?y - golfball ?b - hexagonal_bin)
                      (game-optional
                        (exists (?l ?n - dodgeball)
                          (not
                            (agent_holds ?l)
                          )
                        )
                      )
                    )
                  )
                  (forall (?f - triangular_ramp)
                    (and
                      (exists (?s - hexagonal_bin)
                        (exists (?e - game_object ?h - building)
                          (exists (?t - teddy_bear)
                            (exists (?p - blue_pyramid_block ?a - (either tall_cylindrical_block triangle_block))
                              (game-conserved
                                (agent_holds ?f)
                              )
                            )
                          )
                        )
                      )
                    )
                  )
                )
                (exists (?i - (either dodgeball laptop) ?l - pyramid_block ?r - (either book golfball))
                  (and
                    (exists (?y - hexagonal_bin)
                      (game-optional
                        (not
                          (in_motion rug)
                        )
                      )
                    )
                  )
                )
                (forall (?a - dodgeball)
                  (game-conserved
                    (on ?a)
                  )
                )
              )
              (game-conserved
                (same_color green_golfball ?d)
              )
              (game-conserved
                (same_object ?o)
              )
            )
          )
        )
        (game-optional
          (and
            (and
              (in ?o)
              (in agent ?o)
            )
            (in ?o)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?h - wall ?g - hexagonal_bin ?s - block ?y - hexagonal_bin)
      (and
        (preference preference1
          (exists (?s - cube_block ?v - hexagonal_bin)
            (then
              (once (agent_holds ?y ?v) )
              (hold (not (and (in_motion pink_dodgeball) (not (in_motion ?y ?v) ) (and (not (in_motion agent) ) (not (not (on ?v) ) ) (in_motion ?y) ) ) ) )
              (once (not (agent_holds ?y desk) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (<= (count-total preference1:red) (+ (count-once-per-objects preference1:dodgeball:basketball) (external-forall-maximize (count-once preference1:beachball) ) )
  )
)
(:scoring
  (* (* (* (= (count-once-per-objects preference1:doggie_bed) )
        (+ 1 (< (count-once preference1:dodgeball) 8 )
          (* (count preference1:beachball) (total-score) )
        )
      )
      20
    )
    (external-forall-minimize
      (+ 180 )
    )
  )
)
)


(define (game game-id-120) (:domain medium-objects-room-v1)
(:setup
  (game-optional
    (touch ?xxx)
  )
)
(:constraints
  (and
    (forall (?s - triangular_ramp ?i - dodgeball)
      (and
        (preference preference1
          (exists (?a - chair)
            (then
              (once (and (and (and (or (on ?a) (in ?a) ) (exists (?g - hexagonal_bin) (agent_holds pink_dodgeball ?i) ) ) (< (distance agent agent) 2) ) (agent_holds ?a) ) )
              (once (in_motion ?i) )
              (once (and (in_motion ?a ?a) (not (on ?i ?i) ) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (< (count-measure preference1:basketball) (count-once-per-objects preference1:dodgeball) )
    (>= 2 4 )
  )
)
(:scoring
  (count preference1:yellow)
)
)


(define (game game-id-121) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (agent_holds ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?z - block)
        (then
          (once (and (in ?z) (not (and (open ?z ?z) (> 7 (distance 9 ?z)) ) ) ) )
          (once (not (and (not (agent_holds ?z) ) (in ?z) ) ) )
          (once (agent_holds ?z brown) )
          (hold-while (is_setup_object upright) (and (on ?z ?z) (agent_holds ?z) ) )
          (hold (not (agent_holds ?z) ) )
        )
      )
    )
    (preference preference2
      (exists (?g - dodgeball ?c - (either dodgeball cube_block))
        (at-end
          (agent_crouches ?c)
        )
      )
    )
  )
)
(:terminal
  (or
    (>= 3 (total-time) )
    (>= (count preference2:cube_block) (* 2 (count preference2:basketball) )
    )
  )
)
(:scoring
  (- (count preference1:hexagonal_bin:beachball) )
)
)


(define (game game-id-122) (:domain medium-objects-room-v1)
(:setup
  (and
    (forall (?d - doggie_bed ?z - curved_wooden_ramp)
      (exists (?o - (either golfball doggie_bed) ?m - (either mug dodgeball cellphone cylindrical_block))
        (forall (?d - dodgeball)
          (game-conserved
            (adjacent ?z ?d)
          )
        )
      )
    )
    (game-conserved
      (agent_holds ?xxx west_wall)
    )
    (and
      (and
        (and
          (not
            (exists (?r - hexagonal_bin)
              (game-conserved
                (not
                  (on ?r ?r ?r)
                )
              )
            )
          )
          (forall (?n - cube_block ?v ?r ?u ?w ?p ?i - (either desktop laptop) ?e - (either hexagonal_bin pink))
            (exists (?y - doggie_bed)
              (game-conserved
                (on ?e ?e)
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
    (forall (?z - hexagonal_bin ?d - game_object ?d - hexagonal_bin)
      (and
        (preference preference1
          (exists (?f - hexagonal_bin)
            (then
              (once (<= (distance ?d agent desk) 2) )
              (once (and (agent_holds ?d ?d) (on ?f) ) )
              (once (and (agent_holds agent ?d) (adjacent ?d ?f) ) )
            )
          )
        )
      )
    )
    (forall (?t - chair)
      (and
        (preference preference2
          (exists (?c - hexagonal_bin ?k - ball)
            (then
              (hold-while (and (in_motion ?t) (in_motion ?k) ) (adjacent ?t) )
              (once (and (in_motion ?t) (not (on ?k) ) ) )
              (hold (not (not (not (and (not (and (and (and (not (< (distance desk ?k 1) (distance 2 3)) ) (or (not (and (in_motion ?t) (and (not (not (= 0.5 1) ) ) (agent_holds ?k) ) (in_motion upright ?t) ) ) (agent_holds pink_dodgeball) ) ) (not (in ?k ?t ?k) ) ) (in_motion ?t) ) ) ) ) ) ) )
            )
          )
        )
      )
    )
    (forall (?g - wall)
      (and
        (preference preference3
          (exists (?d ?a - wall ?d - hexagonal_bin)
            (then
              (once (touch ?g ?g) )
              (once (agent_holds ?g ?d) )
              (hold (< 2 0.5) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (<= (+ (total-time) (* (count preference1:blue_dodgeball) )
    )
    (count-shortest preference2:pink)
  )
)
(:scoring
  (external-forall-maximize
    (count preference1:yellow_cube_block)
  )
)
)


(define (game game-id-123) (:domain many-objects-room-v1)
(:setup
  (forall (?n - hexagonal_bin ?s - game_object)
    (exists (?v - (either cellphone pyramid_block cylindrical_block) ?d - hexagonal_bin)
      (game-optional
        (on ?s ?s)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?w - (either mug) ?c - hexagonal_bin)
        (at-end
          (on ?c)
        )
      )
    )
    (forall (?t - (either curved_wooden_ramp golfball dodgeball))
      (and
        (preference preference2
          (exists (?g - ball)
            (then
              (once (agent_holds ?t) )
              (once (same_type sideways ?g) )
              (once-measure (and (in side_table) (same_color agent ?t) ) (distance ?t) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= 4 (* (+ 7 5 )
        6
        (* (count-once-per-objects preference2:red:blue_dodgeball) 9 )
      )
    )
  )
)
(:scoring
  (count-measure preference2:basketball)
)
)


(define (game game-id-124) (:domain medium-objects-room-v1)
(:setup
  (forall (?a - hexagonal_bin ?r - cube_block)
    (game-optional
      (in_motion ?r ?r)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?x - curved_wooden_ramp)
        (then
          (hold-while (adjacent_side agent ?x) (in_motion ?x) (agent_holds ball) )
          (hold (between desk ?x) )
          (once (and (in_motion left ?x) (agent_holds desk) ) )
        )
      )
    )
    (preference preference2
      (exists (?r - (either blue_cube_block laptop) ?u - chair)
        (at-end
          (on ?u)
        )
      )
    )
    (preference preference3
      (exists (?y - block ?y - hexagonal_bin)
        (then
          (once (between ?y) )
          (once (not (is_setup_object ?y) ) )
          (once (object_orientation ?y) )
          (once (< (distance room_center agent) (distance 8 ?y)) )
        )
      )
    )
  )
)
(:terminal
  (<= (* (+ (* (* 4 (* (* (count-once-per-objects preference1:purple) (count preference2:wall) (count-once preference1:beachball) )
              6
            )
          )
          (- (- (* (count preference1:green) (count preference3:green) )
            )
          )
        )
        1
        (count-once-per-objects preference1:pink_dodgeball)
        1
      )
      (count preference2:basketball)
    )
    (count preference2)
  )
)
(:scoring
  (* (count preference3:yellow) (count preference3:pink) )
)
)


(define (game game-id-125) (:domain medium-objects-room-v1)
(:setup
  (exists (?v - dodgeball ?n - curved_wooden_ramp)
    (exists (?m - doggie_bed)
      (exists (?u - color)
        (and
          (and
            (game-optional
              (adjacent ?u)
            )
            (game-conserved
              (touch ?n ?m)
            )
            (game-conserved
              (on ?m)
            )
          )
          (forall (?y - triangular_ramp)
            (and
              (exists (?o - building ?r - block)
                (exists (?o - teddy_bear)
                  (game-conserved
                    (and
                      (not
                        (on agent ?u)
                      )
                      (in ?n ?r)
                      (in_motion ?u ?y)
                    )
                  )
                )
              )
              (game-optional
                (and
                  (agent_holds ?u ?n)
                  (and
                    (not
                      (adjacent ?y)
                    )
                    (agent_holds agent ?y)
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
(:constraints
  (and
    (preference preference1
      (exists (?p ?n - hexagonal_bin)
        (then
          (once (and (in ?n) (exists (?z - teddy_bear) (in_motion ?p side_table) ) (> 3 2) (in ?p ?p) (agent_holds ?n) (exists (?t - building) (in ?t) ) ) )
          (once (adjacent agent) )
          (once (and (not (not (not (not (not (and (agent_holds ?p) ) ) ) ) ) ) (not (in_motion ?n) ) ) )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (- (* (count preference1:yellow) (+ 5 (count preference1:red) )
        )
      )
      (+ (count preference1:orange) (* (* (+ (* (count-once-per-objects preference1:yellow_pyramid_block) (+ 1 (+ 2 (count-total preference1:red:pink) )
                )
              )
              (* 2 3 )
            )
            (+ 6 (count preference1:hexagonal_bin) )
          )
          4
        )
      )
    )
    (not
      (>= (count preference1:yellow) (count preference1:dodgeball) )
    )
  )
)
(:scoring
  2
)
)


(define (game game-id-126) (:domain many-objects-room-v1)
(:setup
  (exists (?b - dodgeball)
    (game-conserved
      (and
        (not
          (and
            (on ?b)
            (and
              (not
                (agent_holds bed)
              )
              (in_motion ?b)
            )
            (not
              (agent_holds bed)
            )
            (not
              (on front)
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (agent_holds ?xxx) )
        (once (touch ?xxx ?xxx) )
        (once (not (not (on desk) ) ) )
      )
    )
  )
)
(:terminal
  (>= 3 7 )
)
(:scoring
  5
)
)


(define (game game-id-127) (:domain medium-objects-room-v1)
(:setup
  (game-optional
    (in ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?s - game_object ?j - game_object)
        (at-end
          (agent_holds top_shelf)
        )
      )
    )
    (preference preference2
      (exists (?n - drawer)
        (at-end
          (game_over ?n ?n)
        )
      )
    )
  )
)
(:terminal
  (not
    (and
      (>= 5 (count preference2:golfball) )
    )
  )
)
(:scoring
  5
)
)


(define (game game-id-128) (:domain many-objects-room-v1)
(:setup
  (and
    (exists (?t - hexagonal_bin)
      (game-optional
        (adjacent ?t)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?w - ball ?x - curved_wooden_ramp ?y - hexagonal_bin)
        (then
          (once (and (and (in_motion ?y ?y) (not (agent_holds ?y ?y) ) (agent_holds ?y ?y) (< 7 1) ) (agent_holds ?y) (exists (?p - golfball) (on ?y) ) ) )
          (hold-while (and (not (in_motion ?y rug) ) (exists (?p - golfball) (agent_holds ?y agent) ) ) (adjacent ?y) )
          (hold-while (touch ?y sideways) (not (= 1 1) ) )
          (once (and (in_motion ?y ?y) ) )
        )
      )
    )
    (preference preference2
      (exists (?o - building ?k - doggie_bed)
        (then
          (hold (in_motion ?k) )
          (hold (not (adjacent desktop) ) )
          (once (touch ?k) )
        )
      )
    )
  )
)
(:terminal
  (>= (total-time) (= (count-once-per-objects preference1:dodgeball:yellow) (* 180 5 (count preference2:golfball) 10 )
    )
  )
)
(:scoring
  (count-once preference2:beachball)
)
)


(define (game game-id-129) (:domain few-objects-room-v1)
(:setup
  (game-optional
    (in_motion ?xxx ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?q - dodgeball ?v - hexagonal_bin)
        (then
          (hold-while (forall (?w - tall_cylindrical_block ?d - ball) (in_motion ?d ?d) ) (not (and (not (in_motion ?v desk) ) (faces ?v ?v) ) ) (in_motion ?v ?v) (= (distance desk ?v) 4) )
          (hold (on ?v) )
          (once (not (in ?v) ) )
          (once (> (distance ?v ?v) (distance room_center ?v)) )
        )
      )
    )
    (forall (?v - wall)
      (and
        (preference preference2
          (exists (?r - shelf)
            (then
              (hold (and (in ?r ?r) (on ?v) (< (distance 2 ?v) 1) (not (and (agent_holds rug ?v ?v) (exists (?f - ball ?h - (either watch laptop)) (< 1 7) ) ) ) ) )
              (once (in_motion ?r) )
              (hold (adjacent ?r) )
            )
          )
        )
        (preference preference3
          (exists (?l - building)
            (then
              (once (forall (?y - building) (same_color agent ?l) ) )
              (hold (and (on ?v) (not (adjacent_side desk) ) ) )
              (once (between ?v) )
            )
          )
        )
        (preference preference4
          (exists (?u - hexagonal_bin ?q - (either bridge_block red pencil) ?u - hexagonal_bin ?e - ball ?c - dodgeball)
            (then
              (once (on ?c ?c) )
              (once (in_motion ?v ?v) )
              (hold (agent_holds ?c ?v) )
            )
          )
        )
      )
    )
    (preference preference5
      (exists (?q - building)
        (then
          (once (not (not (not (agent_holds agent ?q) ) ) ) )
          (once (touch ?q ?q) )
          (once (in_motion ?q ?q) )
        )
      )
    )
    (forall (?o ?m ?t ?d - ball)
      (and
        (preference preference6
          (exists (?w - (either doggie_bed golfball) ?f - cube_block)
            (at-end
              (and
                (in_motion ?f ?t)
                (not
                  (and
                    (and
                      (not
                        (not
                          (on ?d ?m)
                        )
                      )
                      (and
                        (and
                          (and
                            (in_motion ?o)
                            (and
                              (not
                                (agent_holds ?t ?d)
                              )
                              (or
                                (= 1 1)
                                (not
                                  (on ?f)
                                )
                              )
                            )
                          )
                          (or
                            (adjacent ?d ?t)
                            (on agent)
                            (in_motion bed ?m)
                          )
                        )
                        (not
                          (agent_holds bed ?m)
                        )
                      )
                    )
                    (not
                      (on ?o)
                    )
                  )
                )
              )
            )
          )
        )
        (preference preference7
          (then
            (any)
            (hold (and (on ?d ?t) (and (and (and (and (not (agent_holds ?m) ) (same_color ?t) (same_color ?o) (not (<= (distance ?o 0) (distance room_center 3 room_center)) ) ) (= 1 (distance room_center ?t) 0.5) ) (not (in ?d pink_dodgeball) ) ) (adjacent floor agent) ) ) )
            (hold (not (agent_holds ?m ?m) ) )
          )
        )
      )
    )
    (preference preference8
      (exists (?k - hexagonal_bin ?b - dodgeball)
        (then
          (once (touch bed ?b) )
          (once (on ?b) )
          (once (not (in_motion agent) ) )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference5:beachball) (/ (count-once-per-objects preference5:pink) (* (+ (* 6 (count-once-per-objects preference4:triangle_block:yellow) )
          (+ 10 (count-once-per-external-objects preference1:orange:dodgeball) )
        )
        (+ (count-once preference7:book) (* (count-once-per-objects preference1:beachball) (* (count preference3:golfball) (count-once-per-objects preference4:basketball) )
          )
        )
      )
    )
  )
)
(:scoring
  8
)
)


(define (game game-id-130) (:domain many-objects-room-v1)
(:setup
  (and
    (game-optional
      (in upright door)
    )
    (and
      (and
        (game-optional
          (not
            (game_over ?xxx)
          )
        )
      )
    )
    (game-conserved
      (adjacent ?xxx rug)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?s - pyramid_block)
        (then
          (hold (agent_holds ?s floor) )
          (once (agent_holds ?s) )
          (once (touch ?s) )
        )
      )
    )
    (preference preference2
      (exists (?l - cube_block ?f - triangular_ramp)
        (then
          (once (not (on ?f ?f) ) )
          (once (adjacent ?f ?f) )
          (once (and (rug_color_under ?f) (and (exists (?t - wall ?b - hexagonal_bin) (same_color ?b agent) ) (in_motion ?f) (and (and (in ?f ?f) (and (adjacent ?f ?f) (not (not (adjacent ?f rug) ) ) ) (agent_holds desk ?f) (on bed ?f) ) (and (on floor) (not (in_motion ?f) ) ) ) ) ) )
        )
      )
    )
    (preference preference3
      (exists (?s - dodgeball)
        (then
          (hold (and (and (and (not (agent_holds desk) ) (and (in_motion ?s) (agent_holds ?s ?s) (and (< 1 (distance ?s agent)) ) (and (exists (?j - game_object) (not (and (agent_holds ?j) (in bed) (and (in ?j) (and (and (and (agent_holds ?j) (and (in_motion floor ?s) (and (in_motion ?j) (or (on ?j) (agent_holds ?j) ) ) ) (<= 1 (distance ?s ?s)) ) (not (in_motion ?j ?j) ) ) (and (not (and (not (and (and (in_motion ?j) ) (not (not (same_color ?s) ) ) ) ) (not (is_setup_object ?j front_left_corner) ) ) ) (and (not (adjacent ?j) ) (and (and (and (in_motion ?j) (agent_holds ?s) (in ?j ?s) (and (not (and (in_motion ?s) (not (in_motion ?s) ) ) ) (not (in ?j) ) ) ) (agent_holds ?j ?j) ) (not (not (< (distance ?s 6) (distance_side door ?j)) ) ) ) ) (agent_holds rug) (and (in_motion agent ?s) (on ?j ?s) ) ) (and (and (adjacent ?s) (not (exists (?i - hexagonal_bin) (agent_holds ?i ?s) ) ) (in ?s) ) (between agent) ) ) ) ) ) ) (and (and (in_motion ?s) ) (forall (?m ?k - cube_block) (agent_holds ?m ?s) ) (on ?s ?s) (equal_z_position ?s) ) (equal_z_position ?s ?s) (agent_holds ?s ?s) (same_object ?s) (adjacent ?s) ) ) ) (in ?s ?s) ) (and (on ?s ?s) (not (or (game_over agent) (not (on ?s) ) ) ) ) ) )
          (once (and (in_motion ?s brown) (same_color ?s) ) )
          (hold (and (= (distance front 0) (distance ?s front)) (< 8 1) ) )
        )
      )
    )
  )
)
(:terminal
  (>= (* (count preference3:pink) 4 )
    (count-once preference3:pyramid_block)
  )
)
(:scoring
  (* (count preference1:golfball:yellow:hexagonal_bin) )
)
)


(define (game game-id-131) (:domain many-objects-room-v1)
(:setup
  (forall (?w - (either dodgeball beachball) ?m - block ?n - curved_wooden_ramp)
    (game-conserved
      (not
        (opposite ?n bed)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?m - pillow)
        (at-end
          (in_motion ?m)
        )
      )
    )
    (forall (?j - building)
      (and
        (preference preference2
          (exists (?x - teddy_bear ?k - wall)
            (then
              (once (and (agent_holds ?k) (on ?k) ) )
              (once (exists (?s ?x ?w - game_object ?h - game_object) (equal_x_position ?j pink) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (not
    (>= (count-longest preference1:red_pyramid_block) (+ 10 (count-once preference1:red) )
    )
  )
)
(:scoring
  (count-once-per-objects preference2)
)
)


(define (game game-id-132) (:domain many-objects-room-v1)
(:setup
  (exists (?c - dodgeball)
    (exists (?y ?p - hexagonal_bin)
      (exists (?d - teddy_bear)
        (or
          (game-optional
            (not
              (not
                (not
                  (adjacent ?y)
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
    (forall (?n - doggie_bed ?e - game_object)
      (and
        (preference preference1
          (exists (?t - block)
            (then
              (hold-while (agent_holds ?t ?t ?t) (exists (?x - curved_wooden_ramp) (not (in_motion ?x) ) ) )
              (once (and (or (touch ?e ?t) (and (not (agent_holds ?t ?e) ) (same_color pillow) ) ) (and (not (in_motion ?e) ) (< 2 (distance ?e 10 ?e)) ) ) )
              (hold (not (> (distance ?e ?t) (distance agent ?e)) ) )
            )
          )
        )
      )
    )
    (preference preference2
      (exists (?c - building)
        (then
          (hold (= (x_position desk) (x_position ?c room_center)) )
          (once (agent_holds ?c ?c) )
          (hold (< (distance ?c room_center) 2) )
        )
      )
    )
  )
)
(:terminal
  (< (* 30 (total-score) 1 )
    2
  )
)
(:scoring
  (count preference2:doggie_bed)
)
)


(define (game game-id-133) (:domain few-objects-room-v1)
(:setup
  (and
    (game-optional
      (agent_holds ?xxx)
    )
    (and
      (game-conserved
        (on ?xxx)
      )
      (game-optional
        (agent_holds ?xxx)
      )
      (and
        (game-optional
          (not
            (and
              (in ?xxx brown)
              (agent_holds ?xxx)
            )
          )
        )
      )
    )
    (not
      (exists (?s - (either ball pyramid_block))
        (and
          (and
            (and
              (exists (?o - hexagonal_bin)
                (game-conserved
                  (adjacent ?s ?s)
                )
              )
            )
          )
          (game-conserved
            (on ?s ?s)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?w - hexagonal_bin)
        (at-end
          (faces ?w ?w)
        )
      )
    )
  )
)
(:terminal
  (> 4 15 )
)
(:scoring
  (* 7 (count preference1:yellow_cube_block:pink) )
)
)


(define (game game-id-134) (:domain many-objects-room-v1)
(:setup
  (or
    (forall (?m - ball)
      (and
        (exists (?g - dodgeball)
          (game-conserved
            (not
              (in_motion ?g agent)
            )
          )
        )
      )
    )
    (forall (?o - bridge_block)
      (game-optional
        (not
          (in_motion ?o)
        )
      )
    )
    (or
      (exists (?y - drawer ?v - (either triangle_block golfball cd))
        (game-conserved
          (not
            (not
              (and
                (< (distance ?v ?v) 5)
                (touch ?v)
              )
            )
          )
        )
      )
      (and
        (and
          (game-conserved
            (touch agent)
          )
        )
        (exists (?y - chair)
          (or
            (and
              (game-conserved
                (exists (?n - game_object)
                  (agent_holds pink)
                )
              )
            )
            (game-conserved
              (not
                (not
                  (and
                    (in_motion ?y)
                    (agent_holds ?y)
                  )
                )
              )
            )
          )
        )
      )
      (game-optional
        (touch ?xxx)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?b - teddy_bear)
        (then
          (once (agent_holds bed) )
          (once (in_motion ?b agent) )
        )
      )
    )
  )
)
(:terminal
  (> (* 3 (+ (or (count preference1:dodgeball) 5 (count-shortest preference1:beachball) ) (count preference1:hexagonal_bin) 1 )
    )
    (= (or 2 2 (count-once-per-external-objects preference1:dodgeball) ) (count-longest preference1:beachball) )
  )
)
(:scoring
  (count preference1:hexagonal_bin:pink_dodgeball)
)
)


(define (game game-id-135) (:domain medium-objects-room-v1)
(:setup
  (exists (?y - ball)
    (game-optional
      (and
        (touch ?y)
        (in_motion ?y)
      )
    )
  )
)
(:constraints
  (and
    (forall (?t ?w - hexagonal_bin)
      (and
        (preference preference1
          (exists (?b - hexagonal_bin ?s - blinds)
            (then
              (hold (adjacent ?t ?w) )
              (once (and (forall (?d - (either mug chair) ?x - hexagonal_bin) (and (in_motion ?x) (agent_holds ?s ?w) (and (not (in ?t) ) (not (broken ?s) ) ) ) ) (< (x_position 10 ?w) 1) ) )
              (once (not (< (distance 3 1) 2) ) )
            )
          )
        )
      )
    )
    (forall (?o - dodgeball)
      (and
        (preference preference2
          (exists (?n ?w - red_dodgeball)
            (then
              (hold (in_motion ?o ?o) )
              (once (and (in_motion ?o ?o) (agent_holds ?n) ) )
              (once (object_orientation ?n) )
            )
          )
        )
        (preference preference3
          (exists (?k - dodgeball ?f - hexagonal_bin)
            (then
              (once (adjacent ?f) )
              (hold (exists (?w - ball ?i - color) (above ?o green) ) )
              (once (< 2 1) )
            )
          )
        )
      )
    )
    (preference preference4
      (then
        (once (in_motion ?xxx agent) )
        (once (or (and (not (agent_holds ?xxx) ) (not (and (not (agent_holds ?xxx) ) (on ?xxx ?xxx) ) ) (not (not (in_motion ?xxx ?xxx) ) ) ) (in ?xxx) ) )
        (hold (equal_z_position desk ?xxx) )
      )
    )
  )
)
(:terminal
  (> 2 (count preference1:dodgeball) )
)
(:scoring
  5
)
)


(define (game game-id-136) (:domain many-objects-room-v1)
(:setup
  (not
    (game-conserved
      (on ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?b - hexagonal_bin)
        (then
          (once (and (and (touch ?b) (and (and (agent_holds ?b) (in drawer) ) (in_motion ?b) ) (same_type ?b ?b) ) (object_orientation ?b) ) )
          (hold (agent_holds ?b) )
          (once (agent_holds ?b ?b) )
        )
      )
    )
  )
)
(:terminal
  (not
    (or
      (not
        (>= (count preference1:basketball) (+ 6 (count-once preference1:golfball:beachball) )
        )
      )
    )
  )
)
(:scoring
  3
)
)


(define (game game-id-137) (:domain few-objects-room-v1)
(:setup
  (game-optional
    (touch rug)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?h - hexagonal_bin ?o - dodgeball)
        (then
          (once (and (and (in ?o) (and (not (in blinds) ) (agent_holds ?o ?o) (and (not (not (in_motion ?o ?o ?o) ) ) (and (and (in ?o) (and (agent_holds ?o) (in_motion ?o sideways) ) ) (not (on ?o ?o) ) ) ) ) ) (in_motion ?o) (agent_holds ?o rug) (and (in_motion ?o) (not (agent_holds pink_dodgeball) ) ) (in_motion ?o) (exists (?u - dodgeball) (and (agent_holds ?o ?u) (is_setup_object ?o ?o) ) ) (and (rug_color_under ?o ?o) (not (and (agent_holds rug ?o) (equal_z_position ?o) ) ) (and (on ?o) (agent_holds ?o ?o) ) ) ) )
          (hold (in_motion ?o ?o) )
          (once (agent_holds ?o) )
        )
      )
    )
    (forall (?c - red_dodgeball ?s - tall_cylindrical_block)
      (and
        (preference preference2
          (exists (?y - dodgeball ?r - dodgeball)
            (then
              (once (not (adjacent ?r ?s) ) )
              (once (and (on ?s ?s) (in_motion ?r) ) )
              (once (not (and (touch ?r ?r) (in ?s bed) ) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (> (count preference2:blue_cube_block) 5 )
    (or
      (>= (* 3 (count preference2:red_pyramid_block) )
        (count preference1:dodgeball)
      )
      (>= 3 (* 4 (* 2 8 )
          (count-once preference2:cube_block)
          (count preference1:pink)
        )
      )
      (>= (count preference2:hexagonal_bin) (- (count preference1:bed:pink) )
      )
      (or
        (>= (count-once preference2:hexagonal_bin:pink:yellow) (count-once-per-objects preference1:blue_cube_block) )
        (or
          (>= (count-once preference1) (count-once-per-objects preference1:green:orange:yellow_cube_block) )
          (>= (count preference2:blue_dodgeball:green) (- (* (+ (count-measure preference1:purple) 2 (+ 4 3 )
                )
                7
              )
            )
          )
        )
      )
    )
  )
)
(:scoring
  (count preference2:basketball)
)
)


(define (game game-id-138) (:domain many-objects-room-v1)
(:setup
  (forall (?f - (either mug mug bridge_block cd book blue_cube_block laptop))
    (game-conserved
      (not
        (in ?f)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (hold (and (in ?xxx) (exists (?i - pyramid_block) (not (and (not (agent_holds ?i ?i) ) (in_motion ?i ?i) ) ) ) (agent_holds ?xxx ?xxx) ) )
        (hold (adjacent ?xxx agent) )
        (once (on desk block) )
      )
    )
    (preference preference2
      (exists (?p - dodgeball ?r - pyramid_block)
        (then
          (once (and (adjacent rug) (in_motion ?r) (on ?r ?r) ) )
          (hold-while (or (or (same_color desk) (not (not (not (agent_holds door ?r desk) ) ) ) ) (between ?r) ) (and (and (< (distance ?r ?r) 1) (in ?r floor) ) (and (and (in ?r ?r) (in_motion agent ?r) (agent_holds ?r ?r bed) ) (on ?r) ) ) )
          (once (agent_holds ?r ?r) )
        )
      )
    )
    (preference preference3
      (exists (?z ?h ?r ?x ?g - dodgeball)
        (then
          (once (agent_holds ?h) )
          (once (not (agent_holds ?z) ) )
          (once (adjacent ?r desk) )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference1:red:pink:book) (- (count preference2:golfball) )
  )
)
(:scoring
  30
)
)


(define (game game-id-139) (:domain many-objects-room-v1)
(:setup
  (and
    (forall (?t - (either side_table hexagonal_bin))
      (forall (?m - flat_block ?z - wall)
        (or
          (forall (?w - block)
            (game-conserved
              (agent_holds top_shelf)
            )
          )
        )
      )
    )
    (game-conserved
      (in ?xxx)
    )
  )
)
(:constraints
  (and
    (forall (?m ?j - bridge_block)
      (and
        (preference preference1
          (exists (?p - doggie_bed)
            (then
              (once (not (not (object_orientation ?p) ) ) )
              (once (agent_holds ?m desk) )
              (hold (agent_holds ?m ?m ?j) )
            )
          )
        )
        (preference preference2
          (exists (?i - dodgeball)
            (then
              (hold-while (game_start top_drawer) (in_motion agent) )
              (once (and (and (or (>= 7 (distance ?i)) (< 1 3) ) (same_type ?m ?j) ) (touch ?m ?i) ) )
              (once (in_motion ?m) )
              (hold (and (in_motion bed ?i) (not (or (and (on blue sideways) (in_motion ?i) ) (in ?j ?m) ) ) ) )
            )
          )
        )
        (preference preference3
          (exists (?c ?f ?w ?y ?v - hexagonal_bin)
            (then
              (once (not (agent_holds ?j ?y) ) )
              (hold (and (in rug) (not (agent_holds bed ?c) ) (in ?v) ) )
              (once (agent_holds ?y upright) )
            )
          )
        )
      )
    )
    (forall (?q ?k ?y - cube_block)
      (and
        (preference preference4
          (exists (?x - hexagonal_bin)
            (then
              (once (< 3 (distance 4 ?k 6)) )
              (hold-while (in_motion north_wall ?x) (and (touch ?q ?q) (in ?k) ) (agent_holds ?y) )
              (hold (in_motion ?y) )
            )
          )
        )
      )
    )
    (preference preference5
      (exists (?v - game_object)
        (then
          (once (and (and (not (and (in_motion ?v ?v) ) ) (and (not (not (not (equal_z_position brown) ) ) ) (in_motion ?v ?v) ) ) (not (and (on ?v) (in ?v ?v) ) ) (not (agent_holds ?v) ) ) )
          (once (in_motion desk ?v) )
          (once (and (not (same_color ?v) ) (in_motion ?v ?v) ) )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= 4 (count preference2:pink:beachball) )
    (>= (count-once-per-objects preference3:beachball) (* (count preference3:hexagonal_bin:dodgeball) (count preference4:yellow_pyramid_block:hexagonal_bin) )
    )
  )
)
(:scoring
  (count preference5:book:cube_block)
)
)


(define (game game-id-140) (:domain medium-objects-room-v1)
(:setup
  (and
    (game-conserved
      (in_motion ?xxx ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?n - building)
        (at-end
          (in_motion rug ?n)
        )
      )
    )
    (preference preference2
      (then
        (hold (in_motion ?xxx ?xxx) )
        (hold-while (and (not (not (not (adjacent ?xxx) ) ) ) (touch ?xxx) ) (on ?xxx) )
        (hold (on ?xxx) )
      )
    )
    (preference preference3
      (exists (?a - chair)
        (then
          (once (not (not (in_motion ?a ?a) ) ) )
          (once (and (and (same_object ?a ?a) (touch agent) ) (not (and (and (in ?a ?a) (in_motion bed) ) ) ) ) )
          (hold (and (not (between ?a ?a) ) (not (not (and (and (agent_holds ?a ?a) (not (above ?a ?a ?a) ) ) (not (adjacent ?a ?a) ) ) ) ) ) )
        )
      )
    )
    (preference preference4
      (exists (?b - dodgeball ?w - cube_block)
        (then
          (once (and (not (in_motion agent) ) (not (agent_holds ?w) ) (forall (?n - hexagonal_bin) (not (not (adjacent ?n) ) ) ) ) )
          (hold (on ?w) )
          (once (not (< 1 1) ) )
        )
      )
    )
    (preference preference5
      (exists (?a - teddy_bear)
        (then
          (once (in_motion ?a ?a) )
          (hold (same_color ?a ?a) )
          (once (agent_holds ?a rug) )
        )
      )
    )
    (preference preference6
      (exists (?h - ball)
        (at-end
          (agent_holds ?h)
        )
      )
    )
  )
)
(:terminal
  (>= (count preference4:basketball:orange) (- (count preference2:red) (count-once-per-objects preference3:beachball:golfball:beachball) ) )
)
(:scoring
  (count preference2:dodgeball)
)
)


(define (game game-id-141) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (> 1 0.5)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?c - shelf)
        (then
          (hold (and (not (forall (?j - wall) (or (not (< 2 (distance ?j ?j)) ) (not (and (and (on ?c) (agent_holds ?c ?j) (and (not (and (same_object ?j ?j) (not (and (in_motion ?j) (not (or (not (not (above ?j) ) ) (not (not (in_motion ?c) ) ) (in_motion ?c ?j) ) ) ) ) ) ) (on agent) ) ) ) ) ) ) ) (agent_holds rug) ) )
          (hold (< 1 (distance ?c ?c)) )
        )
      )
    )
    (preference preference2
      (exists (?r - hexagonal_bin)
        (then
          (hold (on ?r) )
          (hold (touch ?r ?r) )
          (once (in bed) )
        )
      )
    )
    (preference preference3
      (exists (?x - hexagonal_bin ?o - ball ?j - dodgeball)
        (at-end
          (not
            (exists (?l - chair)
              (not
                (agent_holds ?l)
              )
            )
          )
        )
      )
    )
    (forall (?b - hexagonal_bin)
      (and
        (preference preference4
          (exists (?m - (either cd yellow) ?u - hexagonal_bin)
            (then
              (hold (< 3 1) )
              (once (< (distance room_center ?u) (distance ?u 8 room_center)) )
              (once (game_start agent) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (> (count-overlapping preference4:brown:basketball) 10 )
)
(:scoring
  3
)
)


(define (game game-id-142) (:domain many-objects-room-v1)
(:setup
  (not
    (game-conserved
      (in ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?d - block ?l - building)
        (then
          (once (not (not (in_motion ?l) ) ) )
          (once (not (on ?l) ) )
          (hold-while (agent_holds ?l ?l) (in_motion ?l) )
        )
      )
    )
  )
)
(:terminal
  (<= (* (count preference1:basketball) 30 )
    (count preference1:yellow)
  )
)
(:scoring
  3
)
)


(define (game game-id-143) (:domain few-objects-room-v1)
(:setup
  (and
    (exists (?w - (either doggie_bed triangular_ramp))
      (game-optional
        (< 0.5 10)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?h - color)
        (then
          (hold (on bed ?h) )
          (once (< 10 (distance ?h 7)) )
          (hold (in_motion ?h ?h) )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (* (count preference1:bed) (= (* (count-once-per-external-objects preference1:blue_pyramid_block) (count-once preference1:beachball) )
          (+ 2 (count-once-per-objects preference1:beachball) )
        )
      )
      (< 5 9 )
    )
    (>= (count-once preference1:golfball) 3 )
    (>= (count preference1:hexagonal_bin) (count preference1:blue_dodgeball) )
  )
)
(:scoring
  (* 5 (* (and (- (total-score) )
        10
        (count preference1:dodgeball:blue_pyramid_block:hexagonal_bin)
      )
      4
      10
      (count-overlapping preference1:beachball)
    )
  )
)
)


(define (game game-id-144) (:domain many-objects-room-v1)
(:setup
  (and
    (forall (?x ?s - curved_wooden_ramp)
      (game-optional
        (agent_holds ?x)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?s - building ?j - hexagonal_bin)
        (then
          (once (and (and (< (distance ) (x_position room_center ?j agent)) (exists (?g - dodgeball) (on ?g ?j) ) ) (not (agent_holds ?j ?j) ) ) )
          (hold (agent_holds ?j) )
          (hold (above blue bridge_block) )
        )
      )
    )
  )
)
(:terminal
  (or
    (and
      (> (count preference1:blue_dodgeball) (count-once-per-objects preference1:basketball:pink) )
      (> (count-once-per-objects preference1:golfball) (* (* (count-once-per-objects preference1:dodgeball:dodgeball) 1 (count preference1) )
          50
        )
      )
    )
    (or
      (>= 5 (* (* 300 3 )
          (* (* (* (count preference1:doggie_bed) (count preference1:triangle_block:dodgeball) (total-score) )
              10
            )
            (count preference1:green)
            (count preference1:basketball)
          )
        )
      )
      (or
        (>= 20 2 )
        (<= (* 5 6 )
          5
        )
      )
    )
  )
)
(:scoring
  (count preference1:orange)
)
)


(define (game game-id-145) (:domain medium-objects-room-v1)
(:setup
  (exists (?e - (either key_chain bed basketball doggie_bed))
    (game-conserved
      (and
        (and
          (on agent ?e)
          (is_setup_object ?e)
          (not
            (not
              (exists (?k - hexagonal_bin)
                (is_setup_object ?k)
              )
            )
          )
        )
        (agent_holds ?e ?e)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?j - hexagonal_bin)
        (at-end
          (and
            (in ?j)
            (same_color ?j ?j)
            (in_motion ?j)
          )
        )
      )
    )
  )
)
(:terminal
  (> 180 (count-once-per-objects preference1:dodgeball) )
)
(:scoring
  4
)
)


(define (game game-id-146) (:domain many-objects-room-v1)
(:setup
  (game-optional
    (on ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?h - hexagonal_bin)
        (then
          (hold (< (x_position agent ?h) 9) )
          (once (adjacent floor ?h) )
          (once (adjacent ?h) )
        )
      )
    )
    (preference preference2
      (exists (?x - game_object)
        (then
          (once (adjacent front bed) )
          (hold (in_motion ?x) )
          (hold (in_motion agent) )
        )
      )
    )
    (preference preference3
      (exists (?e - (either alarm_clock cd cube_block) ?v - doggie_bed ?n - dodgeball ?j - chair ?e - hexagonal_bin)
        (at-end
          (and
            (in_motion ?e ?e)
            (not
              (forall (?z - hexagonal_bin)
                (on ?e)
              )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (= (count preference2:pink:beachball) 7 )
)
(:scoring
  (count preference1:basketball:orange)
)
)


(define (game game-id-147) (:domain many-objects-room-v1)
(:setup
  (game-optional
    (in ?xxx)
  )
)
(:constraints
  (and
    (forall (?f - hexagonal_bin)
      (and
        (preference preference1
          (exists (?y - drawer ?p ?n ?l ?y ?a ?m - dodgeball ?u - wall)
            (at-end
              (in front)
            )
          )
        )
      )
    )
    (preference preference2
      (then
        (once (and (forall (?r - hexagonal_bin) (touch ?r) ) (in_motion ?xxx ?xxx) ) )
      )
    )
  )
)
(:terminal
  (>= 5 (+ (count preference1:yellow:basketball) 5 )
  )
)
(:scoring
  (count preference1:pink)
)
)


(define (game game-id-148) (:domain medium-objects-room-v1)
(:setup
  (and
    (and
      (game-conserved
        (not
          (not
            (< (distance desk ?xxx) 1)
          )
        )
      )
      (exists (?c - flat_block)
        (game-optional
          (and
            (not
              (adjacent ?c)
            )
            (agent_holds ?c)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?d - ball)
        (then
          (once (or (not (same_color ?d) ) (agent_holds ?d) ) )
          (once (adjacent desk) )
          (hold (< 1 5) )
        )
      )
    )
    (preference preference2
      (then
        (once (not (on ?xxx ?xxx) ) )
        (hold (not (on ?xxx) ) )
        (hold-while (not (and (exists (?a - ball) (and (agent_holds ?a) (> 6 (distance desk ?a)) ) ) (not (and (touch ?xxx) (object_orientation ?xxx ?xxx) (in_motion desk) (> 6 1) ) ) (agent_holds ?xxx) ) ) (not (<= 1 (distance_side 6 4)) ) )
      )
    )
    (forall (?m ?g ?o - tall_cylindrical_block)
      (and
        (preference preference3
          (exists (?c - hexagonal_bin ?l - book)
            (then
              (once (and (and (exists (?p - building ?w - dodgeball) (in_motion ?m) ) (agent_holds ?l) ) (not (not (on ?o agent) ) ) ) )
              (hold (agent_holds ?o block) )
              (hold-while (on ?m) (in_motion ?g ?l) (is_setup_object ?g) (in ?m) )
            )
          )
        )
        (preference preference4
          (exists (?x - chair ?d - doggie_bed)
            (then
              (once (in_motion ?d ?g) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 2 (count-unique-positions preference4:cube_block:purple:golfball) )
)
(:scoring
  (count preference3:dodgeball)
)
)


(define (game game-id-149) (:domain many-objects-room-v1)
(:setup
  (exists (?z - color ?k - doggie_bed)
    (and
      (and
        (game-conserved
          (on ?k ?k)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?e - hexagonal_bin ?r - cube_block)
        (at-end
          (or
            (and
              (and
                (agent_holds ?r ?r)
                (on ?r)
              )
              (agent_holds desk)
              (on ?r ?r)
            )
            (exists (?o - wall)
              (in bed)
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (+ (* (count preference1:dodgeball) 4 )
      (count-once-per-objects preference1:basketball)
    )
    (or
      (+ (count preference1:yellow:dodgeball) (* (- 2 )
          (+ 3 (external-forall-maximize 2 ) (+ 3 (* (= (count preference1:hexagonal_bin) (count preference1:dodgeball) )
                3
                10
              )
            )
          )
        )
      )
      (count preference1:golfball)
      (total-score)
    )
  )
)
(:scoring
  5
)
)


(define (game game-id-150) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (not
      (in_motion ?xxx)
    )
  )
)
(:constraints
  (and
    (forall (?s - cube_block)
      (and
        (preference preference1
          (exists (?a - beachball ?e - hexagonal_bin)
            (then
              (hold (agent_holds ?e rug) )
              (once (not (in_motion ?e ?s) ) )
              (once (on ?s pink_dodgeball) )
            )
          )
        )
        (preference preference2
          (exists (?p - wall)
            (then
              (hold (agent_holds ) )
              (once (exists (?o - teddy_bear ?l - hexagonal_bin ?i - hexagonal_bin) (agent_holds ?p ?i) ) )
              (once (agent_holds ?s ?p) )
            )
          )
        )
      )
    )
    (preference preference3
      (exists (?x - hexagonal_bin)
        (then
          (once (not (not (and (agent_holds north_wall) (forall (?r - hexagonal_bin ?e - triangular_ramp ?b - (either book desktop) ?y - golfball) (on ?x agent) ) ) ) ) )
          (hold (and (adjacent ?x) (not (and (agent_holds side_table) (agent_holds ?x ?x) ) ) (not (and (not (in_motion ?x ?x) ) (and (and (not (> (distance_side ?x ?x) 0) ) (agent_holds agent) ) (in ?x) ) (not (adjacent ?x) ) ) ) ) )
          (once (and (on ?x) (on ?x ?x) (not (and (in_motion floor ?x) (not (agent_holds ?x) ) ) ) ) )
        )
      )
    )
  )
)
(:terminal
  (> 4 (count preference3:hexagonal_bin:basketball) )
)
(:scoring
  (* (count preference2:yellow) (count preference3:yellow_cube_block) (external-forall-maximize 1 ) (* 4 (total-score) )
  )
)
)


(define (game game-id-151) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (agent_holds ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?h - doggie_bed ?f - game_object)
        (then
          (hold-to-end (in ?f) )
          (once (agent_holds ?f ?f) )
          (hold (not (and (and (on block) (adjacent_side ?f ?f) ) (and (not (and (not (in_motion ?f) ) (not (not (agent_holds ?f floor) ) ) ) ) (and (agent_holds ?f ?f ?f) (agent_holds ?f ?f) ) ) ) ) )
        )
      )
    )
    (forall (?k - dodgeball)
      (and
        (preference preference2
          (exists (?v - building ?e - wall)
            (then
              (once (not (not (agent_holds ?k) ) ) )
              (hold-to-end (touch bed ?k) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (external-forall-maximize (* 6 (+ 1 (count preference1:alarm_clock) )
      )
    )
    (* 4 (count preference1:dodgeball) (count preference2:orange:beachball) (count-once-per-objects preference1:basketball) 10 )
  )
)
(:scoring
  60
)
)


(define (game game-id-152) (:domain few-objects-room-v1)
(:setup
  (exists (?x - cube_block ?h - hexagonal_bin ?e - hexagonal_bin ?f - doggie_bed)
    (game-conserved
      (< 7 2)
    )
  )
)
(:constraints
  (and
    (forall (?o - hexagonal_bin ?k - hexagonal_bin)
      (and
        (preference preference1
          (exists (?t - cube_block ?i - hexagonal_bin)
            (then
              (once (on ?i) )
              (once (and (< 6 (distance_side ?k ?i 9)) (agent_holds bed ?i) ) )
              (once (object_orientation rug) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference1:pink_dodgeball:yellow) (* 2 (count-once-per-objects preference1:blue_cube_block) )
  )
)
(:scoring
  (* (count preference1:basketball) (count preference1:pink:bed) )
)
)


(define (game game-id-153) (:domain medium-objects-room-v1)
(:setup
  (and
    (game-optional
      (and
        (in ?xxx ?xxx)
      )
    )
  )
)
(:constraints
  (and
    (forall (?r - color)
      (and
        (preference preference1
          (exists (?p - dodgeball)
            (then
              (hold-while (agent_holds ?r) (in floor ?p) )
              (once (agent_holds ?p ?r) )
              (once (on ?r) )
            )
          )
        )
        (preference preference2
          (exists (?i - hexagonal_bin)
            (then
              (hold (on ?i) )
              (once (< (distance ?r 5) 3) )
              (hold (< (distance bed ?i) (distance ?r ?i)) )
            )
          )
        )
      )
    )
    (forall (?i - hexagonal_bin ?t - dodgeball)
      (and
        (preference preference3
          (exists (?h - building)
            (then
              (hold (agent_holds ?t) )
              (once (on ?h ?h) )
              (hold-while (agent_holds ?h ?t) (touch ?h) )
              (any)
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (* (+ (* 10 (= (* (+ (+ (- (count preference3:pyramid_block) 50 ) (* (count-once-per-objects preference1:beachball:blue_dodgeball:golfball) 6 )
                )
                (+ (count-once-per-objects preference1:golfball) 5 )
              )
              (count preference3:beachball:pink)
            )
            (total-time)
          )
          1
          (count-once-per-objects preference3:book:basketball)
          10
          2
        )
        (count preference2:beachball)
      )
      (count-once-per-objects preference2:tall_cylindrical_block:beachball)
    )
    (not
      (* 5 50 )
    )
  )
)
(:scoring
  40
)
)


(define (game game-id-154) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (in_motion ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?m ?o - cube_block ?g - hexagonal_bin)
        (then
          (once (exists (?d - dodgeball) (not (and (and (and (or (on desk) (or (forall (?n - dodgeball ?b - dodgeball ?p ?c - shelf) (in_motion ?g) ) (in_motion ?g) ) ) (agent_holds ?d ?d) ) (in ?g) ) (and (agent_holds ?d ?d) (same_type ?d ?d) (< 1 2) ) ) ) ) )
          (once (> (distance 0 ?g) 0.5) )
          (once (on bed) )
        )
      )
    )
    (preference preference2
      (exists (?n - sliding_door ?k - triangular_ramp)
        (at-end
          (not
            (not
              (not
                (and
                  (and
                    (agent_holds ?k ?k)
                    (> 1 5)
                  )
                  (agent_holds ?k ?k)
                )
              )
            )
          )
        )
      )
    )
    (preference preference3
      (exists (?b - (either wall top_drawer) ?p - dodgeball ?d - yellow_pyramid_block)
        (at-end
          (in ?d ?d)
        )
      )
    )
    (preference preference4
      (exists (?b - dodgeball)
        (then
          (any)
          (once (same_color ?b) )
          (once (and (not (on ?b) ) (on ?b ?b) (adjacent_side ?b ?b) (on desk) ) )
        )
      )
    )
    (preference preference5
      (exists (?c - dodgeball ?w - red_dodgeball ?u - teddy_bear ?b - dodgeball ?u - (either cylindrical_block laptop))
        (then
          (hold-while (in_motion agent) (and (agent_holds ?u ?u) (not (on ?u) ) ) (in ?u) (in rug ?u) )
        )
      )
    )
    (preference preference6
      (exists (?j - triangular_ramp)
        (then
          (once-measure (agent_holds ?j ?j) (distance 0 ?j) )
          (hold (and (not (in ?j ?j) ) (in_motion ?j) ) )
          (once-measure (in_motion ?j ?j) (distance 2 ?j) )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference5:dodgeball:basketball) (count-unique-positions preference4:red:pink:dodgeball) )
)
(:scoring
  (count-once-per-objects preference6:brown)
)
)


(define (game game-id-155) (:domain few-objects-room-v1)
(:setup
  (and
    (game-conserved
      (and
        (and
          (and
            (not
              (and
                (in_motion ?xxx ?xxx)
                (agent_holds ?xxx ?xxx ?xxx)
              )
            )
            (in ?xxx)
            (and
              (touch ?xxx)
              (exists (?b - hexagonal_bin)
                (and
                  (on blue)
                  (in_motion ?b ?b)
                )
              )
            )
          )
          (and
            (and
              (agent_holds ?xxx ?xxx)
              (not
                (same_color ?xxx)
              )
              (exists (?h - shelf ?b - tall_cylindrical_block)
                (and
                  (in_motion ?b)
                  (in_motion ?b ?b)
                )
              )
              (and
                (< (distance ?xxx desk) (distance 4 ?xxx ?xxx))
                (is_setup_object ?xxx ?xxx)
                (< 1 (distance bed agent))
              )
            )
            (agent_holds ?xxx)
          )
          (not
            (on ?xxx ?xxx)
          )
        )
        (adjacent_side agent)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?n - yellow_cube_block)
        (then
          (once (and (not (agent_holds ?n) ) (and (not (not (not (agent_holds ?n agent) ) ) ) (agent_holds ?n) (agent_holds pink_dodgeball) ) ) )
          (once (agent_holds ?n ?n) )
          (once (in_motion ?n) )
        )
      )
    )
  )
)
(:terminal
  (<= (+ (count-same-positions preference1:pink_dodgeball) (* (count-once-per-objects preference1:beachball) (count preference1:yellow) )
    )
    3
  )
)
(:scoring
  (* 3 (count-once-per-objects preference1:pink) )
)
)


(define (game game-id-156) (:domain medium-objects-room-v1)
(:setup
  (forall (?j ?o - hexagonal_bin)
    (game-optional
      (and
        (or
          (and
            (= 1)
            (above agent)
          )
          (in_motion ?j ?j)
        )
        (not
          (on ?j)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?a - hexagonal_bin ?o - (either book cube_block))
      (and
        (preference preference1
          (exists (?d - hexagonal_bin)
            (then
              (hold-while (faces ?d) (not (not (in_motion ?o ?d) ) ) )
              (once (exists (?k - game_object ?b - (either cube_block golfball basketball) ?g - hexagonal_bin) (in_motion ?g) ) )
              (hold (on ?d) )
            )
          )
        )
      )
    )
    (preference preference2
      (then
        (once (agent_holds ?xxx ?xxx) )
        (once (exists (?h - hexagonal_bin ?v - hexagonal_bin) (not (exists (?k - block) (agent_holds ?k) ) ) ) )
        (hold (touch rug ?xxx) )
      )
    )
    (preference preference3
      (exists (?v ?p - building)
        (then
          (hold-while (not (adjacent_side agent) ) (and (not (not (not (or (not (< (distance ?v agent) 1) ) (< (distance ) 6) ) ) ) ) (not (agent_holds ?v) ) ) (not (not (not (in_motion ?p) ) ) ) (on ?v ?p) )
          (once (and (agent_holds desk) (not (not (or (in ?p ?p) (adjacent_side ?p) ) ) ) ) )
          (once (on ?p ?p) )
        )
      )
    )
  )
)
(:terminal
  (>= 1 (+ 8 (>= (count preference1:hexagonal_bin) (external-forall-maximize 5 ) )
    )
  )
)
(:scoring
  (- 5 )
)
)


(define (game game-id-157) (:domain medium-objects-room-v1)
(:setup
  (and
    (exists (?p - red_dodgeball ?v - game_object)
      (and
        (forall (?j - game_object)
          (and
            (forall (?d - triangular_ramp)
              (and
                (exists (?i - (either golfball golfball))
                  (game-conserved
                    (agent_holds door)
                  )
                )
              )
            )
          )
        )
        (and
          (game-conserved
            (in_motion agent ?v)
          )
          (and
            (game-conserved
              (in_motion ?v bed)
            )
          )
          (and
            (and
              (exists (?w - ball ?o - hexagonal_bin ?e ?s - hexagonal_bin)
                (exists (?d - hexagonal_bin)
                  (game-optional
                    (not
                      (in_motion ?d ?v)
                    )
                  )
                )
              )
              (and
                (exists (?l - drawer)
                  (game-optional
                    (above ?v)
                  )
                )
              )
            )
          )
        )
        (exists (?c - ball ?r ?h ?z ?a ?q ?c - hexagonal_bin)
          (exists (?t - triangular_ramp)
            (and
              (game-conserved
                (agent_holds ?t ?v)
              )
              (game-conserved
                (in_motion ?z ?a)
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
    (preference preference1
      (exists (?v - (either dodgeball blue_cube_block))
        (then
          (once (not (agent_holds ?v) ) )
          (once (touch ?v) )
          (hold (in ?v) )
        )
      )
    )
    (preference preference2
      (exists (?l - (either golfball cube_block dodgeball) ?i - hexagonal_bin)
        (then
          (once (and (adjacent ?i ?i) (on ?i) (not (not (not (agent_holds ?i) ) ) ) ) )
          (once (or (agent_holds bed) (not (not (not (not (same_color ?i) ) ) ) ) ) )
          (hold (not (agent_holds ?i ?i) ) )
        )
      )
    )
    (preference preference3
      (at-end
        (touch ?xxx ?xxx)
      )
    )
  )
)
(:terminal
  (not
    (not
      (>= (* (count-once-per-objects preference3:yellow) (count-once-per-objects preference2:beachball) (+ 2 (count preference3:basketball:beachball) (+ (external-forall-maximize 5 ) (* (+ (* 2 (* (count preference2:pink_dodgeball) )
                  )
                  (count preference3:blue_cube_block:dodgeball)
                )
                (count preference2:dodgeball:dodgeball)
              )
            )
          )
        )
        2
      )
    )
  )
)
(:scoring
  (count preference3:dodgeball:blue_dodgeball)
)
)


(define (game game-id-158) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (same_color ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?s - dodgeball ?o - hexagonal_bin)
        (then
          (forall-sequence (?w - flat_block)
            (then
              (hold (touch ?w) )
              (hold (in ?o ?o) )
              (hold (not (and (exists (?h - hexagonal_bin) (agent_holds ?o) ) (and (and (and (exists (?e - dodgeball) (agent_holds ?e) ) (on ?o ?o) ) (and (and (not (in_motion ?w ?w) ) (and (in_motion ?w ?o) (and (not (in_motion ?w ?w) ) (and (not (not (not (exists (?p - block ?m ?b - cylindrical_block) (and (on blue ?m) (touch agent) ) ) ) ) ) (< (distance ?o desk) (distance room_center room_center)) ) ) ) ) (on ?o) ) (agent_holds ?o) ) ) ) ) )
            )
          )
          (hold (touch ?o) )
          (once (on ?o) )
        )
      )
    )
    (preference preference2
      (exists (?m - cube_block)
        (then
          (once (not (in_motion ?m back) ) )
          (once (same_type agent ?m) )
          (hold-while (in_motion desk) (and (exists (?r - wall) (faces ?r) ) (exists (?d - game_object) (and (and (not (and (in ?d) (in_motion ?m) ) ) (agent_holds ?m ?d) ) (and (not (not (agent_holds ?d) ) ) (agent_holds ?m) ) (not (in ?d) ) (and (not (forall (?e - color) (agent_holds ?e) ) ) (agent_holds ?d) (same_color ?d) ) ) ) ) (and (is_setup_object bed ?m) (and (and (in ?m ?m) (not (in_motion agent) ) ) (not (in_motion agent side_table) ) (not (in ?m) ) ) ) )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= 2 (count preference2:dodgeball) )
    (= (* (- 5 )
        (count-once-per-objects preference1:pink)
      )
      6
    )
  )
)
(:scoring
  (count preference1:dodgeball:doggie_bed)
)
)


(define (game game-id-159) (:domain many-objects-room-v1)
(:setup
  (and
    (exists (?n - ball)
      (forall (?o - building)
        (game-optional
          (agent_holds ?n)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?t - wall)
      (and
        (preference preference1
          (exists (?h - cube_block)
            (at-end
              (agent_holds ?h ?t)
            )
          )
        )
        (preference preference2
          (exists (?w - hexagonal_bin)
            (then
              (hold (adjacent ?t) )
              (once (exists (?a - dodgeball) (in_motion ?t ?a) ) )
              (hold (and (and (not (in_motion ?t) ) (and (agent_holds ?w) (not (in_motion ?w) ) ) ) ) )
            )
          )
        )
        (preference preference3
          (exists (?h - curved_wooden_ramp ?n - (either tall_cylindrical_block pyramid_block))
            (at-end
              (in_motion agent ?t)
            )
          )
        )
      )
    )
    (preference preference4
      (exists (?r - game_object ?j - dodgeball)
        (then
          (once (not (adjacent ?j) ) )
          (hold (on ?j ?j) )
          (hold (not (agent_holds top_shelf ?j) ) )
          (once (agent_holds ?j ?j) )
        )
      )
    )
    (forall (?a ?x - (either laptop desktop))
      (and
        (preference preference5
          (exists (?k - ball ?v - dodgeball)
            (then
              (once (agent_holds ?a ?v) )
              (once (or (not (and (and (agent_holds ?x) (agent_holds ?a) ) (in_motion ?v front) ) ) (in_motion ?x ?v) ) )
              (once (on ?x) )
            )
          )
        )
      )
    )
    (preference preference6
      (exists (?l - game_object)
        (at-end
          (not
            (same_color ?l)
          )
        )
      )
    )
    (preference preference7
      (exists (?b - teddy_bear)
        (then
          (hold (agent_holds ?b rug) )
          (hold (and (in_motion ?b) (on color) (agent_holds ?b) (and (and (and (touch ?b ?b) (in_motion ?b) ) (equal_z_position ?b) ) (and (and (in_motion desk ?b) (not (not (touch ?b) ) ) ) (rug_color_under ?b) ) ) ) )
          (once (agent_holds ?b) )
        )
      )
    )
    (preference preference8
      (exists (?p - dodgeball)
        (then
          (hold (not (agent_holds ?p ?p) ) )
          (once (and (in_motion desk) (and (on agent ?p) (= (distance ?p ?p) (distance ?p 8)) ) ) )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (* (- 2 )
        3
      )
      5
    )
    (>= (+ (count-once-per-objects preference6:blue_cube_block:beachball) (count-once-per-objects preference2:dodgeball:blue_dodgeball) )
      1
    )
  )
)
(:scoring
  (* (count preference2:dodgeball:top_drawer) 10 (count-total preference2:side_table) )
)
)


(define (game game-id-160) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (agent_holds ?xxx ?xxx)
  )
)
(:constraints
  (and
    (forall (?h - desktop)
      (and
        (preference preference1
          (exists (?o - curved_wooden_ramp)
            (at-end
              (in_motion ?h)
            )
          )
        )
      )
    )
    (preference preference2
      (exists (?p ?h ?o - doggie_bed)
        (then
          (hold (and (agent_holds ?o ?p) (in_motion front ?o) ) )
          (hold (in ?h ?o) )
          (once (agent_holds ?p ?h) )
          (once (and (in_motion ?p upright) (adjacent ?p) ) )
        )
      )
    )
  )
)
(:terminal
  (>= (+ (count preference2:dodgeball:dodgeball) (count-once-per-objects preference2:blue_dodgeball) )
    (* (* (count preference1:pink:pink_dodgeball) 2 )
      (count-once-per-objects preference1:dodgeball)
    )
  )
)
(:scoring
  (count-once-per-objects preference2:dodgeball)
)
)


(define (game game-id-161) (:domain few-objects-room-v1)
(:setup
  (forall (?r - (either cd pyramid_block))
    (game-conserved
      (on ?r ?r)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (at-end
        (not
          (agent_holds ?xxx)
        )
      )
    )
    (preference preference2
      (exists (?f - cube_block)
        (then
          (hold-for 2 (and (agent_holds rug) (not (in ?f) ) ) )
          (once (and (and (rug_color_under ?f) (not (in_motion ?f ?f) ) ) ) )
          (hold (< (distance desk ?f) (distance agent ?f)) )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference2:dodgeball:pink) 1 )
)
(:scoring
  (count preference1:bed)
)
)


(define (game game-id-162) (:domain few-objects-room-v1)
(:setup
  (and
    (exists (?z - bridge_block)
      (forall (?w - rug ?k - building)
        (game-optional
          (agent_holds rug)
        )
      )
    )
    (and
      (exists (?m - book)
        (and
          (and
            (game-conserved
              (and
                (agent_holds ?m)
                (in_motion ?m)
              )
            )
          )
        )
      )
      (and
        (and
          (game-conserved
            (and
              (and
                (not
                  (agent_holds ?xxx)
                )
                (agent_holds ?xxx)
              )
              (in ?xxx)
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?i - cube_block ?o ?d - block)
      (and
        (preference preference1
          (exists (?h - dodgeball)
            (then
              (hold (and (agent_holds ?h ?d) ) )
              (hold (not (agent_holds ?h) ) )
              (once (in agent) )
            )
          )
        )
        (preference preference2
          (exists (?p - cube_block)
            (then
              (once (not (< (distance room_center ?d) (distance desk ?p)) ) )
              (once (agent_holds ?p ?p) )
              (once (touch ?d rug) )
            )
          )
        )
      )
    )
    (forall (?f ?z ?n - (either cube_block basketball))
      (and
        (preference preference3
          (exists (?w - game_object ?l - ball ?p ?g ?k - triangular_ramp)
            (then
              (once (and (and (and (in floor) (and (adjacent ?n ?z) (in ?p) (not (and (and (and (same_color ?z) (and (not (on ?f) ) (agent_holds ?p) ) (and (not (and (and (in_motion ?k) (and (in_motion ?p) (same_color ?k ?g) ) ) (and (and (adjacent rug top_shelf) (agent_holds ?k) ) (agent_holds ?p) (in_motion ?f) ) ) ) (and (> (distance ?z ?k) (distance desk ?p)) (= (distance ?k 9) 1) (same_object agent) ) ) (in_motion agent) (in ?p agent) (exists (?s - dodgeball) (adjacent ?z) ) ) (on ?f) ) (and (and (in_motion door) (and (in_motion ?f) (and (forall (?i - (either basketball cd) ?j - teddy_bear) (agent_holds ?z) ) (not (in_motion ?g) ) ) ) ) (in ?k top_drawer) ) ) ) (object_orientation ?n ?g) ) ) (in_motion ?p ?z) (not (on ?z ?p) ) ) (between ?p) ) )
              (once (not (and (not (= 0 0 (distance back 8)) ) (not (on ?p ?n) ) ) ) )
              (once (not (not (adjacent_side pink_dodgeball) ) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (not
    (<= 4 2 )
  )
)
(:scoring
  5
)
)


(define (game game-id-163) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (and
      (and
        (agent_holds ?xxx ?xxx)
        (not
          (in_motion agent)
        )
      )
      (in_motion agent)
    )
  )
)
(:constraints
  (and
    (forall (?s ?z - teddy_bear ?d - wall)
      (and
        (preference preference1
          (exists (?n - teddy_bear)
            (then
              (once (agent_holds ?n) )
              (once (and (and (equal_z_position ?d) (< 4 (distance room_center ?n ?n)) ) (and (not (in_motion ?n) ) (and (in_motion ?d ?d) (in ?n) ) ) ) )
              (once-measure (agent_holds ?d) (distance ?n ?n) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 2 4 )
)
(:scoring
  (count preference1:pink_dodgeball)
)
)


(define (game game-id-164) (:domain many-objects-room-v1)
(:setup
  (game-optional
    (in_motion ?xxx)
  )
)
(:constraints
  (and
    (forall (?t - hexagonal_bin)
      (and
        (preference preference1
          (exists (?o - dodgeball ?y - shelf)
            (then
              (hold (on ?y) )
              (hold-while (and (not (in ?t rug) ) (faces ?y ?t) ) (and (agent_holds ?t ?t) (not (and (and (in ?y) (agent_crouches ?y ?y) (and (on ?t) (in ?t) (adjacent ?y) (not (in_motion ?t ball) ) ) ) (adjacent ?t ?t) ) ) ) (on ?y ?y) (in_motion ?y) )
              (once (on ?y ?y) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference1:beachball:dodgeball) (* (count preference1) (= (count preference1) (count preference1:green) 6 )
      (* (count-once preference1:basketball:dodgeball) (count-total preference1:beachball) )
      (count preference1:book:red_pyramid_block)
      (count preference1:blue_pyramid_block)
      (* 0 (count-once-per-objects preference1:pink_dodgeball) )
    )
  )
)
(:scoring
  (count-once-per-objects preference1:blue_dodgeball)
)
)


(define (game game-id-165) (:domain few-objects-room-v1)
(:setup
  (forall (?m - building ?v - cube_block)
    (forall (?d - (either dodgeball cd) ?i - building)
      (and
        (game-conserved
          (and
            (on ?i)
            (agent_holds ?v)
            (between bottom_shelf south_west_corner)
            (in_motion ?v)
          )
        )
        (game-optional
          (and
            (and
              (in ?v ?v)
            )
            (rug_color_under ?i)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?a ?t - wall)
      (and
        (preference preference1
          (exists (?z - building)
            (then
              (hold-while (not (agent_holds agent) ) (is_setup_object desk ?a) )
              (once (toggled_on agent) )
              (once (< 8 (distance 5 ?t)) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 1 30 )
)
(:scoring
  (count-once-per-objects preference1:pyramid_block)
)
)


(define (game game-id-166) (:domain medium-objects-room-v1)
(:setup
  (and
    (game-conserved
      (in_motion ?xxx west_wall)
    )
  )
)
(:constraints
  (and
    (forall (?t - block ?u - beachball ?z - beachball ?s - wall)
      (and
        (preference preference1
          (exists (?p - (either dodgeball side_table cellphone) ?p - building)
            (at-end
              (touch ?p ?s)
            )
          )
        )
        (preference preference2
          (exists (?t - chair ?i ?y - building)
            (then
              (once (is_setup_object ?y ?i) )
              (once (in_motion ?i) )
              (once (and (agent_holds floor) (in_motion ?y) (and (and (in_motion ?y) (and (touch ?s) (agent_holds ?y) ) ) (and (agent_holds ?s ?i) (< 1 8) ) ) (in ?y) ) )
            )
          )
        )
        (preference preference3
          (exists (?x - hexagonal_bin)
            (then
              (once (and (< (distance green_golfball) 1) (in_motion ?s) (in_motion floor) ) )
              (once (same_type desk ?s front ?x) )
              (once (in_motion ?x ?s) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference1:dodgeball) 2 )
)
(:scoring
  3
)
)


(define (game game-id-167) (:domain many-objects-room-v1)
(:setup
  (exists (?q ?k - hexagonal_bin)
    (and
      (game-conserved
        (agent_holds ?k)
      )
      (and
        (exists (?v - dodgeball)
          (exists (?w - (either dodgeball lamp))
            (and
              (game-optional
                (agent_holds ?k ?v)
              )
              (and
                (and
                  (game-optional
                    (in_motion ?v agent)
                  )
                  (game-optional
                    (in ?q)
                  )
                )
              )
              (exists (?n - blue_cube_block ?t ?y ?c ?p - (either lamp hexagonal_bin doggie_bed cd) ?x - (either cd cd))
                (and
                  (and
                    (forall (?b - doggie_bed)
                      (and
                        (and
                          (exists (?n - shelf ?n ?j - doggie_bed ?p - building)
                            (and
                              (exists (?g - (either golfball cylindrical_block) ?r - dodgeball)
                                (or
                                  (game-conserved
                                    (not
                                      (agent_holds ?k ?p)
                                    )
                                  )
                                  (exists (?y - game_object ?l - hexagonal_bin)
                                    (exists (?t - hexagonal_bin ?a - building)
                                      (game-optional
                                        (and
                                          (and
                                            (not
                                              (agent_holds agent ?l)
                                            )
                                            (on west_wall)
                                          )
                                          (and
                                            (not
                                              (rug_color_under ?a ?p)
                                            )
                                            (agent_holds ?q)
                                          )
                                          (on agent)
                                          (on ?p ?a)
                                        )
                                      )
                                    )
                                  )
                                )
                              )
                              (game-optional
                                (and
                                  (agent_holds ?q ?w)
                                  (= 1 (distance ?b room_center))
                                )
                              )
                              (or
                                (and
                                  (and
                                    (forall (?n - ball ?m - ball)
                                      (exists (?c - dodgeball ?s ?j - hexagonal_bin ?g - game_object ?t - dodgeball ?r - ball)
                                        (and
                                          (game-conserved
                                            (agent_holds ?b)
                                          )
                                          (forall (?s - hexagonal_bin)
                                            (and
                                              (and
                                                (game-optional
                                                  (game_over blue)
                                                )
                                                (exists (?d - doggie_bed)
                                                  (game-conserved
                                                    (touch ?s ?q)
                                                  )
                                                )
                                              )
                                            )
                                          )
                                          (exists (?z - game_object)
                                            (and
                                              (game-conserved
                                                (agent_holds agent ?x)
                                              )
                                              (game-conserved
                                                (not
                                                  (in ?m)
                                                )
                                              )
                                            )
                                          )
                                        )
                                      )
                                    )
                                  )
                                  (game-conserved
                                    (agent_holds ?w)
                                  )
                                )
                                (exists (?u - blinds)
                                  (game-conserved
                                    (in_motion ?q)
                                  )
                                )
                              )
                            )
                          )
                        )
                        (exists (?z - game_object ?t - (either dodgeball yellow key_chain))
                          (forall (?j - hexagonal_bin)
                            (game-conserved
                              (not
                                (not
                                  (and
                                    (not
                                      (in_motion ?k)
                                    )
                                    (on ?x)
                                  )
                                )
                              )
                            )
                          )
                        )
                      )
                    )
                  )
                  (game-optional
                    (on ?x)
                  )
                )
              )
            )
          )
        )
        (game-conserved
          (<= (distance door ?q) 1)
        )
        (game-conserved
          (not
            (in_motion ?k ?q)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?r - dodgeball)
        (at-end
          (in_motion agent)
        )
      )
    )
  )
)
(:terminal
  (>= (count-shortest preference1:dodgeball) 5 )
)
(:scoring
  (count preference1:basketball)
)
)


(define (game game-id-168) (:domain many-objects-room-v1)
(:setup
  (game-optional
    (agent_holds ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?g - ball)
        (then
          (once (and (not (adjacent ?g ?g) ) (in_motion ?g) ) )
          (once (in agent) )
          (hold-while (in top_drawer ?g) (and (in ?g ?g) (agent_holds brown bridge_block) ) (not (in ?g) ) (and (not (adjacent ?g ?g) ) (not (< 1 1) ) ) )
        )
      )
    )
    (preference preference2
      (exists (?d - dodgeball ?v - doggie_bed ?p ?d - ball)
        (then
          (hold (in_motion desk) )
          (once (and (in_motion ?p) (= 1 1) ) )
          (hold-while (agent_holds ?d ?p) (agent_holds ?d) )
          (once (on ?d ?d) )
        )
      )
    )
  )
)
(:terminal
  (>= (- (count preference2:book) )
    (count-once-per-objects preference2:pink)
  )
)
(:scoring
  (external-forall-maximize
    3
  )
)
)


(define (game game-id-169) (:domain medium-objects-room-v1)
(:setup
  (exists (?o - hexagonal_bin ?d - hexagonal_bin)
    (exists (?g - building ?z - hexagonal_bin)
      (game-conserved
        (not
          (agent_holds ?d ?z)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?x - hexagonal_bin)
        (then
          (once (agent_holds ?x) )
          (once (not (on ?x) ) )
          (once (in_motion ?x ?x) )
        )
      )
    )
    (preference preference2
      (exists (?c - teddy_bear)
        (then
          (hold (and (and (faces ?c agent) (same_color ?c ?c) ) (not (agent_holds ?c) ) ) )
          (hold (touch floor) )
          (hold (adjacent ?c) )
        )
      )
    )
  )
)
(:terminal
  (or
    (> 5 3 )
    (>= (count preference2:blue_cube_block) 2 )
  )
)
(:scoring
  5
)
)


(define (game game-id-170) (:domain medium-objects-room-v1)
(:setup
  (and
    (and
      (and
        (forall (?a - ball)
          (or
            (game-conserved
              (on ?a)
            )
          )
        )
      )
      (game-conserved
        (and
          (and
            (agent_holds ?xxx)
            (and
              (not
                (same_color ?xxx ?xxx)
              )
              (and
                (not
                  (not
                    (and
                      (and
                        (not
                          (adjacent ?xxx ?xxx)
                        )
                        (adjacent_side ?xxx ?xxx)
                        (< (distance_side ?xxx ?xxx) 1)
                      )
                      (and
                        (> (distance ?xxx ?xxx) 9)
                        (in_motion ?xxx)
                      )
                    )
                  )
                )
                (and
                  (and
                    (not
                      (and
                        (= (distance ?xxx ?xxx) (distance bed desk))
                        (not
                          (in_motion ?xxx ?xxx)
                        )
                      )
                    )
                  )
                  (and
                    (in_motion ?xxx north_wall)
                    (and
                      (agent_holds ?xxx ?xxx)
                      (not
                        (and
                          (and
                            (in_motion ?xxx)
                            (exists (?m - hexagonal_bin ?h - desk_shelf)
                              (in_motion ?h)
                            )
                            (and
                              (agent_holds ?xxx)
                              (not
                                (not
                                  (or
                                    (in_motion side_table ?xxx)
                                    (agent_holds desk)
                                    (not
                                      (in_motion ?xxx)
                                    )
                                    (and
                                      (agent_holds ?xxx pink)
                                      (not
                                        (in_motion agent)
                                      )
                                    )
                                  )
                                )
                              )
                            )
                          )
                          (in_motion ?xxx)
                          (opposite bridge_block)
                        )
                      )
                      (in_motion ?xxx)
                      (adjacent ?xxx)
                    )
                  )
                )
                (rug_color_under ?xxx)
              )
            )
          )
        )
      )
      (and
        (game-conserved
          (in_motion ?xxx)
        )
      )
      (and
        (game-optional
          (not
            (agent_holds pink)
          )
        )
        (forall (?q - (either alarm_clock key_chain golfball))
          (game-conserved
            (not
              (and
                (and
                  (in ?q pink_dodgeball)
                  (agent_holds bed pink)
                )
                (exists (?i - dodgeball)
                  (and
                    (agent_holds ?i ?i)
                    (in_motion agent)
                  )
                )
              )
            )
          )
        )
      )
      (and
        (forall (?q - doggie_bed)
          (game-optional
            (adjacent ?q)
          )
        )
        (and
          (exists (?l - (either dodgeball laptop mug))
            (and
              (and
                (exists (?f ?j - wall ?o - hexagonal_bin)
                  (or
                    (forall (?t - cylindrical_block ?r - doggie_bed ?f - tall_cylindrical_block ?u - dodgeball ?g - hexagonal_bin)
                      (game-conserved
                        (on ?l ?g)
                      )
                    )
                  )
                )
              )
              (game-conserved
                (in ?l)
              )
              (and
                (exists (?m - ball)
                  (game-conserved
                    (agent_holds ?l)
                  )
                )
                (exists (?a - flat_block)
                  (and
                    (game-optional
                      (object_orientation ?a ?l)
                    )
                    (and
                      (exists (?p - teddy_bear)
                        (forall (?c - block)
                          (game-optional
                            (in ?l)
                          )
                        )
                      )
                      (game-conserved
                        (forall (?d - ball)
                          (agent_holds ?d)
                        )
                      )
                      (not
                        (or
                          (exists (?p - hexagonal_bin)
                            (game-conserved
                              (in_motion ?p)
                            )
                          )
                        )
                      )
                      (forall (?t - wall)
                        (and
                          (game-conserved
                            (= (distance room_center ?t) 10)
                          )
                          (game-conserved
                            (on ?t)
                          )
                        )
                      )
                      (game-conserved
                        (adjacent_side ?a ?a)
                      )
                    )
                    (and
                      (and
                        (exists (?r - cube_block)
                          (forall (?f ?p - (either hexagonal_bin bridge_block))
                            (exists (?s - hexagonal_bin ?b - (either golfball))
                              (exists (?h ?j - color)
                                (and
                                  (exists (?y - wall ?t - dodgeball)
                                    (and
                                      (game-optional
                                        (and
                                          (not
                                            (and
                                              (is_setup_object floor)
                                              (and
                                                (exists (?k - cube_block)
                                                  (and
                                                    (not
                                                      (and
                                                        (and
                                                          (in_motion ?j)
                                                          (in ?f left)
                                                          (and
                                                            (in_motion ?j ?r)
                                                            (not
                                                              (in_motion ?j ?f)
                                                            )
                                                          )
                                                        )
                                                        (not
                                                          (on ?p ?t)
                                                        )
                                                      )
                                                    )
                                                    (and
                                                      (agent_holds ?b)
                                                      (not
                                                        (on rug ?l)
                                                      )
                                                    )
                                                  )
                                                )
                                                (not
                                                  (in_motion upright ?a)
                                                )
                                                (not
                                                  (in_motion ?b ?t)
                                                )
                                              )
                                            )
                                          )
                                          (and
                                            (not
                                              (touch ?h)
                                            )
                                            (not
                                              (agent_holds ?p)
                                            )
                                          )
                                        )
                                      )
                                    )
                                  )
                                  (game-conserved
                                    (agent_holds ?l ?p)
                                  )
                                  (game-conserved
                                    (same_object desk)
                                  )
                                )
                              )
                            )
                          )
                        )
                        (game-conserved
                          (and
                            (and
                              (and
                                (in ?l ?l)
                                (rug_color_under ?l)
                              )
                              (on ?a)
                            )
                            (agent_holds ?a)
                          )
                        )
                        (exists (?v - (either yellow_cube_block cube_block) ?f - block ?w - ball ?f - wall)
                          (and
                            (game-conserved
                              (in_motion block)
                            )
                            (game-conserved
                              (on ?f ?l)
                            )
                          )
                        )
                      )
                      (game-conserved
                        (in_motion ?a)
                      )
                      (and
                        (game-conserved
                          (and
                            (object_orientation floor ?l)
                            (not
                              (adjacent ?l ?a)
                            )
                          )
                        )
                      )
                    )
                  )
                )
                (game-conserved
                  (agent_holds ?l rug)
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
    (preference preference1
      (exists (?i - teddy_bear)
        (then
          (once (object_orientation ?i) )
          (hold (and (in ?i) (between ?i ?i) ) )
          (hold (game_over ?i) )
        )
      )
    )
    (preference preference2
      (exists (?v - dodgeball)
        (then
          (hold (on ?v ?v) )
          (once (on ) )
          (once (exists (?c - hexagonal_bin) (not (touch ?c) ) ) )
        )
      )
    )
  )
)
(:terminal
  (= (* (count-once preference2:book) 3 )
    (-
      9
      (count preference1:basketball:pink)
    )
  )
)
(:scoring
  3
)
)


(define (game game-id-171) (:domain many-objects-room-v1)
(:setup
  (forall (?h - hexagonal_bin ?b - chair)
    (game-conserved
      (exists (?d - hexagonal_bin)
        (in ?d)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?z - tall_cylindrical_block)
        (then
          (hold (agent_holds ?z) )
          (hold (not (agent_holds ?z bed) ) )
          (hold-while (and (and (and (not (and (and (agent_holds ?z ?z) (on ?z) ) (on ?z) ) ) (in_motion ?z) ) (object_orientation ?z) ) (agent_holds ?z) (not (in_motion ?z ?z) ) ) (agent_holds ?z ?z) )
          (hold (< 1 (distance ?z ?z)) )
        )
      )
    )
    (preference preference2
      (exists (?f - (either pink doggie_bed) ?t - dodgeball)
        (then
          (hold (< (distance ?t 9) (distance ?t agent agent)) )
          (once (and (forall (?x ?m ?r ?e - game_object) (agent_holds agent) ) (exists (?r - pillow) (in_motion ?r) ) ) )
          (once (not (not (on bed) ) ) )
        )
      )
    )
  )
)
(:terminal
  (or
    (or
      (>= (count-once-per-external-objects preference1:pink:yellow) (>= (count preference1:doggie_bed:dodgeball) 3 )
      )
      (>= (= 1 (count-once-per-objects preference1) )
        (count-once preference2:beachball)
      )
    )
    (>= (count-once-per-objects preference2:dodgeball:book) (* 1 (* 3 (* 5 (count preference1:cylindrical_block:dodgeball) )
        )
      )
    )
  )
)
(:scoring
  (+ (not (count-once-per-objects preference2:hexagonal_bin) ) (count preference1:hexagonal_bin:dodgeball:basketball) (count preference1:pink) (count-once-per-objects preference2:golfball) 2 3 )
)
)


(define (game game-id-172) (:domain many-objects-room-v1)
(:setup
  (and
    (and
      (game-optional
        (and
          (agent_holds front)
          (agent_holds agent ?xxx)
        )
      )
    )
    (and
      (forall (?c - hexagonal_bin ?w - block)
        (and
          (exists (?l - doggie_bed ?d - pillow ?e - block ?i - doggie_bed)
            (game-conserved
              (and
                (not
                  (on ?i)
                )
                (in_motion ?i)
              )
            )
          )
          (game-conserved
            (not
              (in_motion ?w ?w)
            )
          )
          (game-conserved
            (agent_holds ?w ?w ?w ?w)
          )
        )
      )
      (and
        (game-conserved
          (agent_holds ?xxx ?xxx)
        )
      )
      (and
        (forall (?u - pillow ?y - block)
          (game-conserved
            (agent_holds ?y ?y)
          )
        )
        (not
          (game-conserved
            (and
              (agent_holds pink_dodgeball)
              (and
                (in_motion ?xxx)
                (in_motion ?xxx ?xxx)
              )
            )
          )
        )
      )
    )
    (exists (?d - chair ?k - block)
      (exists (?a - (either doggie_bed alarm_clock laptop))
        (forall (?y - hexagonal_bin ?o - block)
          (and
            (forall (?c - doggie_bed)
              (game-conserved
                (and
                  (not
                    (and
                      (not
                        (< (distance ?o ?c) 0)
                      )
                      (in_motion bed)
                    )
                  )
                  (in_motion ?k)
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
    (preference preference1
      (exists (?o - dodgeball)
        (then
          (hold (and (in_motion ?o ?o) (in_motion ?o) ) )
          (once (not (not (touch ?o) ) ) )
          (hold-while (not (not (and (not (and (in_motion ?o) (in ?o) ) ) (adjacent ?o ?o) ) ) ) (same_color agent) )
        )
      )
    )
  )
)
(:terminal
  (>= 4 (+ (count preference1:book) (count-once preference1:doggie_bed) )
  )
)
(:scoring
  (count preference1:blue_cube_block:yellow)
)
)


(define (game game-id-173) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (and
      (in_motion brown ?xxx)
      (and
        (in_motion ?xxx ?xxx)
        (adjacent ?xxx ?xxx)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?k - (either laptop rug blue_cube_block) ?v - hexagonal_bin)
        (then
          (once (agent_holds ?v pink_dodgeball) )
          (hold (in_motion ?v) )
        )
      )
    )
    (preference preference2
      (exists (?g - red_dodgeball)
        (then
          (hold (not (< (distance front ?g) 1) ) )
          (hold (and (not (not (not (and (adjacent_side ?g agent) (= 2 (distance 4 ?g)) ) ) ) ) (exists (?x - hexagonal_bin ?c - wall) (agent_holds ?c) ) ) )
          (hold (in_motion agent) )
        )
      )
    )
    (preference preference3
      (exists (?y - doggie_bed ?a - hexagonal_bin)
        (then
          (once (is_setup_object desk) )
          (once (agent_holds bed) )
          (hold (not (in rug) ) )
        )
      )
    )
  )
)
(:terminal
  (>= (* (count preference1:top_drawer) (total-time) )
    100
  )
)
(:scoring
  2
)
)


(define (game game-id-174) (:domain medium-objects-room-v1)
(:setup
  (exists (?c - doggie_bed)
    (exists (?d - dodgeball)
      (forall (?j - cube_block)
        (and
          (exists (?o - hexagonal_bin)
            (game-conserved
              (on ?j ?d)
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?g - cube_block ?o - (either golfball cylindrical_block))
        (at-end
          (not
            (agent_holds ?o ?o)
          )
        )
      )
    )
    (preference preference2
      (exists (?f - game_object)
        (then
          (once (agent_holds ?f) )
          (once (is_setup_object ?f) )
          (once (and (agent_holds ?f) (in ?f) (in_motion ?f) (= 1 1 7) (is_setup_object bed) (and (and (rug_color_under ?f) (not (in ?f ?f) ) (on ?f) ) (agent_holds bed) (in_motion desk agent) ) (exists (?d - hexagonal_bin) (same_color agent ?f) ) (agent_holds ?f rug) (agent_holds ?f) (in ?f) (not (and (not (in_motion ?f) ) (in ?f ?f) ) ) (< (distance ?f agent) 10) ) )
        )
      )
    )
    (preference preference3
      (exists (?o - dodgeball)
        (then
          (once (and (in_motion ?o desk) (and (not (agent_holds ?o) ) (and (adjacent ?o ?o) (on ?o ?o) (and (in_motion ?o desk) (game_over ?o ?o) ) ) ) (agent_holds ) ) )
          (hold (not (in_motion floor) ) )
          (once (and (on south_west_corner ?o) (or (on ?o ?o) ) ) )
        )
      )
    )
  )
)
(:terminal
  (= 300 (and (count preference1:pink) (* (count preference3:red) 4 (+ 3 (count preference1:yellow) )
      )
      3
    )
  )
)
(:scoring
  5
)
)


(define (game game-id-175) (:domain medium-objects-room-v1)
(:setup
  (and
    (game-conserved
      (and
        (not
          (< (distance ?xxx ?xxx) 0.5)
        )
        (not
          (and
            (adjacent blue)
            (in_motion ?xxx)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?x - building)
        (then
          (once (not (not (and (and (touch ?x) (agent_holds ?x) ) (on ?x) (in_motion bed) (agent_holds ?x ?x) ) ) ) )
          (once (agent_holds ?x) )
          (hold (and (not (agent_holds ?x ?x) ) (forall (?u - doggie_bed ?u - dodgeball) (agent_holds ?x ?x) ) ) )
        )
      )
    )
  )
)
(:terminal
  (>= (- (count preference1:dodgeball:beachball) )
    (count-once-per-objects preference1:orange)
  )
)
(:scoring
  2
)
)


(define (game game-id-176) (:domain medium-objects-room-v1)
(:setup
  (exists (?a ?o - curved_wooden_ramp ?b - (either dodgeball cellphone desktop))
    (and
      (and
        (game-conserved
          (and
            (not
              (agent_holds ?b)
            )
            (on ?b ?b)
          )
        )
      )
      (exists (?l - pillow ?k - (either doggie_bed ball cube_block dodgeball))
        (and
          (exists (?d - ball)
            (and
              (exists (?l - dodgeball)
                (game-conserved
                  (and
                    (and
                      (and
                        (in_motion ?l)
                        (not
                          (or
                            (in ?d)
                            (not
                              (in ?b)
                            )
                          )
                        )
                      )
                      (agent_holds ?l ?k)
                    )
                    (and
                      (touch ?d)
                      (object_orientation ?b ?l)
                    )
                  )
                )
              )
            )
          )
        )
      )
      (exists (?x - building ?t - building ?s - block)
        (game-conserved
          (agent_holds ?s)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?u - red_dodgeball)
        (at-end
          (agent_holds agent)
        )
      )
    )
    (forall (?y - hexagonal_bin)
      (and
        (preference preference2
          (exists (?k - doggie_bed)
            (at-end
              (agent_holds ?k)
            )
          )
        )
        (preference preference3
          (exists (?n - block ?g - red_dodgeball)
            (at-end
              (agent_holds ?g)
            )
          )
        )
      )
    )
    (preference preference4
      (exists (?m - dodgeball)
        (then
          (hold (on ?m ?m) )
          (hold (in_motion ?m) )
          (once (in ?m ?m) )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (count preference2:beachball:side_table) (count-once-per-objects preference1:golfball) )
    (>= (* (count-measure preference2:green:basketball) 40 )
      (count preference2:doggie_bed)
    )
  )
)
(:scoring
  (* (- (* (* 8 15 )
        (count-once preference4:beachball:orange)
      )
    )
  )
)
)


(define (game game-id-177) (:domain few-objects-room-v1)
(:setup
  (game-optional
    (agent_holds rug)
  )
)
(:constraints
  (and
    (forall (?q - chair ?w - doggie_bed)
      (and
        (preference preference1
          (exists (?r - (either dodgeball pillow golfball))
            (at-end
              (not
                (not
                  (and
                    (in_motion rug ?r)
                    (and
                      (agent_holds ?w)
                      (agent_holds ?r)
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
)
(:terminal
  (<= (count-once-per-objects preference1:basketball:pink) (external-forall-maximize (count-once-per-objects preference1:basketball:yellow) ) )
)
(:scoring
  4
)
)


(define (game game-id-178) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (and
      (agent_holds ?xxx ?xxx)
      (not
        (in_motion ?xxx)
      )
      (and
        (and
          (in_motion ?xxx ?xxx)
          (agent_holds ?xxx agent)
        )
        (agent_holds ?xxx)
        (agent_holds ?xxx)
      )
      (not
        (forall (?l - (either cylindrical_block triangle_block) ?d - dodgeball ?i - hexagonal_bin)
          (exists (?p - pillow)
            (= 1 (distance ?i ?p))
          )
        )
      )
      (and
        (agent_holds floor)
        (agent_holds ?xxx)
      )
      (agent_holds ?xxx)
      (agent_holds ?xxx ?xxx)
    )
  )
)
(:constraints
  (and
    (forall (?z - cube_block)
      (and
        (preference preference1
          (exists (?y - shelf ?s - hexagonal_bin)
            (at-end
              (agent_holds ?z)
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (* (total-time) (+ (- (* 2 (count preference1:green:dodgeball) )
        )
        (count preference1:blue_dodgeball:beachball)
      )
      3
      (count preference1)
    )
    (+ (count preference1:yellow) (+ (count preference1:hexagonal_bin:doggie_bed) (* (* 5 30 (* (count preference1:purple:alarm_clock) (count-once-per-objects preference1:red_pyramid_block:blue_pyramid_block) )
          )
          (+ (= (count-once preference1:purple) (or 4 (* (<= (count-once-per-objects preference1:pink:purple) 7 )
                  2
                  (>= (count preference1:basketball) (total-time) )
                  (count-unique-positions preference1:yellow_cube_block)
                  (total-score)
                  (* (* 10 (count preference1:dodgeball) )
                    (count-once-per-objects preference1:top_drawer)
                  )
                  (- (count preference1:orange:pink_dodgeball) )
                  (+ (* (* (count preference1:dodgeball) (* (* (* 20 (+ (* (count preference1:blue_dodgeball) (count preference1:beachball) )
                                (- (count preference1:pink) )
                                (* 7 (= (+ (count preference1:beachball) (* (count preference1) (- (count preference1:doggie_bed:golfball) )
                                      )
                                    )
                                    3
                                    (* (* 2 (* (count preference1:basketball) 2 5 )
                                        (count-once-per-objects preference1:blue_pyramid_block)
                                        (* (+ (<= 0 5 )
                                            (count-once-per-external-objects preference1:yellow)
                                          )
                                          (count preference1:pink_dodgeball)
                                        )
                                        (+ (* (* 3 (count preference1:purple) )
                                            (+ 0 5 )
                                          )
                                          9
                                        )
                                        (* (* 10 100 )
                                          (* (and (* 3 15 )
                                            )
                                            3
                                          )
                                          (+ 1 (or (count preference1:beachball) 300 (count-once-per-objects preference1:dodgeball:pink) ) )
                                        )
                                      )
                                      4
                                    )
                                  )
                                )
                              )
                              3
                            )
                            6
                          )
                          (count preference1:blue_dodgeball)
                        )
                        (count preference1:pink)
                      )
                      (count-once-per-objects preference1:beachball)
                    )
                    5
                  )
                  180
                )
                15
              )
            )
          )
        )
      )
    )
  )
)
(:scoring
  2
)
)


(define (game game-id-179) (:domain medium-objects-room-v1)
(:setup
  (forall (?i - (either alarm_clock))
    (exists (?a - (either main_light_switch blue_cube_block) ?d - dodgeball)
      (and
        (game-conserved
          (not
            (open ?i ?d)
          )
        )
        (game-conserved
          (in ?d)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?k - cube_block ?p - wall)
        (then
          (hold-for 4 (on ?p) )
          (once (and (in_motion ?p ?p) (in_motion ?p ?p) (= 1 1) ) )
          (any)
        )
      )
    )
    (preference preference2
      (exists (?v - beachball)
        (then
          (once (exists (?u - hexagonal_bin) (not (on ?v) ) ) )
          (hold (in_motion ?v) )
          (once (in_motion ?v ?v) )
        )
      )
    )
  )
)
(:terminal
  (>= (count-measure preference1:yellow) (+ (count preference1:pink_dodgeball) (count-measure preference1:wall) )
  )
)
(:scoring
  (or
    (count preference2:pink:dodgeball)
  )
)
)


(define (game game-id-180) (:domain medium-objects-room-v1)
(:setup
  (and
    (exists (?j - shelf ?t - ball)
      (game-optional
        (and
          (and
            (exists (?o - hexagonal_bin)
              (agent_holds ?t upright)
            )
            (adjacent ?t rug)
            (and
              (and
                (< (distance ?t room_center) 6)
                (and
                  (in ?t)
                  (< 1 1)
                )
                (not
                  (and
                    (in_motion ?t)
                    (and
                      (adjacent_side ?t)
                      (in_motion ?t ?t)
                    )
                  )
                )
              )
              (not
                (not
                  (and
                    (not
                      (not
                        (adjacent ?t)
                      )
                    )
                    (agent_holds ?t)
                    (in_motion ?t ?t)
                  )
                )
              )
            )
          )
          (in_motion agent)
        )
      )
    )
    (game-conserved
      (and
        (in_motion ?xxx ?xxx)
        (< (distance desk ?xxx) 4)
        (rug_color_under ?xxx ?xxx)
        (or
          (and
            (in_motion ?xxx ?xxx)
            (opposite ?xxx)
          )
          (and
            (in_motion ?xxx)
            (not
              (between ?xxx ?xxx)
            )
            (not
              (not
                (agent_holds ?xxx ?xxx)
              )
            )
            (same_type ?xxx ?xxx)
          )
          (agent_holds ?xxx)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?n - ball ?i - cylindrical_block)
        (at-end
          (in_motion ?i)
        )
      )
    )
    (forall (?h - (either beachball bridge_block))
      (and
        (preference preference2
          (exists (?x - hexagonal_bin)
            (then
              (hold (not (agent_holds ?x desk) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 10 (count preference1:beachball) )
)
(:scoring
  (* 60 (count-unique-positions preference1:pink) )
)
)


(define (game game-id-181) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (< 1 (distance ?xxx ?xxx))
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?d - (either dodgeball cellphone) ?u - blue_cube_block ?r - ball ?c - doggie_bed)
        (then
          (once (and (or (in_motion ?c) ) (agent_holds ?c) ) )
          (once (agent_holds ?c ?c) )
          (once (adjacent north_wall) )
        )
      )
    )
    (preference preference2
      (exists (?m - dodgeball)
        (at-end
          (not
            (in_motion ?m)
          )
        )
      )
    )
  )
)
(:terminal
  (>= 10 6 )
)
(:scoring
  (+ (count-total preference1:dodgeball:pyramid_block) (+ (count preference1:yellow_cube_block) (count preference1:dodgeball:beachball:dodgeball) )
  )
)
)


(define (game game-id-182) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (= 4 (distance_side 0))
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?y - color)
        (then
          (once (agent_holds ?y ?y) )
          (hold (touch agent) )
          (hold (in ?y) )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (count preference1:blue_cube_block) 5 )
    (>= (count-once-per-objects preference1:dodgeball) (* (+ (and (- (count-overlapping preference1:yellow) )
            (count-once preference1:beachball)
            (count preference1:dodgeball)
          )
          300
          (count preference1:dodgeball)
          4
          (count-once-per-objects preference1:blue_dodgeball)
          9
        )
        (count preference1:blue_dodgeball:book)
        3
        (external-forall-maximize
          (+ 1 (count preference1:basketball:cube_block) )
        )
        10
        (* 4 (count-once-per-objects preference1:dodgeball:wall:pink) )
      )
    )
  )
)
(:scoring
  (* (* 300 (* (* 3 (count-once-per-objects preference1:dodgeball) )
        5
      )
    )
    60
  )
)
)


(define (game game-id-183) (:domain medium-objects-room-v1)
(:setup
  (exists (?u - building)
    (game-optional
      (and
        (and
          (< (distance room_center ?u) 3)
          (not
            (agent_holds ?u)
          )
        )
        (same_type ?u ?u ?u)
      )
    )
  )
)
(:constraints
  (and
    (forall (?p - hexagonal_bin)
      (and
        (preference preference1
          (exists (?x - doggie_bed)
            (then
              (hold (and (and (adjacent_side ?p ?x) (open pink_dodgeball) ) (and (agent_holds ?p ?x) (in ?p ?x) ) ) )
              (hold (exists (?t - dodgeball) (and (not (in_motion ?t) ) (= (distance room_center bed) 7) ) ) )
              (once (and (same_type ?p) (not (not (< 5 5) ) ) (not (in_motion ?x) ) (in ?x ?x) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (<= (count-once-per-objects preference1:yellow_cube_block:red) (count-longest preference1:purple) )
)
(:scoring
  10
)
)


(define (game game-id-184) (:domain many-objects-room-v1)
(:setup
  (and
    (game-conserved
      (on ?xxx)
    )
  )
)
(:constraints
  (and
    (forall (?c - dodgeball)
      (and
        (preference preference1
          (exists (?s - chair)
            (then
              (hold-while (on ?c) (and (agent_holds ?c) (and (in_motion ?s) (and (touch ?c ?s) (agent_holds ?c pillow) ) (agent_holds ?c) (in_motion ?s ?c) ) ) )
              (once (< (distance ?s ?c) 9) )
              (hold (and (in_motion ?s) (on ?c) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 1 (or (/ (+ 8 10 )
        (or
          6
        )
      )
    )
  )
)
(:scoring
  (* (* 100 (count-once-per-objects preference1:blue_dodgeball) )
    (+ (count-once-per-objects preference1:cube_block:green) (count-once-per-objects preference1:yellow) (- (count-once preference1:green) )
    )
  )
)
)


(define (game game-id-185) (:domain few-objects-room-v1)
(:setup
  (exists (?n - (either cd chair))
    (exists (?o - building)
      (exists (?s - curved_wooden_ramp)
        (exists (?g - (either pink pyramid_block) ?w - hexagonal_bin)
          (game-optional
            (in_motion ?n)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?c - (either alarm_clock))
        (then
          (once (agent_holds ?c) )
          (once (and (and (adjacent desk ?c) (adjacent front desk) ) (in ?c ?c) (in_motion ?c ?c) ) )
          (hold-while (in_motion bed) (and (in bed ?c) (agent_holds ?c) (in_motion ?c) ) (in_motion ?c) )
        )
      )
    )
    (preference preference2
      (exists (?b - block)
        (then
          (once (in_motion ?b) )
          (once (on agent) )
          (once (and (agent_holds ?b) (and (and (not (forall (?i - hexagonal_bin) (and (< 6 (distance desk ?i)) (not (< (distance agent ?b) (distance 3)) ) ) ) ) (exists (?y - dodgeball ?h - hexagonal_bin) (same_color agent ?b) ) ) (and (on ?b bed) (not (agent_holds ?b) ) (and (in ?b ?b) (in ?b ?b ?b ?b) ) (not (and (and (and (touch ?b ?b) (adjacent ?b ?b) ) (on ?b ?b) ) (not (and (in_motion ?b ?b) (or (exists (?q - block ?y - building) (= 10 (distance ?b room_center) (distance ?y bed)) ) (and (on ?b ?b) (on ?b ?b) (on ?b) (agent_holds ?b brown) (adjacent floor) (and (agent_holds ?b) (rug_color_under ?b) ) (not (agent_holds ?b ?b) ) (agent_holds rug bottom_shelf ?b) ) ) ) ) (not (in_motion ?b ?b) ) ) ) ) ) ) )
        )
      )
    )
  )
)
(:terminal
  (< 2 (* 8 1 (count preference2:basketball) 2 )
  )
)
(:scoring
  3
)
)


(define (game game-id-186) (:domain medium-objects-room-v1)
(:setup
  (exists (?s - doggie_bed)
    (game-optional
      (rug_color_under ?s)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?o - block)
        (then
          (hold (adjacent desk) )
          (hold-to-end (open bridge_block ?o ?o) )
        )
      )
    )
    (forall (?t ?e ?g ?s - (either yellow_cube_block pink) ?s - block ?g - hexagonal_bin)
      (and
        (preference preference2
          (exists (?x - (either cube_block golfball beachball) ?c - ball ?p - red_dodgeball ?c - hexagonal_bin)
            (then
              (once (on ?c) )
              (once (not (not (agent_holds ?c) ) ) )
              (hold-while (on ?g) (exists (?z - teddy_bear) (and (and (and (not (in_motion ?z ?g) ) (in ?g) ) (agent_holds rug) ) (in_motion ?c ?z) ) ) )
            )
          )
        )
        (preference preference3
          (exists (?v - (either doggie_bed))
            (then
              (once (and (on blue ?g) (not (not (agent_holds ?g ?v) ) ) ) )
              (once (agent_holds ?g drawer) )
              (once (on ?g ?g) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 10 (count-total preference2:dodgeball) )
)
(:scoring
  4
)
)


(define (game game-id-187) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (in ?xxx ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?h - bridge_block ?d - building)
        (then
          (hold-while (and (and (agent_holds ) (not (and (and (in ?d) (>= (distance agent 10) (distance_side desk desk)) (or (adjacent top_shelf) (agent_holds ?d) (not (in ?d ?d) ) (and (not (not (not (not (agent_holds ?d ?d) ) ) ) ) (rug_color_under ?d) (agent_holds ?d ?d) ) ) (not (in_motion ?d ?d) ) (touch floor) (agent_holds ?d) (agent_holds agent) ) (in ?d) ) ) ) (in ?d ?d) ) (and (and (adjacent ?d ?d) (not (in_motion ?d) ) (in_motion ?d) ) (agent_holds ?d) ) )
          (hold-while (in_motion ?d ?d) (in ?d ?d) (not (< (distance room_center ?d) 5) ) )
          (hold-while (forall (?n - cube_block ?g - (either pink laptop) ?k - triangular_ramp) (game_over pillow) ) (on ?d) )
        )
      )
    )
    (forall (?h - hexagonal_bin)
      (and
        (preference preference2
          (then
            (once (adjacent ?h ?h) )
            (hold-for 5 (in block bed) )
            (hold-while (and (< (distance_side ?h 10) 2) (adjacent ?h) ) (not (adjacent bed) ) )
          )
        )
        (preference preference3
          (exists (?w - dodgeball ?g - game_object ?i - golfball ?u - hexagonal_bin)
            (then
              (hold-while (not (exists (?v - chair ?i - cube_block) (in_motion ?h) ) ) (on ?u agent) (agent_holds top_drawer ?h) )
              (once (agent_holds ?u ?u) )
              (once (forall (?t - cylindrical_block) (and (and (and (in_motion ?h ?h) (or (not (on agent ?h) ) (in_motion ?h) (and (not (agent_holds ?u) ) (on ?h ?u) ) ) ) (< (distance 6 2) 1) (in ?h ?u) (in ?u) (exists (?b - ball) (not (not (not (and (not (forall (?g - (either tall_cylindrical_block golfball)) (toggled_on ?g agent) ) ) (in ?b) ) ) ) ) ) (and (in_motion ?u ?h) (on floor ?h) ) (not (agent_holds ?t) ) (exists (?k - dodgeball) (on ?t) ) ) (in_motion ?h ?u) ) ) )
              (hold-to-end (and (same_color agent ?h) (in_motion ?h) ) )
            )
          )
        )
        (preference preference4
          (exists (?l - ball)
            (then
              (once (in_motion upside_down ?h) )
              (hold-while (not (>= 3 1) ) (agent_holds agent ?h) )
              (once (adjacent ?h ?h) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (* (count preference3:cube_block:dodgeball) (count-once-per-objects preference1:dodgeball) )
    (count preference2:red)
  )
)
(:scoring
  (+ (+ (count preference3:yellow) 3 )
    (count-same-positions preference3:dodgeball)
  )
)
)


(define (game game-id-188) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (agent_holds floor)
  )
)
(:constraints
  (and
    (forall (?w - sliding_door ?t - chair)
      (and
        (preference preference1
          (exists (?k - cube_block)
            (then
              (once (on ?t) )
              (hold-while (> 7 2) (in_motion ?t ?t) )
              (hold (in_motion ?k ?t) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (= (external-forall-minimize 4 ) (+ (< (+ (* (count-once-per-objects preference1:doggie_bed) (count preference1:block:pink_dodgeball) )
          2
        )
        (count-overlapping preference1:purple:beachball)
      )
      (count preference1:red:cube_block)
    )
  )
)
(:scoring
  (* 2 (count preference1:red_pyramid_block) )
)
)


(define (game game-id-189) (:domain many-objects-room-v1)
(:setup
  (and
    (exists (?g - (either alarm_clock golfball desktop))
      (forall (?w - hexagonal_bin)
        (forall (?i - block)
          (and
            (and
              (forall (?b - hexagonal_bin ?o - dodgeball)
                (and
                  (game-optional
                    (in_motion agent)
                  )
                  (and
                    (game-optional
                      (agent_holds ?g)
                    )
                    (exists (?q - desktop)
                      (or
                        (and
                          (and
                            (game-optional
                              (agent_holds ?g)
                            )
                          )
                        )
                        (exists (?p - (either dodgeball lamp pink cube_block bridge_block top_drawer cube_block) ?f - ball ?y - teddy_bear)
                          (game-optional
                            (agent_holds ?w ?q)
                          )
                        )
                        (and
                          (game-conserved
                            (not
                              (same_type ?o)
                            )
                          )
                        )
                      )
                    )
                    (game-conserved
                      (in_motion ?o)
                    )
                  )
                )
              )
            )
            (game-optional
              (not
                (in_motion ?g)
              )
            )
          )
        )
      )
    )
    (forall (?e - building ?s - hexagonal_bin)
      (game-conserved
        (not
          (same_color ?s)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (agent_holds agent) )
        (once (on bed) )
        (once (not (or (on ?xxx rug) (on ?xxx) ) ) )
      )
    )
    (forall (?o - hexagonal_bin)
      (and
        (preference preference2
          (exists (?g - (either laptop dodgeball))
            (then
              (hold-for 8 (not (and (agent_holds ?o ?o floor) (agent_holds bed blue) ) ) )
              (once (in ?o) )
              (once-measure (not (between agent) ) (distance ?g room_center) )
            )
          )
        )
        (preference preference3
          (exists (?s - (either bed dodgeball))
            (at-end
              (not
                (and
                  (adjacent_side ?o ?s)
                  (agent_holds floor bed)
                )
              )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (< (* 1 6 20 )
    (external-forall-maximize
      10
    )
  )
)
(:scoring
  4
)
)


(define (game game-id-190) (:domain medium-objects-room-v1)
(:setup
  (forall (?x - chair ?o - hexagonal_bin)
    (and
      (and
        (and
          (and
            (game-conserved
              (same_object ?o ?o)
            )
          )
          (exists (?t - ball)
            (game-conserved
              (in_motion ?o main_light_switch)
            )
          )
        )
      )
      (game-optional
        (>= 1 (distance room_center room_center))
      )
      (and
        (game-conserved
          (agent_holds ?o)
        )
        (not
          (forall (?c - bridge_block)
            (and
              (not
                (forall (?t - (either tall_cylindrical_block golfball) ?w - dodgeball ?t - color)
                  (game-optional
                    (in ?t ?t)
                  )
                )
              )
              (game-conserved
                (is_setup_object ?o rug)
              )
              (game-conserved
                (on ?c)
              )
            )
          )
        )
        (exists (?u - beachball)
          (exists (?n - wall)
            (game-conserved
              (and
                (adjacent ?n ?o)
                (not
                  (same_type ?n)
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
    (preference preference1
      (exists (?k - ball ?h ?s - chair)
        (at-end
          (in west_wall)
        )
      )
    )
    (forall (?z - block)
      (and
        (preference preference2
          (exists (?m ?i - triangular_ramp ?i - dodgeball ?b - hexagonal_bin)
            (then
              (once (exists (?d - chair) (and (and (on ?z) (in_motion ?z ?z) ) (in_motion ?z) ) ) )
              (once (on ?b ?z) )
              (once (or (forall (?i - building ?i - (either side_table cd)) (object_orientation ?i) ) (not (not (on agent) ) ) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (count-once-per-objects preference1:alarm_clock) 3 )
    (and
      (>= 20 (count preference2:dodgeball) )
      (or
        (>= 5 (* (count preference1:golfball) (total-score) (count-total preference1:dodgeball) )
        )
      )
      (not
        (> 4 5 )
      )
    )
  )
)
(:scoring
  (+ (+ (count preference2:basketball) (count-once preference2:green:red) )
    (* (count-once-per-external-objects preference1:golfball) (+ (* (* (count preference1:beachball:basketball) (count preference2:white:red) )
          (- (count preference2:dodgeball) )
        )
        3
      )
    )
  )
)
)


(define (game game-id-191) (:domain medium-objects-room-v1)
(:setup
  (exists (?v - building)
    (game-conserved
      (and
        (adjacent rug)
        (in_motion floor ?v)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (on ?xxx rug) )
      )
    )
    (preference preference2
      (exists (?v ?o - sliding_door ?y - hexagonal_bin)
        (at-end
          (agent_holds pink_dodgeball brown)
        )
      )
    )
    (preference preference3
      (exists (?h - (either dodgeball floor))
        (then
          (once (and (in_motion ?h ?h) ) )
          (once (in_motion ?h upside_down) )
          (hold (in_motion ?h) )
        )
      )
    )
  )
)
(:terminal
  (>= (* 4 5 )
    (total-score)
  )
)
(:scoring
  (count-once-per-objects preference2:dodgeball)
)
)


(define (game game-id-192) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (not
      (in_motion ?xxx ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?f - dodgeball)
        (then
          (hold-while (agent_holds ?f) (in_motion ?f ?f ?f) )
          (hold (on ?f ?f) )
          (once (and (object_orientation ?f rug) (agent_holds ?f) (agent_holds floor) ) )
        )
      )
    )
  )
)
(:terminal
  (> (* 3 (external-forall-maximize 6 ) )
    (count-once-per-objects preference1:dodgeball)
  )
)
(:scoring
  2
)
)


(define (game game-id-193) (:domain many-objects-room-v1)
(:setup
  (forall (?u - teddy_bear)
    (and
      (forall (?e - hexagonal_bin)
        (game-conserved
          (in_motion ?u)
        )
      )
      (and
        (game-conserved
          (agent_holds ?u)
        )
        (game-optional
          (and
            (not
              (on ?u bed)
            )
            (in_motion agent)
          )
        )
        (or
          (exists (?s - (either dodgeball))
            (and
              (game-conserved
                (in_motion ?u ?u)
              )
            )
          )
        )
      )
      (game-conserved
        (agent_holds ?u)
      )
    )
  )
)
(:constraints
  (and
    (forall (?u - dodgeball ?w - cube_block)
      (and
        (preference preference1
          (exists (?g - cube_block)
            (then
              (once (on floor) )
              (once (on ?g) )
              (once (and (< 10 (distance room_center ?g door)) (touch ?w ?g) ) )
            )
          )
        )
      )
    )
    (preference preference2
      (exists (?y - game_object)
        (at-end
          (agent_holds ?y)
        )
      )
    )
  )
)
(:terminal
  (>= (+ 10 (count preference2:golfball:alarm_clock) )
    (count preference1:basketball)
  )
)
(:scoring
  (* (* (+ (count preference1:dodgeball) 2 )
      2
    )
    2
    (count-once-per-objects preference1:dodgeball)
    (external-forall-maximize
      (count preference1:dodgeball:yellow)
    )
    (external-forall-maximize
      (count-once-per-objects preference2:golfball)
    )
    (count-once-per-objects preference1:dodgeball)
  )
)
)


(define (game game-id-194) (:domain few-objects-room-v1)
(:setup
  (game-optional
    (exists (?r - hexagonal_bin)
      (in_motion floor)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?l - cube_block)
        (then
          (hold (and (faces ?l front rug) (not (on floor) ) ) )
          (once (in_motion ?l ?l) )
          (any)
        )
      )
    )
    (preference preference2
      (at-end
        (agent_holds block)
      )
    )
  )
)
(:terminal
  (>= 2 (count-once-per-objects preference2:book) )
)
(:scoring
  (* 12 4 (* 3 (+ (count-once-per-objects preference1:basketball) 2 )
      (* (count preference1:golfball) (count-unique-positions preference1:doggie_bed:basketball) )
    )
  )
)
)


(define (game game-id-195) (:domain few-objects-room-v1)
(:setup
  (exists (?e - hexagonal_bin)
    (game-optional
      (opposite ?e ?e)
    )
  )
)
(:constraints
  (and
    (forall (?l - hexagonal_bin ?f - hexagonal_bin)
      (and
        (preference preference1
          (exists (?g - hexagonal_bin)
            (then
              (once (and (in ?g ?f) (and (not (< (distance 4 ?g) 1) ) ) (> 10 (distance ?f room_center)) ) )
              (hold (in pink_dodgeball ?f) )
              (hold (not (agent_holds ?g) ) )
            )
          )
        )
      )
    )
    (forall (?y - triangular_ramp ?s - doggie_bed)
      (and
        (preference preference2
          (exists (?a - beachball ?m - hexagonal_bin ?v - dodgeball)
            (then
              (once (on ?s) )
              (once (in ?s) )
              (once (agent_holds agent ?v) )
            )
          )
        )
        (preference preference3
          (exists (?q - color)
            (then
              (once (in ?q) )
              (once (in_motion pink_dodgeball) )
              (hold (on ?q ?q) )
              (once (not (not (and (= (distance agent ?q) (distance 2 1)) (and (and (in_motion ?s) (not (agent_holds desk ?s) ) (exists (?o - dodgeball) (in_motion ?s) ) ) (same_type ?q ?q) ) ) ) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (and
    (= (< (* 1 30 )
        2
      )
      (count preference3:dodgeball)
    )
    (or
      (and
        (> (- (* (count-once-per-external-objects preference2:pink) 6 )
          )
          (* 3 (count-once preference2:yellow_pyramid_block:yellow) (+ 3 6 )
            3
            10
            5
            (count-once-per-objects preference2:golfball)
          )
        )
      )
    )
    (>= 5 (total-time) )
  )
)
(:scoring
  (count-once-per-external-objects preference3:tan)
)
)


(define (game game-id-196) (:domain few-objects-room-v1)
(:setup
  (exists (?z - game_object)
    (game-conserved
      (agent_holds ?z ?z)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (hold (on ?xxx) )
        (hold (not (not (not (agent_holds ?xxx) ) ) ) )
        (once (and (< (distance ?xxx ?xxx) (distance ?xxx ?xxx)) (and (and (agent_holds ?xxx) (= 8 (distance desk agent)) (not (adjacent agent) ) ) (on ?xxx agent) ) (in agent) ) )
      )
    )
    (preference preference2
      (exists (?k - pillow ?b - game_object)
        (then
          (once (and (not (agent_holds pink_dodgeball) ) (not (agent_holds ?b ?b) ) ) )
          (once (in_motion ?b left) )
          (once (in ?b) )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference1:hexagonal_bin:green) 8 )
)
(:scoring
  4
)
)


(define (game game-id-197) (:domain medium-objects-room-v1)
(:setup
  (and
    (and
      (exists (?k - dodgeball ?m - dodgeball)
        (game-conserved
          (on ?m ?m)
        )
      )
      (game-optional
        (in_motion ?xxx)
      )
      (exists (?j - hexagonal_bin)
        (game-conserved
          (touch ?j)
        )
      )
    )
    (exists (?l - hexagonal_bin)
      (exists (?r ?v - game_object ?p - hexagonal_bin ?k - hexagonal_bin)
        (game-conserved
          (not
            (and
              (agent_holds bed ?k)
              (on ?k ?l)
              (agent_holds ?k ?k)
              (not
                (same_type ?l ?k ?l)
              )
              (agent_holds ?l ?l)
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?u - hexagonal_bin ?f - hexagonal_bin ?t - ball)
        (then
          (hold-while (agent_holds ?t) (in_motion ?t ?t) (rug_color_under ?t) )
          (hold-to-end (not (agent_holds ?t ?t) ) )
          (any)
        )
      )
    )
  )
)
(:terminal
  (>= 5 (count-once-per-objects preference1) )
)
(:scoring
  (= (count preference1:red) (count preference1:green:pink_dodgeball) )
)
)


(define (game game-id-198) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (and
      (in_motion ?xxx)
      (not
        (not
          (and
            (and
              (and
                (not
                  (and
                    (in_motion blinds ?xxx)
                    (on ?xxx)
                  )
                )
                (in_motion ?xxx ?xxx)
                (in_motion ?xxx)
              )
              (adjacent ?xxx)
            )
            (agent_holds ?xxx ?xxx)
            (agent_holds ?xxx)
          )
        )
      )
      (not
        (not
          (agent_holds ?xxx)
        )
      )
      (not
        (or
          (in_motion ?xxx)
          (adjacent rug ?xxx)
          (open rug)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (at-end
        (not
          (not
            (in_motion ?xxx)
          )
        )
      )
    )
    (forall (?x - ball ?o - (either golfball key_chain))
      (and
        (preference preference2
          (exists (?m - cube_block)
            (at-end
              (agent_holds ?m)
            )
          )
        )
      )
    )
    (preference preference3
      (at-end
        (and
          (in ?xxx)
          (not
            (in_motion ?xxx)
          )
        )
      )
    )
    (forall (?q - building)
      (and
        (preference preference4
          (exists (?r - game_object ?j - cube_block)
            (at-end
              (and
                (touch ?j)
                (agent_holds ?q)
                (and
                  (and
                    (on ?q)
                  )
                  (in_motion ?j)
                )
              )
            )
          )
        )
      )
    )
    (preference preference5
      (exists (?u - (either golfball beachball))
        (then
          (once (and (and (agent_holds ?u ?u) (adjacent ?u ?u) ) (and (forall (?j ?v - hexagonal_bin) (not (not (agent_holds agent ?u) ) ) ) (in_motion ?u ?u) ) ) )
          (once (and (on ?u) (not (between ?u ?u) ) ) )
          (once (not (between ?u) ) )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= 10 2 )
    (and
      (or
        (>= (+ (count-once-per-external-objects preference4:dodgeball) (count preference3:yellow) )
          0.7
        )
        (>= (count preference2:pink_dodgeball) (count-once-per-objects preference5:golfball) )
      )
      (or
        (< 10 (count-once preference2) )
      )
      (>= (external-forall-minimize (* (count preference5:yellow) (count preference1:dodgeball:pink) (count-total preference1:cube_block) (+ (count-once-per-objects preference5:bed:pink_dodgeball) 15 (count preference3:hexagonal_bin) 6 )
            (count preference4:hexagonal_bin)
            (count preference5:hexagonal_bin:book)
          )
        )
        (count-once-per-objects preference2:beachball)
      )
    )
  )
)
(:scoring
  3
)
)


(define (game game-id-199) (:domain few-objects-room-v1)
(:setup
  (forall (?j - flat_block)
    (forall (?p - ball)
      (forall (?m - red_dodgeball)
        (exists (?i ?z ?c - hexagonal_bin)
          (and
            (forall (?o - pyramid_block)
              (exists (?s - curved_wooden_ramp ?y - dodgeball)
                (exists (?d - hexagonal_bin)
                  (and
                    (and
                      (game-conserved
                        (or
                          (above ?c)
                          (in ?c agent)
                        )
                      )
                      (exists (?k - building)
                        (exists (?l - color)
                          (game-conserved
                            (agent_holds ?o ?l)
                          )
                        )
                      )
                      (and
                        (game-conserved
                          (and
                            (and
                              (in ?m)
                              (adjacent ?z)
                            )
                            (object_orientation ?c)
                          )
                        )
                        (game-optional
                          (not
                            (not
                              (in_motion ?y ?d)
                            )
                          )
                        )
                      )
                    )
                  )
                )
              )
            )
            (game-conserved
              (adjacent ?z ?j)
            )
            (and
              (exists (?a - doggie_bed)
                (game-conserved
                  (not
                    (on bed)
                  )
                )
              )
              (and
                (game-conserved
                  (and
                    (not
                      (forall (?r - ball)
                        (and
                          (>= 1 (distance room_center desk))
                          (and
                            (and
                              (> 1 1)
                              (not
                                (and
                                  (in_motion front)
                                  (adjacent ?c)
                                )
                              )
                            )
                            (agent_holds ?z ?p)
                          )
                          (and
                            (adjacent ?i)
                            (in_motion ?m)
                          )
                        )
                      )
                    )
                    (and
                      (in_motion ?z)
                      (not
                        (agent_holds ?z bed)
                      )
                    )
                    (and
                      (> (distance ?i 10) 1)
                    )
                  )
                )
                (game-conserved
                  (touch ?z)
                )
                (game-optional
                  (and
                    (not
                      (and
                        (agent_holds rug ?c)
                        (agent_holds ?m ?j)
                        (not
                          (not
                            (same_color desk ?i)
                          )
                        )
                        (in_motion ?m ?p)
                      )
                    )
                    (in_motion ?z desk)
                    (agent_holds ?c)
                  )
                )
              )
              (and
                (or
                  (and
                    (game-optional
                      (and
                        (is_setup_object ?i)
                        (not
                          (and
                            (agent_holds ?z ?i ?c)
                            (not
                              (and
                                (in_motion ?i)
                                (agent_holds blue ?m)
                              )
                            )
                          )
                        )
                        (in ?p ?c)
                      )
                    )
                    (and
                      (game-optional
                        (agent_holds ?j)
                      )
                    )
                  )
                )
                (forall (?d - doggie_bed)
                  (exists (?f - pyramid_block ?v - chair)
                    (exists (?o - (either alarm_clock game_object) ?k - block ?g - game_object)
                      (forall (?x - block)
                        (game-conserved
                          (not
                            (< (distance ?x room_center) 2)
                          )
                        )
                      )
                    )
                  )
                )
                (exists (?k - hexagonal_bin ?o - hexagonal_bin)
                  (game-optional
                    (not
                      (and
                        (in_motion ?j bed)
                        (and
                          (game_start bed)
                          (not
                            (not
                              (in_motion ?j ?m)
                            )
                          )
                          (and
                            (faces ?p bed)
                            (not
                              (on ?c ?z bed)
                            )
                          )
                        )
                      )
                    )
                  )
                )
              )
              (and
                (game-conserved
                  (agent_holds ?m)
                )
                (game-optional
                  (above ?c)
                )
                (exists (?t - (either cube_block blue_cube_block))
                  (forall (?l ?y - hexagonal_bin ?h ?v - dodgeball ?w - building)
                    (game-optional
                      (in_motion ?j agent)
                    )
                  )
                )
              )
              (and
                (forall (?a - (either dodgeball lamp curved_wooden_ramp) ?f - ball ?r - dodgeball)
                  (and
                    (exists (?s - cube_block ?n - curved_wooden_ramp ?h - green_triangular_ramp)
                      (game-conserved
                        (and
                          (in_motion ?h ?z)
                          (in_motion ?c)
                        )
                      )
                    )
                  )
                )
              )
            )
            (exists (?l - building)
              (game-conserved
                (not
                  (< 1 1)
                )
              )
            )
            (and
              (forall (?n - (either cellphone bridge_block cd) ?s - ball ?s - game_object)
                (exists (?h - hexagonal_bin ?d - hexagonal_bin)
                  (game-optional
                    (adjacent ?j)
                  )
                )
              )
              (and
                (exists (?n - dodgeball)
                  (exists (?r - building)
                    (game-optional
                      (not
                        (in_motion ?z ?n)
                      )
                    )
                  )
                )
              )
              (and
                (and
                  (forall (?y - (either doggie_bed pyramid_block alarm_clock doggie_bed))
                    (game-conserved
                      (not
                        (in_motion ?j ?p)
                      )
                    )
                  )
                  (and
                    (forall (?h - pyramid_block)
                      (forall (?r - teddy_bear)
                        (game-conserved
                          (not
                            (not
                              (agent_holds ?z)
                            )
                          )
                        )
                      )
                    )
                  )
                  (forall (?e - hexagonal_bin)
                    (forall (?n - cube_block)
                      (game-conserved
                        (not
                          (and
                            (on ?z ?i)
                            (and
                              (< 6 (distance agent ?j))
                              (agent_holds ?j pink)
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
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?q - wall ?l - block)
        (at-end
          (or
            (object_orientation ?l)
            (< 1 (distance ?l desk))
          )
        )
      )
    )
    (preference preference2
      (exists (?o - hexagonal_bin)
        (then
          (once (and (agent_holds ?o) (agent_holds pink_dodgeball ?o) ) )
          (hold-while (not (on ?o) ) (not (same_color ?o ?o) ) )
          (hold (not (and (agent_holds ?o) (exists (?k - shelf) (agent_holds ?k) ) ) ) )
        )
      )
    )
    (preference preference3
      (exists (?a - chair)
        (then
          (once (in ?a) )
          (once (in_motion ?a) )
          (once (in ?a) )
        )
      )
    )
  )
)
(:terminal
  (>= 18 10 )
)
(:scoring
  3
)
)


(define (game game-id-200) (:domain medium-objects-room-v1)
(:setup
  (and
    (game-conserved
      (not
        (equal_z_position ?xxx)
      )
    )
    (game-optional
      (in_motion ?xxx ?xxx ?xxx ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?s - teddy_bear)
        (at-end
          (in_motion ?s)
        )
      )
    )
    (forall (?q ?n - teddy_bear)
      (and
        (preference preference2
          (exists (?d - wall)
            (then
              (once (not (in ?d) ) )
              (hold (in_motion ?d) )
              (once (agent_holds ?q ?n) )
            )
          )
        )
        (preference preference3
          (exists (?c - dodgeball)
            (then
              (any)
              (hold-while (in_motion ?c ?c) (and (and (and (and (adjacent ?q ?n) (in_motion floor) (agent_holds ?c) ) ) (not (agent_holds ?c) ) ) (or (in ?c ?n) (and (adjacent ?n ?n) (not (not (agent_holds bed ?n) ) ) ) (in_motion ?c ?c) ) (not (agent_holds ?q ?q ?c) ) ) )
            )
          )
        )
      )
    )
    (preference preference4
      (exists (?m - hexagonal_bin ?a - game_object)
        (then
          (once (< (distance desk agent) (distance room_center desk)) )
          (hold-while (same_color ?a ?a) (in ?a) (and (agent_holds ?a) (= (distance ?a side_table)) ) )
          (once (agent_holds ?a) )
        )
      )
    )
    (preference preference5
      (exists (?s - game_object)
        (then
          (once (same_type ?s) )
          (hold (in ?s ?s) )
          (once (on ?s) )
        )
      )
    )
    (forall (?j - block ?h - hexagonal_bin)
      (and
        (preference preference6
          (exists (?r - (either yellow_cube_block cd) ?q - hexagonal_bin)
            (then
              (once (on ?q) )
              (once (and (not (and (adjacent ?h ?h) (on ?q) ) ) (on agent) ) )
              (once (and (touch ?q) (in_motion ?q ?q) ) )
            )
          )
        )
        (preference preference7
          (at-end
            (on ?h)
          )
        )
      )
    )
  )
)
(:terminal
  (and
    (>= 3 (+ 50 (count preference2:dodgeball) )
    )
    (< 100 4 )
    (or
      (>= 4 (count-shortest preference6:blue_dodgeball) )
      (> (* (count preference2:beachball:pink_dodgeball) (count preference5:dodgeball) )
        2
      )
    )
  )
)
(:scoring
  (count-once-per-objects preference2:blue_pyramid_block)
)
)


(define (game game-id-201) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (open right)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?z - sliding_door ?e - hexagonal_bin)
        (then
          (any)
          (once (agent_holds ?e) )
          (hold (and (not (in_motion ?e ?e right) ) (exists (?h ?q - game_object ?r - game_object) (not (agent_holds ?e ?e) ) ) ) )
          (once-measure (and (not (agent_holds blinds rug) ) (in_motion ?e) (and (< 1 1) (not (> 1 (distance ?e ?e)) ) ) (agent_holds ?e ?e) ) (distance ?e room_center) )
        )
      )
    )
  )
)
(:terminal
  (>= (external-forall-maximize (external-forall-maximize (* (* (count-once preference1:hexagonal_bin) 1 )
          10
        )
      )
    )
    (count preference1:dodgeball)
  )
)
(:scoring
  (* (count-same-positions preference1:blue_dodgeball:basketball) (- 5 )
  )
)
)


(define (game game-id-202) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (and
      (in_motion ?xxx)
      (touch ?xxx)
    )
  )
)
(:constraints
  (and
    (forall (?s - teddy_bear)
      (and
        (preference preference1
          (exists (?y - ball ?k - block)
            (at-end
              (agent_holds ?k agent)
            )
          )
        )
      )
    )
    (forall (?k - pillow ?o - hexagonal_bin)
      (and
        (preference preference2
          (then
            (hold-while (not (in_motion ?o desk) ) (exists (?y - dodgeball) (agent_holds front ?y) ) )
            (hold (not (and (not (in_motion ?o) ) (agent_holds ?o ?o) (agent_holds ?o) ) ) )
            (hold (not (not (in_motion desk agent) ) ) )
          )
        )
        (preference preference3
          (exists (?z - ball)
            (then
              (once (<= (distance ?o ?z) (distance ?o ?o)) )
              (once (not (on ?o) ) )
              (once (adjacent agent ?o ?z) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (> 2 2 )
)
(:scoring
  (count-shortest preference1:yellow)
)
)


(define (game game-id-203) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (not
      (not
        (agent_holds ?xxx)
      )
    )
  )
)
(:constraints
  (and
    (forall (?q - wall ?w - hexagonal_bin)
      (and
        (preference preference1
          (then
            (once (in_motion ?w) )
            (hold-to-end (in_motion ?w ?w) )
            (once (= 1 (distance desk ?w)) )
          )
        )
        (preference preference2
          (exists (?h - dodgeball)
            (then
              (once (in_motion ?h) )
              (once (and (= (distance ?w bed) (distance ?w) (distance agent agent)) (not (in floor) ) ) )
              (once (not (adjacent ?h) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (not
      (or
        (>= (* (count-overlapping preference2:dodgeball:hexagonal_bin) (total-score) )
          (count preference2:dodgeball)
        )
        (<= (= (count preference1:basketball:alarm_clock) (* (count-once-per-objects preference1:yellow:dodgeball:blue_dodgeball) (count-once-per-objects preference1:dodgeball) )
            (count-once-per-objects preference2:dodgeball)
          )
          (total-time)
        )
      )
    )
    (< (total-score) (* (count-once-per-objects preference1:wall:red) 1 )
    )
    (and
      (>= (count-same-positions preference2:yellow) (count preference2:doggie_bed) )
    )
  )
)
(:scoring
  6
)
)


(define (game game-id-204) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (and
      (< 1 (distance ?xxx ?xxx))
      (and
        (and
          (in_motion ?xxx ?xxx)
          (not
            (forall (?a - dodgeball)
              (and
                (between brown ?a)
                (and
                  (agent_holds ?a)
                  (and
                    (and
                      (not
                        (same_type ?a ?a)
                      )
                      (in_motion ?a ?a)
                    )
                    (not
                      (not
                        (agent_holds agent)
                      )
                    )
                    (on ?a)
                  )
                )
                (adjacent ?a)
              )
            )
          )
        )
        (in_motion ?xxx ?xxx)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?m - dodgeball ?c - chair)
        (then
          (hold (in floor ?c) )
          (once (not (on ?c) ) )
          (hold (not (and (between ?c ?c) (and (not (not (< (distance ?c ?c) 0) ) ) (agent_holds pink_dodgeball) ) ) ) )
        )
      )
    )
    (preference preference2
      (exists (?l - dodgeball)
        (then
          (once (and (in_motion ?l ?l) (and (agent_holds drawer ?l) (agent_holds ?l) ) ) )
          (hold (faces ?l) )
          (once (on ?l ?l) )
        )
      )
    )
    (preference preference3
      (exists (?l - cylindrical_block)
        (then
          (hold (same_color ?l) )
          (once (in ?l) )
          (once (exists (?h - teddy_bear) (on ?l green) ) )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference3:beachball) (not (* (count-overlapping preference2:yellow:white) (* (total-time) (count-once-per-objects preference2:blue_cube_block) )
      )
    )
  )
)
(:scoring
  (* (count-once preference2:pink) 1 )
)
)


(define (game game-id-205) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (not
      (and
        (not
          (in_motion ?xxx)
        )
        (on ?xxx)
        (touch ?xxx ?xxx)
      )
    )
  )
)
(:constraints
  (and
    (forall (?g - cube_block)
      (and
        (preference preference1
          (exists (?e - building ?r - cube_block)
            (then
              (once (not (in_motion ?g) ) )
              (once (not (in ?g ?r) ) )
              (once (and (not (in ?r ?g) ) (not (in door ?g) ) ) )
            )
          )
        )
      )
    )
    (preference preference2
      (exists (?e - dodgeball ?l - dodgeball)
        (then
          (once (agent_holds rug bed) )
          (hold (in_motion yellow) )
          (hold (in_motion ?l) )
        )
      )
    )
  )
)
(:terminal
  (>= (* (not (* 2 (count preference1:purple) )
      )
      0
      (* (count-once-per-objects preference2:basketball) 10 )
    )
    (* (count preference2:blue_dodgeball) 10 )
  )
)
(:scoring
  (count-longest preference1:pink:alarm_clock:bed)
)
)


(define (game game-id-206) (:domain few-objects-room-v1)
(:setup
  (forall (?t - (either main_light_switch alarm_clock))
    (and
      (and
        (forall (?o - shelf)
          (and
            (exists (?c ?j ?b ?k ?s ?n ?a ?m ?w ?y - (either yellow doggie_bed blue_cube_block))
              (not
                (game-conserved
                  (in ?s ?n)
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
    (preference preference1
      (exists (?h - triangular_ramp)
        (then
          (once (not (and (and (exists (?f - ball ?b - game_object) (on ?h ?h) ) (in_motion ?h) ) (in_motion ?h ?h) ) ) )
          (once (agent_holds ?h) )
          (hold (in_motion ?h ?h) )
        )
      )
    )
    (preference preference2
      (then
        (once (in_motion rug) )
        (once (in_motion ?xxx) )
        (once (agent_holds ?xxx) )
      )
    )
  )
)
(:terminal
  (or
    (> 6 2 )
    (or
      (>= 6 (* (count preference1:wall) (count preference2:book) )
      )
      (>= (* (count-once-per-objects preference2:dodgeball) (+ (count preference2:golfball) (count preference1:dodgeball) (count preference2:blue_dodgeball) (+ (count-once-per-objects preference2:red:side_table) (count-longest preference1:beachball) )
          )
          (- 5 )
        )
        3
      )
    )
    (not
      (> (not (count-once-per-objects preference1:hexagonal_bin) ) (* (count-measure preference2:dodgeball:dodgeball) (* (count preference1:dodgeball) 3 4 5 (count preference2:hexagonal_bin:dodgeball) (count-once preference2:bed) )
        )
      )
    )
  )
)
(:scoring
  (count preference2:red)
)
)


(define (game game-id-207) (:domain medium-objects-room-v1)
(:setup
  (exists (?o - teddy_bear ?n - hexagonal_bin)
    (and
      (and
        (forall (?m - doggie_bed ?r - hexagonal_bin ?x - block ?m - cube_block ?z - (either key_chain pyramid_block))
          (game-conserved
            (touch ?n top_shelf)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?y - game_object)
        (then
          (hold (not (or (agent_holds ?y ?y ?y) (not (not (on ?y) ) ) (not (not (same_type ?y) ) ) ) ) )
          (once (not (not (and (on desk ?y) (not (rug_color_under ?y ?y) ) ) ) ) )
          (once (= (distance ?y 10 ?y) (distance ?y ?y) (distance ?y side_table)) )
        )
      )
    )
    (preference preference2
      (exists (?d - triangular_ramp)
        (at-end
          (in_motion ?d)
        )
      )
    )
  )
)
(:terminal
  (and
    (> (- (+ (+ 1 (* (count preference2:purple:purple) 2 )
          )
          (total-time)
        )
      )
      1
    )
    (<= (count preference1:dodgeball:golfball) 7 )
    (not
      (> (count-once preference1:golfball) (* 7 100 )
      )
    )
  )
)
(:scoring
  (count-once-per-objects preference2:hexagonal_bin)
)
)


(define (game game-id-208) (:domain few-objects-room-v1)
(:setup
  (game-optional
    (same_color floor agent)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?p - (either dodgeball beachball))
        (then
          (hold (on ?p) )
          (once (agent_holds ?p) )
          (hold-while (in_motion ?p) (in_motion ?p ?p) )
        )
      )
    )
  )
)
(:terminal
  (<= 4 10 )
)
(:scoring
  (count preference1:beachball)
)
)


(define (game game-id-209) (:domain many-objects-room-v1)
(:setup
  (forall (?h - (either pyramid_block pyramid_block pencil))
    (game-conserved
      (game_over bed)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?z - hexagonal_bin)
        (at-end
          (< 1 (distance ?z door))
        )
      )
    )
    (preference preference2
      (exists (?d - red_dodgeball ?s - ball)
        (then
          (once (in_motion ?s) )
          (once (adjacent_side ?s) )
          (once (in_motion ?s ?s) )
        )
      )
    )
  )
)
(:terminal
  (> (count preference1:green:pink_dodgeball) (count preference2:basketball) )
)
(:scoring
  10
)
)


(define (game game-id-210) (:domain medium-objects-room-v1)
(:setup
  (exists (?c - dodgeball)
    (and
      (game-optional
        (in_motion ?c ?c)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?n - doggie_bed ?h - hexagonal_bin ?j - hexagonal_bin)
        (then
          (hold (or (in_motion ?j ?j) (in_motion ?j ?j) (on ?j) (adjacent ?j) ) )
          (hold-while (touch ?j) (agent_holds ?j ?j) )
          (hold-for 6 (and (>= (distance agent ?j desk) (distance door desk)) (= 2 (distance desk ?j)) ) )
        )
      )
    )
    (preference preference2
      (exists (?l - (either dodgeball cellphone) ?t - hexagonal_bin)
        (at-end
          (agent_holds ?t)
        )
      )
    )
    (forall (?t - building)
      (and
        (preference preference3
          (exists (?o - (either cylindrical_block chair))
            (at-end
              (agent_holds ?t ?o)
            )
          )
        )
        (preference preference4
          (exists (?n - block ?k - hexagonal_bin)
            (then
              (hold (not (in_motion ?t ?t) ) )
              (hold (agent_holds ?k ?t) )
              (once (forall (?y - hexagonal_bin) (adjacent ?k) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count-once preference4:orange) 3 )
)
(:scoring
  2
)
)


(define (game game-id-211) (:domain many-objects-room-v1)
(:setup
  (and
    (game-conserved
      (< 1 (distance desk room_center))
    )
    (game-conserved
      (not
        (and
          (on ?xxx ?xxx)
          (in_motion ?xxx ?xxx)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?y - shelf ?u - curved_wooden_ramp)
        (then
          (once (in_motion ?u ?u) )
          (once (agent_holds ?u) )
          (once (equal_x_position ?u ?u) )
          (hold-while (on ?u) (on ?u) )
        )
      )
    )
    (preference preference2
      (exists (?e - hexagonal_bin ?g - game_object)
        (then
          (hold-while (in ?g ?g) (and (not (and (on ?g ?g) (and (in_motion ?g ?g) (agent_holds ?g ?g) ) ) ) (in_motion ?g) ) (not (and (and (in_motion ?g ?g) (touch ?g ?g agent) ) (not (in ?g) ) ) ) (on ?g ?g) )
          (hold (not (in_motion ?g) ) )
          (hold (faces ?g ?g) )
        )
      )
    )
  )
)
(:terminal
  (not
    (>= (external-forall-maximize (count-once-per-objects preference2:beachball) ) 10 )
  )
)
(:scoring
  (* (count preference1:pink) (- 6 )
    3
    (+ (count-once preference2:basketball) (external-forall-maximize 5 ) (count-measure preference2:block) 4 (external-forall-maximize (external-forall-maximize (* (* 10 (* (count-same-positions preference1:alarm_clock) (- (* (external-forall-maximize 2 ) (count-overlapping preference2:green) (count preference2:dodgeball:dodgeball) 0 (count preference2:red) (count-measure preference2:yellow:rug:book) )
                )
                (total-time)
              )
            )
            (count-longest preference2:dodgeball)
          )
        )
      )
      (* (external-forall-maximize (count preference1:dodgeball) ) (count preference2:yellow) 3 )
      1
    )
    (= (* (+ 2 (+ (+ (* (count preference2:blue_pyramid_block:golfball:pink_dodgeball) (count-once-per-objects preference1:blue_dodgeball) )
              (+ (count-overlapping preference2:basketball) (not (count-once preference1:basketball) ) 3 )
            )
          )
        )
        10
      )
      5
      100
    )
    (* (count-once preference1:triangle_block) (count preference1:cube_block) )
    6
  )
)
)


(define (game game-id-212) (:domain many-objects-room-v1)
(:setup
  (forall (?v - building ?w - hexagonal_bin)
    (forall (?b - ball)
      (exists (?r - hexagonal_bin)
        (not
          (game-optional
            (in_motion ?r)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?f - curved_wooden_ramp)
        (at-end
          (same_color ball)
        )
      )
    )
    (forall (?u - game_object)
      (and
        (preference preference2
          (exists (?o - (either basketball watch) ?t - ball)
            (then
              (once (agent_holds bed) )
              (once (in_motion agent ?u) )
              (hold (not (and (touch ?t ?t) (or (same_color ?u) (adjacent ?t ?t) (same_color ?t) ) (not (in_motion ?t) ) ) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference2:pink_dodgeball) 5 )
)
(:scoring
  (count preference2:pink)
)
)


(define (game game-id-213) (:domain medium-objects-room-v1)
(:setup
  (exists (?e - red_dodgeball)
    (exists (?b - dodgeball ?l - drawer ?l - wall ?g - dodgeball ?z - desk_shelf)
      (game-conserved
        (not
          (not
            (and
              (or
                (and
                  (> 2 (distance ?z ?z front))
                  (toggled_on ?e ?z)
                  (and
                    (and
                      (and
                        (and
                          (and
                            (and
                              (agent_holds bed)
                              (adjacent ?e)
                            )
                            (in_motion bed)
                          )
                          (and
                            (in_motion ?e ?e)
                            (and
                              (not
                                (and
                                  (agent_holds ?e)
                                  (in ?z)
                                  (in_motion ?z)
                                )
                              )
                              (agent_holds )
                            )
                            (and
                              (not
                                (and
                                  (and
                                    (in ?e)
                                  )
                                  (agent_holds ?e)
                                  (and
                                    (or
                                      (adjacent ?e ?z)
                                      (object_orientation bed)
                                    )
                                  )
                                  (not
                                    (on ?z)
                                  )
                                  (in_motion ?e desk rug)
                                  (in_motion ?e)
                                  (and
                                    (forall (?q - wall)
                                      (in_motion ?e ?q ?z)
                                    )
                                    (in_motion ?e ?z)
                                  )
                                  (not
                                    (not
                                      (not
                                        (and
                                          (not
                                            (and
                                              (and
                                                (object_orientation ?e)
                                                (exists (?c - dodgeball ?l - cube_block)
                                                  (adjacent_side ?e ?z)
                                                )
                                              )
                                              (exists (?d - hexagonal_bin ?o - hexagonal_bin)
                                                (in_motion ?z agent)
                                              )
                                            )
                                          )
                                          (adjacent ?e bed)
                                        )
                                      )
                                    )
                                  )
                                  (on ?e)
                                  (in blue ?z)
                                )
                              )
                              (in_motion ?z)
                            )
                            (in_motion ?z)
                            (is_setup_object ?e ?e)
                            (not
                              (and
                                (in_motion desk ?e)
                                (forall (?f - pyramid_block)
                                  (adjacent desk)
                                )
                                (not
                                  (and
                                    (agent_holds ?e ?z)
                                    (not
                                      (in_motion ?z)
                                    )
                                  )
                                )
                                (and
                                  (agent_holds ?z ?e)
                                  (and
                                    (>= (distance ?e agent) (distance ?z ?z ?e))
                                    (in_motion ?e)
                                  )
                                )
                                (agent_holds ?e ?z)
                                (in_motion ?e ?e)
                              )
                            )
                            (not
                              (in_motion ?z ?e)
                            )
                            (not
                              (adjacent ?e)
                            )
                          )
                        )
                        (not
                          (agent_holds ?z)
                        )
                      )
                      (in_motion ?e)
                    )
                    (in_motion ?e)
                  )
                  (agent_holds ?z agent)
                )
                (in_motion ?e ?z ?e)
              )
              (agent_holds ?z top_shelf)
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?x - teddy_bear)
        (then
          (hold (and (and (and (on ?x) (and (agent_holds agent ?x) (in_motion agent) ) ) (and (not (agent_holds ?x ?x) ) (rug_color_under ?x agent) ) ) (not (in_motion ?x) ) ) )
          (once (agent_holds ?x top_shelf) )
          (once (and (rug_color_under brown ?x) (or (not (and (not (and (on ?x ?x) (and (not (not (exists (?y - curved_wooden_ramp ?n - pillow) (in_motion back) ) ) ) (in_motion ?x ?x) ) ) ) (agent_holds ?x bed) ) ) (in_motion pink_dodgeball) ) ) )
        )
      )
    )
    (forall (?e - teddy_bear ?o - ball)
      (and
        (preference preference2
          (exists (?w ?a - hexagonal_bin)
            (then
              (hold (in_motion ?a) )
              (once (and (not (and (< 2 1) (in ?a) ) ) (not (agent_holds ?o ?w) ) (not (not (and (in agent ?a) (not (in agent door) ) ) ) ) ) )
              (once (not (and (< (distance room_center 8 room_center) (distance agent ?o)) (and (>= (distance room_center ?a) 1) (and (agent_holds ?o ?a) (>= 9 1) ) (touch ?w agent) ) ) ) )
            )
          )
        )
        (preference preference3
          (exists (?t - hexagonal_bin)
            (then
              (hold (not (and (above ?t ?t) (on ?t) ) ) )
              (once (and (in_motion yellow) (= 1 0.5) ) )
              (once (in_motion ?o ?t ?t) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference1:dodgeball) (count-shortest preference2:basketball:bed) )
)
(:scoring
  (- (* (count-once-per-objects preference2:pink) (count-once preference3:basketball) )
  )
)
)


(define (game game-id-214) (:domain many-objects-room-v1)
(:setup
  (game-optional
    (in_motion ?xxx ?xxx)
  )
)
(:constraints
  (and
    (forall (?e - block)
      (and
        (preference preference1
          (exists (?l - hexagonal_bin ?x - building ?p - shelf)
            (then
              (once (not (on upright desk) ) )
              (hold (< (distance ?p) (distance )) )
              (once (> 1 1) )
            )
          )
        )
        (preference preference2
          (exists (?k - curved_wooden_ramp)
            (then
              (once (and (on ?e ?e) (and (between ?e ?e) (and (< (distance side_table agent ?e) 7) (same_color ?k ?k) ) ) (and (and (adjacent_side ?e) (and (in ?e) (on ?e bed) ) ) (and (on ?k) (and (not (and (not (is_setup_object agent ?e) ) (on ?k ?e) (adjacent ?e ?e) ) ) (and (in_motion pillow) (adjacent ?k) ) ) ) ) ) )
              (hold (not (agent_holds ?k) ) )
              (hold (adjacent agent bridge_block) )
            )
          )
        )
      )
    )
    (preference preference3
      (exists (?f - ball ?u - (either desktop bed))
        (then
          (once (in ?u) )
          (once (and (adjacent rug ?u) (< 1 4) ) )
          (once (and (not (agent_holds ?u) ) (in_motion ?u ?u) ) )
        )
      )
    )
  )
)
(:terminal
  (or
    (<= (count preference1:hexagonal_bin) (count-overlapping preference3:hexagonal_bin) )
    (>= (count-increasing-measure preference3:basketball) (* (* (count preference2:alarm_clock) (* 60 10 (count preference2:dodgeball) (count-total preference2:pink:dodgeball) )
        )
        (count-once-per-objects preference1:hexagonal_bin:pink)
      )
    )
  )
)
(:scoring
  2
)
)


(define (game game-id-215) (:domain many-objects-room-v1)
(:setup
  (exists (?e - dodgeball ?l - dodgeball)
    (and
      (game-conserved
        (rug_color_under ?l ?l)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?b - hexagonal_bin)
        (then
          (once (and (on ?b) (agent_holds ?b) ) )
          (once (not (= (distance room_center desk) 1 1) ) )
          (hold-while (< 10 5) (in bed ?b) (not (and (not (object_orientation ?b) ) (and (agent_holds ?b) (in_motion ?b ?b ?b) ) ) ) (and (not (not (agent_holds ?b) ) ) (not (> 1 (distance ?b ?b)) ) (and (and (> (distance desk ?b) 10) (not (in pink_dodgeball upright) ) (not (in_motion bed ?b) ) (and (and (and (and (same_type ?b) (>= 1 (distance ?b 10)) ) ) (> 1 2) ) (not (not (and (same_object ?b ?b) (in_motion ?b) (in ?b) (in_motion ?b ?b) (= 1 (distance ?b room_center)) (>= 0.5 (distance 0 room_center)) (exists (?j - hexagonal_bin ?e - hexagonal_bin) (not (adjacent block) ) ) ) ) ) ) (not (forall (?v - building ?h - hexagonal_bin) (on ?h) ) ) (same_color ?b) (agent_holds agent ?b) ) (and (agent_holds agent) (not (> (distance door 10) (distance room_center 1)) ) ) ) ) )
        )
      )
    )
    (preference preference2
      (exists (?k - hexagonal_bin)
        (then
          (once (not (not (and (in front) (on agent) ) ) ) )
          (hold (and (and (is_setup_object ) (not (touch ?k) ) ) (< 1 (distance room_center ?k)) ) )
          (once (in_motion south_west_corner) )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (total-score) (count-increasing-measure preference2:pink) )
    (and
      (not
        (not
          (>= (+ (count-once preference1:dodgeball) )
            1
          )
        )
      )
    )
  )
)
(:scoring
  1
)
)


(define (game game-id-216) (:domain few-objects-room-v1)
(:setup
  (exists (?b - color ?n - cube_block)
    (game-conserved
      (in_motion ?n)
    )
  )
)
(:constraints
  (and
    (forall (?g - curved_wooden_ramp ?y ?n - ball)
      (and
        (preference preference1
          (exists (?t - dodgeball ?r - (either blue_cube_block pink) ?j - hexagonal_bin)
            (at-end
              (not
                (or
                  (< 1 1)
                  (agent_holds desk)
                )
              )
            )
          )
        )
        (preference preference2
          (exists (?z - pyramid_block)
            (then
              (once (not (in_motion ?y) ) )
              (hold (exists (?s - hexagonal_bin ?c - shelf ?a - block) (equal_x_position ?y) ) )
              (once (in_motion ?z agent) )
            )
          )
        )
        (preference preference3
          (exists (?w - ball)
            (then
              (once (in_motion agent) )
              (once (on ?n) )
              (once (in_motion ?y) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (> (count-increasing-measure preference2:blue_pyramid_block) (count-same-positions preference1:pink_dodgeball) )
)
(:scoring
  5
)
)


(define (game game-id-217) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (and
      (in_motion ?xxx)
      (in_motion ?xxx ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?c ?p - dodgeball)
        (then
          (once (and (agent_holds ?p ?p) (and (or (and (in_motion ?p ?c) (not (object_orientation ?p rug) ) (and (agent_holds ?c) (and (>= (distance_side door room_center) (distance ?c ?c)) (in ?c) (not (or (agent_holds ?c) ) ) ) ) ) (in_motion ?p) ) (and (not (agent_holds rug) ) (adjacent ?p ?c) ) ) (adjacent desk) (and (> 9 (distance ?p 5)) (and (agent_holds ?c ?p) (on ?c ?c) ) ) ) )
          (hold (agent_holds ?p) )
          (once (in_motion tan block) )
        )
      )
    )
  )
)
(:terminal
  (>= (* (and 3 (+ (count-once-per-objects preference1:yellow) (count-same-positions preference1:beachball) )
        (count-once-per-objects preference1:yellow)
      )
      (+ (* 4 (count preference1:basketball) )
        (count-total preference1:pink)
      )
    )
    2
  )
)
(:scoring
  4
)
)


(define (game game-id-218) (:domain few-objects-room-v1)
(:setup
  (and
    (and
      (and
        (and
          (exists (?k - doggie_bed ?f - ball)
            (game-conserved
              (in desk ?f)
            )
          )
        )
        (exists (?p ?n ?d - hexagonal_bin)
          (game-optional
            (in ?p)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?l - hexagonal_bin)
      (and
        (preference preference1
          (at-end
            (in_motion ?l)
          )
        )
      )
    )
  )
)
(:terminal
  (>= 7 (- 0 )
  )
)
(:scoring
  (+ (count-once-per-objects preference1:beachball:triangle_block) (+ 8 (+ (+ 2 6 )
        (* 0 2 )
      )
      (count preference1:dodgeball)
    )
  )
)
)


(define (game game-id-219) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (not
      (on ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?g - hexagonal_bin ?k - (either cylindrical_block game_object) ?q - (either alarm_clock triangle_block) ?d - blue_cube_block)
        (then
          (once (in_motion ?d ?d) )
          (once (above ?d) )
          (hold (in_motion desk desk ?d) )
        )
      )
    )
    (forall (?e - (either block))
      (and
        (preference preference2
          (exists (?l ?r - hexagonal_bin)
            (then
              (once (exists (?j - tall_cylindrical_block) (between ?j) ) )
              (hold-while (not (in ?e ?r) ) (in_motion ?r) )
              (once (agent_holds ?e) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= 2 (* (count-total preference2:tan) 2 (count-once-per-objects preference1:golfball:red) )
    )
    (>= 10 10 )
  )
)
(:scoring
  (* (- (- 10 )
    )
    10
  )
)
)


(define (game game-id-220) (:domain few-objects-room-v1)
(:setup
  (and
    (forall (?o ?z - dodgeball)
      (and
        (game-optional
          (in_motion ?o)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?l - hexagonal_bin)
        (then
          (once (agent_holds ?l) )
          (once (in_motion ?l ?l) )
          (once (and (in ?l) (or (agent_holds ?l agent) (not (in ?l) ) (and (in_motion agent) (not (agent_holds ?l ?l) ) ) ) ) )
        )
      )
    )
  )
)
(:terminal
  (>= (* 5 )
    (count-once-per-objects preference1:triangle_block)
  )
)
(:scoring
  1
)
)


(define (game game-id-221) (:domain medium-objects-room-v1)
(:setup
  (and
    (and
      (and
        (game-optional
          (agent_holds ?xxx ?xxx)
        )
        (exists (?o - dodgeball)
          (game-optional
            (in ?o)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?j - ball ?w - doggie_bed)
      (and
        (preference preference1
          (exists (?t - cube_block)
            (then
              (hold (< (distance ?t ?w) (distance ?w desk)) )
              (once (agent_holds desk ?w) )
              (once (agent_holds ?t ?w) )
            )
          )
        )
      )
    )
    (preference preference2
      (exists (?q - flat_block)
        (then
          (once (agent_holds ?q ?q) )
          (hold-while (and (on ?q) (and (not (and (or (exists (?n - (either mug teddy_bear) ?x - doggie_bed) (between ?x) ) ) (exists (?t - game_object ?e ?d - ball) (in_motion pink_dodgeball) ) ) ) (on ?q ?q) (not (on ?q) ) (not (agent_holds ?q) ) (and (on ?q agent) (and (in ?q ?q) (in_motion ?q ?q) ) ) (in_motion ?q ?q) (agent_holds bed ?q) ) ) (on ?q) (not (and (adjacent ?q ?q) (and (and (not (object_orientation ?q) ) (in_motion ?q) ) (not (touch ?q) ) ) ) ) )
          (hold (between ?q) )
        )
      )
    )
  )
)
(:terminal
  (or
    (= (total-score) (count-increasing-measure preference2:blue_dodgeball) )
    (>= (count-once-per-objects preference2:dodgeball) (count-once-per-objects preference2:dodgeball) )
  )
)
(:scoring
  (+ (count-unique-positions preference2:pyramid_block:yellow) 5 )
)
)


(define (game game-id-222) (:domain many-objects-room-v1)
(:setup
  (forall (?a - bridge_block ?z - hexagonal_bin)
    (exists (?r - hexagonal_bin)
      (game-conserved
        (on ?z)
      )
    )
  )
)
(:constraints
  (and
    (forall (?k - dodgeball)
      (and
        (preference preference1
          (then
            (hold (not (forall (?n - hexagonal_bin) (agent_holds ?k ?k) ) ) )
            (once (not (or (adjacent_side ?k) (not (agent_holds ?k) ) (agent_holds ?k) (and (not (= 4 0 (distance ?k ?k) 1) ) (and (agent_holds ?k west_wall) (< (distance side_table ?k) 1) ) ) ) ) )
            (once (and (and (and (agent_holds ?k) (agent_holds ?k) ) (not (adjacent ?k) ) ) (in_motion ?k) (in_motion ?k) ) )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count-unique-positions preference1:red) (* (+ (count preference1:pink_dodgeball) (* 4 (count preference1:alarm_clock:book) )
        (count-once-per-objects preference1:beachball:dodgeball:blue_dodgeball)
      )
      (count preference1:basketball)
    )
  )
)
(:scoring
  10
)
)


(define (game game-id-223) (:domain few-objects-room-v1)
(:setup
  (and
    (game-conserved
      (agent_holds agent ?xxx)
    )
    (game-optional
      (and
        (< (distance 10 room_center) (distance desk ?xxx))
        (on ?xxx)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?s - (either hexagonal_bin pyramid_block) ?h - dodgeball ?s - building)
        (at-end
          (in_motion ?s)
        )
      )
    )
    (forall (?t - hexagonal_bin ?q - hexagonal_bin)
      (and
        (preference preference2
          (exists (?j - pyramid_block)
            (then
              (once (and (agent_holds ?j ?j) (and (and (and (>= 1 (distance ?j agent ?q)) (in ?q) ) (agent_holds agent) ) (= 7 (distance ?j agent)) ) (rug_color_under ?q ?q) ) )
              (hold (in_motion ?j) )
              (once (in ?j) )
            )
          )
        )
        (preference preference3
          (exists (?n - (either basketball floor cellphone))
            (at-end
              (< (distance desk ?n) 7)
            )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= 1 2 )
    (> 2 (count preference1:bed:dodgeball) )
    (> 1 (- (count-once-per-objects preference3:hexagonal_bin) )
    )
  )
)
(:scoring
  (- (count preference2:dodgeball) )
)
)


(define (game game-id-224) (:domain few-objects-room-v1)
(:setup
  (and
    (game-conserved
      (and
        (in ?xxx desk)
        (agent_holds ?xxx ?xxx)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?f ?s - (either cd cellphone bed))
        (at-end
          (exists (?o - hexagonal_bin)
            (in pillow)
          )
        )
      )
    )
    (preference preference2
      (then
        (hold-to-end (adjacent agent) )
        (hold (and (or (in_motion ?xxx ?xxx) (rug_color_under ?xxx ?xxx) ) (and (in_motion agent ?xxx) ) ) )
        (once (or (in_motion bed ?xxx) (in_motion ?xxx) ) )
      )
    )
  )
)
(:terminal
  (>= (count-once-per-objects preference2:dodgeball) (* (* 4 (count-once-per-external-objects preference2:yellow) )
      (count preference1:dodgeball)
      60
      3
      (count-once-per-objects preference1:basketball)
      (= (= 7 )
        (+ 30 (* (count-longest preference1:pink) (* (+ (* (- (count-shortest preference1:pink) )
                  (* (* (count preference2:yellow) 2 (count preference1:pink) 2 )
                    2
                  )
                )
                (/
                  (/
                    (- (external-forall-maximize (count preference2:dodgeball) ) )
                    40
                  )
                  (count preference1:red)
                )
              )
              (count-once preference2:pink)
            )
          )
        )
      )
      (total-score)
    )
  )
)
(:scoring
  (* (count preference2:alarm_clock) )
)
)


(define (game game-id-225) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (in_motion ?xxx ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?f - ball)
        (then
          (hold (agent_holds ?f ?f) )
          (hold (in_motion ?f) )
        )
      )
    )
    (preference preference2
      (then
        (hold (is_setup_object agent agent) )
        (hold (not (in_motion ?xxx agent) ) )
        (once (in_motion ?xxx ?xxx) )
      )
    )
    (preference preference3
      (exists (?l - (either book cube_block golfball))
        (at-end
          (agent_holds ?l)
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (count preference1:blue_dodgeball) (- (count-once preference1:basketball) )
    )
    (>= 5 (+ (* (or (+ (count-once-per-objects preference1:pink) 8 )
            (count preference2:yellow)
            300
          )
          10
        )
        3
        (* 3 3 )
      )
    )
    (>= (count preference3:pink) 6 )
  )
)
(:scoring
  2
)
)


(define (game game-id-226) (:domain few-objects-room-v1)
(:setup
  (game-optional
    (and
      (in ?xxx)
      (in_motion ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?f - red_dodgeball)
        (at-end
          (agent_holds agent ?f)
        )
      )
    )
  )
)
(:terminal
  (>= (* 5 (= (* (* (count-once-per-objects preference1:dodgeball) 10 (* (count preference1:basketball) (count preference1:dodgeball) )
          )
          6
        )
        (+ (not (+ (count preference1:basketball:pyramid_block) 2 (count preference1:yellow:dodgeball) )
          )
          (total-score)
        )
      )
    )
    (count-total preference1:doggie_bed)
  )
)
(:scoring
  (count preference1:dodgeball)
)
)


(define (game game-id-227) (:domain few-objects-room-v1)
(:setup
  (game-optional
    (agent_holds ?xxx)
  )
)
(:constraints
  (and
    (forall (?j - ball)
      (and
        (preference preference1
          (exists (?b - bridge_block ?l - teddy_bear)
            (at-end
              (same_color agent ?l bed)
            )
          )
        )
        (preference preference2
          (exists (?v - pillow ?v ?h - dodgeball)
            (at-end
              (> 1 2)
            )
          )
        )
      )
    )
    (preference preference3
      (exists (?w ?v - dodgeball)
        (then
          (forall-sequence (?s - hexagonal_bin ?i - (either bridge_block pyramid_block))
            (then
              (once (in_motion agent) )
              (once (touch ?v) )
            )
          )
          (hold (and (same_color ?w) (adjacent ?v) ) )
          (once (and (in ?w ?w) (in_motion ?w) ) )
        )
      )
    )
    (forall (?n - cube_block ?s - (either doggie_bed doggie_bed))
      (and
        (preference preference4
          (then
            (once-measure (and (>= 6 1) (not (and (not (not (touch ?s) ) ) (exists (?i - wall) (not (touch blue) ) ) ) ) ) (distance room_center) )
            (once (in_motion bed) )
            (once (above ?s) )
          )
        )
      )
    )
    (preference preference5
      (exists (?n - beachball ?q - hexagonal_bin)
        (then
          (once (in_motion ?q) )
          (once (in_motion agent) )
          (any)
          (once (in_motion ?q) )
        )
      )
    )
  )
)
(:terminal
  (>= (* (* 7 (total-time) (* (count-once-per-objects preference3:yellow_cube_block) (count preference4:green) )
      )
      8
    )
    (count preference2:block)
  )
)
(:scoring
  10
)
)


(define (game game-id-228) (:domain few-objects-room-v1)
(:setup
  (game-optional
    (not
      (agent_holds ?xxx rug)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?i - doggie_bed)
        (then
          (hold (agent_holds ?i green) )
          (hold (and (and (not (and (and (adjacent agent) ) (agent_holds ?i) ) ) (in_motion ?i ?i) ) ) )
          (once (and (and (in_motion ?i ?i) (in_motion ?i) ) (not (in_motion ?i) ) ) )
        )
      )
    )
    (forall (?q - desk_shelf)
      (and
        (preference preference2
          (exists (?x - game_object)
            (then
              (once (and (touch ?q) (agent_holds ?q rug) ) )
              (once (and (and (agent_holds ?x ?x) ) (not (in_motion ?x) ) ) )
              (once (and (same_type ?q ?x) (< (distance door ?x) (distance_side ?x 0)) ) )
            )
          )
        )
      )
    )
    (preference preference3
      (exists (?l - cube_block)
        (then
          (hold-while (not (exists (?z - ball) (not (agent_holds ?l) ) ) ) (in ?l) )
          (once (same_type ?l) )
          (once (agent_holds ?l ?l) )
          (once (agent_holds ?l ?l) )
        )
      )
    )
  )
)
(:terminal
  (or
    (< 1 10 )
    (< (* (+ (* 7 )
          (* (count-once-per-external-objects preference2:golfball) (* (count-once-per-objects preference2:dodgeball) (* (count-once-per-objects preference1:alarm_clock) (count preference2:dodgeball) (count preference2:rug) (count preference2:golfball) )
            )
          )
        )
        (count preference3:blue_dodgeball)
      )
      (- (* (* (* (count-once-per-objects preference2:pink:beachball) (count-once-per-objects preference3:dodgeball:beachball) )
            (count preference2:doggie_bed)
            3
          )
          3
        )
      )
    )
    (> 2 1 )
  )
)
(:scoring
  5
)
)


(define (game game-id-229) (:domain few-objects-room-v1)
(:setup
  (forall (?c - hexagonal_bin)
    (game-conserved
      (and
        (in ?c ?c)
        (adjacent_side ?c)
        (on bed)
      )
    )
  )
)
(:constraints
  (and
    (forall (?n - hexagonal_bin)
      (and
        (preference preference1
          (exists (?g - hexagonal_bin ?z - shelf)
            (then
              (once (on ?n) )
              (hold-while (and (and (adjacent ?n) (is_setup_object ?n ?n) ) (agent_holds ?z) ) (= (distance ?z ?n) 0.5 (distance )) )
              (once (agent_holds agent ?z) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (or
      (or
        (>= (count preference1) 10 )
        (> (> 5 (* (* (count preference1:blue_cube_block:yellow) (- 9 )
              )
              (= (count-once-per-objects preference1:pyramid_block:golfball) (count preference1:blue_dodgeball:red_pyramid_block) (count preference1:beachball) )
            )
          )
          (count-measure preference1:blue_dodgeball)
        )
        (>= (count preference1:doggie_bed) (* 3 (count preference1:dodgeball) (count-once-per-objects preference1:tall_cylindrical_block) )
        )
      )
      (>= 6 (count preference1:dodgeball) )
    )
    (> (* 8 0.5 )
      (* (* (* 100 (count preference1:dodgeball) )
          3
        )
        (total-time)
        (* 10 5 (external-forall-maximize (count preference1:yellow:basketball) ) )
        (* (external-forall-maximize 10 ) 18 )
        (> 4 (- (= 50 (* (* (external-forall-minimize (* 10 )
                  )
                  1
                )
                (/
                  (count preference1:dodgeball:dodgeball)
                  6
                )
                9
                40
                9
                (* (count preference1:purple:pink_dodgeball) (count-once-per-objects preference1) )
              )
            )
          )
        )
        (- 5 )
      )
    )
  )
)
(:scoring
  (count preference1:beachball)
)
)


(define (game game-id-230) (:domain many-objects-room-v1)
(:setup
  (exists (?x - (either red cd))
    (forall (?w - hexagonal_bin)
      (game-optional
        (in_motion agent ?x)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?y - hexagonal_bin)
        (then
          (once (and (in_motion ?y) (not (in_motion ?y ?y) ) ) )
          (once (not (in_motion ?y) ) )
          (hold (and (not (on ?y) ) (agent_holds ?y) (not (adjacent_side rug) ) ) )
        )
      )
    )
    (preference preference2
      (exists (?u - cube_block ?v - hexagonal_bin ?v - doggie_bed ?f - hexagonal_bin)
        (then
          (once (on ?f ?f) )
          (once (not (in_motion ?f) ) )
          (once (on ?f ?f) )
        )
      )
    )
  )
)
(:terminal
  (>= (* 10 3 )
    (* (* 5 (count-once preference2:pink) )
      0
    )
  )
)
(:scoring
  (count-once-per-objects preference2:yellow)
)
)


(define (game game-id-231) (:domain few-objects-room-v1)
(:setup
  (exists (?u - dodgeball)
    (forall (?v - (either basketball book))
      (game-optional
        (and
          (exists (?d - book ?j - doggie_bed ?k - doggie_bed)
            (and
              (= 7 9)
              (and
                (agent_holds ?k ?u)
                (and
                  (in_motion ?u)
                  (in ?u)
                )
              )
            )
          )
          (in_motion ?u ?u)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?a - ball)
        (at-end
          (same_color ?a ?a)
        )
      )
    )
    (preference preference2
      (exists (?i - golfball)
        (then
          (once (agent_holds ?i) )
          (hold (and (in ?i) (> 1 (distance ?i 5)) ) )
          (once (equal_x_position ?i) )
        )
      )
    )
    (preference preference3
      (exists (?t - hexagonal_bin)
        (then
          (once (not (<= 1 1) ) )
          (once (< 4 (distance desk ?t)) )
          (once (not (and (in_motion ?t) (and (in_motion ?t ?t) (agent_holds ?t) ) ) ) )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (count preference3:beachball) (count-increasing-measure preference1:pink) )
    (not
      (or
        (>= (total-score) (count preference3:dodgeball:golfball) )
        (>= (* (count preference3:blue_pyramid_block) (<= (* (* (count-once-per-objects preference2:pink_dodgeball:pink_dodgeball) (count preference3:green) )
                5
                (count-once-per-external-objects preference3:golfball)
              )
              (+ (* (* 5 (count-same-positions preference1) )
                  (+ 8 (total-score) )
                )
                (+ 5 (+ (external-forall-maximize (- (count-once-per-external-objects preference2:pink:blue_pyramid_block) )
                    )
                    (count-once-per-objects preference2:alarm_clock)
                  )
                )
              )
            )
          )
          2
        )
      )
    )
  )
)
(:scoring
  3
)
)


(define (game game-id-232) (:domain medium-objects-room-v1)
(:setup
  (and
    (exists (?r ?s - hexagonal_bin ?s - doggie_bed)
      (and
        (forall (?b - dodgeball)
          (exists (?x - dodgeball ?w - pyramid_block ?c - hexagonal_bin)
            (and
              (game-conserved
                (agent_holds rug)
              )
            )
          )
        )
        (forall (?x - chair)
          (exists (?p - ball)
            (and
              (game-conserved
                (on ?p)
              )
              (forall (?h - hexagonal_bin)
                (exists (?a - drawer ?o ?w - (either cylindrical_block dodgeball book))
                  (exists (?j - ball ?y - block)
                    (game-optional
                      (exists (?c - tall_cylindrical_block)
                        (rug_color_under ?y ?y)
                      )
                    )
                  )
                )
              )
              (and
                (game-optional
                  (agent_holds pillow)
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
    (preference preference1
      (exists (?h - (either hexagonal_bin key_chain) ?h - hexagonal_bin)
        (then
          (once (and (and (not (in ?h) ) (in ?h ?h ?h) ) (in ?h ?h) ) )
          (once (and (and (not (in ?h ?h) ) (not (not (in ?h) ) ) ) (< 1 (distance ?h ?h)) ) )
          (hold (touch color ?h) )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (external-forall-maximize (total-score) ) (* (+ (total-time) (count-once-per-objects preference1:basketball:dodgeball) )
        1
      )
    )
    (> (count preference1:pink_dodgeball:red:dodgeball) (+ (> 30 (count-once-per-objects preference1:purple:red) )
        3
      )
    )
  )
)
(:scoring
  (count preference1:dodgeball)
)
)


(define (game game-id-233) (:domain few-objects-room-v1)
(:setup
  (forall (?m - hexagonal_bin ?w - dodgeball)
    (game-optional
      (agent_holds agent ?w)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?g - (either blue_cube_block block))
        (at-end
          (agent_holds ?g ?g)
        )
      )
    )
  )
)
(:terminal
  (<= (+ 3 (* (- 6 )
        (count preference1:doggie_bed)
      )
      (external-forall-maximize
        (count-once preference1:pink:beachball)
      )
    )
    3
  )
)
(:scoring
  (count-once preference1:basketball)
)
)


(define (game game-id-234) (:domain many-objects-room-v1)
(:setup
  (exists (?u - green_triangular_ramp)
    (game-optional
      (agent_holds ?u ?u)
    )
  )
)
(:constraints
  (and
    (forall (?w - triangular_ramp)
      (and
        (preference preference1
          (exists (?j - dodgeball)
            (then
              (hold-while (and (on ?j ?w) (in_motion ?w) ) (rug_color_under ?w) )
              (hold (in ?w) )
              (once (not (agent_holds ?w ?j) ) )
            )
          )
        )
        (preference preference2
          (exists (?e - triangular_ramp)
            (at-end
              (forall (?d - (either basketball dodgeball))
                (in_motion upright ?e)
              )
            )
          )
        )
      )
    )
    (preference preference3
      (exists (?k - block)
        (then
          (hold (touch ?k) )
          (once (in_motion ?k) )
          (hold (not (agent_holds ?k) ) )
        )
      )
    )
  )
)
(:terminal
  (and
    (or
      (or
        (or
          (or
            (and
              (>= 0 (count-same-positions preference2:side_table) )
            )
            (>= (+ (count-once-per-objects preference2) (* (count-once-per-external-objects preference3) (count preference1:basketball) )
              )
              (count preference2:red)
            )
            (>= 30 (count-once-per-objects preference2:purple:block) )
            (>= (* 1 (+ (count-longest preference3:dodgeball) (* (count preference1:beachball:hexagonal_bin:dodgeball) 5 )
                )
                (count-overlapping preference1:dodgeball)
                (* (* (and (- (count preference3:beachball) )
                    )
                    (count preference1)
                  )
                  (count preference3:yellow_cube_block:purple)
                )
              )
              (count preference1:doggie_bed:beachball)
            )
          )
          (> 8 2 )
        )
        (and
          (>= 3 1 )
          (>= 18 9 )
          (<= (total-score) (count preference2) )
        )
      )
    )
    (or
      (or
        (>= (+ (+ (count-once-per-objects preference3:pink_dodgeball) 180 )
            (* (count preference1:basketball:golfball) 3 (count-once-per-objects preference2:hexagonal_bin) (* (* (* (count preference3:blue_dodgeball) 2 )
                  10
                )
                (count-once-per-external-objects preference3:dodgeball:hexagonal_bin)
              )
            )
            (count preference1:purple)
          )
          (-
            (* (count-once-per-objects preference3:beachball:yellow) (count-once-per-objects preference1:yellow) )
            (* (count preference1:beachball:red) (* 10 (* (total-score) (* (count preference2:doggie_bed:pink) 10 )
                )
              )
            )
          )
        )
        (or
          (>= 30 (count preference1:golfball:dodgeball) )
          (>= 5 (+ (* (* (count preference1:beachball) (count-once-per-objects preference2:dodgeball) (count preference2:pink) (or 10 ) )
                (+ (- (* 4 (count-increasing-measure preference2:pink:dodgeball:beachball) (* (count-unique-positions preference1:cylindrical_block:dodgeball:white) (* (+ (count preference2:beachball) (* 10 (count preference1:pink:green) )
                          )
                          (+ 3 (- (count-once-per-objects preference3:alarm_clock:book:purple) )
                            2
                          )
                        )
                      )
                    )
                  )
                  (count-once preference3:dodgeball)
                )
              )
              60
            )
          )
          (>= (= 2 (count preference1:orange) )
            (+ (- (* (count-once-per-objects preference1:beachball) (count preference1:yellow) )
              )
              (count-once-per-objects preference3:tall_cylindrical_block:pink_dodgeball)
            )
          )
        )
      )
      (and
        (and
          (or
            (not
              (> (count-same-positions preference2) (count-once-per-objects preference2:golfball) )
            )
            (>= 10 (* (count preference2:pink_dodgeball:golfball) (count-once-per-objects preference3:dodgeball:orange) )
            )
          )
        )
      )
    )
    (or
      (or
        (or
          (>= (count-once-per-external-objects preference2:pink) 4 )
          (or
            (>= (* 1 5 )
              (total-time)
            )
            (>= (count preference3:wall) (* 4 (count preference2:pink) )
            )
          )
        )
      )
      (>= (count-once preference1:dodgeball) 3 )
    )
  )
)
(:scoring
  (* 1 20 )
)
)


(define (game game-id-235) (:domain few-objects-room-v1)
(:setup
  (game-optional
    (agent_holds ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?t - hexagonal_bin ?x ?j - (either cube_block dodgeball triangular_ramp pillow basketball yellow_cube_block cube_block) ?w - cube_block ?t - shelf)
        (then
          (once (not (agent_holds ?t ?t) ) )
          (once (faces ?t) )
        )
      )
    )
  )
)
(:terminal
  (> (- (* (* 6 15 8 (count preference1:cube_block:dodgeball:top_drawer) )
        (= 5 )
      )
    )
    (* (count-once-per-objects preference1:yellow) (count-once-per-objects preference1:dodgeball) )
  )
)
(:scoring
  (count preference1:hexagonal_bin)
)
)


(define (game game-id-236) (:domain medium-objects-room-v1)
(:setup
  (game-optional
    (on ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?h - green_triangular_ramp ?s - wall)
        (then
          (once (in_motion ?s top_shelf) )
          (hold-while (and (in ?s ?s) ) (same_type ?s) )
        )
      )
    )
  )
)
(:terminal
  (or
    (or
      (>= (count preference1:golfball) (* (count-once-per-objects preference1:beachball:pyramid_block) (count-once preference1:blue_dodgeball) )
      )
    )
    (or
      (or
        (>= (count-once preference1:book) (count preference1:golfball) )
        (= (count-once-per-objects preference1:beachball:side_table) (* (* (count-once-per-objects preference1:pink_dodgeball) 10 10 (+ 2 10 )
            )
            (- (count preference1:pink) )
          )
        )
      )
    )
  )
)
(:scoring
  (- (+ 1 (- (+ (count preference1:dodgeball) (+ (count preference1:red) 8 (- (external-forall-maximize (count-once-per-objects preference1:purple) ) )
            2
            (count-same-positions preference1:golfball)
            (count preference1:beachball)
          )
        )
      )
      (total-score)
    )
  )
)
)


(define (game game-id-237) (:domain few-objects-room-v1)
(:setup
  (exists (?f - chair)
    (forall (?n - hexagonal_bin ?y ?u - color)
      (exists (?h - dodgeball)
        (game-optional
          (not
            (not
              (and
                (and
                  (exists (?v - (either credit_card cylindrical_block))
                    (and
                      (exists (?a ?g - triangular_ramp)
                        (exists (?z - hexagonal_bin ?p - color ?e - hexagonal_bin ?w - shelf)
                          (not
                            (in_motion ?f)
                          )
                        )
                      )
                      (and
                        (not
                          (not
                            (agent_crouches ?u ?y)
                          )
                        )
                        (<= (distance 2 ?u) 1)
                      )
                    )
                  )
                  (in_motion pink ?h)
                )
                (agent_holds ?y)
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
    (preference preference1
      (exists (?k - hexagonal_bin ?y - ball)
        (then
          (hold (not (on ?y ?y) ) )
          (hold (agent_holds ?y) )
          (once (not (agent_crouches agent ?y) ) )
        )
      )
    )
    (preference preference2
      (exists (?c - dodgeball ?l - dodgeball)
        (then
          (once (and (and (on ?l ?l) (in_motion ?l ?l) (agent_holds ?l ?l) ) (exists (?i - (either golfball dodgeball mug cellphone) ?z - building) (not (and (not (agent_holds ?l ?z) ) (and (exists (?q - curved_wooden_ramp) (and (not (not (not (on ?z ?z) ) ) ) (not (not (on ?z) ) ) (in_motion ?q) ) ) (agent_holds ?z) (forall (?g - doggie_bed) (not (not (exists (?p ?a - hexagonal_bin) (in pillow) ) ) ) ) ) ) ) ) ) )
          (once (in_motion upright ?l) )
          (hold (not (and (or (adjacent desk) (agent_holds ?l) ) (exists (?d - hexagonal_bin) (agent_holds ?l ?d) ) (not (in_motion ?l) ) ) ) )
        )
      )
    )
  )
)
(:terminal
  (>= (count-overlapping preference2:dodgeball) (and 3 16 ) )
)
(:scoring
  (count-once preference2:blue_cube_block)
)
)


(define (game game-id-238) (:domain many-objects-room-v1)
(:setup
  (and
    (and
      (forall (?e - chair ?f - building)
        (exists (?n - chair ?l ?i ?z ?o ?h ?v - dodgeball ?s - dodgeball)
          (and
            (game-optional
              (on ?s ?f)
            )
          )
        )
      )
    )
    (game-conserved
      (in ?xxx ?xxx)
    )
    (exists (?d - hexagonal_bin)
      (game-conserved
        (in_motion ?d ?d)
      )
    )
  )
)
(:constraints
  (and
    (forall (?v - block)
      (and
        (preference preference1
          (exists (?g ?i - hexagonal_bin)
            (then
              (once (not (adjacent ?v) ) )
              (once (adjacent ?v) )
              (once (and (game_start ?v) (and (agent_holds ?i ?i) (in_motion ?v) ) ) )
            )
          )
        )
        (preference preference2
          (exists (?s - building)
            (then
              (once (agent_holds ?v ?s) )
              (hold (in ?v) )
              (once (not (same_object agent ?s) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (not
    (not
      (> 5 (total-score) )
    )
  )
)
(:scoring
  (count preference2:dodgeball:beachball)
)
)


(define (game game-id-239) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (not
      (and
        (not
          (not
            (agent_holds ?xxx)
          )
        )
        (and
          (agent_holds sideways)
          (agent_holds ?xxx ?xxx)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?h - (either mug book) ?h - hexagonal_bin)
      (and
        (preference preference1
          (then
            (hold (= (distance ?h ?h) (distance ?h ?h ?h) 1) )
            (hold (and (agent_holds ?h) (on upside_down) ) )
            (once (equal_z_position ?h ?h) )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count-once preference1:pink) (count-once preference1:alarm_clock:yellow_cube_block) )
)
(:scoring
  (* (* (total-time) (count preference1:dodgeball:dodgeball:golfball) )
    5
    4
  )
)
)


(define (game game-id-240) (:domain many-objects-room-v1)
(:setup
  (exists (?g - hexagonal_bin ?k - game_object)
    (game-conserved
      (in_motion ?k ?k)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?n ?z - block)
        (then
          (hold-while (and (in_motion ?z ?n) (in_motion ?z) ) (on ?z) )
          (once (on ?z) )
          (once (on agent ?n) )
        )
      )
    )
    (forall (?v - triangular_ramp ?t - dodgeball)
      (and
        (preference preference2
          (exists (?l - red_pyramid_block ?j - (either flat_block dodgeball))
            (then
              (once (not (not (exists (?b - block) (agent_holds ?b bed) ) ) ) )
              (hold (and (not (and (not (in_motion ?j ?j) ) (same_color ?j) (adjacent agent ?t) ) ) (and (not (on ?t ?j) ) (in ?t ?j) ) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (< 5 (total-time) )
)
(:scoring
  15
)
)


(define (game game-id-241) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (and
      (not
        (and
          (and
            (or
              (in ?xxx)
              (in_motion ?xxx ?xxx)
            )
            (and
              (agent_holds ?xxx ?xxx)
              (not
                (not
                  (and
                    (in_motion ?xxx)
                    (agent_holds ?xxx ?xxx)
                  )
                )
              )
            )
          )
          (in_motion ?xxx)
        )
      )
      (in bed ?xxx)
      (agent_holds ?xxx)
    )
  )
)
(:constraints
  (and
    (forall (?k - game_object ?m - hexagonal_bin)
      (and
        (preference preference1
          (exists (?c - building)
            (then
              (once (and (adjacent ?c) (in_motion ?c) ) )
              (once (and (and (in ?m) (not (and (not (not (in ?m) ) ) (or (>= 2 1) ) ) ) (and (and (exists (?r - pillow) (agent_holds ?r) ) (not (on ?m ?c ?c) ) ) (>= 1 1) ) ) (on ?c) ) )
              (once (and (agent_holds ?c) (not (agent_holds upright ?m) ) ) )
            )
          )
        )
      )
    )
    (preference preference2
      (then
        (once (in agent ?xxx) )
        (once (not (agent_holds ?xxx) ) )
        (hold (not (not (not (on ?xxx ?xxx) ) ) ) )
      )
    )
  )
)
(:terminal
  (and
    (>= 1 4 )
  )
)
(:scoring
  (* (count preference1:dodgeball) (count preference2:dodgeball) )
)
)


(define (game game-id-242) (:domain medium-objects-room-v1)
(:setup
  (and
    (exists (?s - beachball)
      (game-conserved
        (and
          (in_motion ?s)
          (agent_holds ?s)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?u - (either blue_cube_block chair))
      (and
        (preference preference1
          (exists (?p - hexagonal_bin)
            (then
              (once (not (on ?p ?p) ) )
              (once (on ?p) )
              (once (adjacent ?p ?u) )
            )
          )
        )
        (preference preference2
          (exists (?l - dodgeball ?k - dodgeball)
            (then
              (hold (same_object ?k) )
              (hold (agent_holds ?k ?u) )
              (hold (in_motion upright) )
            )
          )
        )
      )
    )
    (preference preference3
      (at-end
        (on ?xxx ?xxx ?xxx)
      )
    )
    (forall (?s - color ?b - (either cube_block pyramid_block) ?n ?t - dodgeball)
      (and
        (preference preference4
          (exists (?s ?e ?i - hexagonal_bin)
            (at-end
              (in_motion agent)
            )
          )
        )
      )
    )
  )
)
(:terminal
  (< (external-forall-minimize (count-once-per-objects preference1:white) ) 2 )
)
(:scoring
  (/
    (count-once-per-external-objects preference1:hexagonal_bin)
    (count-once-per-external-objects preference2:green:golfball)
  )
)
)


(define (game game-id-243) (:domain medium-objects-room-v1)
(:setup
  (and
    (forall (?g - (either cube_block golfball yellow_cube_block key_chain doggie_bed alarm_clock blue_cube_block))
      (exists (?b - building)
        (exists (?e - hexagonal_bin)
          (game-optional
            (adjacent_side ?g)
          )
        )
      )
    )
    (and
      (and
        (or
          (and
            (forall (?d - wall)
              (game-conserved
                (is_setup_object ?d agent)
              )
            )
            (exists (?p - hexagonal_bin)
              (and
                (exists (?s - chair)
                  (forall (?d ?k ?i - cube_block)
                    (forall (?w - (either dodgeball cylindrical_block))
                      (game-conserved
                        (adjacent ?d ?i)
                      )
                    )
                  )
                )
              )
            )
          )
          (exists (?s - pillow ?i - (either alarm_clock cellphone))
            (exists (?f - chair)
              (game-optional
                (and
                  (on ?f ?f)
                  (not
                    (not
                      (agent_holds bed ?f)
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
)
(:constraints
  (and
    (preference preference1
      (exists (?m - doggie_bed)
        (then
          (once (in_motion ?m) )
          (once-measure (not (in ?m) ) (distance room_center ?m) )
          (hold-while (agent_holds ?m) (not (in_motion ?m) ) )
        )
      )
    )
  )
)
(:terminal
  (>= 9 (count preference1:dodgeball) )
)
(:scoring
  (>= (count-once preference1:dodgeball) (count-once-per-external-objects preference1:pyramid_block:cube_block) )
)
)


(define (game game-id-244) (:domain many-objects-room-v1)
(:setup
  (exists (?f - hexagonal_bin ?i - hexagonal_bin)
    (game-conserved
      (in ?i)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?q - (either dodgeball dodgeball hexagonal_bin) ?i - hexagonal_bin)
        (at-end
          (not
            (agent_holds ?i)
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference1:yellow:cube_block) (count-once-per-objects preference1:green) )
)
(:scoring
  (/
    7
    3
  )
)
)


(define (game game-id-245) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (agent_holds ?xxx ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?l - game_object ?h ?q - watch)
        (then
          (once (on front) )
          (once (and (same_type ?h ?h) (in_motion ?h) ) )
          (once (on ?q) )
        )
      )
    )
    (preference preference2
      (then
        (once (agent_holds ?xxx) )
        (once (on ?xxx ?xxx) )
        (hold (adjacent_side ?xxx ?xxx) )
        (once (exists (?i - (either golfball triangle_block cube_block)) (agent_holds ?i ?i) ) )
      )
    )
    (preference preference3
      (exists (?q - blue_pyramid_block)
        (at-end
          (not
            (not
              (not
                (= (distance_side ?q) 1 1)
              )
            )
          )
        )
      )
    )
    (preference preference4
      (exists (?y - dodgeball)
        (then
          (once (on ?y) )
          (hold (not (is_setup_object ?y) ) )
          (hold-while (agent_holds ?y) (forall (?r - (either chair lamp) ?z - hexagonal_bin ?h - triangular_ramp) (on ?y ?h) ) (agent_holds ?y ?y block) )
        )
      )
    )
  )
)
(:terminal
  (or
    (not
      (>= (+ (count-once preference3:blue_dodgeball) (+ (* 8 300 )
            (count-once preference1:dodgeball:yellow)
          )
          180
          (count-once preference2:dodgeball:red)
        )
        (count preference2:green)
      )
    )
    (>= 3 (count preference1:beachball:basketball) )
  )
)
(:scoring
  1
)
)


(define (game game-id-246) (:domain few-objects-room-v1)
(:setup
  (exists (?f - hexagonal_bin)
    (exists (?a - (either key_chain ball) ?i - teddy_bear)
      (game-conserved
        (and
          (in_motion ?i)
          (exists (?k - hexagonal_bin ?x - hexagonal_bin ?l - building)
            (in floor)
          )
          (adjacent pink_dodgeball)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?s - ball)
        (at-end
          (on ?s ?s)
        )
      )
    )
  )
)
(:terminal
  (>= (external-forall-maximize (count-overlapping preference1:dodgeball) ) 5 )
)
(:scoring
  (count-once preference1:dodgeball)
)
)


(define (game game-id-247) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (forall (?t ?a - ball)
      (and
        (in_motion ?t)
        (in_motion ?a ?t)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?i - teddy_bear ?v - cube_block)
        (then
          (hold (agent_holds ?v) )
          (hold (not (agent_holds ?v) ) )
        )
      )
    )
    (preference preference2
      (exists (?s ?y ?t - cube_block ?f - hexagonal_bin)
        (then
          (once (agent_holds ?f) )
          (once (and (touch ?f) (in_motion ?f) ) )
          (once (agent_holds ?f ?f) )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference2:blue_dodgeball) (* (+ 5 (count preference2:dodgeball) )
      30
    )
  )
)
(:scoring
  3
)
)


(define (game game-id-248) (:domain medium-objects-room-v1)
(:setup
  (and
    (forall (?k ?q - dodgeball)
      (game-conserved
        (not
          (agent_holds ?k)
        )
      )
    )
    (game-conserved
      (is_setup_object ?xxx)
    )
    (game-conserved
      (on ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?z - dodgeball)
        (then
          (once (<= 1 1) )
          (hold-while (and (and (in_motion ?z ?z) (on ?z ?z) ) (is_setup_object ?z) ) (in_motion ?z) )
          (once (and (equal_z_position agent ?z) (exists (?p - hexagonal_bin ?e - block) (agent_holds ?z ?e) ) ) )
        )
      )
    )
    (preference preference2
      (exists (?p - chair ?f - (either yellow_cube_block golfball golfball))
        (then
          (once (not (agent_holds ?f) ) )
          (hold (in_motion ?f) )
          (once (adjacent ?f ?f) )
        )
      )
    )
  )
)
(:terminal
  (> (external-forall-maximize 2 ) (total-time) )
)
(:scoring
  (count-once-per-objects preference2:red)
)
)


(define (game game-id-249) (:domain few-objects-room-v1)
(:setup
  (forall (?v - wall)
    (game-conserved
      (on ?v)
    )
  )
)
(:constraints
  (and
    (forall (?p - shelf ?i - yellow_pyramid_block)
      (and
        (preference preference1
          (exists (?x - hexagonal_bin)
            (then
              (once (same_object ?i) )
              (hold (not (agent_holds ?x) ) )
              (once (exists (?g - chair ?b - wall) (adjacent ?i brown) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (not
      (>= 2 4 )
    )
    (>= (+ 5 (* 3 (* (/ 8 (count-unique-positions preference1:golfball) ) (+ 15 (* (* (= (* (total-time) 5 )
                    (count-once-per-external-objects preference1:book)
                  )
                  (* (<= (not 4 ) (count preference1:basketball) )
                    10
                  )
                )
                (* 3 (count-once-per-objects preference1:pink) )
              )
            )
          )
        )
        (count preference1:pyramid_block)
        (count preference1:yellow:yellow)
        (count-once-per-objects preference1:doggie_bed:book)
        (+ 3 (total-time) (count preference1:red) (count-once preference1) 3 (+ (- (count preference1:hexagonal_bin:pink) )
            4
            (count-once-per-objects preference1:doggie_bed)
            (* (count-once-per-objects preference1:brown) 25 )
            30
            4
            30
          )
        )
      )
      100
    )
  )
)
(:scoring
  5
)
)


(define (game game-id-250) (:domain few-objects-room-v1)
(:setup
  (forall (?e - hexagonal_bin ?t - building ?z - hexagonal_bin)
    (game-conserved
      (and
        (and
          (not
            (on ?z ?z)
          )
          (exists (?y - block ?x - (either cellphone mug golfball))
            (not
              (not
                (object_orientation ?z ?x)
              )
            )
          )
        )
        (in ?z ?z)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?p - game_object)
        (then
          (once (adjacent ?p) )
          (hold (in_motion ?p) )
          (once (and (and (and (in_motion ?p) (in_motion ?p) ) (in_motion ?p ?p ?p) ) (in_motion ?p rug) ) )
        )
      )
    )
  )
)
(:terminal
  (> 10 10 )
)
(:scoring
  (external-forall-maximize
    5
  )
)
)


(define (game game-id-251) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (agent_holds ?xxx ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (at-end
        (and
          (and
            (and
              (forall (?i - doggie_bed)
                (not
                  (agent_holds ?i agent)
                )
              )
              (not
                (and
                  (and
                    (in_motion agent)
                    (on ?xxx ?xxx)
                  )
                  (rug_color_under ?xxx)
                )
              )
            )
            (on ?xxx)
          )
          (in_motion ?xxx ?xxx)
          (adjacent ?xxx ?xxx)
          (not
            (in ?xxx)
          )
        )
      )
    )
    (preference preference2
      (exists (?t - triangular_ramp ?q - (either cylindrical_block key_chain bridge_block))
        (then
          (once (and (agent_holds ?q door) (and (touch bed ?q) (and (and (agent_holds ?q) (not (in_motion bed) ) ) (on ?q rug) ) ) ) )
          (once (and (agent_holds ?q) (not (touch ?q ?q) ) ) )
          (hold (agent_holds agent) )
        )
      )
    )
  )
)
(:terminal
  (> (count preference2:beachball) 3 )
)
(:scoring
  (count preference1:pink_dodgeball)
)
)


(define (game game-id-252) (:domain many-objects-room-v1)
(:setup
  (forall (?w ?d - ball ?y - hexagonal_bin)
    (exists (?l - cube_block ?k - cylindrical_block)
      (forall (?d ?n ?p ?m - curved_wooden_ramp ?l - dodgeball)
        (and
          (game-optional
            (on ?y)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?p - dodgeball)
        (then
          (hold (on agent) )
          (hold (in_motion ?p) )
          (hold-to-end (and (same_color rug) (adjacent ?p ?p) ) )
        )
      )
    )
    (forall (?i - dodgeball)
      (and
        (preference preference2
          (exists (?d - dodgeball)
            (then
              (once (and (on upright ?i) (and (not (not (agent_holds ?d) ) ) (and (in_motion top_drawer ?d) (adjacent_side ?d ?d) (and (adjacent floor agent ?d) (and (and (in_motion floor) (= 2 0) ) (touch ?i) ) (= (distance ?d 1) (distance ?d)) (and (in ?i ?d) (on ?i ?d) ) ) ) ) ) )
              (once (agent_holds pink_dodgeball ?d) )
              (hold (not (object_orientation ?d) ) )
              (once (in_motion ?i) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (> (count-overlapping preference1:beachball:blue_dodgeball) (count preference2:basketball) )
    (<= (- 6 )
      4
    )
  )
)
(:scoring
  (count-once-per-objects preference2:triangle_block)
)
)


(define (game game-id-253) (:domain many-objects-room-v1)
(:setup
  (exists (?u - (either desktop doggie_bed))
    (game-optional
      (in_motion ?u)
    )
  )
)
(:constraints
  (and
    (forall (?e - ball)
      (and
        (preference preference1
          (exists (?c - curved_wooden_ramp)
            (at-end
              (in_motion ?e ?e)
            )
          )
        )
        (preference preference2
          (exists (?v - dodgeball)
            (then
              (once (or (agent_holds ?v) (in_motion desk) ) )
              (once (agent_holds ?v ?v) )
              (hold (agent_holds ?e) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (not
    (<= 2 (count preference1:beachball) )
  )
)
(:scoring
  (count preference2:golfball)
)
)


(define (game game-id-254) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (or
      (and
        (or
          (and
            (and
              (> (distance ?xxx agent) 1)
              (and
                (agent_holds ?xxx)
                (not
                  (equal_z_position ?xxx)
                )
              )
              (on ?xxx)
            )
            (in_motion ?xxx)
            (not
              (agent_holds ?xxx ?xxx)
            )
            (and
              (is_setup_object ?xxx ?xxx top_shelf ?xxx)
              (and
                (not
                  (and
                    (forall (?r - dodgeball ?e ?z ?h ?o ?v ?u ?j ?t ?l ?d - game_object)
                      (touch ?o ?o bed)
                    )
                    (adjacent ?xxx)
                  )
                )
                (agent_holds ?xxx)
              )
            )
            (agent_holds agent ?xxx)
            (and
              (agent_holds ?xxx)
              (in_motion ?xxx ?xxx)
              (not
                (agent_holds ?xxx)
              )
            )
          )
          (in_motion ?xxx ?xxx)
        )
        (and
          (agent_holds ?xxx ?xxx)
          (agent_holds top_drawer)
        )
      )
      (< 6 (distance ))
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?i - game_object)
        (at-end
          (and
            (agent_holds ?i)
            (not
              (and
                (in ?i)
                (not
                  (agent_holds ?i ?i)
                )
              )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (count-same-positions preference1:purple) (and (count preference1:blue_dodgeball:dodgeball) ) )
    (>= 10 (count preference1:yellow_cube_block) )
  )
)
(:scoring
  (count preference1:yellow_pyramid_block)
)
)


(define (game game-id-255) (:domain few-objects-room-v1)
(:setup
  (forall (?q - cube_block)
    (and
      (and
        (forall (?a - cube_block)
          (game-conserved
            (adjacent ?q)
          )
        )
      )
      (exists (?s - curved_wooden_ramp)
        (exists (?v - hexagonal_bin)
          (exists (?k - dodgeball ?d - yellow_cube_block)
            (forall (?h ?j - hexagonal_bin ?c - hexagonal_bin)
              (forall (?l - doggie_bed)
                (or
                  (game-optional
                    (and
                      (agent_holds upright)
                      (in sideways ?d)
                    )
                  )
                )
              )
            )
          )
        )
      )
      (game-optional
        (in_motion ?q)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?u - hexagonal_bin)
        (then
          (once (and (on desk) (not (in_motion ?u ?u) ) (not (and (not (on ?u) ) (exists (?b ?i - (either lamp alarm_clock triangle_block)) (exists (?j - hexagonal_bin) (on ?u) ) ) ) ) ) )
          (hold (not (in ?u ?u) ) )
          (hold (agent_holds ?u) )
        )
      )
    )
    (preference preference2
      (then
        (once (and (agent_holds ?xxx ?xxx) (not (not (and (and (and (or (exists (?x - game_object ?k - pyramid_block ?g - building ?a - red_dodgeball ?n - dodgeball) (agent_holds ?n) ) ) (not (agent_holds ?xxx agent) ) ) (agent_holds ?xxx ?xxx) ) (in_motion door) ) ) ) (same_type ?xxx) ) )
        (once (on agent) )
        (hold (and (not (not (agent_holds ?xxx) ) ) (same_type ?xxx) ) )
      )
    )
  )
)
(:terminal
  (<= 50 (count-once-per-objects preference2:pink) )
)
(:scoring
  (+ (count preference2:beachball) 1 )
)
)

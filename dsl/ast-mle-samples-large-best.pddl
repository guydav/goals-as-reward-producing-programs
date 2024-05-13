(define (game game-id-36) (:domain medium-objects-room-v1)
(:setup
  (and
    (forall (?e - dodgeball ?g - cube_block)
      (exists (?j - hexagonal_bin)
        (forall (?e - golfball ?r - block ?p - tall_cylindrical_block)
          (exists (?o - (either game_object desktop))
            (and
              (game-optional
                (between ?g)
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
    (forall (?h - teddy_bear)
      (and
        (preference preference1
          (exists (?y - shelf)
            (then
              (hold-while (touch ?y) (not (and (agent_holds ?y ?h) (in_motion ?h) (not (on ?y) ) ) ) (<= (distance 4 ?y) (distance agent desk ?y)) )
              (once (in_motion ?y) )
              (hold-while (rug_color_under ?h) (in_motion ?y) )
            )
          )
        )
      )
    )
    (forall (?u - hexagonal_bin ?b - (either pyramid_block dodgeball))
      (and
        (preference preference2
          (exists (?y - pillow)
            (then
              (once (game_start ?b) )
              (once (adjacent ?y) )
              (once (in ?b) )
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
(define (game game-id-43) (:domain many-objects-room-v1)
(:setup
  (exists (?e - flat_block)
    (forall (?b - (either blue_cube_block main_light_switch))
      (and
        (forall (?x - red_dodgeball)
          (game-conserved
            (agent_holds ?b ?b)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?k - ball ?j - wall ?e - ball ?n ?h - cube_block ?a - block)
      (and
        (preference preference1
          (exists (?x - (either golfball golfball))
            (then
              (hold (not (agent_holds ) ) )
              (hold (is_setup_object ?a ?x) )
              (once (agent_holds ?a ?x) )
            )
          )
        )
      )
    )
    (preference preference2
      (exists (?s - triangular_ramp ?p - wall)
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
(define (game game-id-58) (:domain few-objects-room-v1)
(:setup
  (forall (?x - ball ?w - chair ?m - tall_cylindrical_block)
    (and
      (game-optional
        (not
          (or
            (and
              (or
                (on ?m)
                (not
                  (agent_holds ?m)
                )
              )
              (in_motion ?m ?m)
            )
            (in ?m)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?r - blue_pyramid_block)
      (and
        (preference preference1
          (exists (?o - (either dodgeball cube_block) ?q - ball)
            (then
              (once (not (open ?r) ) )
              (hold-while (is_setup_object ?q) (in_motion ?q) )
              (once (agent_holds ?q) )
            )
          )
        )
      )
    )
    (preference preference2
      (exists (?k ?h - dodgeball ?t - golfball ?c - yellow_cube_block ?f - pyramid_block)
        (then
          (hold (<= 10 1) )
          (once (agent_holds ?f ?f) )
          (hold (on ?f) )
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
(define (game game-id-64) (:domain many-objects-room-v1)
(:setup
  (exists (?t - beachball)
    (and
      (or
        (forall (?q - building)
          (exists (?p - hexagonal_bin)
            (forall (?r ?e ?h ?k ?u ?v - ball ?u - (either flat_block cube_block) ?r ?x - hexagonal_bin)
              (and
                (game-optional
                  (not
                    (on ?q)
                  )
                )
                (exists (?k - game_object)
                  (game-optional
                    (on ?k ?r)
                  )
                )
              )
            )
          )
        )
        (game-conserved
          (in green ?t)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?g - game_object)
      (and
        (preference preference1
          (exists (?l - building)
            (then
              (once (in_motion ?l ?g) )
              (hold-while (in_motion ?l) (in ?l) )
              (once (in_motion ?g) )
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
(define (game game-id-73) (:domain many-objects-room-v1)
(:setup
  (exists (?g - teddy_bear ?e - (either pyramid_block cylindrical_block))
    (game-conserved
      (forall (?u - hexagonal_bin)
        (in floor ?e)
      )
    )
  )
)
(:constraints
  (and
    (forall (?n ?o ?v ?q ?j ?e - hexagonal_bin ?g - (either key_chain book))
      (and
        (preference preference1
          (exists (?y - curved_wooden_ramp)
            (then
              (once (in_motion ?y ?g) )
              (hold (not (exists (?n - ball ?z - wall) (in brown) ) ) )
              (hold-to-end (touch ?y rug) )
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
(define (game game-id-113) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (in ?xxx)
  )
)
(:constraints
  (and
    (forall (?w - block)
      (and
        (preference preference1
          (exists (?s - hexagonal_bin)
            (then
              (hold (not (touch ?w ?s) ) )
              (hold (and (adjacent ?s ?s) (> 7 (distance_side agent back)) ) )
              (hold (agent_holds agent) )
              (hold (agent_holds ?w ?w) )
              (once (adjacent_side ?w ?w ?s) )
            )
          )
        )
      )
    )
    (forall (?h - game_object)
      (and
        (preference preference2
          (at-end
            (open ?h bridge_block)
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
(define (game game-id-147) (:domain many-objects-room-v1)
(:setup
  (game-optional
    (in ?xxx)
  )
)
(:constraints
  (and
    (forall (?q - hexagonal_bin)
      (and
        (preference preference1
          (exists (?u - drawer ?p ?k ?w ?u ?e ?n - dodgeball ?z - wall)
            (at-end
              (in front)
            )
          )
        )
      )
    )
    (preference preference2
      (then
        (once (and (forall (?l - hexagonal_bin) (touch ?l) ) (in_motion ?xxx ?xxx) ) )
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
(define (game game-id-152) (:domain few-objects-room-v1)
(:setup
  (exists (?m - cube_block ?x - hexagonal_bin ?t - hexagonal_bin ?q - doggie_bed)
    (game-conserved
      (< 7 2)
    )
  )
)
(:constraints
  (and
    (forall (?o - hexagonal_bin ?n - hexagonal_bin)
      (and
        (preference preference1
          (exists (?c - cube_block ?r - hexagonal_bin)
            (then
              (once (on ?r) )
              (once (and (< 6 (distance_side ?n ?r 9)) (agent_holds bed ?r) ) )
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
    (forall (?f ?h - teddy_bear ?d - wall)
      (and
        (preference preference1
          (exists (?k - teddy_bear)
            (then
              (once (agent_holds ?k) )
              (once (and (and (equal_z_position ?d) (< 4 (distance room_center ?k ?k)) ) (and (not (in_motion ?k) ) (and (in_motion ?d ?d) (in ?k) ) ) ) )
              (once-measure (agent_holds ?d) (distance ?k ?k) )
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
(define (game game-id-165) (:domain few-objects-room-v1)
(:setup
  (forall (?y - building ?j - cube_block)
    (forall (?d - (either dodgeball cd) ?r - building)
      (and
        (game-conserved
          (and
            (on ?r)
            (agent_holds ?j)
            (between bottom_shelf south_west_corner)
            (in_motion ?j)
          )
        )
        (game-optional
          (and
            (and
              (in ?j ?j)
            )
            (rug_color_under ?r)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?e ?c - wall)
      (and
        (preference preference1
          (exists (?h - building)
            (then
              (hold-while (not (agent_holds agent) ) (is_setup_object desk ?e) )
              (once (toggled_on agent) )
              (once (< 8 (distance 5 ?c)) )
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
(define (game game-id-191) (:domain medium-objects-room-v1)
(:setup
  (exists (?j - building)
    (game-conserved
      (and
        (adjacent rug)
        (in_motion floor ?j)
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
      (exists (?j ?o - sliding_door ?u - hexagonal_bin)
        (at-end
          (agent_holds pink_dodgeball brown)
        )
      )
    )
    (preference preference3
      (exists (?x - (either dodgeball floor))
        (then
          (once (and (in_motion ?x ?x) ) )
          (once (in_motion ?x upside_down) )
          (hold (in_motion ?x) )
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
(define (game game-id-210) (:domain medium-objects-room-v1)
(:setup
  (exists (?a - dodgeball)
    (and
      (game-optional
        (in_motion ?a ?a)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?k - doggie_bed ?x - hexagonal_bin ?s - hexagonal_bin)
        (then
          (hold (or (in_motion ?s ?s) (in_motion ?s ?s) (on ?s) (adjacent ?s) ) )
          (hold-while (touch ?s) (agent_holds ?s ?s) )
          (hold-for 6 (and (>= (distance agent ?s desk) (distance door desk)) (= 2 (distance desk ?s)) ) )
        )
      )
    )
    (preference preference2
      (exists (?w - (either dodgeball cellphone) ?c - hexagonal_bin)
        (at-end
          (agent_holds ?c)
        )
      )
    )
    (forall (?c - building)
      (and
        (preference preference3
          (exists (?o - (either cylindrical_block chair))
            (at-end
              (agent_holds ?c ?o)
            )
          )
        )
        (preference preference4
          (exists (?k - block ?n - hexagonal_bin)
            (then
              (hold (not (in_motion ?c ?c) ) )
              (hold (agent_holds ?n ?c) )
              (once (forall (?u - hexagonal_bin) (adjacent ?u) ) )
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
(define (game game-id-250) (:domain few-objects-room-v1)
(:setup
  (forall (?t - hexagonal_bin ?c - building ?h - hexagonal_bin)
    (game-conserved
      (and
        (and
          (not
            (on ?h ?h)
          )
          (exists (?u - block ?m - (either cellphone mug golfball))
            (not
              (not
                (object_orientation ?m ?h)
              )
            )
          )
        )
        (in ?h ?h)
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
(define (game game-id-302) (:domain few-objects-room-v1)
(:setup
  (and
    (game-conserved
      (in_motion ?xxx)
    )
    (exists (?a - hexagonal_bin ?g - doggie_bed)
      (exists (?j - cube_block ?o - ball)
        (and
          (game-conserved
            (not
              (agent_holds ?o ?g)
            )
          )
          (game-conserved
            (not
              (agent_holds ?g)
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
      (exists (?s ?h - game_object)
        (then
          (hold (same_color ?s ?h) )
          (hold (adjacent_side agent) )
          (hold-to-end (not (same_object ?s ?s) ) )
        )
      )
    )
  )
)
(:terminal
  (>= 10 (>= 30 1 )
  )
)
(:scoring
  6
)
)
(define (game game-id-315) (:domain few-objects-room-v1)
(:setup
  (game-optional
    (in ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?y - hexagonal_bin ?x - (either golfball cylindrical_block) ?q - shelf)
        (then
          (once (on ?q rug) )
          (once (agent_holds ?q) )
          (once (and (not (on ?q ?q) ) (adjacent ?q ?q) (and (not (agent_holds rug ?q) ) (agent_holds ?q) ) (in_motion ?q) ) )
        )
      )
    )
  )
)
(:terminal
  (< 10 10 )
)
(:scoring
  0
)
)
(define (game game-id-322) (:domain medium-objects-room-v1)
(:setup
  (game-optional
    (and
      (in_motion ?xxx)
      (broken ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?a - building)
        (then
          (once (not (in_motion ?a) ) )
          (hold (in_motion ?a) )
          (hold-while (on top_drawer) (< 2 (distance room_center desk ?a)) (same_color ?a ?a) )
        )
      )
    )
  )
)
(:terminal
  (>= 2 (total-time) )
)
(:scoring
  8
)
)
(define (game game-id-347) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (in_motion ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?k - ball)
        (then
          (once (not (in_motion ?k ?k) ) )
          (once (or (on ?k ?k) (agent_holds ?k) ) )
          (hold (forall (?q - (either cube_block doggie_bed)) (not (agent_holds ?q ?k) ) ) )
        )
      )
    )
    (preference preference2
      (exists (?e - (either mug beachball))
        (then
          (once (not (and (same_color ?e) (in_motion ?e) ) ) )
          (hold (or (game_start ?e ?e) (in_motion ?e) ) )
          (hold-while (on bed) (adjacent ?e) )
        )
      )
    )
    (forall (?e - ball)
      (and
        (preference preference3
          (then
            (once (agent_holds ?e) )
            (once (in_motion ?e ?e) )
            (once (and (not (touch ?e ?e) ) (not (agent_holds ?e) ) (< (distance ?e ?e) 2) ) )
          )
        )
        (preference preference4
          (exists (?w - cube_block)
            (then
              (once (and (agent_holds agent) ) )
              (once (not (same_object brown) ) )
              (once (in_motion pink_dodgeball) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 100 10 )
)
(:scoring
  (count preference4:pink_dodgeball)
)
)
(define (game game-id-364) (:domain medium-objects-room-v1)
(:setup
  (exists (?c - cube_block)
    (game-conserved
      (not
        (agent_holds agent)
      )
    )
  )
)
(:constraints
  (and
    (forall (?e - ball)
      (and
        (preference preference1
          (exists (?a - (either chair) ?z - shelf ?o - teddy_bear)
            (then
              (hold (in ?o) )
              (hold (exists (?y - building) (and (same_color ?o) (> 1 1) ) ) )
              (once (adjacent ?e) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 3 (count-once preference1:beachball) )
)
(:scoring
  10
)
)
(define (game game-id-377) (:domain few-objects-room-v1)
(:setup
  (forall (?r - red_dodgeball)
    (exists (?k - teddy_bear ?e - ball ?j - ball)
      (and
        (and
          (or
            (game-conserved
              (in_motion ?j)
            )
          )
          (game-optional
            (agent_holds ?j)
          )
        )
        (exists (?v - tall_cylindrical_block)
          (exists (?o - hexagonal_bin)
            (game-conserved
              (in ?r desk)
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
      (exists (?h - dodgeball)
        (at-end
          (and
            (exists (?z - (either cube_block golfball))
              (on floor)
            )
            (agent_holds ?h ?h)
          )
        )
      )
    )
  )
)
(:terminal
  (>= 6 0 )
)
(:scoring
  3
)
)
(define (game game-id-383) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (and
      (in_motion ?xxx ?xxx)
      (and
        (rug_color_under ?xxx)
        (agent_holds ?xxx)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?q - hexagonal_bin)
        (at-end
          (exists (?m - triangular_ramp)
            (< 4 1)
          )
        )
      )
    )
    (preference preference2
      (exists (?g - cylindrical_block)
        (then
          (once (on ?g) )
          (hold (in_motion blue) )
          (once (on ?g ?g) )
        )
      )
    )
  )
)
(:terminal
  (= (count preference1:purple) 4 )
)
(:scoring
  (* (total-score) (count preference1:pyramid_block) )
)
)
(define (game game-id-391) (:domain few-objects-room-v1)
(:setup
  (exists (?n - (either golfball bridge_block) ?x - hexagonal_bin)
    (and
      (game-conserved
        (in_motion ?x)
      )
      (exists (?c - hexagonal_bin)
        (game-conserved
          (game_over ?c ?c)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?h - triangular_ramp ?u ?f - dodgeball)
        (then
          (once (in_motion ?u ?u) )
          (once (not (not (touch ?u) ) ) )
          (once (agent_holds bridge_block ?f) )
          (once (not (and (in ?f) (not (in_motion ?f) ) ) ) )
          (once (agent_holds ?u) )
        )
      )
    )
  )
)
(:terminal
  (>= (* 10 2 180 )
    5
  )
)
(:scoring
  3
)
)
(define (game game-id-411) (:domain few-objects-room-v1)
(:setup
  (game-optional
    (on ?xxx agent)
  )
)
(:constraints
  (and
    (forall (?l - curved_wooden_ramp)
      (and
        (preference preference1
          (then
            (once (on ?l) )
            (once (and (in_motion ?l ?l) (on ?l ?l) (in_motion ?l ?l) ) )
            (once (agent_holds ?l) )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 2 15 )
)
(:scoring
  (count-measure preference1:yellow)
)
)
(define (game game-id-447) (:domain medium-objects-room-v1)
(:setup
  (game-optional
    (agent_holds ?xxx ?xxx)
  )
)
(:constraints
  (and
    (forall (?t - wall)
      (and
        (preference preference1
          (exists (?r - dodgeball)
            (then
              (once (in agent) )
              (hold (and (on ?t) (object_orientation ?t) ) )
              (hold (adjacent_side ?t) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference1:doggie_bed) 10 )
)
(:scoring
  (* (- (external-forall-maximize (count preference1:cube_block) ) )
    3
  )
)
)
(define (game game-id-486) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (or
      (agent_holds top_shelf ?xxx)
      (not
        (in_motion ?xxx)
      )
    )
  )
)
(:constraints
  (and
    (forall (?l - hexagonal_bin)
      (and
        (preference preference1
          (exists (?m - (either dodgeball golfball cube_block) ?u - (either top_drawer yellow_cube_block dodgeball) ?d - building ?g - block)
            (then
              (hold-for 2 (same_color ?g) )
              (once (in_motion ?g door) )
              (once (and (not (touch ?g) ) (and (not (in_motion ?g) ) ) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference1:yellow:dodgeball) (count-once-per-objects preference1:basketball) )
)
(:scoring
  (count-once-per-objects preference1:yellow)
)
)
(define (game game-id-490) (:domain few-objects-room-v1)
(:setup
  (forall (?r - (either dodgeball curved_wooden_ramp))
    (game-conserved
      (in_motion ?r)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?y - cube_block ?h ?a - curved_wooden_ramp)
        (then
          (once (agent_holds ?h) )
          (once (and (agent_holds ?a ?h) (above ?h) (in_motion ?h) ) )
          (hold (adjacent ?h) )
        )
      )
    )
    (preference preference2
      (exists (?h - ball ?k - dodgeball)
        (then
          (once (in ?k) )
          (once (agent_holds ?k) )
        )
      )
    )
  )
)
(:terminal
  (< 3 3 )
)
(:scoring
  (count-once-per-objects preference2:golfball)
)
)
(define (game game-id-514) (:domain few-objects-room-v1)
(:setup
  (and
    (and
      (forall (?p - hexagonal_bin)
        (game-conserved
          (in ?p)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (agent_holds ?xxx ?xxx) )
        (hold (on ?xxx) )
        (once (not (agent_holds ?xxx) ) )
      )
    )
  )
)
(:terminal
  (<= (> 3 0 )
    3
  )
)
(:scoring
  7
)
)
(define (game game-id-526) (:domain few-objects-room-v1)
(:setup
  (and
    (game-conserved
      (is_setup_object ?xxx)
    )
    (game-conserved
      (not
        (in ?xxx ?xxx)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?x - (either pink pink))
        (then
          (hold (not (agent_holds ?x floor) ) )
          (once (in_motion ?x ?x) )
          (once (agent_holds agent ?x) )
        )
      )
    )
  )
)
(:terminal
  (>= 4 (external-forall-maximize 3 ) )
)
(:scoring
  5
)
)
(define (game game-id-577) (:domain few-objects-room-v1)
(:setup
  (game-optional
    (and
      (touch ?xxx ?xxx)
      (on ?xxx)
    )
  )
)
(:constraints
  (and
    (forall (?u - curved_wooden_ramp)
      (and
        (preference preference1
          (then
            (once (not (agent_holds ?u ?u) ) )
            (once (in_motion bed) )
            (once (on ?u) )
          )
        )
      )
    )
  )
)
(:terminal
  (> (count preference1:yellow) 4 )
)
(:scoring
  (total-score)
)
)
(define (game game-id-593) (:domain medium-objects-room-v1)
(:setup
  (exists (?c - dodgeball)
    (exists (?u - game_object ?v - hexagonal_bin)
      (game-conserved
        (in ?v ?v ?c)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?a - ball ?p - ball)
        (then
          (hold (not (in_motion agent ?p) ) )
          (once (and (agent_holds ?p) (on ?p ?p) (agent_holds ?p) ) )
          (once (in_motion ?p desk) )
        )
      )
    )
    (forall (?s ?u ?e - game_object)
      (and
        (preference preference2
          (exists (?m - hexagonal_bin)
            (at-end
              (in_motion ?m pink_dodgeball)
            )
          )
        )
      )
    )
  )
)
(:terminal
  (= (count preference2:golfball:dodgeball) (count preference1:pink_dodgeball) )
)
(:scoring
  (total-time)
)
)
(define (game game-id-604) (:domain many-objects-room-v1)
(:setup
  (exists (?d - pyramid_block)
    (game-conserved
      (in_motion ?d ?d)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (at-end
        (in ?xxx)
      )
    )
  )
)
(:terminal
  (= 3 (= 15 4 )
  )
)
(:scoring
  (* 3 5 )
)
)
(define (game game-id-624) (:domain medium-objects-room-v1)
(:setup
  (game-optional
    (in_motion ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?y - hexagonal_bin ?f - building ?r ?m ?z ?x ?h - wall)
        (then
          (once (agent_holds floor ?x) )
          (hold (agent_holds ?x) )
          (once (on ?x) )
        )
      )
    )
  )
)
(:terminal
  (>= 7 (count-once-per-objects preference1:red) )
)
(:scoring
  (count preference1:pink_dodgeball)
)
)
(define (game game-id-628) (:domain few-objects-room-v1)
(:setup
  (and
    (exists (?j - dodgeball)
      (game-optional
        (or
          (not
            (in_motion ?j)
          )
          (on ?j)
        )
      )
    )
    (game-conserved
      (touch pink_dodgeball)
    )
    (and
      (game-optional
        (on ?xxx ?xxx)
      )
      (forall (?k - (either laptop dodgeball pyramid_block) ?j - ball)
        (game-conserved
          (not
            (on ?j)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?p - block ?t - (either desktop dodgeball) ?n ?s ?z - dodgeball)
        (then
          (hold (in ?s ?s) )
          (once (adjacent ?n) )
          (once (agent_holds ?z) )
        )
      )
    )
    (forall (?g - hexagonal_bin ?p - cube_block)
      (and
        (preference preference2
          (exists (?q - hexagonal_bin ?s - dodgeball ?o - bridge_block ?y - flat_block)
            (then
              (hold (touch ?y) )
              (once (in ?p ?y) )
              (once (in ?y) )
            )
          )
        )
      )
    )
    (preference preference3
      (exists (?v - flat_block)
        (then
          (once (= (distance bed) 8 (distance ?v 4)) )
          (hold (touch agent desk) )
          (once (in_motion ?v ?v ?v) )
        )
      )
    )
  )
)
(:terminal
  (>= (count-shortest preference2:hexagonal_bin:triangle_block) (count preference3:dodgeball) )
)
(:scoring
  3
)
)
(define (game game-id-638) (:domain few-objects-room-v1)
(:setup
  (exists (?d - wall)
    (game-conserved
      (on ?d)
    )
  )
)
(:constraints
  (and
    (forall (?w - block ?u - game_object ?w - hexagonal_bin ?a - hexagonal_bin)
      (and
        (preference preference1
          (exists (?j - hexagonal_bin)
            (then
              (once (in ?a) )
              (once (touch top_drawer) )
              (once (in ?a) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (> 15 (count preference1:orange) )
)
(:scoring
  (count-unique-positions preference1:doggie_bed)
)
)
(define (game game-id-649) (:domain medium-objects-room-v1)
(:setup
  (exists (?t - flat_block)
    (game-conserved
      (in_motion ?t upright)
    )
  )
)
(:constraints
  (and
    (forall (?p - hexagonal_bin ?e - chair)
      (and
        (preference preference1
          (exists (?m - doggie_bed ?c - hexagonal_bin)
            (then
              (once (on ?e) )
              (hold (not (= (distance room_center ?c) (distance ?c)) ) )
              (once (in ?e ?e) )
            )
          )
        )
        (preference preference2
          (exists (?m - hexagonal_bin)
            (then
              (once (and (agent_holds ?e ?e) (on agent desktop) ) )
              (hold (and (< 5 5) ) )
              (once (< 7 (distance )) )
            )
          )
        )
      )
    )
    (preference preference3
      (at-end
        (agent_holds ?xxx ?xxx)
      )
    )
  )
)
(:terminal
  (= 2 (count preference2:dodgeball) )
)
(:scoring
  40
)
)
(define (game game-id-656) (:domain many-objects-room-v1)
(:setup
  (and
    (forall (?m - block ?k - (either golfball key_chain))
      (game-conserved
        (on ?k)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?d - block ?i - curved_wooden_ramp ?f - hexagonal_bin)
        (then
          (once (in_motion ?f ?f) )
          (once (in_motion ?f ?f) )
          (hold-for 9 (above ?f ?f) )
        )
      )
    )
    (preference preference2
      (exists (?t - dodgeball)
        (at-end
          (and
            (= (distance desk ?t) (distance 0 desk ?t))
            (and
              (agent_holds agent)
              (on ?t)
              (agent_holds ?t)
            )
          )
        )
      )
    )
    (preference preference3
      (exists (?t - wall)
        (then
          (once (in ?t ?t) )
          (once (not (in_motion agent) ) )
          (once (in_motion ?t) )
        )
      )
    )
  )
)
(:terminal
  (>= 25 1 )
)
(:scoring
  (count-once-per-objects preference2:basketball)
)
)
(define (game game-id-660) (:domain many-objects-room-v1)
(:setup
  (exists (?o ?d - dodgeball)
    (game-optional
      (agent_holds ?d)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?f - cube_block)
        (then
          (once (or (in_motion ?f) (not (agent_holds ?f ?f) ) ) )
          (hold (above bridge_block) )
          (hold (in bed) )
        )
      )
    )
  )
)
(:terminal
  (>= 1 3 )
)
(:scoring
  (count preference1:basketball:basketball)
)
)
(define (game game-id-681) (:domain many-objects-room-v1)
(:setup
  (game-optional
    (and
      (and
        (agent_holds ?xxx ?xxx)
        (agent_holds ?xxx ?xxx)
        (and
          (not
            (in_motion ?xxx ?xxx)
          )
          (and
            (in_motion ?xxx)
            (in ?xxx)
          )
        )
        (is_setup_object ?xxx yellow)
      )
      (not
        (and
          (touch ?xxx rug)
          (on bed)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?s - ball)
      (and
        (preference preference1
          (exists (?r - cube_block)
            (then
              (once (and (in_motion ?s ?r) (not (not (object_orientation bed) ) ) ) )
              (hold (not (agent_holds ?r desk) ) )
              (once (in_motion ?r) )
            )
          )
        )
      )
    )
    (forall (?c - bridge_block)
      (and
        (preference preference2
          (exists (?h - block ?h - wall)
            (at-end
              (same_type ?h)
            )
          )
        )
      )
    )
  )
)
(:terminal
  (<= (count preference1:pink) (count-once-per-external-objects preference1:purple) )
)
(:scoring
  (count preference2:dodgeball)
)
)
(define (game game-id-703) (:domain many-objects-room-v1)
(:setup
  (and
    (exists (?n - ball)
      (and
        (exists (?z - hexagonal_bin ?f - watch)
          (game-conserved
            (in_motion ?f agent)
          )
        )
        (game-optional
          (not
            (and
              (on ?n)
              (agent_holds ?n)
            )
          )
        )
      )
    )
    (game-optional
      (in_motion ?xxx ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?a - dodgeball)
        (at-end
          (same_color top_shelf agent)
        )
      )
    )
  )
)
(:terminal
  (< 5 (or 2 2 10 ) )
)
(:scoring
  300
)
)
(define (game game-id-705) (:domain many-objects-room-v1)
(:setup
  (or
    (exists (?o - ball)
      (game-conserved
        (agent_holds pink_dodgeball pink)
      )
    )
    (game-optional
      (in_motion ?xxx ?xxx)
    )
  )
)
(:constraints
  (and
    (forall (?k - shelf)
      (and
        (preference preference1
          (exists (?l - game_object ?g - building)
            (then
              (hold (in_motion ?g) )
              (once (same_color ?k ?k) )
              (hold (not (agent_holds ?k) ) )
            )
          )
        )
        (preference preference2
          (exists (?t - dodgeball)
            (at-end
              (agent_holds ?k ?t)
            )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (external-forall-minimize 30 ) 2 )
  )
)
(:scoring
  (count preference2:basketball)
)
)
(define (game game-id-707) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (exists (?a - dodgeball ?s - teddy_bear ?o - game_object)
      (in_motion ?o)
    )
  )
)
(:constraints
  (and
    (forall (?o - dodgeball)
      (and
        (preference preference1
          (exists (?h - cylindrical_block)
            (at-end
              (not
                (and
                  (in ?o ?o)
                  (agent_holds ?h ?h)
                )
              )
            )
          )
        )
        (preference preference2
          (exists (?e - triangular_ramp ?a - cube_block)
            (at-end
              (and
                (equal_z_position ?a ?o)
                (in_motion ?o)
              )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 2 7 )
)
(:scoring
  (count preference1:bed)
)
)
(define (game game-id-737) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (or
      (object_orientation ?xxx)
      (in_motion ?xxx agent)
    )
  )
)
(:constraints
  (and
    (forall (?t - game_object)
      (and
        (preference preference1
          (exists (?r - hexagonal_bin ?w - (either cylindrical_block dodgeball))
            (then
              (once (<= 2 1) )
              (once (agent_holds ?t) )
              (once-measure (on bridge_block ?w ?t) (distance ?t 7) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (external-forall-maximize 100 ) (total-time) )
)
(:scoring
  (count-once preference1:dodgeball)
)
)
(define (game game-id-747) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (in ?xxx)
  )
)
(:constraints
  (and
    (forall (?u - hexagonal_bin)
      (and
        (preference preference1
          (exists (?s - wall)
            (at-end
              (agent_holds ?u ?s)
            )
          )
        )
      )
    )
  )
)
(:terminal
  (> (count-once-per-objects preference1:beachball) 2 )
)
(:scoring
  (count preference1:dodgeball)
)
)
(define (game game-id-763) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (and
      (on ?xxx ?xxx)
      (not
        (in_motion rug)
      )
      (adjacent ?xxx ?xxx)
    )
  )
)
(:constraints
  (and
    (forall (?a - hexagonal_bin ?c ?m - ball)
      (and
        (preference preference1
          (exists (?z - ball)
            (then
              (hold (on ?c) )
              (hold (agent_holds ?c) )
              (once (< 1 (distance ?z ?z)) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference1:red:blue_cube_block) (count preference1:red) )
)
(:scoring
  (count-once-per-objects preference1:block)
)
)
(define (game game-id-776) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (agent_holds ?xxx ?xxx)
  )
)
(:constraints
  (and
    (forall (?c ?y - dodgeball)
      (and
        (preference preference1
          (at-end
            (and
              (same_color ?c)
              (and
                (agent_holds ?y)
                (< 5 2)
              )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (<= 10 (* 3 3 )
  )
)
(:scoring
  (* (* 3 (count preference1:dodgeball) )
    (count-overlapping preference1:dodgeball)
  )
)
)
(define (game game-id-778) (:domain few-objects-room-v1)
(:setup
  (not
    (game-optional
      (on ?xxx)
    )
  )
)
(:constraints
  (and
    (forall (?k - dodgeball)
      (and
        (preference preference1
          (exists (?q - pillow ?t - building ?g - pillow)
            (then
              (once (agent_holds ?g) )
              (once (and (in ?g) (same_color block east_sliding_door) ) )
              (hold-while (and (agent_holds ?k) (and (< 1 (distance ?k ?g)) (in_motion ?g) (adjacent ?k) ) ) (agent_holds agent ?k) (in_motion ?k ?k) (agent_holds ?g ?g) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference1:dodgeball) (count-once-per-objects preference1:cube_block) )
)
(:scoring
  (count preference1:beachball)
)
)
(define (game game-id-809) (:domain many-objects-room-v1)
(:setup
  (forall (?p - game_object)
    (game-optional
      (exists (?v - cube_block)
        (touch ?v)
      )
    )
  )
)
(:constraints
  (and
    (forall (?y ?s - teddy_bear)
      (and
        (preference preference1
          (exists (?j - ball)
            (then
              (once (< 1 0.5) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (<= (count-once-per-objects preference1:pink) (count-once-per-external-objects preference1:beachball) )
)
(:scoring
  (count-once-per-objects preference1:golfball:yellow)
)
)
(define (game game-id-863) (:domain many-objects-room-v1)
(:setup
  (exists (?a - game_object)
    (and
      (game-optional
        (in ?a)
      )
      (game-optional
        (< (distance ?a 7) 10)
      )
    )
  )
)
(:constraints
  (and
    (forall (?f - ball)
      (and
        (preference preference1
          (exists (?o - wall)
            (then
              (hold (not (on ?o) ) )
              (once (touch ?f) )
              (once (not (agent_holds ?f ?o) ) )
            )
          )
        )
        (preference preference2
          (then
            (once (not (adjacent_side top_drawer) ) )
            (once (in_motion ?f bed ?f) )
            (hold (and (agent_holds ?f) (in ?f) ) )
            (once (on bed ?f) )
            (once (object_orientation ?f) )
          )
        )
      )
    )
    (preference preference3
      (exists (?k - pillow ?u - game_object)
        (then
          (once (in_motion ?u) )
          (once (in_motion ?u) )
          (once (agent_holds back) )
        )
      )
    )
  )
)
(:terminal
  (>= 10 (count-once preference2:green:dodgeball) )
)
(:scoring
  (count preference1:basketball:basketball:green)
)
)
(define (game game-id-897) (:domain few-objects-room-v1)
(:setup
  (exists (?z - teddy_bear)
    (exists (?s - beachball ?h - building ?h - chair ?x - dodgeball)
      (game-optional
        (not
          (on ?x)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?y - curved_wooden_ramp)
      (and
        (preference preference1
          (exists (?b - hexagonal_bin)
            (then
              (once (open ?y) )
              (hold (touch ?b) )
              (once (in_motion ?y) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 3 (+ 1 (count preference1:alarm_clock) )
  )
)
(:scoring
  (count-increasing-measure preference1:tall_cylindrical_block)
)
)
(define (game game-id-903) (:domain few-objects-room-v1)
(:setup
  (exists (?q - ball ?c - dodgeball)
    (and
      (exists (?a - doggie_bed)
        (exists (?z - dodgeball ?v - cube_block ?f - chair)
          (forall (?q - building)
            (exists (?w - hexagonal_bin)
              (game-optional
                (in bed)
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
      (exists (?i - dodgeball ?z - block)
        (then
          (once (< 5 (distance ?z ?z)) )
          (once (adjacent_side ?z) )
          (once (on pink_dodgeball) )
        )
      )
    )
  )
)
(:terminal
  (>= 50 5 )
)
(:scoring
  (count preference1:purple)
)
)
(define (game game-id-915) (:domain few-objects-room-v1)
(:setup
  (exists (?m - hexagonal_bin ?k - (either pencil basketball laptop))
    (game-optional
      (and
        (agent_holds ?k)
        (not
          (in_motion rug ?k)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?u - hexagonal_bin ?y - hexagonal_bin)
        (then
          (once (forall (?f - (either wall cylindrical_block) ?d - triangular_ramp) (agent_holds ?d) ) )
          (once (agent_holds ?y) )
        )
      )
    )
  )
)
(:terminal
  (>= 10 15 )
)
(:scoring
  5
)
)
(define (game game-id-954) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (in_motion ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?a - flat_block)
        (then
          (once (in_motion ?a) )
          (once (in_motion ?a floor) )
          (once (not (not (in_motion agent) ) ) )
        )
      )
    )
    (preference preference2
      (exists (?t - hexagonal_bin ?c - (either cube_block cellphone) ?v ?h - hexagonal_bin)
        (then
          (hold (agent_holds ?v bed) )
          (once (on ?v) )
          (once (same_object agent) )
        )
      )
    )
    (preference preference3
      (exists (?y - ball ?n - (either pyramid_block bridge_block))
        (then
          (once (in ?n front) )
          (hold (not (or (and (adjacent_side agent) ) (agent_holds ?n ?n) ) ) )
          (once (agent_holds ?n) )
        )
      )
    )
  )
)
(:terminal
  (>= 15 3 )
)
(:scoring
  (count preference1:pyramid_block)
)
)
(define (game game-id-976) (:domain medium-objects-room-v1)
(:setup
  (game-optional
    (agent_holds ?xxx ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?o - hexagonal_bin)
        (then
          (hold (is_setup_object rug) )
          (once (is_setup_object front) )
          (once (and (and (and (not (not (agent_holds ?o ?o) ) ) (in_motion ?o ?o) ) (between desk ?o) ) (agent_holds ?o ?o) (< 8 9) (in ?o) ) )
        )
      )
    )
  )
)
(:terminal
  (>= 3 10 )
)
(:scoring
  5
)
)
(define (game game-id-1001) (:domain many-objects-room-v1)
(:setup
  (not
    (forall (?s - (either laptop alarm_clock rug) ?q - ball)
      (game-conserved
        (agent_holds ?q)
      )
    )
  )
)
(:constraints
  (and
    (forall (?o ?x - doggie_bed ?y - building)
      (and
        (preference preference1
          (exists (?v - red_dodgeball)
            (then
              (hold (agent_holds ?v) )
              (once (agent_holds floor) )
              (once (and (on ?v) (exists (?r ?x - hexagonal_bin ?i - (either teddy_bear tall_cylindrical_block) ?r ?l - wall ?d - ball) (not (adjacent ?y ?y) ) ) ) )
            )
          )
        )
      )
    )
    (preference preference2
      (exists (?e - ball ?b - wall)
        (then
          (hold (and (agent_holds desk pink_dodgeball) (touch agent) (in_motion agent ?b) (agent_holds ?b agent) (in_motion ?b bed) (in_motion ?b) ) )
          (hold (on ?b) )
          (once (< (distance 1 ?b) 3) )
        )
      )
    )
    (preference preference3
      (then
        (hold (agent_holds ?xxx ?xxx) )
        (once (agent_holds top_shelf) )
        (once (agent_holds ?xxx) )
      )
    )
  )
)
(:terminal
  (>= (external-forall-maximize 4 ) (count preference1:purple) )
)
(:scoring
  (count preference2:blue_dodgeball)
)
)
(define (game game-id-1030) (:domain many-objects-room-v1)
(:setup
  (game-optional
    (agent_holds ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?b ?g ?q ?a ?k ?n - dodgeball)
        (at-end
          (and
            (or
              (and
                (in_motion ?q)
                (same_object ?k ?b)
              )
            )
            (in_motion ?n)
            (and
              (same_color floor ?b)
              (on ?b)
              (agent_holds ?q right)
            )
            (and
              (same_color ?b rug)
              (not
                (on ?b)
              )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 8 5 )
)
(:scoring
  (count preference1)
)
)
(define (game game-id-1051) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (same_color ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?q ?h - building)
        (then
          (once (in ?q agent) )
          (once (in_motion ?q) )
          (hold-to-end (in_motion ?q) )
        )
      )
    )
  )
)
(:terminal
  (>= 5 30 )
)
(:scoring
  (count preference1:pink)
)
)
(define (game game-id-1058) (:domain few-objects-room-v1)
(:setup
  (exists (?i - beachball)
    (game-optional
      (agent_holds ?i)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?z - building)
        (then
          (once (in ?z ?z) )
          (once (above ?z ?z) )
          (once (agent_holds ?z) )
        )
      )
    )
    (preference preference2
      (exists (?p - (either hexagonal_bin basketball dodgeball) ?l - hexagonal_bin)
        (then
          (once (on ?l agent) )
          (hold (and (on ?l ?l) (or (touch ?l ?l ?l) (on ?l) ) ) )
          (once (touch agent pink_dodgeball) )
        )
      )
    )
    (preference preference3
      (exists (?f - game_object)
        (then
          (hold (in_motion ?f ?f) )
          (once (not (not (agent_holds ?f ?f ?f) ) ) )
        )
      )
    )
  )
)
(:terminal
  (>= 3 5 )
)
(:scoring
  (- (count preference1:beachball) )
)
)
(define (game game-id-1121) (:domain many-objects-room-v1)
(:setup
  (and
    (and
      (and
        (game-conserved
          (and
            (exists (?y - wall)
              (on rug)
            )
            (agent_holds floor)
          )
        )
      )
    )
    (game-conserved
      (adjacent_side ?xxx)
    )
  )
)
(:constraints
  (and
    (forall (?j - wall ?b - game_object)
      (and
        (preference preference1
          (exists (?f - triangular_ramp)
            (then
              (once (not (in_motion ?f) ) )
              (once (touch ?f ?b) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (not
    (>= 2 3 )
  )
)
(:scoring
  (count-once-per-objects preference1:block)
)
)
(define (game game-id-1125) (:domain medium-objects-room-v1)
(:setup
  (and
    (and
      (not
        (game-optional
          (not
            (on door ?xxx ?xxx)
          )
        )
      )
      (exists (?p - dodgeball ?d ?w - color)
        (game-conserved
          (in_motion ?d ?d)
        )
      )
      (game-conserved
        (and
          (and
            (and
              (in ?xxx ?xxx)
              (between ?xxx)
            )
            (touch yellow ?xxx)
          )
          (agent_holds bed ?xxx)
        )
      )
    )
    (game-conserved
      (is_setup_object ?xxx ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?x - teddy_bear)
        (then
          (hold (in_motion desk ?x) )
          (once (agent_holds ?x) )
          (once (agent_holds ?x) )
          (once (not (in_motion ?x ?x) ) )
          (once (in ?x) )
          (once (in_motion ?x) )
        )
      )
    )
    (preference preference2
      (exists (?e - curved_wooden_ramp)
        (then
          (hold (on ?e ?e) )
          (hold-while (agent_holds ?e) (not (= 8) ) (not (and (on ?e agent) (agent_holds ?e ?e) ) ) )
          (hold-while (agent_holds floor) (on ?e) )
        )
      )
    )
  )
)
(:terminal
  (< 3 2 )
)
(:scoring
  (- 3 )
)
)
(define (game game-id-1127) (:domain many-objects-room-v1)
(:setup
  (or
    (exists (?j - ball)
      (forall (?b - (either blue_cube_block dodgeball))
        (and
          (game-conserved
            (< 1 1)
          )
          (forall (?y - flat_block ?s - block)
            (and
              (game-conserved
                (on ?b)
              )
              (forall (?e - ball ?u - shelf)
                (game-optional
                  (not
                    (agent_holds ?s)
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
    (forall (?x - pillow ?p - building)
      (and
        (preference preference1
          (exists (?z - ball)
            (at-end
              (in_motion ?p)
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 9 30 )
)
(:scoring
  (count preference1:pink_dodgeball)
)
)
(define (game game-id-1134) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (not
      (agent_holds ?xxx ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?x - ball)
        (at-end
          (and
            (agent_holds ?x ?x)
            (< (distance room_center ?x) 0)
          )
        )
      )
    )
    (preference preference2
      (exists (?k ?b ?c ?w - ball ?c - curved_wooden_ramp)
        (then
          (once (in_motion pink_dodgeball) )
          (hold (or (agent_holds ?c ?c) (adjacent ?c) (and (and (in ?c ?c) (in_motion ?c) ) (and (adjacent ?c ?c) (in ?c) (< (distance ?c 4 ?c) 0.5) (not (in_motion ?c floor) ) ) (faces ?c) ) (not (adjacent ?c) ) ) )
          (hold (< 1 (distance ?c ?c)) )
        )
      )
    )
  )
)
(:terminal
  (= 3 (<= 2 3 )
  )
)
(:scoring
  5
)
)
(define (game game-id-1136) (:domain few-objects-room-v1)
(:setup
  (exists (?v - bridge_block)
    (game-conserved
      (agent_holds ?v)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?g - (either yellow_cube_block teddy_bear))
        (at-end
          (in_motion ?g ?g)
        )
      )
    )
  )
)
(:terminal
  (>= 1 5 )
)
(:scoring
  (* (count-once-per-objects preference1:rug) 30 (count-same-positions preference1:yellow:purple) )
)
)
(define (game game-id-1146) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (in ?xxx south_west_corner)
  )
)
(:constraints
  (and
    (forall (?r - game_object ?p - hexagonal_bin)
      (and
        (preference preference1
          (exists (?o - tall_cylindrical_block)
            (then
              (hold (not (agent_holds ?p ?p) ) )
              (once (agent_holds ?o ?p) )
              (once (on side_table) )
              (hold-while (in_motion ?o ?p) (in_motion ?o south_west_corner) (in_motion ?o ?p) (in_motion ?o) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (< 10 40 )
)
(:scoring
  (count-once preference1:basketball)
)
)
(define (game game-id-1154) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (not
      (in_motion rug)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?o - hexagonal_bin)
        (then
          (once (and (< (distance ) 1) (rug_color_under ?o) ) )
          (hold (and (in_motion ?o desk) (not (in_motion agent) ) ) )
          (once (on ?o) )
        )
      )
    )
    (preference preference2
      (exists (?r - cube_block ?t - desk_shelf)
        (then
          (once (in agent) )
          (once (and (in_motion desktop ?t) (same_color ?t) ) )
        )
      )
    )
  )
)
(:terminal
  (>= 300 (count-once-per-external-objects preference1) )
)
(:scoring
  8
)
)
(define (game game-id-1189) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (above ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?w - (either golfball dodgeball))
        (then
          (hold-while (touch ?w) (or (in_motion ?w) ) (and (in_motion ?w) (object_orientation top_shelf rug) (in ?w) (not (in_motion ?w ?w) ) (not (and (adjacent_side ?w) (agent_holds ?w ?w) ) ) (in_motion ?w) (in_motion ?w) ) )
          (once (adjacent ?w) )
          (once (and (not (in_motion ?w desk) ) (on ?w) ) )
        )
      )
    )
    (forall (?j - ball ?o - hexagonal_bin ?w - beachball)
      (and
        (preference preference2
          (exists (?n - sliding_door)
            (then
              (hold-to-end (same_color ?w) )
              (hold (exists (?b - (either cylindrical_block desktop) ?x - hexagonal_bin) (in_motion ?w ?x) ) )
              (hold (object_orientation ?n) )
            )
          )
        )
      )
    )
    (preference preference3
      (exists (?t - hexagonal_bin ?w - wall)
        (at-end
          (not
            (< (distance room_center room_center) 1)
          )
        )
      )
    )
  )
)
(:terminal
  (<= 4 (count preference2:red:blue_cube_block) )
)
(:scoring
  (* (count preference2:beachball:red) 15 )
)
)
(define (game game-id-1197) (:domain few-objects-room-v1)
(:setup
  (and
    (forall (?y - cube_block)
      (exists (?s - color)
        (exists (?z - dodgeball)
          (game-conserved
            (and
              (and
                (in_motion agent ?y)
                (in ?y ?z)
              )
              (in_motion ?s)
            )
          )
        )
      )
    )
    (exists (?x - hexagonal_bin ?e - dodgeball)
      (game-conserved
        (agent_holds ?e ?e)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?v - (either block bed) ?c - (either golfball game_object golfball pillow key_chain))
        (then
          (hold-while (opposite ?c ?c) (in_motion bed) )
          (hold-for 9 (in_motion agent) )
          (once (agent_holds ?c) )
        )
      )
    )
  )
)
(:terminal
  (>= 6 2 )
)
(:scoring
  (* 1 3 )
)
)
(define (game game-id-1200) (:domain medium-objects-room-v1)
(:setup
  (and
    (exists (?y - block)
      (game-conserved
        (in_motion ?y ?y)
      )
    )
    (game-optional
      (adjacent desk ?xxx)
    )
    (and
      (forall (?k - chair)
        (exists (?z - curved_wooden_ramp)
          (forall (?t - (either pink pillow))
            (game-conserved
              (equal_z_position ?k)
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
      (exists (?d - hexagonal_bin)
        (then
          (once (agent_holds top_shelf) )
          (any)
          (once (adjacent ?d) )
        )
      )
    )
  )
)
(:terminal
  (>= (count-once-per-objects preference1:hexagonal_bin) 2 )
)
(:scoring
  (count preference1:blue_cube_block)
)
)
(define (game game-id-1204) (:domain medium-objects-room-v1)
(:setup
  (and
    (forall (?z - (either dodgeball))
      (game-conserved
        (agent_holds ?z ?z)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?k ?y - hexagonal_bin)
        (at-end
          (adjacent ?k)
        )
      )
    )
    (preference preference2
      (exists (?j - tall_cylindrical_block)
        (then
          (once (adjacent ?j) )
          (once (in ?j) )
          (once (on ?j) )
        )
      )
    )
  )
)
(:terminal
  (> 2 (external-forall-maximize 10 ) )
)
(:scoring
  (external-forall-minimize
    6
  )
)
)
(define (game game-id-1226) (:domain few-objects-room-v1)
(:setup
  (forall (?p ?m - hexagonal_bin)
    (game-conserved
      (and
        (not
          (in ?p pink)
        )
        (in_motion ?m ?m)
      )
    )
  )
)
(:constraints
  (and
    (forall (?o - (either cellphone dodgeball wall) ?i - hexagonal_bin)
      (and
        (preference preference1
          (exists (?j - dodgeball ?f - watch)
            (at-end
              (agent_holds agent)
            )
          )
        )
      )
    )
    (preference preference2
      (exists (?t - doggie_bed ?b - dodgeball ?y - ball)
        (then
          (hold (not (not (in_motion ?y ?y) ) ) )
          (hold (and (exists (?g - pyramid_block) (agent_holds ?g) ) (and (agent_holds ?y floor) (object_orientation ?y ?y) ) ) )
        )
      )
    )
  )
)
(:terminal
  (>= 3 (count-once-per-objects preference1:golfball) )
)
(:scoring
  (count preference1:beachball)
)
)
(define (game game-id-1268) (:domain few-objects-room-v1)
(:setup
  (forall (?e - doggie_bed)
    (game-optional
      (and
        (in ?e)
        (< (distance ?e room_center) 2)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?j - yellow_cube_block)
        (then
          (hold (adjacent ?j) )
          (hold (same_type ?j ?j) )
          (once (in ?j ?j) )
          (hold-while (same_type ?j ?j) (agent_holds ?j ?j) )
          (once (adjacent green_golfball) )
        )
      )
    )
    (preference preference2
      (then
        (once (= 1) )
        (once (not (agent_holds ?xxx ?xxx) ) )
        (hold (not (in_motion ?xxx) ) )
      )
    )
  )
)
(:terminal
  (>= 10 2 )
)
(:scoring
  20
)
)
(define (game game-id-1331) (:domain few-objects-room-v1)
(:setup
  (forall (?s - curved_wooden_ramp)
    (not
      (forall (?l - building ?m - color)
        (forall (?n - cube_block ?k - hexagonal_bin)
          (and
            (exists (?z - block)
              (not
                (game-conserved
                  (on ?z ?m)
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
    (forall (?q - curved_wooden_ramp ?l ?a - dodgeball ?a ?u - hexagonal_bin ?z - chair)
      (and
        (preference preference1
          (exists (?u - game_object)
            (at-end
              (not
                (not
                  (not
                    (> 5 (distance bed door agent))
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
  (>= (* 3 3 )
    (* (count-unique-positions preference1:hexagonal_bin) (count preference1:beachball) )
  )
)
(:scoring
  2
)
)
(define (game game-id-1350) (:domain many-objects-room-v1)
(:setup
  (exists (?t - hexagonal_bin)
    (forall (?d - desk_shelf)
      (game-optional
        (on ?d)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?y - block)
        (then
          (hold-for 5 (and (exists (?o - triangular_ramp ?n - dodgeball ?u - hexagonal_bin) (on ?y ?u) ) (and (and (in_motion ?y ?y) (in_motion ?y ?y) ) (not (in_motion ?y) ) ) ) )
          (hold (and (or (in_motion ?y bed) (in_motion ?y ?y) ) (and (not (not (rug_color_under ?y ?y) ) ) (and (not (not (same_type rug agent) ) ) ) ) (agent_holds ?y) ) )
          (once (< 1 1) )
        )
      )
    )
    (forall (?p - hexagonal_bin)
      (and
        (preference preference2
          (exists (?e - shelf)
            (then
              (once (not (on ?p) ) )
              (once (agent_holds ?e) )
              (once (agent_holds rug ?e) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count-increasing-measure preference2:golfball) (count preference2:red) )
)
(:scoring
  (+ (* 10 9 )
    25
  )
)
)
(define (game game-id-1391) (:domain many-objects-room-v1)
(:setup
  (exists (?k - tall_cylindrical_block)
    (and
      (game-conserved
        (not
          (agent_holds ?k ?k)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?v - cylindrical_block ?t - golfball)
      (and
        (preference preference1
          (exists (?l - dodgeball)
            (then
              (once (agent_holds ?t block) )
              (any)
              (once (adjacent agent) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference1:red) (count preference1:rug) )
)
(:scoring
  (* (count-increasing-measure preference1:golfball) (count preference1:golfball:blue_pyramid_block) )
)
)
(define (game game-id-1403) (:domain few-objects-room-v1)
(:setup
  (game-optional
    (agent_holds ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?j - red_dodgeball)
        (then
          (hold (on ?j) )
          (once (in_motion ?j) )
          (hold (object_orientation ?j ?j) )
        )
      )
    )
  )
)
(:terminal
  (> 5 (total-time) )
)
(:scoring
  (* (count preference1) 5 )
)
)
(define (game game-id-1414) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (same_color )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?f - cube_block)
        (at-end
          (and
            (on ?f)
            (not
              (open floor ?f rug)
            )
          )
        )
      )
    )
    (preference preference2
      (exists (?r - wall)
        (then
          (hold (not (not (agent_holds ?r) ) ) )
          (once (not (and (in ?r) (and (agent_holds rug) (< (x_position ?r 9) (distance )) ) ) ) )
          (once (and (same_color ?r bed) (and (not (in_motion ?r ?r) ) (not (not (in ?r ?r) ) ) ) ) )
        )
      )
    )
  )
)
(:terminal
  (>= (- (count-once preference2) )
    (total-time)
  )
)
(:scoring
  3
)
)
(define (game game-id-1416) (:domain few-objects-room-v1)
(:setup
  (forall (?p - hexagonal_bin ?z - (either cube_block bridge_block))
    (game-conserved
      (not
        (agent_holds right)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?l - game_object ?q - (either cube_block key_chain) ?n - triangular_ramp)
        (at-end
          (exists (?r - hexagonal_bin ?s - hexagonal_bin)
            (agent_holds ?n)
          )
        )
      )
    )
  )
)
(:terminal
  (and
    (>= 16 300 )
  )
)
(:scoring
  10
)
)
(define (game game-id-1425) (:domain medium-objects-room-v1)
(:setup
  (and
    (exists (?j - (either floor pink laptop))
      (forall (?n - hexagonal_bin)
        (exists (?h - dodgeball)
          (game-conserved
            (and
              (adjacent ?h)
              (exists (?z - color ?o - block)
                (in ?h ?n)
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
    (forall (?q - curved_wooden_ramp)
      (and
        (preference preference1
          (exists (?s - (either tall_cylindrical_block laptop))
            (then
              (hold (< 4 (distance ?q ?q)) )
              (once (in ?s) )
              (hold (not (not (agent_holds ?s ?q) ) ) )
            )
          )
        )
        (preference preference2
          (exists (?r - dodgeball ?v - game_object ?d - dodgeball ?h - (either cube_block flat_block))
            (then
              (hold (in_motion ?q) )
              (once (not (< 1 (distance room_center)) ) )
              (once (opposite ?h ?h) )
            )
          )
        )
      )
    )
    (forall (?m ?n ?g - dodgeball)
      (and
        (preference preference3
          (exists (?b - cube_block ?v - (either cylindrical_block golfball) ?s - dodgeball)
            (then
              (once (agent_holds ?s) )
              (once (agent_holds desk ?m) )
              (once (agent_holds ?n) )
            )
          )
        )
      )
    )
    (preference preference4
      (then
        (once (in_motion ?xxx front) )
        (once (not (touch ?xxx bridge_block) ) )
        (hold (> 7 (distance ?xxx agent)) )
      )
    )
  )
)
(:terminal
  (<= (count preference2:dodgeball:hexagonal_bin) 1 )
)
(:scoring
  3
)
)
(define (game game-id-1427) (:domain few-objects-room-v1)
(:setup
  (and
    (forall (?b ?m ?i - wall)
      (game-conserved
        (agent_holds ?i)
      )
    )
  )
)
(:constraints
  (and
    (forall (?b - dodgeball)
      (and
        (preference preference1
          (exists (?p - (either key_chain teddy_bear key_chain))
            (then
              (hold (in_motion ?b) )
              (hold (not (< 5 3) ) )
              (once (not (exists (?l - cylindrical_block ?h - building) (< (distance ?p agent) (distance desk door)) ) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (= (count preference1:green) (count preference1:red) )
)
(:scoring
  (count-once-per-objects preference1:red)
)
)
(define (game game-id-1444) (:domain many-objects-room-v1)
(:setup
  (and
    (forall (?f - hexagonal_bin)
      (exists (?j - dodgeball ?u - hexagonal_bin)
        (game-optional
          (not
            (not
              (not
                (agent_holds ?u ?u)
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
      (exists (?k - hexagonal_bin ?v - shelf ?u - ball)
        (then
          (once (and (same_color ?u) (agent_holds ?u agent) ) )
          (hold-while (and (not (and (agent_holds ?u) (touch bed) (and (open ?u ?u) (<= (distance 5 door) 3) ) ) ) (touch ?u ?u) ) (agent_holds ?u) (exists (?f - dodgeball ?v - game_object) (not (opposite ?v) ) ) )
          (once (and (same_type ?u) (on ?u) ) )
        )
      )
    )
    (preference preference2
      (exists (?y - hexagonal_bin ?k - game_object)
        (then
          (hold-while (< 9 (distance 8 3)) (agent_holds ?k ?k) )
          (once (not (in ?k) ) )
          (hold (agent_holds ?k ?k) )
        )
      )
    )
  )
)
(:terminal
  (>= 10 4 )
)
(:scoring
  (count-once preference2:pink)
)
)
(define (game game-id-1488) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (agent_holds ?xxx bed)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?d - cube_block)
        (at-end
          (exists (?a - block ?s - (either dodgeball cube_block) ?m - ball)
            (object_orientation ?d)
          )
        )
      )
    )
  )
)
(:terminal
  (>= 3 6 )
)
(:scoring
  (* 10 (count-shortest preference1:pink) )
)
)
(define (game game-id-1502) (:domain few-objects-room-v1)
(:setup
  (forall (?r - hexagonal_bin ?e - hexagonal_bin)
    (game-conserved
      (not
        (forall (?z - hexagonal_bin)
          (is_setup_object ?e)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?h - hexagonal_bin ?m - block)
        (then
          (hold (in ?m bed) )
          (once (touch ?m ?m) )
          (once (in_motion ?m bed) )
        )
      )
    )
  )
)
(:terminal
  (or
    (> 7 0 )
  )
)
(:scoring
  2
)
)
(define (game game-id-1517) (:domain many-objects-room-v1)
(:setup
  (exists (?g - chair)
    (exists (?v - game_object ?i - (either cylindrical_block dodgeball))
      (exists (?u - (either pen beachball))
        (forall (?y ?f ?e ?n - hexagonal_bin ?b - ball ?f - chair)
          (and
            (game-conserved
              (in_motion ?i)
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?a - flat_block ?y - game_object)
      (and
        (preference preference1
          (exists (?n - ball)
            (at-end
              (adjacent ?y)
            )
          )
        )
        (preference preference2
          (exists (?b ?o ?u - dodgeball)
            (then
              (once (agent_holds ?b) )
              (hold (agent_holds ?o ?o) )
              (hold (agent_holds ?b) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (<= (count preference1:doggie_bed) (count-once-per-external-objects preference1:basketball) )
)
(:scoring
  (or
    6
  )
)
)
(define (game game-id-1565) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (in_motion ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (hold (and (not (agent_holds ?xxx top_shelf) ) (not (and (= 0 (distance ?xxx ?xxx)) (on ?xxx ?xxx) ) ) ) )
        (once (open ?xxx) )
        (once (not (exists (?s - ball) (above ?s ?s) ) ) )
      )
    )
    (preference preference2
      (exists (?z - (either blue_cube_block cylindrical_block) ?d - (either basketball doggie_bed) ?o - curved_wooden_ramp)
        (then
          (hold (agent_holds ?o front) )
          (once (in_motion ?o ?o) )
          (once (on ?o ?o) )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= 4 3 )
  )
)
(:scoring
  3
)
)
(define (game game-id-1570) (:domain medium-objects-room-v1)
(:setup
  (exists (?t - wall ?a - pyramid_block)
    (game-conserved
      (and
        (agent_holds ?a)
        (and
          (in_motion ?a)
          (not
            (on ?a ?a)
          )
        )
        (on ?a ?a)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?q - dodgeball)
        (then
          (once (adjacent_side ?q) )
          (once (agent_holds ?q) )
          (once (not (in_motion blue) ) )
        )
      )
    )
  )
)
(:terminal
  (>= 5 30 )
)
(:scoring
  3
)
)
(define (game game-id-1606) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (exists (?n - hexagonal_bin ?l - dodgeball)
      (in_motion ?l)
    )
  )
)
(:constraints
  (and
    (forall (?h - chair ?a ?k ?g ?p ?m - dodgeball)
      (and
        (preference preference1
          (exists (?r - game_object)
            (then
              (once (and (same_color ?r) (or (and (same_color ?k) (and (< 1 5) (agent_holds ?a) ) ) (and (in_motion ?a ?g) (in ?k ?a) ) (in_motion ?m) ) ) )
              (once (and (in_motion ?r ?m) (agent_holds ?k) ) )
              (once (object_orientation ?p) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 6 2 )
)
(:scoring
  (> (count preference1:hexagonal_bin) (count-once preference1:blue_dodgeball:beachball) )
)
)
(define (game game-id-1612) (:domain few-objects-room-v1)
(:setup
  (exists (?v - (either golfball laptop))
    (and
      (game-conserved
        (agent_holds bed green_golfball)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?o - cube_block)
        (then
          (once (not (not (agent_holds sideways ?o) ) ) )
        )
      )
    )
    (forall (?a - (either dodgeball dodgeball))
      (and
        (preference preference2
          (exists (?j ?z - wall)
            (then
              (once (agent_holds ?j ?j) )
              (once (not (and (agent_holds ?j) (in_motion ?j) ) ) )
              (hold (and (not (agent_holds ?a) ) (and (agent_holds ?j bed) (on ?a) ) (in_motion ?a west_wall) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count-once preference2:beachball:basketball) (count preference2:basketball) )
)
(:scoring
  180
)
)
(define (game game-id-1631) (:domain few-objects-room-v1)
(:setup
  (exists (?i ?j - hexagonal_bin ?q - doggie_bed ?b - curved_wooden_ramp)
    (exists (?j - (either book cellphone) ?e - hexagonal_bin)
      (and
        (game-optional
          (touch ?e)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?p - game_object)
        (at-end
          (and
            (and
              (agent_holds ?p)
              (in_motion ?p)
            )
          )
        )
      )
    )
  )
)
(:terminal
  (<= 3 5 )
)
(:scoring
  6
)
)
(define (game game-id-1647) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (not
      (adjacent_side main_light_switch)
    )
  )
)
(:constraints
  (and
    (forall (?v - cube_block)
      (and
        (preference preference1
          (exists (?u - dodgeball ?l - game_object)
            (then
              (once-measure (and (in_motion ?l ?l) (touch ?l) ) (distance ?l ?v) )
              (hold (on ?v) )
              (once (agent_holds ?l ?v) )
            )
          )
        )
        (preference preference2
          (exists (?k - hexagonal_bin ?k - hexagonal_bin)
            (then
              (once (adjacent ?k) )
              (once-measure (adjacent ?v) (distance ?v ?k) )
              (forall-sequence (?n - cube_block)
                (then
                  (once (in ?n ?v) )
                  (once (in_motion ?v) )
                  (once (in_motion rug) )
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
  (>= (+ 5 (count-once-per-objects preference2:beachball) )
    (- 5 )
  )
)
(:scoring
  (external-forall-maximize
    (count-once-per-objects preference1:golfball:dodgeball)
  )
)
)
(define (game game-id-1657) (:domain medium-objects-room-v1)
(:setup
  (game-optional
    (on bed)
  )
)
(:constraints
  (and
    (forall (?g ?r ?n - dodgeball ?q - ball)
      (and
        (preference preference1
          (exists (?u ?p ?z - dodgeball)
            (then
              (once (agent_holds ?q agent) )
            )
          )
        )
      )
    )
    (forall (?k - dodgeball)
      (and
        (preference preference2
          (exists (?p - (either cube_block alarm_clock flat_block wall basketball top_drawer yellow_cube_block) ?r - hexagonal_bin)
            (at-end
              (and
                (in_motion ?r ?k)
                (< 1 1)
              )
            )
          )
        )
      )
    )
    (preference preference3
      (exists (?y - building)
        (then
          (once (on ?y) )
          (hold (and (in_motion ?y) (same_object ?y ?y) ) )
          (once (on ?y ?y) )
        )
      )
    )
  )
)
(:terminal
  (and
    (>= 0.5 (count-overlapping preference2:pink_dodgeball) )
  )
)
(:scoring
  5
)
)
(define (game game-id-1696) (:domain many-objects-room-v1)
(:setup
  (exists (?t - color ?t - hexagonal_bin)
    (and
      (game-conserved
        (not
          (not
            (exists (?p - dodgeball ?v - (either bridge_block alarm_clock))
              (in_motion agent ?v)
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?b - building ?x - ball ?n - beachball)
      (and
        (preference preference1
          (exists (?s - (either dodgeball key_chain cylindrical_block))
            (then
              (once (agent_holds ?s ?s) )
              (once (on ?n ?n) )
              (hold (in_motion ?n ?n) )
              (once (agent_holds ?n) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (- 1 )
    2
  )
)
(:scoring
  (count-once-per-objects preference1:dodgeball)
)
)
(define (game game-id-1698) (:domain medium-objects-room-v1)
(:setup
  (exists (?o - hexagonal_bin ?o - drawer)
    (exists (?t - doggie_bed)
      (exists (?d - desk_shelf)
        (forall (?l - ball ?e - hexagonal_bin ?u - dodgeball)
          (game-optional
            (in agent ?o)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?k - chair ?p - (either triangular_ramp pyramid_block))
      (and
        (preference preference1
          (exists (?y - hexagonal_bin)
            (at-end
              (not
                (in ?p ?y)
              )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (not
    (= 2 10 )
  )
)
(:scoring
  (/
    (count-once-per-objects preference1:orange:green)
    3
  )
)
)
(define (game game-id-1767) (:domain medium-objects-room-v1)
(:setup
  (and
    (exists (?x - ball ?x - teddy_bear ?l - (either cylindrical_block cube_block beachball) ?r - (either dodgeball pillow))
      (forall (?f - chair)
        (exists (?x - pillow)
          (and
            (exists (?z - hexagonal_bin ?q - (either main_light_switch golfball) ?n - cube_block ?w - red_dodgeball)
              (game-optional
                (agent_holds ?f ?r)
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
    (forall (?t - dodgeball ?z ?c - cube_block ?n - (either mug ball))
      (and
        (preference preference1
          (exists (?q - ball)
            (then
              (once (touch floor) )
            )
          )
        )
      )
    )
    (preference preference2
      (exists (?u - shelf)
        (then
          (once (and (in_motion floor ?u ?u) (same_type ?u) (and (in ?u ?u) (agent_holds ?u ?u) ) (on desk ?u) (same_object ?u ?u) (agent_holds ?u) ) )
          (once (not (not (adjacent ?u) ) ) )
          (hold (in_motion ?u) )
        )
      )
    )
  )
)
(:terminal
  (> (* (count-once-per-objects preference1:triangle_block) (count-once-per-objects preference1:golfball) )
    2
  )
)
(:scoring
  (count-once-per-objects preference1:dodgeball:green)
)
)
(define (game game-id-1783) (:domain many-objects-room-v1)
(:setup
  (exists (?s - cube_block)
    (game-optional
      (and
        (and
          (agent_holds ?s)
          (not
            (not
              (> 2 1)
            )
          )
        )
        (not
          (agent_holds ?s ?s)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?s - cube_block ?x - (either pyramid_block hexagonal_bin cylindrical_block desktop desktop cube_block mug))
        (then
          (once (and (on ?x) (same_type front) ) )
          (once (agent_holds ?x ?x) )
          (hold (= (distance ?x ?x ?x) 1) )
        )
      )
    )
    (preference preference2
      (exists (?q - chair)
        (at-end
          (in_motion ?q)
        )
      )
    )
    (preference preference3
      (exists (?p - hexagonal_bin ?t - hexagonal_bin)
        (then
          (once (not (< (distance door desk) 1) ) )
          (once (in_motion ?t ?t) )
          (hold (= (distance desk room_center) 10) )
        )
      )
    )
  )
)
(:terminal
  (>= 3 (+ 6 15 )
  )
)
(:scoring
  5
)
)
(define (game game-id-1786) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (agent_holds ?xxx ?xxx)
  )
)
(:constraints
  (and
    (forall (?m - (either cd key_chain book))
      (and
        (preference preference1
          (exists (?u - cube_block)
            (then
              (hold-while (> (x_position ?u 2) 2) (agent_holds ?m ?m) (in ?u ?m) )
              (hold (above ?u) )
              (once (not (in ?m) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (< (count-once-per-objects preference1:purple) (count preference1:beachball:pink_dodgeball:doggie_bed) )
)
(:scoring
  60
)
)
(define (game game-id-1830) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (agent_holds agent)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?p - hexagonal_bin ?v - (either dodgeball cellphone))
        (then
          (once (in_motion ?v ?v) )
          (once (< 5 1) )
          (once (in_motion ?v) )
        )
      )
    )
    (preference preference2
      (exists (?v - dodgeball)
        (then
          (once (agent_holds ?v ?v) )
          (once (agent_holds ?v ?v) )
          (once (in_motion ?v upright) )
        )
      )
    )
  )
)
(:terminal
  (>= 1 100 )
)
(:scoring
  (* (total-score) (not (total-time) ) (count-once-per-external-objects preference2:pink_dodgeball:pink_dodgeball) (count-once-per-objects preference1:yellow) (count preference1:basketball) (count preference1:golfball) (count-overlapping preference2:blue_dodgeball:blue_cube_block) )
)
)
(define (game game-id-1833) (:domain medium-objects-room-v1)
(:setup
  (and
    (exists (?y - hexagonal_bin)
      (exists (?u - building)
        (forall (?j - block ?r - ball)
          (and
            (not
              (game-conserved
                (agent_holds ?u ?r)
              )
            )
            (exists (?k - game_object ?e - game_object)
              (exists (?k - (either hexagonal_bin alarm_clock golfball))
                (game-conserved
                  (and
                    (in_motion ?u ?u)
                    (agent_holds ?r)
                    (not
                      (exists (?x - hexagonal_bin ?x - hexagonal_bin)
                        (not
                          (in_motion ?r)
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
      (exists (?u - shelf)
        (then
          (once (in ?u ?u) )
          (hold-while (in_motion ?u sideways) (and (in ?u) (not (and (agent_holds ?u) (on ?u ?u) (agent_holds ?u ?u) ) ) ) )
          (once (agent_holds ?u ?u) )
          (once (and (in_motion ?u ?u) (in_motion ?u) ) )
          (once (touch ?u) )
        )
      )
    )
  )
)
(:terminal
  (>= (total-time) 2 )
)
(:scoring
  12
)
)
(define (game game-id-1886) (:domain few-objects-room-v1)
(:setup
  (and
    (forall (?p - cylindrical_block)
      (forall (?n ?t ?q - color ?z - block ?d - teddy_bear)
        (game-conserved
          (on desk)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?x - desk_shelf)
        (at-end
          (< (distance ?x) 2)
        )
      )
    )
    (forall (?r - golfball)
      (and
        (preference preference2
          (exists (?c - dodgeball)
            (then
              (hold (and (in_motion rug ?r) (object_orientation ?r ?r) ) )
              (once (agent_holds ?r) )
              (once (in_motion ?c ?c) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (* (count preference2:dodgeball:dodgeball) )
    (count preference1:pink_dodgeball:golfball)
  )
)
(:scoring
  (count preference2:dodgeball)
)
)
(define (game game-id-1907) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (on ?xxx)
  )
)
(:constraints
  (and
    (forall (?z - game_object)
      (and
        (preference preference1
          (exists (?b - wall)
            (then
              (once (agent_holds ?z ?z) )
              (once (agent_holds ?b) )
              (hold (on ?b) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (not
    (or
      (>= (or 5 ) 1 )
    )
  )
)
(:scoring
  (* (count preference1:dodgeball) 15 )
)
)
(define (game game-id-1908) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (touch ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?j - dodgeball)
        (then
          (once (not (and (on ?j) (not (on ?j) ) ) ) )
          (hold (same_color ?j pink_dodgeball) )
          (once (agent_holds ?j) )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= 9 100 )
    (>= 3 4 )
  )
)
(:scoring
  3
)
)
(define (game game-id-1961) (:domain medium-objects-room-v1)
(:setup
  (and
    (game-conserved
      (not
        (not
          (in ?xxx ?xxx)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?z - pyramid_block)
      (and
        (preference preference1
          (exists (?u - ball)
            (at-end
              (on rug)
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 10 3 )
)
(:scoring
  (count preference1:yellow)
)
)
(define (game game-id-2019) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (on ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?v - hexagonal_bin)
        (then
          (once (agent_holds ) )
          (once (open ?v) )
        )
      )
    )
    (preference preference2
      (exists (?g - (either bed ball))
        (then
          (once (and (in ?g ?g) (agent_holds ?g) ) )
          (once (in_motion ?g ?g) )
          (once (not (in_motion ?g ?g) ) )
        )
      )
    )
    (preference preference3
      (exists (?a - hexagonal_bin)
        (then
          (once (exists (?b - triangular_ramp ?u - hexagonal_bin ?v - chair) (not (= 1 1) ) ) )
          (once (not (on ?a) ) )
          (hold (agent_holds agent) )
        )
      )
    )
    (forall (?r - (either yellow_cube_block dodgeball))
      (and
        (preference preference4
          (exists (?a - (either golfball basketball))
            (then
              (hold (and (not (and (in_motion ?a) (in door) ) ) (not (in_motion agent ?a) ) ) )
              (hold-while (< 1 0.5) (in ?r) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (= (total-time) 1 )
)
(:scoring
  (count preference4:golfball)
)
)
(define (game game-id-2023) (:domain many-objects-room-v1)
(:setup
  (exists (?j - hexagonal_bin)
    (game-conserved
      (and
        (in ?j ?j)
        (agent_holds ?j)
      )
    )
  )
)
(:constraints
  (and
    (forall (?z - ball)
      (and
        (preference preference1
          (exists (?k - game_object)
            (then
              (hold-while (in_motion pink_dodgeball) (not (agent_holds desk front) ) )
              (once (in_motion ?z) )
              (once (in_motion ?k ?k) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (not
    (>= 10 (count preference1:dodgeball) )
  )
)
(:scoring
  (= 1 2 )
)
)
(define (game game-id-2058) (:domain many-objects-room-v1)
(:setup
  (exists (?k - (either dodgeball doggie_bed ball))
    (exists (?x - cube_block)
      (game-conserved
        (on ?x)
      )
    )
  )
)
(:constraints
  (and
    (forall (?i - (either mug) ?f - game_object)
      (and
        (preference preference1
          (at-end
            (adjacent bed)
          )
        )
        (preference preference2
          (exists (?v - doggie_bed)
            (then
              (once (and (in_motion ?f) (< 2 3) ) )
              (once (on ?v ?v) )
              (once (adjacent ?v ?f) )
            )
          )
        )
      )
    )
    (preference preference3
      (exists (?k - dodgeball)
        (at-end
          (in ?k ?k)
        )
      )
    )
  )
)
(:terminal
  (>= (* 8 3 )
    (and
      (count preference1:basketball)
    )
  )
)
(:scoring
  5
)
)
(define (game game-id-2065) (:domain few-objects-room-v1)
(:setup
  (game-optional
    (in top_drawer ?xxx)
  )
)
(:constraints
  (and
    (forall (?i - (either alarm_clock book))
      (and
        (preference preference1
          (exists (?l - game_object)
            (then
              (once (agent_holds ?l bridge_block) )
              (hold (not (in ?l) ) )
              (once (touch ?l) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (< 5 (count-once-per-objects preference1:pink) )
)
(:scoring
  (* (count-once preference1:basketball) (+ (count preference1:beachball:bed) (+ 50 3 )
    )
  )
)
)
(define (game game-id-2068) (:domain many-objects-room-v1)
(:setup
  (game-optional
    (and
      (and
        (exists (?x - shelf)
          (in_motion sideways ?x)
        )
        (or
          (not
            (not
              (not
                (< 4 1)
              )
            )
          )
          (not
            (in ?xxx)
          )
        )
      )
      (broken ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?a - doggie_bed ?p - hexagonal_bin)
        (at-end
          (in_motion agent)
        )
      )
    )
  )
)
(:terminal
  (>= 300 6 )
)
(:scoring
  (not
    (total-time)
  )
)
)
(define (game game-id-2096) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (agent_holds ?xxx)
  )
)
(:constraints
  (and
    (forall (?j - dodgeball ?o - cube_block ?y - pillow)
      (and
        (preference preference1
          (exists (?r - cube_block ?f - (either laptop bridge_block teddy_bear desktop))
            (then
              (once (in_motion ?f ?y) )
              (hold (in_motion ?y) )
            )
          )
        )
      )
    )
    (preference preference2
      (exists (?d - chair)
        (then
          (once (not (in_motion ?d) ) )
          (hold (not (on pink_dodgeball) ) )
          (hold (in_motion ?d) )
        )
      )
    )
  )
)
(:terminal
  (>= 8 10 )
)
(:scoring
  (count preference1:dodgeball:basketball)
)
)
(define (game game-id-2122) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (< (distance room_center ?xxx) 2)
  )
)
(:constraints
  (and
    (forall (?x - hexagonal_bin)
      (and
        (preference preference1
          (exists (?z - (either golfball cellphone yellow_cube_block golfball bridge_block yellow alarm_clock) ?k - block)
            (then
              (once (exists (?u - curved_wooden_ramp) (not (in ?k ?x) ) ) )
              (once (not (touch south_wall) ) )
              (once-measure (in_motion ?k) (distance desk 4) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (> (count preference1:green) (* (count preference1:beachball) (count preference1) )
  )
)
(:scoring
  (- (count-once-per-objects preference1:dodgeball) )
)
)
(define (game game-id-2124) (:domain medium-objects-room-v1)
(:setup
  (forall (?e - hexagonal_bin)
    (and
      (game-conserved
        (= (distance room_center ?e) (distance ?e room_center))
      )
    )
  )
)
(:constraints
  (and
    (forall (?l - teddy_bear)
      (and
        (preference preference1
          (exists (?n - ball ?r - tall_cylindrical_block)
            (then
              (once (not (in ?l) ) )
              (hold (in_motion ?r ?r) )
              (once (and (on ?r) (not (and (agent_holds ?l ?r) (in_motion blue ?l) (in_motion sideways) ) ) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (< (count preference1:beachball) (count preference1:red) )
)
(:scoring
  (total-time)
)
)
(define (game game-id-2163) (:domain medium-objects-room-v1)
(:setup
  (forall (?u - cube_block)
    (game-conserved
      (agent_holds ?u)
    )
  )
)
(:constraints
  (and
    (forall (?i - game_object)
      (and
        (preference preference1
          (exists (?r - hexagonal_bin)
            (then
              (hold (adjacent ?i) )
              (hold-while (not (agent_holds ?i) ) (touch ?i ?i) (agent_holds ?r) )
              (once (in color) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (- (count-once preference1:green) )
    1
  )
)
(:scoring
  10
)
)
(define (game game-id-2172) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (= 1 (distance ?xxx ?xxx))
  )
)
(:constraints
  (and
    (forall (?b - color ?w - green_triangular_ramp)
      (and
        (preference preference1
          (exists (?a - (either doggie_bed side_table blue_cube_block))
            (at-end
              (same_object ?a)
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference1:dodgeball) (count-same-positions preference1:beachball:beachball) )
)
(:scoring
  10
)
)
(define (game game-id-2184) (:domain many-objects-room-v1)
(:setup
  (and
    (game-optional
      (and
        (agent_holds ?xxx)
        (agent_holds ?xxx ?xxx)
        (agent_holds ?xxx)
      )
    )
    (exists (?d - block)
      (game-optional
        (not
          (on ?d ?d)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?t - triangular_ramp)
        (then
          (once (not (> 1 1) ) )
          (once (<= 2 9) )
          (once (and (toggled_on ?t) (on ?t) ) )
        )
      )
    )
    (forall (?o - teddy_bear ?n - wall)
      (and
        (preference preference2
          (exists (?s - cube_block)
            (then
              (hold (same_type ?n ?s) )
              (once (not (< (distance 2 room_center) (distance desk room_center)) ) )
              (hold-while (< 2 1) (touch ?n bed) (and (not (agent_holds ?s) ) (not (same_color ?s ?s) ) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (= (count-total preference2:golfball:dodgeball) (count-once-per-external-objects preference2:dodgeball:golfball) )
)
(:scoring
  5
)
)
(define (game game-id-2202) (:domain few-objects-room-v1)
(:setup
  (game-optional
    (agent_holds ?xxx ?xxx)
  )
)
(:constraints
  (and
    (forall (?y - hexagonal_bin)
      (and
        (preference preference1
          (exists (?t - shelf)
            (then
              (once (on ?t ?t) )
              (hold (agent_holds ?t ?y) )
              (once (is_setup_object agent) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 3 (count preference1:dodgeball) )
)
(:scoring
  (* 3 (* 6 (count-once-per-objects preference1:cylindrical_block:yellow) (count-once-per-objects preference1:cube_block) )
  )
)
)
(define (game game-id-2235) (:domain medium-objects-room-v1)
(:setup
  (and
    (game-optional
      (in_motion door ?xxx)
    )
  )
)
(:constraints
  (and
    (forall (?w - block)
      (and
        (preference preference1
          (exists (?g - teddy_bear)
            (at-end
              (broken ?g ?g)
            )
          )
        )
        (preference preference2
          (exists (?f - hexagonal_bin)
            (at-end
              (is_setup_object ?f)
            )
          )
        )
        (preference preference3
          (exists (?p - dodgeball ?g - game_object)
            (then
              (once (in_motion ?w ?w) )
              (once (on ?g) )
              (once (on ?g sideways) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 60 3 )
)
(:scoring
  (count-total preference1:purple)
)
)
(define (game game-id-2264) (:domain medium-objects-room-v1)
(:setup
  (exists (?r - cube_block ?t - hexagonal_bin ?p - hexagonal_bin ?q - curved_wooden_ramp)
    (forall (?z - curved_wooden_ramp)
      (exists (?g - (either pen laptop laptop))
        (game-conserved
          (not
            (in_motion agent ?q)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?h - hexagonal_bin)
      (and
        (preference preference1
          (exists (?j - hexagonal_bin)
            (then
              (hold (adjacent ?h) )
              (once (in_motion ?h ?j) )
              (once (on ?j ?h) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (not
    (>= (count preference1:tall_cylindrical_block) 3 )
  )
)
(:scoring
  (* (count-overlapping preference1:dodgeball) 5 )
)
)
(define (game game-id-2288) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (on ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?g - hexagonal_bin)
        (then
          (hold (on ?g ?g) )
          (once (agent_holds ?g) )
          (once (in_motion floor ?g) )
          (hold-while (and (touch desk ?g) (in ?g) ) (in ?g) )
        )
      )
    )
    (preference preference2
      (exists (?c - triangular_ramp)
        (at-end
          (and
            (not
              (< (distance desk ?c) 1)
            )
            (agent_holds ?c rug)
          )
        )
      )
    )
  )
)
(:terminal
  (>= 10 (count-once-per-objects preference2:golfball) )
)
(:scoring
  (count preference1:dodgeball)
)
)
(define (game game-id-2329) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (not
      (and
        (in_motion rug)
        (on door)
      )
    )
  )
)
(:constraints
  (and
    (forall (?b - chair ?x - teddy_bear)
      (and
        (preference preference1
          (exists (?f - dodgeball ?m - triangular_ramp)
            (then
              (once (in desk rug) )
              (once (not (agent_holds ?m ?m) ) )
              (hold (in ?m ?x) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 4 (count-overlapping preference1:purple:pink) )
)
(:scoring
  (count-once-per-objects preference1:basketball)
)
)
(define (game game-id-2380) (:domain medium-objects-room-v1)
(:setup
  (forall (?p - hexagonal_bin)
    (exists (?a - shelf)
      (game-optional
        (agent_holds ?p)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?v - game_object ?k ?f ?b ?o ?u ?s - (either yellow_cube_block cube_block))
        (at-end
          (agent_holds ?f)
        )
      )
    )
  )
)
(:terminal
  (>= (total-score) 20 )
)
(:scoring
  9
)
)
(define (game game-id-2386) (:domain few-objects-room-v1)
(:setup
  (forall (?g - pyramid_block)
    (not
      (and
        (game-optional
          (agent_holds ?g)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?l - ball)
      (and
        (preference preference1
          (exists (?m - (either desktop cd))
            (at-end
              (and
                (same_type ?m ?l)
                (and
                  (touch ?l)
                  (not
                    (on door)
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
  (>= (count-once-per-objects preference1) 1 )
)
(:scoring
  (count preference1:hexagonal_bin)
)
)
(define (game game-id-2390) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (agent_holds floor)
  )
)
(:constraints
  (and
    (forall (?o - (either book dodgeball) ?i ?d - cube_block)
      (and
        (preference preference1
          (then
            (hold (and (in_motion rug) (and (= 1 (distance ?d ?d)) (agent_holds ?i ?d) ) ) )
            (once-measure (and (not (in ?d ?d) ) (agent_holds ?i) ) (distance bed ?i) )
            (once (= 1 (distance ?i agent)) )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (external-forall-maximize (count-once-per-objects preference1:block) ) (- (count preference1:doggie_bed:green) )
  )
)
(:scoring
  10
)
)
(define (game game-id-2409) (:domain many-objects-room-v1)
(:setup
  (and
    (exists (?e - building)
      (exists (?p - cube_block ?i - (either dodgeball curved_wooden_ramp))
        (forall (?q - cylindrical_block ?g - triangular_ramp)
          (game-conserved
            (on ?e)
          )
        )
      )
    )
    (game-optional
      (and
        (agent_holds ?xxx)
        (not
          (not
            (and
              (not
                (and
                  (on ?xxx)
                  (and
                    (in_motion rug ?xxx)
                    (on ?xxx)
                    (in_motion ?xxx)
                  )
                )
              )
              (not
                (not
                  (touch ?xxx rug)
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
      (exists (?b - hexagonal_bin)
        (then
          (hold (exists (?z - hexagonal_bin) (in ?b ?z) ) )
          (hold-while (agent_holds ?b ?b) (in_motion ?b bed) )
          (once (is_setup_object ?b agent) )
        )
      )
    )
    (forall (?b - sliding_door)
      (and
        (preference preference2
          (exists (?a ?q - (either curved_wooden_ramp dodgeball))
            (at-end
              (on ?b rug)
            )
          )
        )
      )
    )
    (preference preference3
      (exists (?z - pyramid_block)
        (then
          (hold (not (and (same_type ?z) (in_motion ?z) ) ) )
          (hold-for 5 (not (in_motion ?z ?z) ) )
        )
      )
    )
  )
)
(:terminal
  (>= 3 15 )
)
(:scoring
  (count-once-per-objects preference2:alarm_clock)
)
)
(define (game game-id-2415) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (in ?xxx ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?f - cube_block)
        (then
          (once (in_motion ?f ?f) )
          (once (in ?f) )
          (hold (and (touch ?f) (on ?f) ) )
          (once (touch rug) )
        )
      )
    )
  )
)
(:terminal
  (> 4 (* 2 10 )
  )
)
(:scoring
  (+ 2 (* 16 40 )
    (and
      (* 5 3 )
    )
    5
  )
)
)
(define (game game-id-2417) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (in ?xxx ?xxx)
  )
)
(:constraints
  (and
    (forall (?l - teddy_bear)
      (and
        (preference preference1
          (exists (?d - pillow)
            (then
              (once (on ?l) )
              (hold (not (agent_holds ?l) ) )
              (once (agent_holds ?l) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (= (count preference1:yellow_cube_block:blue_dodgeball:basketball) (count-once preference1:beachball) )
)
(:scoring
  (count preference1:red_pyramid_block:golfball)
)
)
(define (game game-id-2423) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (in ?xxx desk)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?a - hexagonal_bin ?c - red_pyramid_block)
        (at-end
          (agent_holds ?c)
        )
      )
    )
    (preference preference2
      (exists (?k - (either tall_cylindrical_block ball laptop))
        (then
          (hold (agent_holds ?k) )
          (hold-while (adjacent ?k) (not (not (on ?k) ) ) )
          (once (on ?k) )
        )
      )
    )
  )
)
(:terminal
  (>= (total-score) 10 )
)
(:scoring
  1
)
)
(define (game game-id-2447) (:domain many-objects-room-v1)
(:setup
  (and
    (forall (?x - wall ?m - pillow)
      (game-conserved
        (and
          (in_motion ?m)
          (not
            (or
              (in_motion ?m)
              (in agent)
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
      (exists (?c - dodgeball)
        (then
          (once (not (= 1) ) )
          (once (in_motion ?c ?c) )
          (once (not (is_setup_object pink ?c) ) )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= 5 3 )
  )
)
(:scoring
  3
)
)
(define (game game-id-2453) (:domain many-objects-room-v1)
(:setup
  (game-optional
    (on agent)
  )
)
(:constraints
  (and
    (forall (?v - pillow)
      (and
        (preference preference1
          (exists (?j - dodgeball)
            (at-end
              (in ?v ?j)
            )
          )
        )
      )
    )
    (preference preference2
      (exists (?y - wall)
        (then
          (once (game_over ?y rug) )
          (hold (adjacent ?y ?y) )
          (hold-while (exists (?i - hexagonal_bin) (< 1 6) ) (in ?y) )
        )
      )
    )
  )
)
(:terminal
  (>= (count-same-positions preference1:beachball) 3 )
)
(:scoring
  4
)
)
(define (game game-id-2458) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (agent_holds ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?h - book)
        (then
          (once (agent_holds ?h ?h) )
          (once (and (on ?h) (not (adjacent pink_dodgeball) ) ) )
          (forall-sequence (?t - (either dodgeball))
            (then
              (once (on ?h ?t) )
              (once (agent_holds ?h) )
              (hold (and (in ?h) (on ?h ?t) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 3 0 )
)
(:scoring
  (count preference1:dodgeball:purple)
)
)
(define (game game-id-2467) (:domain many-objects-room-v1)
(:setup
  (forall (?w - (either bridge_block cellphone blue_cube_block))
    (and
      (game-optional
        (in_motion block ?w)
      )
      (forall (?u - (either pillow pyramid_block yellow_cube_block))
        (exists (?x - (either flat_block cylindrical_block blue_cube_block) ?o - drawer)
          (game-conserved
            (in ?o)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?i ?n - dodgeball)
        (then
          (once (in ?i) )
          (once (in ?n agent) )
          (once (in_motion ?n ?i) )
          (hold (not (not (exists (?s - hexagonal_bin) (touch ?s ?s) ) ) ) )
        )
      )
    )
  )
)
(:terminal
  (>= 5 (+ 25 10 )
  )
)
(:scoring
  180
)
)
(define (game game-id-2476) (:domain medium-objects-room-v1)
(:setup
  (exists (?o - hexagonal_bin)
    (game-conserved
      (on bed)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?c - (either cube_block key_chain) ?a - (either ball book desktop))
        (at-end
          (and
            (and
              (in_motion agent)
              (adjacent ?a)
            )
            (in_motion ?a door)
          )
        )
      )
    )
  )
)
(:terminal
  (> (+ 2 10 )
    6
  )
)
(:scoring
  6
)
)
(define (game game-id-2509) (:domain many-objects-room-v1)
(:setup
  (and
    (game-optional
      (not
        (adjacent ?xxx)
      )
    )
    (forall (?i - golfball ?c - building ?l - hexagonal_bin ?m - desk_shelf)
      (game-optional
        (agent_holds top_shelf ?m)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?j - block)
        (at-end
          (in_motion ?j)
        )
      )
    )
  )
)
(:terminal
  (>= (- 3 )
    (- 5 )
  )
)
(:scoring
  (total-time)
)
)
(define (game game-id-2518) (:domain few-objects-room-v1)
(:setup
  (and
    (and
      (game-conserved
        (in_motion ?xxx)
      )
    )
  )
)
(:constraints
  (and
    (forall (?a - shelf)
      (and
        (preference preference1
          (exists (?i - hexagonal_bin)
            (then
              (once (in_motion agent) )
              (hold (is_setup_object ?i) )
            )
          )
        )
      )
    )
    (preference preference2
      (at-end
        (touch ?xxx ?xxx)
      )
    )
  )
)
(:terminal
  (>= (count preference1:yellow) 3 )
)
(:scoring
  (* 7 15 )
)
)
(define (game game-id-2549) (:domain many-objects-room-v1)
(:setup
  (not
    (game-conserved
      (= (distance ?xxx ?xxx ?xxx) 10)
    )
  )
)
(:constraints
  (and
    (forall (?o ?x ?g - hexagonal_bin ?w - wall)
      (and
        (preference preference1
          (exists (?l - game_object)
            (at-end
              (agent_holds rug ?l)
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (* 5 10 )
    (count-once-per-objects preference1:alarm_clock)
  )
)
(:scoring
  3
)
)
(define (game game-id-2551) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (and
      (in_motion ?xxx)
      (agent_holds ?xxx ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?y - chair)
        (then
          (hold (agent_holds ?y bridge_block) )
          (hold (agent_holds ?y ?y) )
          (once (agent_holds ?y ?y) )
        )
      )
    )
  )
)
(:terminal
  (>= 10 2 )
)
(:scoring
  (total-score)
)
)
(define (game game-id-2558) (:domain medium-objects-room-v1)
(:setup
  (game-optional
    (and
      (and
        (same_type desk rug ?xxx)
        (exists (?p - hexagonal_bin)
          (in_motion top_shelf)
        )
      )
      (not
        (in_motion ?xxx ?xxx)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?x - shelf ?g - block)
        (then
          (once (above ?g) )
          (once (not (in ?g) ) )
          (hold-to-end (exists (?w - block) (on ?w ?w) ) )
        )
      )
    )
  )
)
(:terminal
  (<= (total-time) 20 )
)
(:scoring
  (count-increasing-measure preference1:yellow_pyramid_block)
)
)
(define (game game-id-2577) (:domain many-objects-room-v1)
(:setup
  (forall (?i - doggie_bed)
    (exists (?s - bridge_block)
      (game-optional
        (object_orientation ?i)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?k - (either cd bridge_block cd))
        (then
          (hold (on ?k ?k) )
          (once (in_motion ?k ?k) )
          (hold (agent_holds ?k ?k) )
        )
      )
    )
  )
)
(:terminal
  (> 30 3 )
)
(:scoring
  (count preference1:yellow:dodgeball)
)
)
(define (game game-id-2589) (:domain many-objects-room-v1)
(:setup
  (forall (?j - dodgeball ?b - golfball)
    (game-optional
      (and
        (in ?b)
        (not
          (agent_holds ?b)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?y - (either dodgeball golfball dodgeball) ?q - dodgeball)
      (and
        (preference preference1
          (exists (?f - beachball ?f - dodgeball)
            (then
              (hold (in_motion ?f) )
              (once (agent_holds ?f) )
              (once (agent_holds ?f upright) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (and
    (>= (count preference1:yellow_cube_block) 6 )
  )
)
(:scoring
  (* 10 2 )
)
)
(define (game game-id-2592) (:domain medium-objects-room-v1)
(:setup
  (forall (?j - (either book cylindrical_block))
    (exists (?m - flat_block ?a ?y - hexagonal_bin)
      (forall (?e ?s - (either cube_block))
        (game-optional
          (on pink)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?b - cube_block)
        (then
          (once (not (not (agent_holds rug ?b) ) ) )
          (once (not (agent_holds ?b) ) )
          (hold (or (in_motion ?b desk) ) )
        )
      )
    )
  )
)
(:terminal
  (> 3 (count preference1) )
)
(:scoring
  (< (- 2 )
    1
  )
)
)
(define (game game-id-2612) (:domain many-objects-room-v1)
(:setup
  (and
    (game-conserved
      (or
        (agent_holds ?xxx)
        (not
          (not
            (and
              (adjacent ?xxx)
              (and
                (on ?xxx)
                (adjacent ?xxx)
                (is_setup_object bed)
              )
              (not
                (not
                  (in_motion ?xxx ?xxx)
                )
              )
              (adjacent ?xxx agent)
            )
          )
        )
        (in_motion ?xxx ?xxx)
        (in_motion ?xxx)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?o - (either cellphone key_chain))
        (then
          (hold (in_motion ?o) )
          (hold-for 7 (in ?o) )
          (hold (game_start tan) )
        )
      )
    )
    (preference preference2
      (exists (?w - block)
        (at-end
          (not
            (object_orientation ?w)
          )
        )
      )
    )
  )
)
(:terminal
  (>= 1 (total-score) )
)
(:scoring
  1
)
)
(define (game game-id-2625) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (on ?xxx ?xxx)
  )
)
(:constraints
  (and
    (forall (?k - wall)
      (and
        (preference preference1
          (at-end
            (not
              (not
                (forall (?e - chair)
                  (not
                    (and
                      (in_motion ?k ?e)
                      (adjacent ?k ?e)
                    )
                  )
                )
              )
            )
          )
        )
        (preference preference2
          (exists (?u - bridge_block)
            (at-end
              (not
                (agent_holds ?u ?k)
              )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (* 18 (count preference1:pink) )
    (count-once-per-objects preference2:dodgeball)
  )
)
(:scoring
  (* (count-once preference1:basketball:dodgeball) (count-once preference1:beachball) )
)
)
(define (game game-id-2662) (:domain medium-objects-room-v1)
(:setup
  (and
    (game-optional
      (not
        (in_motion ?xxx)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?z - hexagonal_bin)
        (then
          (once (agent_holds ?z ?z) )
          (once (not (not (adjacent ?z ?z) ) ) )
          (hold (in_motion ?z) )
        )
      )
    )
  )
)
(:terminal
  (>= (total-score) 1 )
)
(:scoring
  50
)
)
(define (game game-id-2696) (:domain few-objects-room-v1)
(:setup
  (forall (?t - (either cylindrical_block tall_cylindrical_block yellow_cube_block))
    (game-optional
      (in ?t rug)
    )
  )
)
(:constraints
  (and
    (forall (?l - wall ?p ?o ?f - red_dodgeball ?d - building)
      (and
        (preference preference1
          (exists (?q - hexagonal_bin)
            (then
              (once (and (not (< (distance 5 ?d) 1) ) (agent_holds ?d) ) )
              (once (agent_holds ?q ?d) )
              (hold (in_motion upright ?d) )
            )
          )
        )
      )
    )
    (preference preference2
      (exists (?q ?o ?y ?z ?d ?u - shelf ?r - curved_wooden_ramp)
        (at-end
          (in_motion ?r ?r)
        )
      )
    )
  )
)
(:terminal
  (= 5 (external-forall-maximize (count preference1:green) ) )
)
(:scoring
  5
)
)
(define (game game-id-2770) (:domain few-objects-room-v1)
(:setup
  (game-optional
    (on desk ?xxx)
  )
)
(:constraints
  (and
    (forall (?r - dodgeball)
      (and
        (preference preference1
          (exists (?h - dodgeball)
            (at-end
              (< 1 (distance 9 ?h))
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 10 3 )
)
(:scoring
  (count-once-per-objects preference1:brown)
)
)
(define (game game-id-2774) (:domain few-objects-room-v1)
(:setup
  (and
    (game-optional
      (not
        (opposite bottom_shelf)
      )
    )
    (game-conserved
      (not
        (agent_holds ?xxx)
      )
    )
    (forall (?q - dodgeball)
      (and
        (game-optional
          (agent_holds ?q)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (agent_holds ) )
      )
    )
  )
)
(:terminal
  (>= 50 (total-score) )
)
(:scoring
  6
)
)
(define (game game-id-2778) (:domain few-objects-room-v1)
(:setup
  (game-optional
    (agent_holds bed ?xxx)
  )
)
(:constraints
  (and
    (forall (?r - hexagonal_bin)
      (and
        (preference preference1
          (exists (?n - ball ?g - cylindrical_block ?p - chair)
            (at-end
              (is_setup_object ?p)
            )
          )
        )
      )
    )
    (forall (?k - doggie_bed ?f - shelf)
      (and
        (preference preference2
          (exists (?q - hexagonal_bin ?u - beachball)
            (then
              (once (in_motion ?u) )
              (once (in_motion ?f) )
              (hold (on agent ?u ?f) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (<= (count preference1:dodgeball) (count preference1:blue_cube_block:pink) )
)
(:scoring
  (* (count-once-per-objects preference1:yellow_cube_block) (count preference1:blue_cube_block) 3 )
)
)
(define (game game-id-2823) (:domain medium-objects-room-v1)
(:setup
  (and
    (and
      (and
        (game-conserved
          (in ?xxx ?xxx)
        )
      )
      (forall (?z - dodgeball)
        (game-optional
          (and
            (in ?z)
            (in agent)
          )
        )
      )
      (and
        (forall (?a - hexagonal_bin)
          (exists (?j ?g - color)
            (game-optional
              (not
                (and
                  (in_motion ?g)
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
    (forall (?v - hexagonal_bin ?a - dodgeball ?v - dodgeball)
      (and
        (preference preference1
          (exists (?h ?k - block)
            (at-end
              (> (distance 0 desk) (distance room_center room_center))
            )
          )
        )
        (preference preference2
          (exists (?k - block)
            (then
              (once (and (adjacent ?k) (object_orientation ?k) ) )
              (hold (same_object agent) )
              (once (agent_holds bed desk) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference2:dodgeball:hexagonal_bin) 1 )
)
(:scoring
  (count preference1:pink)
)
)
(define (game game-id-2828) (:domain few-objects-room-v1)
(:setup
  (exists (?p ?d ?n ?b - hexagonal_bin ?g - hexagonal_bin)
    (game-conserved
      (in agent)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?u - triangular_ramp)
        (then
          (once (in_motion desk ?u) )
          (once (agent_holds ?u) )
          (once (agent_holds ?u) )
        )
      )
    )
  )
)
(:terminal
  (>= (count-measure preference1:dodgeball) 10 )
)
(:scoring
  (external-forall-maximize
    (count-once-per-external-objects preference1:red_pyramid_block)
  )
)
)
(define (game game-id-2838) (:domain many-objects-room-v1)
(:setup
  (game-optional
    (in_motion ?xxx bed)
  )
)
(:constraints
  (and
    (forall (?r - hexagonal_bin)
      (and
        (preference preference1
          (exists (?s - doggie_bed)
            (at-end
              (and
                (exists (?n - dodgeball)
                  (same_type ?r ?s)
                )
                (not
                  (in_motion south_west_corner ?r)
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
  (>= 3 5 )
)
(:scoring
  (= (count-once-per-objects preference1:hexagonal_bin:blue_cube_block) (count-overlapping preference1:pink) (count-shortest preference1:side_table) )
)
)
(define (game game-id-2853) (:domain medium-objects-room-v1)
(:setup
  (game-optional
    (agent_holds ?xxx)
  )
)
(:constraints
  (and
    (forall (?s - beachball ?u - (either yellow cube_block lamp tall_cylindrical_block cd yellow_cube_block basketball) ?r - dodgeball)
      (and
        (preference preference1
          (exists (?a - building ?f - cube_block)
            (at-end
              (not
                (not
                  (object_orientation ?f ?f)
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
  (> (count-once preference1:wall) (count preference1:doggie_bed) )
)
(:scoring
  (* (- (count-once preference1:basketball) 5 ) (count-once-per-objects preference1:book) )
)
)
(define (game game-id-2873) (:domain medium-objects-room-v1)
(:setup
  (exists (?c - chair ?t - color ?w - triangular_ramp)
    (exists (?y - shelf)
      (exists (?m - cube_block ?p - (either cellphone tall_cylindrical_block))
        (and
          (exists (?l - dodgeball ?i - yellow_cube_block)
            (game-conserved
              (agent_holds ?p)
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?k - dodgeball)
      (and
        (preference preference1
          (exists (?i - cylindrical_block ?g - building)
            (then
              (once (same_color ?k ?k) )
              (once (agent_holds ?k) )
              (hold-while (forall (?q - curved_wooden_ramp ?a - hexagonal_bin) (in_motion ?k) ) (not (= 8 (distance 4 4)) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference1:yellow_cube_block) 5 )
)
(:scoring
  3
)
)
(define (game game-id-2881) (:domain medium-objects-room-v1)
(:setup
  (and
    (game-conserved
      (and
        (not
          (in ?xxx ?xxx agent)
        )
        (and
          (in_motion left)
        )
      )
    )
    (game-conserved
      (and
        (agent_holds ?xxx)
        (agent_holds ?xxx ?xxx)
        (and
          (not
            (agent_holds ?xxx)
          )
          (in agent)
        )
        (agent_holds rug)
      )
    )
  )
)
(:constraints
  (and
    (forall (?n - ball)
      (and
        (preference preference1
          (exists (?r - hexagonal_bin)
            (then
              (once (not (in ?n) ) )
              (once (not (agent_holds ?n) ) )
              (once (on ?r ?r) )
            )
          )
        )
      )
    )
    (preference preference2
      (exists (?v - dodgeball)
        (then
          (hold-to-end (in_motion ?v) )
          (hold (and (adjacent ?v ?v) (and (agent_holds ?v) (on ?v ?v) ) ) )
          (hold (not (forall (?x - triangular_ramp) (= (distance ?v 1) (distance room_center 4)) ) ) )
        )
      )
    )
  )
)
(:terminal
  (< (count preference1:tall_cylindrical_block) 10 )
)
(:scoring
  (count preference1:pink)
)
)
(define (game game-id-2925) (:domain medium-objects-room-v1)
(:setup
  (or
    (not
      (game-conserved
        (agent_holds door ?xxx)
      )
    )
    (game-conserved
      (in ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?b - curved_wooden_ramp)
        (then
          (once (agent_holds ?b) )
          (once (not (agent_holds ?b) ) )
          (once (agent_holds ?b agent) )
        )
      )
    )
  )
)
(:terminal
  (>= (count-once preference1:dodgeball:yellow) 15 )
)
(:scoring
  (count preference1:yellow)
)
)
(define (game game-id-2952) (:domain few-objects-room-v1)
(:setup
  (and
    (and
      (game-optional
        (and
          (in_motion ?xxx)
          (in_motion ?xxx bed)
        )
      )
      (game-conserved
        (object_orientation ?xxx floor)
      )
      (forall (?n - ball)
        (game-optional
          (agent_holds ?n ?n)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?b - teddy_bear ?p - ball)
      (and
        (preference preference1
          (exists (?r - ball)
            (then
              (once (in_motion ?p) )
              (hold (and (not (same_color ?r ?p) ) (or (and (not (on bed) ) (forall (?z - color ?m ?v ?s ?c - (either laptop bridge_block) ?c - curved_wooden_ramp) (same_object ?r) ) ) (on rug) ) ) )
              (once (and (not (touch ?p) ) (agent_holds ?r) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (> 8 6 )
)
(:scoring
  (count preference1:tan:doggie_bed)
)
)
(define (game game-id-2953) (:domain medium-objects-room-v1)
(:setup
  (exists (?i ?s - cube_block)
    (and
      (game-optional
        (agent_holds )
      )
    )
  )
)
(:constraints
  (and
    (forall (?n - shelf)
      (and
        (preference preference1
          (then
            (hold-while (not (and (in_motion bed bed) (on pink_dodgeball) ) ) (on ?n agent ?n) (agent_holds ?n) )
            (once (same_object ?n ?n) )
            (once (in_motion ?n) )
          )
        )
      )
    )
  )
)
(:terminal
  (= (count preference1:dodgeball) (count preference1:pink:beachball) )
)
(:scoring
  10
)
)
(define (game game-id-2990) (:domain medium-objects-room-v1)
(:setup
  (and
    (game-conserved
      (in_motion ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (hold (on yellow) )
        (once (agent_holds ?xxx ?xxx) )
        (once (open ?xxx) )
      )
    )
    (preference preference2
      (exists (?t - cube_block)
        (at-end
          (in ?t)
        )
      )
    )
  )
)
(:terminal
  (> 4 4 )
)
(:scoring
  (- 50 )
)
)
(define (game game-id-3100) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (in )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (agent_holds ?xxx) )
        (hold (< (distance ?xxx room_center) 1) )
        (once (and (same_type floor ?xxx) (in ?xxx) ) )
      )
    )
  )
)
(:terminal
  (= 10 6 )
)
(:scoring
  (count preference1:golfball)
)
)
(define (game game-id-3144) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (not
      (agent_holds rug ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?z - hexagonal_bin)
        (at-end
          (not
            (agent_holds ?z ?z)
          )
        )
      )
    )
  )
)
(:terminal
  (>= 2 (+ 5 15 )
  )
)
(:scoring
  (* 10 (+ 8 10 )
  )
)
)
(define (game game-id-3201) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (and
      (in_motion ?xxx rug)
      (in ?xxx ?xxx)
      (in_motion floor)
      (and
        (agent_holds ?xxx)
        (and
          (not
            (in_motion ?xxx ?xxx)
          )
          (on agent)
        )
        (on ?xxx ?xxx)
      )
    )
  )
)
(:constraints
  (and
    (forall (?f - block ?p - dodgeball)
      (and
        (preference preference1
          (exists (?b ?y - ball)
            (then
              (once (< (distance ?y ?y) (distance front 4)) )
              (once (and (and (and (agent_holds ?y ?y) (not (opposite ?b ?b) ) ) (is_setup_object ?p) (agent_holds ?b ?y) ) (not (in_motion ?y top_shelf) ) ) )
              (once-measure (touch ?p) (distance ) )
              (hold (and (in ?b) (agent_holds ?b) ) )
              (once (touch ?p) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count-once-per-external-objects preference1:basketball) (count-once-per-objects preference1) )
)
(:scoring
  10
)
)
(define (game game-id-3204) (:domain many-objects-room-v1)
(:setup
  (game-optional
    (opposite ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?n - curved_wooden_ramp)
        (at-end
          (agent_holds ?n ?n)
        )
      )
    )
  )
)
(:terminal
  (>= (> 3 10 )
    3
  )
)
(:scoring
  3
)
)
(define (game game-id-3215) (:domain few-objects-room-v1)
(:setup
  (and
    (forall (?f - hexagonal_bin ?l ?t - curved_wooden_ramp)
      (game-conserved
        (not
          (not
            (not
              (not
                (on ?t)
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
      (exists (?o - doggie_bed)
        (at-end
          (< 1 (distance ?o 8))
        )
      )
    )
  )
)
(:terminal
  (>= 12 3 )
)
(:scoring
  9
)
)
(define (game game-id-3219) (:domain few-objects-room-v1)
(:setup
  (exists (?q - hexagonal_bin)
    (exists (?h - triangular_ramp ?n - wall)
      (forall (?j - game_object ?e - game_object)
        (game-conserved
          (not
            (in rug)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?f - (either dodgeball laptop) ?n - block)
        (then
          (hold (in_motion agent) )
          (hold-to-end (on ?n) )
          (hold (>= (distance ?n desk) (distance agent ?n)) )
        )
      )
    )
  )
)
(:terminal
  (>= 3 7 )
)
(:scoring
  7
)
)
(define (game game-id-3227) (:domain medium-objects-room-v1)
(:setup
  (game-optional
    (not
      (and
        (not
          (agent_holds ?xxx ?xxx)
        )
        (not
          (not
            (agent_holds ?xxx pink_dodgeball)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?o - building)
        (then
          (once (agent_holds ?o) )
          (once (agent_holds ?o) )
          (once (and (agent_holds ?o) (equal_x_position desk) ) )
        )
      )
    )
  )
)
(:terminal
  (= (* 5 30 )
    5
  )
)
(:scoring
  (count preference1:beachball:pink)
)
)
(define (game game-id-3228) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (not
      (not
        (not
          (and
            (adjacent ?xxx)
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
      (exists (?f - game_object)
        (then
          (hold (agent_holds ) )
          (once (< 0 4) )
        )
      )
    )
  )
)
(:terminal
  (> 5 1 )
)
(:scoring
  6
)
)
(define (game game-id-3252) (:domain few-objects-room-v1)
(:setup
  (and
    (and
      (exists (?v - hexagonal_bin)
        (game-conserved
          (game_start top_shelf desk)
        )
      )
      (game-conserved
        (same_type ?xxx ?xxx)
      )
    )
  )
)
(:constraints
  (and
    (forall (?b - block)
      (and
        (preference preference1
          (exists (?o - pillow)
            (then
              (once (< 1 1) )
              (hold-to-end (forall (?v - dodgeball ?n ?k - cube_block) (same_object ?k agent) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference1:blue_dodgeball:blue_dodgeball) (count-once preference1:dodgeball) )
)
(:scoring
  (* 15 (total-score) )
)
)
(define (game game-id-3293) (:domain medium-objects-room-v1)
(:setup
  (exists (?e - hexagonal_bin)
    (game-conserved
      (adjacent ?e)
    )
  )
)
(:constraints
  (and
    (forall (?o - (either yellow beachball floor))
      (and
        (preference preference1
          (exists (?y - block ?x - dodgeball)
            (then
              (once (in green_golfball ?x) )
              (once (agent_holds ?x) )
              (once-measure (exists (?g - building) (agent_holds ?o ?o) ) (distance_side desk 10) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 20 (count preference1:pink) )
)
(:scoring
  0.7
)
)
(define (game game-id-3305) (:domain few-objects-room-v1)
(:setup
  (forall (?y - (either cylindrical_block dodgeball))
    (and
      (and
        (game-conserved
          (adjacent ?y)
        )
      )
      (exists (?t - shelf)
        (game-optional
          (and
            (not
              (same_color ?y desk)
            )
            (adjacent ?t ?y)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?j - (either cellphone cylindrical_block basketball))
      (and
        (preference preference1
          (exists (?w - (either beachball red hexagonal_bin))
            (then
              (once-measure (and (and (agent_holds ?w ?j) (agent_holds ?j ?w) ) (on ?w) ) (distance ?j ?j) )
              (hold (agent_holds ?w ?j) )
              (hold (and (= 10 1 1) (on green_golfball) ) )
            )
          )
        )
        (preference preference2
          (exists (?s - hexagonal_bin)
            (then
              (once (and (not (in ?s) ) (not (in_motion ?j) ) ) )
              (once (exists (?y - wall) (on pink_dodgeball agent) ) )
              (hold (not (agent_holds desk ?j) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (< 10 (count preference2:yellow) )
)
(:scoring
  (count preference1:beachball:basketball)
)
)
(define (game game-id-3329) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (not
      (not
        (on ?xxx ?xxx)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?x - (either desktop dodgeball floor))
        (then
          (once (in_motion ?x ?x) )
          (once (in_motion pink) )
          (hold (< 2 1) )
        )
      )
    )
  )
)
(:terminal
  (>= 3 10 )
)
(:scoring
  (count-once-per-external-objects preference1:yellow)
)
)
(define (game game-id-3346) (:domain many-objects-room-v1)
(:setup
  (forall (?m - dodgeball)
    (game-conserved
      (not
        (not
          (not
            (adjacent ?m)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?e - hexagonal_bin)
      (and
        (preference preference1
          (exists (?f - block)
            (then
              (once (in_motion agent) )
              (hold (in ?e) )
              (once (in_motion ?f ?e) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 3 3 )
)
(:scoring
  (count-once-per-objects preference1:blue_pyramid_block:book)
)
)
(define (game game-id-3350) (:domain many-objects-room-v1)
(:setup
  (game-optional
    (agent_holds ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?l - teddy_bear ?o - curved_wooden_ramp)
        (then
          (once (is_setup_object ?o ?o) )
          (hold (equal_z_position agent) )
          (once (touch ?o ?o) )
        )
      )
    )
  )
)
(:terminal
  (>= 7 (count-once-per-objects preference1:beachball) )
)
(:scoring
  (* 300 (* 6 10 )
  )
)
)
(define (game game-id-3404) (:domain many-objects-room-v1)
(:setup
  (exists (?t - wall)
    (exists (?j - drawer)
      (game-conserved
        (< (distance ?j ?j ?j) 1)
      )
    )
  )
)
(:constraints
  (and
    (forall (?d - (either golfball wall))
      (and
        (preference preference1
          (exists (?n - hexagonal_bin ?h - pillow)
            (then
              (hold (agent_holds ?d) )
              (once (adjacent ?d ?h) )
              (hold (not (< 3 6) ) )
              (once (adjacent front ?d) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference1:blue_pyramid_block) (count-once-per-objects preference1:brown) )
)
(:scoring
  (count-once-per-objects preference1:hexagonal_bin)
)
)
(define (game game-id-3422) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (not
      (not
        (on ?xxx)
      )
    )
  )
)
(:constraints
  (and
    (forall (?d - game_object)
      (and
        (preference preference1
          (exists (?r - hexagonal_bin)
            (then
              (once (on ?d) )
              (once (not (not (in_motion ?d ?d) ) ) )
              (once (same_color ?d ?d) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (* (count preference1:beachball:cylindrical_block) 30 )
    15
  )
)
(:scoring
  2
)
)
(define (game game-id-3459) (:domain many-objects-room-v1)
(:setup
  (and
    (forall (?z - dodgeball)
      (game-optional
        (agent_holds ?z)
      )
    )
  )
)
(:constraints
  (and
    (forall (?q ?y - dodgeball)
      (and
        (preference preference1
          (exists (?b - dodgeball ?e - chair ?w - dodgeball)
            (then
              (once (not (on agent) ) )
              (hold (and (not (in_motion ?q) ) (agent_holds ?y ?q) ) )
              (once (agent_holds rug ?q) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (= (count-total preference1:dodgeball) (* (count-once preference1:dodgeball:basketball) (count-once preference1:pink:dodgeball) )
  )
)
(:scoring
  (count preference1:beachball)
)
)
(define (game game-id-3483) (:domain many-objects-room-v1)
(:setup
  (game-optional
    (in ?xxx ?xxx)
  )
)
(:constraints
  (and
    (forall (?x - (either pillow rug desktop) ?j - triangular_ramp)
      (and
        (preference preference1
          (exists (?t - hexagonal_bin)
            (then
              (hold (and (in_motion ?j) (not (not (in_motion bed ?j) ) ) ) )
              (once (in_motion ?j ?j) )
              (hold (on agent) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 5 10 )
)
(:scoring
  (count preference1:blue_dodgeball:hexagonal_bin)
)
)
(define (game game-id-3486) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (in_motion upright ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (once (or (in_motion ?xxx) (agent_holds ?xxx ?xxx) ) )
        (hold (not (and (and (in_motion ?xxx) ) (and (in_motion ?xxx ?xxx) (and (not (adjacent ?xxx) ) (is_setup_object ?xxx) ) ) ) ) )
        (hold (not (in ?xxx ?xxx) ) )
      )
    )
  )
)
(:terminal
  (>= 2 1 )
)
(:scoring
  1
)
)
(define (game game-id-3503) (:domain few-objects-room-v1)
(:setup
  (game-optional
    (and
      (and
        (and
          (not
            (same_color ?xxx)
          )
          (and
            (not
              (not
                (not
                  (< (distance room_center desk) 2)
                )
              )
            )
            (in_motion ?xxx)
          )
        )
        (not
          (not
            (adjacent ?xxx ?xxx)
          )
        )
      )
      (and
        (same_type ?xxx ?xxx ?xxx)
        (agent_holds ?xxx ?xxx)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?q - dodgeball)
        (then
          (once (or (and (agent_holds ?q) (in_motion ?q) ) (not (< (distance 10 2) (distance 8 ?q)) ) ) )
          (once (and (and (and (= (distance ?q room_center) (distance ?q ?q ?q)) (agent_holds ?q) (agent_holds ?q ?q) ) (and (same_type top_shelf) (not (and (in door ?q) (agent_holds bed) ) ) ) (adjacent ?q ?q) ) (same_color ?q ?q) ) )
          (once (in_motion agent ?q) )
        )
      )
    )
  )
)
(:terminal
  (>= (total-score) 10 )
)
(:scoring
  (* 6 (total-score) )
)
)
(define (game game-id-3592) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (game_over ?xxx ?xxx)
  )
)
(:constraints
  (and
    (forall (?s - game_object)
      (and
        (preference preference1
          (exists (?v - pillow)
            (at-end
              (agent_holds ?v ?v)
            )
          )
        )
      )
    )
  )
)
(:terminal
  (<= (count preference1:basketball) (count-longest preference1:pink) )
)
(:scoring
  (* (- 50 )
    (-
      (count-once preference1:yellow)
      (count-once-per-objects preference1:doggie_bed)
    )
  )
)
)
(define (game game-id-3599) (:domain few-objects-room-v1)
(:setup
  (game-optional
    (agent_holds ?xxx)
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (hold-while (in_motion ?xxx ?xxx) (same_type ?xxx) )
        (once (touch desk) )
        (once (in ?xxx ?xxx) )
      )
    )
  )
)
(:terminal
  (>= 3 5 )
)
(:scoring
  (count-once-per-objects preference1:dodgeball)
)
)
(define (game game-id-3622) (:domain many-objects-room-v1)
(:setup
  (game-optional
    (not
      (< (distance ?xxx room_center) 1)
    )
  )
)
(:constraints
  (and
    (forall (?u - doggie_bed)
      (and
        (preference preference1
          (exists (?b - dodgeball)
            (then
              (hold (same_color ?u ?u) )
              (once (touch ?b) )
              (once (agent_holds ?u floor) )
              (once (not (touch ?u) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (= (count-once-per-objects preference1:rug) 5 )
)
(:scoring
  15
)
)
(define (game game-id-3632) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (not
      (in_motion ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?i - teddy_bear ?k - block)
        (then
          (once (<= (distance agent agent) (distance agent ?k)) )
          (hold (adjacent_side ?k) )
          (hold (in_motion floor) )
        )
      )
    )
    (forall (?e - (either cd cellphone) ?d - building)
      (and
        (preference preference2
          (exists (?v - color)
            (then
              (once (not (< 1 (x_position 9 ?v)) ) )
              (once (adjacent ?v) )
              (once (< 6 (distance ?v 7 ?v)) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference2:doggie_bed:blue_dodgeball) 5 )
)
(:scoring
  5
)
)
(define (game game-id-3639) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (and
      (same_color ?xxx)
      (not
        (agent_holds ?xxx ?xxx)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (then
        (hold (adjacent ?xxx ?xxx) )
        (once (< 9 9) )
        (once (object_orientation ?xxx) )
        (once (in_motion ?xxx ?xxx) )
      )
    )
  )
)
(:terminal
  (>= 3 6 )
)
(:scoring
  (count preference1:dodgeball)
)
)
(define (game game-id-3642) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (agent_holds ?xxx)
  )
)
(:constraints
  (and
    (forall (?h - doggie_bed)
      (and
        (preference preference1
          (exists (?i - dodgeball)
            (then
              (once (agent_holds ?h) )
              (once (and (adjacent ?i ?h ?h) (not (in ?h) ) ) )
              (once (in_motion ?i) )
            )
          )
        )
        (preference preference2
          (exists (?c - flat_block ?j - (either cd alarm_clock))
            (then
              (once (same_type ?j ?h) )
              (once (and (above desk desk) (on ?h) ) )
              (any)
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count-once preference2:dodgeball) 3 )
)
(:scoring
  (count preference2:orange:dodgeball)
)
)
(define (game game-id-3645) (:domain few-objects-room-v1)
(:setup
  (not
    (exists (?s - hexagonal_bin)
      (exists (?n - doggie_bed)
        (game-optional
          (object_orientation west_wall ?n)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?x - doggie_bed)
      (and
        (preference preference1
          (then
            (hold (in_motion ?x ?x) )
            (once (on ?x ?x) )
            (once (touch ?x) )
          )
        )
        (preference preference2
          (exists (?s - triangular_ramp)
            (at-end
              (not
                (= 1 (distance agent 3))
              )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (<= (- (count-overlapping preference1:basketball) )
    (count-measure preference2:dodgeball:golfball)
  )
)
(:scoring
  (count-once-per-objects preference1:dodgeball)
)
)
(define (game game-id-3666) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (agent_holds rug)
  )
)
(:constraints
  (and
    (forall (?t - hexagonal_bin)
      (and
        (preference preference1
          (exists (?k - hexagonal_bin)
            (then
              (once (and (agent_holds ?t door) (in brown) ) )
              (hold-to-end (on agent) )
              (once (touch ?k) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (> (count preference1:golfball) (count preference1:dodgeball) )
)
(:scoring
  100
)
)
(define (game game-id-3717) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (agent_holds agent)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?r - block)
        (then
          (once (and (not (and (touch ?r) (on ?r ?r) ) ) (in_motion ?r) ) )
          (once (agent_holds ?r) )
          (once (in_motion ?r ?r) )
        )
      )
    )
  )
)
(:terminal
  (>= (/ (total-time) (total-score) ) (count preference1:golfball) )
)
(:scoring
  (count preference1:dodgeball)
)
)
(define (game game-id-3732) (:domain few-objects-room-v1)
(:setup
  (game-optional
    (>= (distance ?xxx agent) 10)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?c - (either pencil basketball))
        (at-end
          (in_motion ?c)
        )
      )
    )
  )
)
(:terminal
  (> 3 3 )
)
(:scoring
  (count-once-per-external-objects preference1:green:dodgeball)
)
)
(define (game game-id-3819) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (or
      (and
        (and
          (agent_holds ?xxx)
          (not
            (< 7 (distance ?xxx 2 ?xxx))
          )
          (agent_holds ?xxx ?xxx ?xxx)
          (agent_holds ?xxx)
        )
        (in_motion ?xxx)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?u ?r ?q ?i ?a ?h - dodgeball)
        (then
          (hold (in_motion ?u block) )
          (once (in_motion ?a ?a) )
          (hold-while (agent_holds ?i ?r) (in_motion ?i) )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= 3 5 )
  )
)
(:scoring
  2
)
)
(define (game game-id-3825) (:domain few-objects-room-v1)
(:setup
  (game-optional
    (< (distance 3 ?xxx 7) 8)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?a - triangular_ramp)
        (then
          (hold (and (agent_holds ?a) (not (in_motion ?a ?a) ) ) )
          (hold (in_motion ?a ?a) )
          (once (= 6 3 (distance )) )
        )
      )
    )
  )
)
(:terminal
  (or
    (> 15 5 )
  )
)
(:scoring
  5
)
)
(define (game game-id-3829) (:domain few-objects-room-v1)
(:setup
  (game-optional
    (and
      (agent_holds ?xxx)
      (agent_holds ?xxx)
      (agent_holds ?xxx)
    )
  )
)
(:constraints
  (and
    (forall (?k - wall)
      (and
        (preference preference1
          (exists (?p - game_object ?f - (either laptop dodgeball))
            (then
              (once (on ?f ?k) )
              (once (and (and (and (on ?k) (in_motion green_golfball) ) (agent_holds back ?f) ) (and (in_motion ?f agent ?f) (in ?f) ) ) )
              (once (agent_holds upright) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (not
    (>= (count-once-per-objects preference1:beachball) (count-once preference1:dodgeball) )
  )
)
(:scoring
  10
)
)
(define (game game-id-3846) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (in_motion ?xxx ?xxx)
  )
)
(:constraints
  (and
    (forall (?s - hexagonal_bin)
      (and
        (preference preference1
          (exists (?b - dodgeball ?b - golfball)
            (then
              (once (object_orientation pink ?b) )
              (once (on floor ?b) )
              (once (and (in_motion ?b) (in_motion ?b) ) )
            )
          )
        )
        (preference preference2
          (exists (?z - pyramid_block)
            (at-end
              (agent_holds ?s)
            )
          )
        )
      )
    )
  )
)
(:terminal
  (> (count-once-per-objects preference1:golfball:blue_dodgeball) (count-once-per-objects preference2:top_drawer:yellow) )
)
(:scoring
  (* (count-once-per-objects preference1:basketball) (external-forall-minimize 0 ) )
)
)
(define (game game-id-3853) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (< (distance ?xxx agent) (distance ?xxx ?xxx ?xxx))
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?f - (either dodgeball laptop))
        (at-end
          (not
            (and
              (in_motion ?f)
              (in_motion ?f ?f)
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 4 (total-time) )
)
(:scoring
  (count-unique-positions preference1:doggie_bed)
)
)
(define (game game-id-3856) (:domain medium-objects-room-v1)
(:setup
  (forall (?f - ball)
    (game-optional
      (not
        (not
          (not
            (and
              (agent_holds ?f agent)
              (not
                (adjacent ?f)
              )
              (and
                (in_motion ?f ?f)
                (touch ?f)
              )
              (agent_holds ?f ?f)
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
      (at-end
        (and
          (< 3 1)
          (and
            (not
              (adjacent_side ?xxx bed)
            )
            (exists (?b - (either golfball basketball yellow_cube_block) ?u - dodgeball)
              (object_orientation ?u)
            )
          )
        )
      )
    )
    (forall (?o - color)
      (and
        (preference preference2
          (exists (?w - ball)
            (then
              (once (on ?o) )
              (hold (and (not (exists (?n - dodgeball) (not (forall (?s - chair) (not (and (agent_holds ?n) (adjacent ?s ?w) ) ) ) ) ) ) (agent_holds ?o) ) )
              (hold (and (not (in_motion ?o) ) (= 1 (distance 4 agent)) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference2:golfball) 4 )
)
(:scoring
  (not
    (count-once-per-objects preference2:dodgeball)
  )
)
)
(define (game game-id-3878) (:domain few-objects-room-v1)
(:setup
  (and
    (game-conserved
      (exists (?y - triangular_ramp)
        (adjacent ?y)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?n - doggie_bed ?m - ball)
        (then
          (hold (in_motion ?m ?m) )
          (hold-while (not (on upright ?m) ) (in_motion ?m desk) (in_motion ?m) )
          (hold (not (agent_holds ?m) ) )
        )
      )
    )
    (preference preference2
      (exists (?a - dodgeball ?n - triangular_ramp)
        (at-end
          (not
            (adjacent ?n ?n)
          )
        )
      )
    )
  )
)
(:terminal
  (>= 0 5 )
)
(:scoring
  (count-total preference1:basketball)
)
)
(define (game game-id-3898) (:domain many-objects-room-v1)
(:setup
  (exists (?r - hexagonal_bin ?m - hexagonal_bin)
    (game-optional
      (forall (?u - hexagonal_bin)
        (same_color ?m ?u)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?y - pillow)
        (at-end
          (in_motion bed)
        )
      )
    )
  )
)
(:terminal
  (< 60 (total-time) )
)
(:scoring
  100
)
)
(define (game game-id-3901) (:domain medium-objects-room-v1)
(:setup
  (exists (?h - game_object ?w - (either cube_block floor) ?o - pyramid_block ?g - game_object)
    (forall (?s - (either beachball floor))
      (game-conserved
        (in_motion ?g ?g)
      )
    )
  )
)
(:constraints
  (and
    (forall (?b - cube_block)
      (and
        (preference preference1
          (exists (?y - hexagonal_bin)
            (then
              (once (not (touch ?y) ) )
              (once (in_motion main_light_switch front) )
              (once (agent_holds ?y) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (not
    (<= (count-once-per-objects preference1:yellow_cube_block:golfball:dodgeball) 2 )
  )
)
(:scoring
  180
)
)
(define (game game-id-3923) (:domain many-objects-room-v1)
(:setup
  (and
    (exists (?z ?t - flat_block)
      (game-optional
        (not
          (and
            (in_motion rug desk)
            (in_motion ?z)
          )
        )
      )
    )
    (game-conserved
      (in_motion ?xxx ?xxx)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?t - dodgeball)
        (then
          (once (not (agent_holds ?t ?t) ) )
          (hold-while (not (agent_holds ?t) ) (on ?t ?t) )
          (hold (and (is_setup_object ?t ?t) (and (on right ?t) (on ?t) ) ) )
        )
      )
    )
    (preference preference2
      (exists (?p - dodgeball)
        (then
          (hold (in rug) )
          (once (and (touch agent) (object_orientation ?p agent) ) )
          (once (on ?p ?p) )
        )
      )
    )
    (preference preference3
      (exists (?q - ball)
        (then
          (once (on ?q rug ?q) )
          (once (agent_holds ?q) )
          (hold (agent_holds ?q top_shelf) )
        )
      )
    )
  )
)
(:terminal
  (> 10 2 )
)
(:scoring
  4
)
)
(define (game game-id-3943) (:domain few-objects-room-v1)
(:setup
  (exists (?e - cube_block ?z - red_dodgeball)
    (and
      (game-conserved
        (agent_holds ?z ?z)
      )
      (game-optional
        (agent_holds ?z)
      )
    )
  )
)
(:constraints
  (and
    (forall (?x - shelf)
      (and
        (preference preference1
          (exists (?c - ball ?n - drawer)
            (then
              (once (< 2 (distance 3 desk)) )
              (once (not (< 1 (distance ?x ?n)) ) )
              (once (> (distance ?x desk) 4) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (count preference1:pink_dodgeball) 8 )
  )
)
(:scoring
  15
)
)
(define (game game-id-3980) (:domain medium-objects-room-v1)
(:setup
  (exists (?q - color)
    (game-optional
      (adjacent ?q)
    )
  )
)
(:constraints
  (and
    (forall (?x - hexagonal_bin)
      (and
        (preference preference1
          (exists (?n - (either golfball cube_block) ?l - dodgeball ?f - ball)
            (then
              (once (in_motion ?f) )
              (hold (in_motion bed ?f) )
              (hold (not (< (distance_side 9 ?x) 4) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference1:dodgeball) (external-forall-minimize (- (count preference1:yellow_cube_block) )
    )
  )
)
(:scoring
  (count-once-per-objects preference1:golfball)
)
)
(define (game game-id-4028) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (exists (?u - dodgeball)
      (and
        (between ?u ?u)
        (and
          (not
            (not
              (and
                (and
                  (not
                    (not
                      (agent_holds ?u)
                    )
                  )
                  (agent_holds ?u ?u)
                  (not
                    (in_motion ?u)
                  )
                )
                (not
                  (in_motion ?u rug)
                )
              )
            )
          )
          (touch floor ?u agent)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?d - (either bridge_block cylindrical_block) ?v - ball)
      (and
        (preference preference1
          (exists (?s - beachball ?g - hexagonal_bin)
            (then
              (hold-while (not (adjacent_side bed) ) (and (or (and (in ?v door) (> (distance_side desk 3) 8) (in_motion ?v) ) ) (touch ?v ?g) ) (agent_holds ?v agent) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference1:top_drawer) 2 )
)
(:scoring
  (count-once-per-external-objects preference1:red:dodgeball)
)
)
(define (game game-id-4044) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (not
      (< (distance 9 desk) 1)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?j - dodgeball)
        (then
          (once (agent_holds ?j) )
          (once (not (adjacent ?j) ) )
          (once (adjacent_side agent ?j) )
        )
      )
    )
  )
)
(:terminal
  (>= 5 (count-measure preference1:dodgeball) )
)
(:scoring
  (+ 5 (count preference1:beachball) (count-once-per-objects preference1:yellow_cube_block:beachball) )
)
)
(define (game game-id-4051) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (not
      (and
        (not
          (agent_crouches ?xxx ?xxx)
        )
        (agent_holds ?xxx ?xxx)
        (and
          (on blinds)
          (not
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
      (exists (?o - dodgeball)
        (then
          (hold (and (not (and (and (in_motion ?o) (in_motion ?o) ) ) ) (and (and (and (agent_holds ?o rug) (on ?o) ) (agent_holds agent ?o) ) (agent_holds ?o) ) ) )
          (hold (in_motion door) )
        )
      )
    )
  )
)
(:terminal
  (>= 3 (count-once-per-objects preference1) )
)
(:scoring
  50
)
)
(define (game game-id-4055) (:domain few-objects-room-v1)
(:setup
  (forall (?m - dodgeball)
    (and
      (exists (?v - wall ?b - doggie_bed)
        (and
          (forall (?u ?f - triangular_ramp ?j - beachball ?j ?r - ball)
            (game-conserved
              (in_motion ?j ?b)
            )
          )
          (game-conserved
            (not
              (adjacent ?m)
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
      (exists (?u - curved_wooden_ramp)
        (then
          (hold-while (> 1 (x_position ?u 10)) (and (touch ?u) (on upside_down) ) )
          (once (agent_holds ?u) )
          (hold (agent_holds agent) )
        )
      )
    )
  )
)
(:terminal
  (<= 3 5 )
)
(:scoring
  2
)
)

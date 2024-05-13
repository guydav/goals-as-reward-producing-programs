(define (game mcmc-96-0-2-8-9-10-16-19-23-30-31-32-33-36-37-38-40-49-57-58-59-60-70-72-76-78-79-81-82-85-88-89-91-94-100-101-107-110-113-125-126-145-147-161-164-168-172-188-189-197-203-211-226-228-239-240-243-245-248-249-255-267-270-275-287-295-300-301-303-308-313-318-325-327-336-341-342-362-382-392-403-416-419-422-426-434-452-457) (:domain medium-objects-room-v1)
(:setup
  (forall (?c - hexagonal_bin)
    (and
      (and
        (exists (?p - dodgeball)
          (exists (?e - ball)
            (and
              (and
                (game-conserved
                  (same_type ?e ?p)
                )
              )
              (exists (?f - dodgeball ?q - hexagonal_bin)
                (forall (?m - dodgeball)
                  (exists (?n - hexagonal_bin)
                    (game-conserved
                      (agent_holds ?m)
                    )
                  )
                )
              )
            )
          )
        )
        (and
          (and
            (forall (?t - dodgeball)
              (exists (?a - dodgeball ?p - ball)
                (and
                  (and
                    (game-conserved
                      (< (distance room_center ?t) (x_position ?c 8))
                    )
                    (exists (?q - ball ?g - hexagonal_bin)
                      (game-optional
                        (object_orientation ?c ?q)
                      )
                    )
                    (and
                      (game-conserved
                        (forall (?b - ball)
                          (exists (?w - dodgeball)
                            (and
                              (agent_holds ?b)
                              (on agent)
                            )
                          )
                        )
                      )
                    )
                  )
                  (game-conserved
                    (agent_holds ?a)
                  )
                  (game-conserved
                    (in_motion ?a)
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
    (forall (?g - teddy_bear)
      (and
        (preference preference1
          (then
            (once (in_motion ?g) )
            (once (in_motion ?g) )
            (once (or (in_motion ?g) ) )
          )
        )
        (preference preference2
          (exists (?w - cube_block)
            (at-end
              (in_motion ?w)
            )
          )
        )
      )
    )
    (forall (?r - (either dodgeball golfball laptop))
      (and
        (preference preference3
          (exists (?g - (either key_chain game_object))
            (then
              (once (agent_holds ?r) )
              (hold (and (in bed ?g) (touch ?r ?r) ) )
              (once (not (not (and (in_motion ?r) (and (and (not (agent_holds ?r) ) (agent_holds ?r) ) (in_motion ?r) ) ) ) ) )
            )
          )
        )
      )
    )
    (forall (?x ?m - dodgeball)
      (and
        (preference preference5
          (exists (?b - ball ?k - hexagonal_bin ?r ?h - cube_block)
            (then
              (hold (agent_holds ?m) )
              (hold (agent_holds ?k ?h) )
              (hold (in_motion ?h) )
            )
          )
        )
        (preference preference6
          (exists (?p - (either bridge_block laptop))
            (then
              (once (in_motion ?m) )
              (once (agent_holds ?x) )
              (once (in_motion ?x) )
            )
          )
        )
      )
    )
    (forall (?n - dodgeball)
      (and
        (preference preference6
          (exists (?t ?g - teddy_bear ?f - game_object)
            (then
              (hold-while (and (not (not (object_orientation agent ?n) ) ) (not (adjacent ?f) ) (agent_holds ?n) ) (not (in_motion ?n) ) )
              (hold (not (adjacent ?g) ) )
              (once (in_motion ?f) )
            )
          )
        )
      )
    )
    (preference preference7
      (exists (?p - (either tall_cylindrical_block dodgeball))
        (at-end
          (same_color ?p agent)
        )
      )
    )
    (preference preference10
      (exists (?a - dodgeball ?t - green_triangular_ramp ?q - golfball)
        (at-end
          (agent_holds ?a)
        )
      )
    )
  )
)
(:terminal
  (not
    (>= (count-unique-positions preference3:yellow) (* (* (* (+ (count-overlapping preference6:blue_pyramid_block) (count preference6:beachball) )
            (count-once preference3:yellow:pink)
          )
          (* (count-overlapping preference5:yellow) (= (- (count preference1:block) )
              (count-once-per-objects preference7)
            )
          )
        )
        10
      )
    )
  )
)
(:scoring
  20
)
)(define (game mcmc-47-2-6-14-15-20-34-47-53-57-58-60-63-65-76-83-94-103-124-129-146-150-162) (:domain few-objects-room-v1)
(:setup
  (forall (?p - blinds)
    (game-optional
      (not
        (or
          (not
            (< 8 0.5)
          )
          (in_motion ?p)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?w - cube_block ?z - game_object)
        (then
          (hold (in_motion ?w) )
          (hold (in_motion ?w) )
          (once (and (agent_holds ?z) (on desk ?z) ) )
        )
      )
    )
  )
)
(:terminal
  (> 15 2 )
)
(:scoring
  (count-once-per-objects preference1)
)
)(define (game mcmc-13-5-9-10-15-19-23-28-29-30-31-34-36-37-41-43-44-46-55-61-64-67-70-72-73) (:domain few-objects-room-v1)
(:setup
  (exists (?d - ball)
    (game-conserved
      (in_motion bed)
    )
  )
)
(:constraints
  (and
    (forall (?s - book)
      (and
        (preference preference1
          (at-end
            (not
              (on bed ?s)
            )
          )
        )
      )
    )
  )
)
(:terminal
  (> 5 (- (count preference1:red) )
  )
)
(:scoring
  (= 50 7 )
)
)(define (game mcmc-28-1-2-3-13-15-20-26-31-37-42-44-60-65-68-73-82-88-91) (:domain few-objects-room-v1)
(:setup
  (exists (?a - cube_block)
    (exists (?q - chair)
      (game-conserved
        (in_motion ?a)
      )
    )
  )
)
(:constraints
  (and
    (forall (?s - chair)
      (and
        (preference preference1
          (exists (?e ?n - ball)
            (then
              (once (not (in_motion ?n) ) )
              (hold-while (agent_holds ?e) (not (not (in_motion ?s) ) ) (not (agent_holds ?n) ) )
              (once (agent_holds ?n) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (and
    (>= 2 0 )
    (>= 9 5 )
  )
)
(:scoring
  (count preference1:dodgeball)
)
)(define (game mcmc-63-3-5-7-8-9-10-12-15-17-18-19-25-27-28-31-32-45-46-48-50-60-64-68-79-82-89-91-104-108-113-128-139-146-156-160-161-163) (:domain few-objects-room-v1)
(:setup
  (forall (?k - (either dodgeball))
    (game-conserved
      (in_motion ?k)
    )
  )
)
(:constraints
  (and
    (forall (?w ?g ?j ?k ?i ?r - beachball)
      (and
        (preference preference1
          (exists (?l - curved_wooden_ramp ?p - curved_wooden_ramp)
            (at-end
              (not
                (agent_holds bridge_block)
              )
            )
          )
        )
      )
    )
    (forall (?w - triangular_ramp)
      (and
        (preference preference2
          (exists (?j - dodgeball)
            (at-end
              (and
                (above ?w ?j)
                (agent_holds ?j)
              )
            )
          )
        )
      )
    )
    (preference preference3
      (exists (?t - dodgeball)
        (then
          (once (not (on bed ?t) ) )
          (once (> 0.5 2) )
          (hold (in_motion ?t) )
        )
      )
    )
  )
)
(:terminal
  (< (* 5 (* (count preference1:dodgeball) (* (count-once-per-objects preference3:purple) (count-once preference2:yellow) )
      )
    )
    10
  )
)
(:scoring
  3
)
)(define (game mcmc-14-1-3-10-13-16-25-27-29) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (not
      (forall (?u - game_object)
        (agent_holds ?u)
      )
    )
  )
)
(:constraints
  (and
    (forall (?k - (either dodgeball golfball) ?h - dodgeball)
      (and
        (preference preference1
          (then
            (once (in_motion upright) )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 1 (count-once-per-objects preference1:golfball) )
)
(:scoring
  (- (count preference1:beachball) )
)
)(define (game mcmc-4-0-2-3-9-11-19-20-24-26-35-40-45-57-60-61-81-100-104-105-117-134-145-148-150-164-165-167-170) (:domain medium-objects-room-v1)
(:setup
  (forall (?h - dodgeball)
    (and
      (exists (?p - dodgeball)
        (and
          (game-conserved
            (agent_holds ?h)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?c - red_dodgeball)
      (and
        (preference preference1
          (exists (?w - (either dodgeball blue_cube_block tall_cylindrical_block blue_cube_block credit_card dodgeball cd))
            (then
              (any)
              (hold (not (not (agent_holds ?c) ) ) )
              (once (agent_holds ?w) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (and
    (>= 30 30 )
    (>= (count preference1:bed) 6 )
  )
)
(:scoring
  10
)
)(define (game mcmc-43-0-1-8-13-17-19-25-35-48-59-71-74-75-78-81-94-95-104-111-122-137-140-144) (:domain medium-objects-room-v1)
(:setup
  (exists (?u - block ?n - golfball)
    (game-conserved
      (< (distance_side ?n ?n) (distance ?u ?n))
    )
  )
)
(:constraints
  (and
    (forall (?o - triangular_ramp)
      (and
        (preference preference2
          (exists (?n - hexagonal_bin)
            (at-end
              (adjacent ?o ?d)
            )
          )
        )
      )
    )
    (preference preference2
      (exists (?w - red_dodgeball)
        (at-end
          (in_motion ?w)
        )
      )
    )
  )
)
(:terminal
  (= 8 (total-score) )
)
(:scoring
  (* (count-unique-positions preference2:yellow) 20 )
)
)(define (game mcmc-85-0-2-3-5-6-8-10-11-15-16-24-28-29-33-41-44-48-65-85-88-92-100-107-111-112) (:domain few-objects-room-v1)
(:setup
  (forall (?m - hexagonal_bin)
    (game-conserved
      (opposite ?m bed)
    )
  )
)
(:constraints
  (and
    (forall (?l - ball)
      (and
        (preference preference1
          (exists (?x - dodgeball)
            (then
              (hold (agent_holds ?x) )
              (once (agent_holds ?l) )
              (hold (agent_holds ?l) )
            )
          )
        )
      )
    )
    (preference preference3
      (exists (?a - block)
        (then
          (hold (not (agent_holds ?a) ) )
          (once (agent_holds ?a) )
          (hold (adjacent ?a desk) )
        )
      )
    )
    (preference preference4
      (exists (?y - cube_block)
        (at-end
          (in_motion ?y)
        )
      )
    )
  )
)
(:terminal
  (or
    (not
      (>= (count-once-per-objects preference1:pyramid_block) (count preference4:hexagonal_bin) )
    )
    (> (external-forall-maximize (count preference3:doggie_bed) ) 1 )
  )
)
(:scoring
  3
)
)(define (game mcmc-89-1-11-12-15-17-21-23-24-28-33-47-54-71-85-97) (:domain medium-objects-room-v1)
(:setup
  (forall (?r - teddy_bear)
    (game-conserved
      (and
        (agent_holds upright)
        (in_motion ?r floor)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?p - (either dodgeball cube_block))
        (then
          (hold (agent_holds ?p) )
          (hold (not (and (agent_holds ?p) (agent_holds ?p) ) ) )
          (hold (is_setup_object ?p) )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference1) (count preference1) )
)
(:scoring
  (- 5 )
)
)(define (game mcmc-44-0-1-2-6-7-10-19-37-44-53-60-69-71-80-82-84-92-104-117-134-148-169) (:domain many-objects-room-v1)
(:setup
  (exists (?o - cube_block)
    (forall (?u - color ?f - (either book bridge_block) ?n - chair)
      (game-conserved
        (forall (?p - hexagonal_bin ?x - (either dodgeball golfball cd))
          (not
            (= 1 3)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?k - (either golfball))
        (then
          (once (agent_holds ?k) )
          (hold (not (agent_holds ?k) ) )
          (once (agent_holds ?k) )
        )
      )
    )
  )
)
(:terminal
  (>= 0 (count-once-per-objects preference1:dodgeball) )
)
(:scoring
  10
)
)(define (game mcmc-72-6-7-8-10-11-14-17-19-22-23-24-25-26-32-34-43-48-65-67-80-88-95-104-109-113-128) (:domain many-objects-room-v1)
(:setup
  (and
    (exists (?g - building)
      (and
        (game-optional
          (< 1 (distance 5 agent))
        )
        (exists (?d - hexagonal_bin)
          (game-conserved
            (exists (?t - game_object ?u - doggie_bed)
              (and
                (not
                  (not
                    (in_motion ?t)
                  )
                )
                (not
                  (on agent ?d)
                )
                (on ?g ?u)
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
    (forall (?d - block)
      (and
        (preference preference1
          (exists (?y - game_object)
            (then
              (once (agent_holds ?d) )
              (once (agent_holds ?d) )
              (once (in_motion ?y) )
            )
          )
        )
      )
    )
    (preference preference2
      (exists (?f - pillow)
        (then
          (hold (toggled_on ?f) )
          (hold (on rug ?f) )
          (hold (agent_holds ?f) )
        )
      )
    )
  )
)
(:terminal
  (= 0.5 2 )
)
(:scoring
  (external-forall-minimize
    (* (* (count preference2:pyramid_block) 5 3 )
      (count preference1:pink)
    )
  )
)
)(define (game mcmc-98-1-6-9-13-20-24-27-33-35-36-38-40-46-48-49-63-73-74-81-83-96-100-111-115-116-117-121-126-135) (:domain many-objects-room-v1)
(:setup
  (exists (?y - dodgeball ?n - golfball)
    (exists (?u - hexagonal_bin)
      (forall (?i - chair)
        (game-conserved
          (not
            (< 5 1)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?k - (either main_light_switch tall_cylindrical_block dodgeball) ?v - hexagonal_bin)
      (and
        (preference preference1
          (exists (?e - ball)
            (then
              (once (and (not (and (in_motion ?e) (object_orientation ?v ?e) ) ) (in left upright) ) )
              (once-measure (in ?k) (distance ?k) )
              (hold (agent_holds ?e) )
              (once (not (< 1 9) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count-same-positions preference1:pyramid_block:golfball) (count preference1:blue_dodgeball) )
)
(:scoring
  (count preference1)
)
)(define (game mcmc-33-0-3-9-10-12-13-15-16-20-21-22-24-27-39-50-51-52-53-54-60-68-70-79-83-85-86-90-92-105-106-110-112-118-119-121-123-140-141-159-166-167-171-178-189-196-199-220-222) (:domain medium-objects-room-v1)
(:setup
  (exists (?z - dodgeball)
    (game-optional
      (in_motion ?z)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?s - beachball)
        (then
          (hold (not (not (or (touch agent ?s) (agent_holds ?s) ) ) ) )
          (once (and (agent_holds ?s) (not (in_motion ?s agent) ) (or (in_motion ?s) (and (agent_holds ?s) (not (touch ?s) ) ) ) ) )
          (hold (in_motion ?s ?s) )
        )
      )
    )
    (forall (?m - dodgeball ?z - ball)
      (and
        (preference preference2
          (exists (?f - game_object ?s - hexagonal_bin)
            (then
              (once (is_setup_object ?f) )
              (hold (not (agent_holds ?f) ) )
              (once (agent_holds ?f) )
            )
          )
        )
        (preference preference3
          (exists (?f - hexagonal_bin)
            (then
              (once (agent_holds ?z) )
              (once (agent_holds ?m) )
              (once (and (agent_holds ?m) (not (and (in ?z) (in_motion ?z) ) ) ) )
            )
          )
        )
      )
    )
    (preference preference4
      (exists (?t - dodgeball)
        (then
          (once (and (not (agent_holds ?t agent) ) (agent_holds ?t) ) )
          (hold (>= (distance ?t agent) 1) )
          (once (in_motion ?t) )
        )
      )
    )
    (preference preference5
      (exists (?w - (either wall) ?q - red_dodgeball)
        (at-end
          (agent_holds ?q)
        )
      )
    )
  )
)
(:terminal
  (>= (count-once-per-external-objects preference3:blue_cube_block) (count-increasing-measure preference1:beachball) )
)
(:scoring
  (* (* (= (count-once-per-objects preference1:basketball) (+ (count preference5:basketball:orange) 5 )
      )
      (total-score)
    )
    (count-once-per-objects preference4:tall_cylindrical_block)
    (count preference2:dodgeball:dodgeball)
  )
)
)(define (game mcmc-76-0-13-15-22-36-44-53-54-58-77) (:domain medium-objects-room-v1)
(:setup
  (and
    (forall (?k - (either cube_block golfball))
      (and
        (game-optional
          (and
            (in_motion ?k)
            (not
              (not
                (agent_holds ?k)
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
      (exists (?a - hexagonal_bin ?n - hexagonal_bin)
        (at-end
          (same_color ?a ?n)
        )
      )
    )
  )
)
(:terminal
  (>= (count preference1:golfball) 15 )
)
(:scoring
  (count preference1:pyramid_block)
)
)(define (game mcmc-6-0-1-2-3-5-7-9-14-21-23-24-25-36-38-39-42-48-52-56-64-70-73-80-81-95-97-98-112-114-125-126-130-150-156-168-174-191-197-201-204-212-220-228-243-246-252-270-274-281-287-292-294-298-302-303-316-318) (:domain medium-objects-room-v1)
(:setup
  (and
    (game-optional
      (not
        (in color)
      )
    )
    (exists (?p - (either tall_cylindrical_block yellow_cube_block))
      (game-optional
        (not
          (and
            (agent_holds ?p)
            (not
              (in_motion ?p)
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?f - (either book key_chain golfball))
      (and
        (preference preference1
          (then
            (hold-while (and (agent_holds ?f) (on rug) ) (< (distance_side ?f ?f) 10) (agent_holds ?f) )
            (hold-while (in_motion ?f) (in_motion ?f agent) (open ?f) (< (distance ?f ?f) (distance ?f ?f ?f)) )
            (once (agent_holds ?f ?f) )
          )
        )
        (preference preference2
          (exists (?c - block)
            (at-end
              (in_motion ?f)
            )
          )
        )
      )
    )
    (preference preference3
      (exists (?y - ball)
        (then
          (hold (agent_holds ?y) )
          (once-measure (agent_holds ?y) (distance ?y ?y) )
          (hold-while (in_motion south_west_corner ?y) (in_motion ?y) )
        )
      )
    )
    (preference preference4
      (exists (?f - dodgeball)
        (then
          (once (not (agent_holds ?f) ) )
          (once (not (in_motion ?f) ) )
          (once (not (not (and (and (or (agent_holds ?f) (adjacent desk bed) (not (in_motion ?f) ) ) (agent_holds ?f) ) (exists (?l - dodgeball ?y - block ?v - hexagonal_bin) (forall (?c ?g - rug) (not (in ?v ?f) ) ) ) ) ) ) )
        )
      )
    )
    (preference preference6
      (exists (?c - game_object)
        (then
          (hold (in_motion ) )
          (hold (not (on bed agent) ) )
          (hold (and (not (not (in_motion ?c) ) ) (same_color ?c) ) )
        )
      )
    )
    (forall (?j - blinds)
      (and
        (preference preference6
          (exists (?n - doggie_bed)
            (at-end
              (and
                (> (distance ?j ?j ?j) (distance ?n ?n))
                (not
                  (not
                    (< 1 (distance ?j))
                  )
                )
              )
            )
          )
        )
      )
    )
    (forall (?z - (either doggie_bed alarm_clock))
      (and
        (preference preference8
          (exists (?g - ball)
            (then
              (once (touch ?g floor) )
              (hold-while (not (same_object ?g ?z) ) (touch ?g) (agent_holds ?g) )
              (once (exists (?c - beachball) (agent_holds ?n) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (<= (* 0 (count preference3:red) (* 10 (count preference4:beachball) )
      6
      (external-forall-maximize
        (* 10 (count preference2:red) )
      )
    )
    (count-same-positions preference1:beachball)
  )
)
(:scoring
  (count-once preference6:wall)
)
)(define (game mcmc-67-1-8-14-20-21-41-42-46-49-59-69-76-79) (:domain few-objects-room-v1)
(:setup
  (exists (?c - chair)
    (exists (?y - ball)
      (game-conserved
        (and
          (in_motion ?y)
          (agent_holds ?y)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?x - yellow_pyramid_block ?n - dodgeball)
      (and
        (preference preference1
          (exists (?u - ball)
            (then
              (once (< (distance room_center room_center) 7) )
              (once (agent_holds ?n) )
              (once (touch ?u ?x) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (<= (count-overlapping preference1) (- 3 )
  )
)
(:scoring
  (* (count-once-per-objects preference1:beachball) 10 )
)
)(define (game mcmc-79-4-5-6-7-8-9-11-13-15-16-17-24-28-32-33-34-41-46-53-55-58-68-72-73-75-82-84-86-107-118-124-130-134-142) (:domain many-objects-room-v1)
(:setup
  (exists (?k - hexagonal_bin)
    (exists (?w - hexagonal_bin)
      (forall (?n - hexagonal_bin)
        (game-conserved
          (faces ?f ?k)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?i - (either triangle_block laptop))
      (and
        (preference preference1
          (exists (?x - ball ?t - hexagonal_bin)
            (then
              (hold (in_motion ?i) )
              (once (not (= (distance ?i ?i) (distance ?t ?i)) ) )
              (once (agent_holds ?x) )
            )
          )
        )
        (preference preference2
          (exists (?w - hexagonal_bin)
            (then
              (once (is_setup_object ?w) )
              (hold (not (agent_holds ?f) ) )
              (once (agent_holds ?i) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (total-score) 3 )
)
(:scoring
  (* (count preference2) (+ 3 (* (+ (count preference1:golfball) 4 )
        (* 2 (count preference2:blue_pyramid_block) )
      )
    )
    10
  )
)
)(define (game mcmc-84-0-5-8-9-10-12-15-21-25-28-31-42-53-59-63) (:domain medium-objects-room-v1)
(:setup
  (and
    (game-conserved
      (in pink)
    )
  )
)
(:constraints
  (and
    (forall (?u - pillow)
      (and
        (preference preference1
          (exists (?v - hexagonal_bin)
            (then
              (hold-for 0 (not (not (rug_color_under ?u ?v) ) ) )
              (once (agent_holds ?u) )
              (hold (on ?u ?v) )
            )
          )
        )
        (preference preference2
          (exists (?r - block)
            (then
              (hold-while (and (on rug ?r) (agent_holds ?u) ) (adjacent ?u) )
              (any)
              (hold (touch floor pillow) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 1 (* (+ (+ (+ 5 2 )
          (* 8 (* 5 (count-once-per-objects preference1) (count preference1:red) (* 1 2 (* (count-once-per-objects preference2:beachball) (or 2 10 3 ) )
                (* 0 0 )
                (count preference2:pink)
                100
                3
                10
                5
              )
              100
              5
            )
          )
        )
        (external-forall-maximize
          5
        )
      )
      5
    )
  )
)
(:scoring
  5
)
)(define (game mcmc-92-1-5) (:domain medium-objects-room-v1)
(:setup
  (exists (?c - ball)
    (game-optional
      (in_motion ?c)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?m - block)
        (at-end
          (agent_holds ?m)
        )
      )
    )
  )
)
(:terminal
  (>= 15 (count preference1:basketball) )
)
(:scoring
  12
)
)(define (game mcmc-39-0-1-14-21-23-33-35-36-46-67-75-86-91) (:domain medium-objects-room-v1)
(:setup
  (exists (?n - hexagonal_bin)
    (game-optional
      (adjacent ?n east_sliding_door)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?x - game_object)
        (then
          (hold (in_motion ?x) )
          (hold (not (agent_holds ?x) ) )
          (once (in_motion ?x) )
        )
      )
    )
  )
)
(:terminal
  (or
    (> 2 6 )
    (not
      (>= 3 (count preference1:hexagonal_bin:beachball) )
    )
  )
)
(:scoring
  10
)
)(define (game mcmc-24-0-3-5-8-11-13-19-22-25-34-36-43-45-46-47-48-49-55-64-70-71-77-78-83-86-98-113-115-124-128-129-130-141-142-146-155-168-172-182-202-207-209-210-226-234-245-254-258-273-276-295-296) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (exists (?g - cube_block)
      (> (distance ?g ?g) (distance 10 ?g))
    )
  )
)
(:constraints
  (and
    (forall (?a - (either key_chain dodgeball) ?w - hexagonal_bin)
      (and
        (preference preference1
          (exists (?b - hexagonal_bin)
            (then
              (hold (not (and (agent_holds ?a) (not (not (= (distance ?w ?b) 1) ) ) ) ) )
              (once (on ?a) )
              (hold (in_motion ?b ?a) )
            )
          )
        )
      )
    )
    (forall (?t - dodgeball)
      (and
        (preference preference2
          (exists (?z - dodgeball)
            (then
              (forall-sequence (?n - (either dodgeball side_table game_object))
                (then
                  (once (and (not (and (on bed ?t) (same_type ?z ?t) ) ) (in_motion ?t) ) )
                  (once-measure (and (in_motion ?z) (agent_holds ?z) ) (distance ?n ?t) )
                  (once (or (in_motion ?r) (and (agent_holds ?z) (< (distance ?t ?z) (distance ?t ?z)) ) ) )
                )
              )
              (once (agent_holds ?t) )
              (once (in_motion ?t) )
            )
          )
        )
        (preference preference3
          (exists (?t - dodgeball)
            (at-end
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
(:terminal
  (>= 7 (+ (count-total preference1:golfball) 5 (* (count-once-per-objects preference3:tan:doggie_bed) 2 (count preference3:green) 10 (count preference3:basketball) 5 5 )
      4
    )
  )
)
(:scoring
  (count preference2:basketball)
)
)(define (game mcmc-10-9-10-17-18-19-24-29-35-36-37-38-39-40-50-51-52-57-58) (:domain few-objects-room-v1)
(:setup
  (and
    (forall (?r - cube_block)
      (exists (?v - hexagonal_bin)
        (exists (?f - cube_block)
          (and
            (exists (?d - doggie_bed)
              (game-optional
                (on ?f ?r)
              )
            )
            (game-optional
              (and
                (agent_holds ?r)
                (agent_holds ?f)
                (on ?r ?f)
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
    (forall (?o - hexagonal_bin)
      (and
        (preference preference1
          (exists (?c - (either wall game_object pyramid_block) ?d - ball)
            (at-end
              (agent_holds ?d)
            )
          )
        )
        (preference preference2
          (exists (?n - shelf)
            (at-end
              (in_motion agent ?o)
            )
          )
        )
      )
    )
    (preference preference2
      (exists (?q - (either cd cube_block dodgeball))
        (then
          (once (and (in_motion ?q) (in_motion ?q) ) )
          (once (not (in_motion ?q) ) )
          (once (on ?q) )
        )
      )
    )
  )
)
(:terminal
  (or
    (<= 3 (count preference2:pink_dodgeball) )
    (>= 3 (external-forall-maximize 3 ) )
    (= 50 (+ (+ 5 5 5 )
        (count-same-positions preference1:cylindrical_block)
      )
    )
  )
)
(:scoring
  15
)
)(define (game mcmc-55-0-1-6-7-11-13-15-17-22-28-34-36-56) (:domain few-objects-room-v1)
(:setup
  (and
    (forall (?t - game_object)
      (game-optional
        (not
          (agent_holds ?t)
        )
      )
    )
    (game-conserved
      (agent_holds desk)
    )
  )
)
(:constraints
  (and
    (forall (?d - hexagonal_bin)
      (and
        (preference preference1
          (exists (?i ?k - game_object)
            (then
              (once (on side_table ?i) )
              (once (on ?d ?k) )
              (hold-for 9 (agent_holds ?k) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (> (count preference1:basketball) 10 )
)
(:scoring
  (not
    (total-time)
  )
)
)(define (game mcmc-11-9-11-12-14-15-17-18-21-28-31-32-35-39-46-53-61-63-74-86-89-90-100-104-111) (:domain many-objects-room-v1)
(:setup
  (exists (?p - hexagonal_bin)
    (exists (?m - pillow)
      (game-conserved
        (not
          (= 8 1)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?n - cube_block)
        (then
          (hold (agent_holds ?n) )
          (hold (on blue sideways) )
          (once (agent_holds pink_dodgeball) )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (+ (count preference1:hexagonal_bin) 10 (* (total-score) 6 )
        1
      )
      2
    )
    (>= 1 3 )
  )
)
(:scoring
  (count preference1:dodgeball)
)
)(define (game mcmc-36-0-1-6-12-16-20-21-24-27-38-41-54-58-61-68-70-75) (:domain medium-objects-room-v1)
(:setup
  (exists (?p - block)
    (and
      (exists (?n - game_object)
        (and
          (and
            (forall (?d - cube_block)
              (game-optional
                (in_motion ?d)
              )
            )
            (exists (?y - dodgeball)
              (game-optional
                (in_motion ?p)
              )
            )
            (and
              (and
                (game-conserved
                  (in_motion ?n)
                )
                (and
                  (game-conserved
                    (same_object ?p front)
                  )
                )
                (game-conserved
                  (same_color ?p ?n)
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
      (exists (?q - dodgeball)
        (at-end
          (and
            (same_color ?q)
            (not
              (in_motion ?q)
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count-once preference1) 3 )
)
(:scoring
  15
)
)(define (game mcmc-30-0-9-20-38) (:domain many-objects-room-v1)
(:setup
  (exists (?h - teddy_bear)
    (and
      (game-optional
        (and
          (not
            (in_motion ?h)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?z - game_object)
        (at-end
          (agent_holds ?z)
        )
      )
    )
  )
)
(:terminal
  (not
    (>= (count preference1:basketball) 7 )
  )
)
(:scoring
  2
)
)(define (game mcmc-81-1-10-11-12-15-16-17-27-30-31-33-34-35-38-43-46-50-52-62-75-76-78-87-90-93-97-98-101-108-109-111-119-125-133-151-153) (:domain medium-objects-room-v1)
(:setup
  (forall (?e - block)
    (and
      (game-conserved
        (agent_holds ?e)
      )
      (game-conserved
        (not
          (is_setup_object ?e)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?s - hexagonal_bin)
      (and
        (preference preference1
          (exists (?b - teddy_bear)
            (then
              (once (agent_holds ?b) )
              (once (not (in ?b ?s) ) )
              (once (agent_holds ?b) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= 5 2 )
    (> 30 (count preference1:green) )
  )
)
(:scoring
  180
)
)(define (game mcmc-75-4-5-12-13-14-17-19-29-35-46-50) (:domain few-objects-room-v1)
(:setup
  (exists (?x - bridge_block)
    (game-optional
      (agent_holds ?x)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?c ?v - red_dodgeball)
        (then
          (once (agent_holds ?v) )
          (hold-to-end (in_motion ?c) )
          (once (and (agent_holds ?c) (= 2 1) ) )
          (hold (exists (?b - (either bridge_block pink pyramid_block)) (adjacent ?b front) ) )
          (hold-while (in_motion ?v) (adjacent_side ?c ?v) )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference1:pink_dodgeball) 1 )
)
(:scoring
  (* (count-once-per-external-objects preference1:block) (count preference1:bed) )
)
)(define (game mcmc-91-1-3-7-8-12-15-19-20-23-37-44-63-70-72) (:domain few-objects-room-v1)
(:setup
  (exists (?h - hexagonal_bin)
    (and
      (game-optional
        (not
          (agent_holds ?h)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?i - (either desktop cylindrical_block golfball pencil dodgeball cube_block golfball) ?z - (either cylindrical_block laptop))
      (and
        (preference preference1
          (then
            (hold-while (agent_holds ?z) (game_over bed ?z) )
            (once (same_color ?z ?i) )
            (hold (is_setup_object ?i) )
          )
        )
      )
    )
  )
)
(:terminal
  (not
    (or
      (>= (count preference1:pink_dodgeball) 12 )
      (>= (count-once preference1:hexagonal_bin:beachball) 1 )
      (>= 20 (count preference1:basketball) )
    )
  )
)
(:scoring
  3
)
)(define (game mcmc-95-1-6-8-10-11-14-15-16-26-28-29-33-41-42-45-46-47-49-62-64-66-79-80-92) (:domain medium-objects-room-v1)
(:setup
  (exists (?f - game_object)
    (game-conserved
      (not
        (not
          (and
            (in_motion ?f)
            (agent_holds ?f)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?s - ball)
        (then
          (once (< 1 8) )
          (hold (in_motion ?s) )
          (hold (>= (distance door 3) 0) )
        )
      )
    )
    (preference preference2
      (exists (?e - ball ?h - dodgeball ?h - doggie_bed)
        (at-end
          (in_motion pink_dodgeball)
        )
      )
    )
  )
)
(:terminal
  (or
    (> 3 2 )
    (>= (count preference2:tall_cylindrical_block:dodgeball) 40 )
  )
)
(:scoring
  (count-once preference1:brown)
)
)(define (game mcmc-32-3-5-6-7-10-15-16-23-25-36-40) (:domain few-objects-room-v1)
(:setup
  (exists (?v - hexagonal_bin ?t - hexagonal_bin)
    (game-conserved
      (in ?t green_golfball)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?f ?v ?x ?y - dodgeball)
        (then
          (hold (in_motion ?f) )
          (hold (in_motion ?x) )
          (once (in_motion ?v) )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (count-once-per-objects preference1:basketball) (* 10 7 )
    )
    (>= 5 4 )
  )
)
(:scoring
  (count-unique-positions preference1:beachball)
)
)(define (game mcmc-9-4-5-6-10-11-15-20-28-31-32-46-57-61) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (in_motion floor agent)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?l - cube_block)
        (then
          (once (agent_holds ?l) )
          (once (agent_holds ?l) )
          (once (agent_holds ?l) )
        )
      )
    )
    (forall (?p - (either dodgeball beachball))
      (and
        (preference preference2
          (exists (?q - (either cylindrical_block floor) ?s - hexagonal_bin)
            (then
              (hold (object_orientation ?p agent) )
              (once (in_motion ?p) )
              (hold (< (distance ?p ?s) (distance 9 ?q)) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (< 5 (* 10 (* (* 100 (count-measure preference2:pink) )
        (count preference1:basketball)
      )
    )
  )
)
(:scoring
  20
)
)(define (game mcmc-26-3-4-6-8-10-11-13-15-16-22-28-46-47-49-53-62-74-85-99-101-109-123-124-134-139-152-157-159-160-175-193-199-213-228-232-246-248-264) (:domain few-objects-room-v1)
(:setup
  (exists (?h - (either pen yellow_cube_block))
    (and
      (forall (?w - doggie_bed ?t - game_object)
        (forall (?u - flat_block)
          (and
            (game-conserved
              (adjacent ?u right)
            )
            (and
              (forall (?c - doggie_bed)
                (game-conserved
                  (object_orientation ?h upright)
                )
              )
              (exists (?z - ball)
                (and
                  (exists (?a - dodgeball ?o - hexagonal_bin)
                    (game-conserved
                      (forall (?k - hexagonal_bin)
                        (not
                          (and
                            (in_motion ?a ?o)
                            (not
                              (and
                                (in_motion ?a)
                                (not
                                  (agent_holds ?z)
                                )
                                (agent_holds ?a)
                              )
                            )
                            (toggled_on ?k)
                            (not
                              (in_motion ?z)
                            )
                            (in_motion ?a)
                          )
                        )
                      )
                    )
                  )
                  (and
                    (and
                      (game-conserved
                        (agent_holds ?u)
                      )
                      (exists (?r - cube_block)
                        (not
                          (exists (?c - block)
                            (game-conserved
                              (adjacent ?r ?c)
                            )
                          )
                        )
                      )
                      (game-optional
                        (not
                          (same_type pink_dodgeball ?t)
                        )
                      )
                    )
                  )
                  (game-conserved
                    (in_motion ?z)
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
      (exists (?l - game_object)
        (at-end
          (in ?l ?l)
        )
      )
    )
    (forall (?f - hexagonal_bin)
      (and
        (preference preference2
          (exists (?m - block)
            (at-end
              (and
                (in_motion ?f ?m)
                (in_motion ?m)
                (not
                  (agent_holds ?m)
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
    (and
      (> (total-time) 7 )
    )
    (<= (- 6 )
      (count-once preference1:dodgeball:side_table)
    )
    (>= (- (count preference2:doggie_bed) )
      (count-once-per-objects preference1:pink_dodgeball)
    )
  )
)
(:scoring
  10
)
)(define (game mcmc-83-0-1-4-5-7-8-9-11-13-16-19-24-29-44) (:domain medium-objects-room-v1)
(:setup
  (exists (?m - dodgeball)
    (exists (?x - triangular_ramp)
      (forall (?i - chair ?h ?j ?o ?w ?g - doggie_bed)
        (exists (?m - block)
          (and
            (and
              (game-conserved
                (same_color ?i ?h)
              )
              (exists (?q - golfball ?n - hexagonal_bin)
                (exists (?l - hexagonal_bin)
                  (game-optional
                    (on ?j ?m)
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
    (forall (?x - triangular_ramp ?l - (either dodgeball pencil))
      (and
        (preference preference1
          (exists (?q - block ?t - chair ?j - hexagonal_bin ?p - block)
            (at-end
              (in_motion ?l)
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference1:dodgeball) 100 )
)
(:scoring
  6
)
)(define (game mcmc-12-0-20-22-24-45-50-52-57-61) (:domain medium-objects-room-v1)
(:setup
  (and
    (exists (?e - hexagonal_bin ?m - hexagonal_bin)
      (exists (?i - (either basketball flat_block pillow) ?n - building)
        (exists (?t ?l - block)
          (game-conserved
            (not
              (agent_holds desk)
            )
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?m - rug ?f - curved_wooden_ramp)
      (and
        (preference preference1
          (exists (?i - bridge_block)
            (then
              (hold (on ?m ?i) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (total-score) (count-once-per-objects preference1:hexagonal_bin:basketball) )
    (>= 2 10 )
  )
)
(:scoring
  (not
    3
  )
)
)(define (game mcmc-23-2-3-6-10-14-16-17-20-21-29-30-31-35-44-47-57-66-84) (:domain medium-objects-room-v1)
(:setup
  (forall (?l - curved_wooden_ramp)
    (game-optional
      (on agent ?l)
    )
  )
)
(:constraints
  (and
    (forall (?k - blue_pyramid_block)
      (and
        (preference preference1
          (exists (?e - triangular_ramp)
            (then
              (hold (in_motion ?k) )
              (hold (not (rug_color_under ?k ?e) ) )
            )
          )
        )
        (preference preference2
          (exists (?n - hexagonal_bin)
            (at-end
              (on door)
            )
          )
        )
      )
    )
    (preference preference2
      (exists (?p - block)
        (then
          (hold (adjacent pink) )
          (once (not (adjacent ?p) ) )
          (once (adjacent agent ?p) )
          (once (not (agent_holds ?p) ) )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= 4 (total-score) )
    (>= (count-once-per-objects preference2:hexagonal_bin) (count preference1:dodgeball) )
  )
)
(:scoring
  (count preference1)
)
)(define (game mcmc-37-0-5-7-10-11-14-15-16-29-32-34-35-44-45-47-50) (:domain medium-objects-room-v1)
(:setup
  (forall (?a - hexagonal_bin)
    (exists (?k - game_object)
      (and
        (forall (?j - chair)
          (game-optional
            (in_motion ?k)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?a - (either cylindrical_block hexagonal_bin) ?u - dodgeball)
      (and
        (preference preference1
          (exists (?v - wall ?n - yellow_cube_block)
            (then
              (once (agent_holds ?j) )
              (once (not (in_motion ?u) ) )
              (once (< 1 1) )
            )
          )
        )
      )
    )
    (forall (?v ?z - game_object ?a - dodgeball)
      (and
        (preference preference2
          (exists (?d ?t - (either dodgeball bridge_block) ?f - hexagonal_bin)
            (then
              (hold (and (in ?f ?d) (adjacent_side ?a ?z) (on ?z) ) )
              (once (faces ?d desk) )
              (hold (agent_holds ?v) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (= (count preference2:dodgeball) (count preference1:dodgeball) )
)
(:scoring
  20
)
)(define (game mcmc-34-0-1-7-11-24-25-28-32-43-53-60) (:domain medium-objects-room-v1)
(:setup
  (game-conserved
    (agent_holds upright)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?w - dodgeball)
        (then
          (once (not (agent_holds ?w) ) )
          (once (in_motion ?w) )
        )
      )
    )
    (forall (?b - red_pyramid_block)
      (and
        (preference preference2
          (exists (?b - doggie_bed ?w - dodgeball)
            (at-end
              (and
                (on desk ?m)
                (on agent ?b)
              )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (+ 5 3 )
    (- (* 30 (* (count preference1:dodgeball) (* (count-total preference2:yellow_pyramid_block:green) 5 )
        )
      )
    )
  )
)
(:scoring
  4
)
)(define (game mcmc-2-4-10-11-12-13-14-17-21-24-33-45-46-47-55-59-71-73-84-90-105-118-119-123-128-140-148-152-158-171-176-182) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (not
      (exists (?c - game_object)
        (in_motion ?c)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?s - beachball ?k - game_object)
        (then
          (hold (exists (?m - golfball) (agent_holds ?s) ) )
          (once (in_motion ?k) )
          (once (agent_holds ?s) )
        )
      )
    )
    (preference preference2
      (exists (?j - wall ?g - dodgeball)
        (at-end
          (not
            (adjacent ?j left)
          )
        )
      )
    )
    (preference preference3
      (exists (?o - pillow ?m - game_object)
        (then
          (forall-sequence (?c - cube_block)
            (then
              (once (agent_holds ?o) )
              (once (not (in_motion ?m) ) )
              (once (not (adjacent ?c ?m) ) )
            )
          )
          (once (>= 1 (distance 5 desk)) )
          (once (agent_holds ?m) )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= 2 (+ (count preference3:basketball:yellow_pyramid_block) (+ 10 )
      )
    )
    (>= 20 (count-once-per-objects preference2) )
  )
)
(:scoring
  (count-unique-positions preference1:dodgeball)
)
)(define (game mcmc-49-0-5-6-8-9-14-15-18-20-21-24-33-35-39-54-61-74-84-101-106-114-121-127-139-140-147-152) (:domain many-objects-room-v1)
(:setup
  (forall (?y - ball)
    (exists (?y - game_object ?t - dodgeball ?z - block)
      (game-conserved
        (< 5 0)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?e - block ?s - ball ?u - red_dodgeball ?x - (either doggie_bed desktop) ?l - (either mug dodgeball))
        (then
          (once (agent_holds ?s) )
          (once (in upright ?x) )
          (hold (and (in_motion ?l) (on ?e ?u) ) )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference1:basketball) 3 )
)
(:scoring
  3
)
)(define (game mcmc-59-3-4-5-7-14-15-21-27-38-41-46-56-59-60-62-66) (:domain medium-objects-room-v1)
(:setup
  (exists (?i - cube_block)
    (and
      (game-optional
        (not
          (and
            (touch agent ?i)
            (not
              (agent_holds ?i)
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
      (exists (?r - beachball)
        (then
          (hold (and (not (adjacent ?r) ) (not (agent_holds ?r) ) ) )
          (once (agent_holds ?r) )
          (hold (and (not (not (in_motion desk ?r) ) ) (agent_holds ?r) ) )
        )
      )
    )
    (forall (?k - red_dodgeball)
      (and
        (preference preference2
          (exists (?n - cube_block)
            (at-end
              (<= 1 (distance 9))
            )
          )
        )
      )
    )
  )
)
(:terminal
  (<= (count preference2:red) 5 )
)
(:scoring
  (count preference1:dodgeball)
)
)(define (game mcmc-15-1-2-3-8-10-16-21-23-26-30-32-35-36-41-42-45-55-61-63-72-75-76-83-92) (:domain many-objects-room-v1)
(:setup
  (exists (?m - chair ?d - (either teddy_bear dodgeball))
    (forall (?u - ball)
      (game-conserved
        (agent_holds desk bed)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?j - dodgeball)
        (at-end
          (adjacent agent bed)
        )
      )
    )
    (preference preference2
      (exists (?x - dodgeball ?b - ball)
        (then
          (once (agent_holds ?b) )
          (hold-for 6 (and (opposite ?x) (not (agent_holds ?b) ) ) )
          (once (in_motion ?b) )
        )
      )
    )
    (preference preference3
      (exists (?y - dodgeball)
        (then
          (once (in_motion ?y) )
          (once-measure (< 6 (distance desk room_center)) (building_size ?y 1) )
          (once (open ?y) )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference3:pyramid_block) (* 15 (* (total-time) 3 )
      (count-once-per-objects preference1:blue_pyramid_block)
      (* (* (count preference2:yellow:block) 2 )
        5
      )
      10
      2
    )
  )
)
(:scoring
  15
)
)(define (game mcmc-18-4-20-30) (:domain medium-objects-room-v1)
(:setup
  (forall (?l - hexagonal_bin)
    (forall (?a - hexagonal_bin)
      (game-conserved
        (agent_holds desk)
      )
    )
  )
)
(:constraints
  (and
    (forall (?v - dodgeball ?b - ball)
      (and
        (preference preference1
          (at-end
            (agent_holds ?b)
          )
        )
      )
    )
  )
)
(:terminal
  (>= 1 (count preference1:dodgeball) )
)
(:scoring
  (external-forall-maximize
    (* (count preference1:hexagonal_bin) (total-score) )
  )
)
)(define (game mcmc-99-0-1-3-4-8-10-14-15-16-19-23-29-30-31-35-36-57-58-60-61-69-80-81-84-89) (:domain medium-objects-room-v1)
(:setup
  (exists (?t - ball)
    (game-conserved
      (not
        (touch door ?t)
      )
    )
  )
)
(:constraints
  (and
    (forall (?c - (either yellow_cube_block doggie_bed))
      (and
        (preference preference1
          (at-end
            (same_color ?c agent)
          )
        )
        (preference preference2
          (exists (?j - doggie_bed)
            (then
              (once (open ?c) )
              (once (agent_holds ?j) )
              (hold-while (same_color ?c agent) (same_color ?c agent) (and (toggled_on ?j) (agent_holds ?c ?j) ) )
            )
          )
        )
      )
    )
    (preference preference3
      (exists (?j - block)
        (then
          (once (in_motion ?j) )
          (hold (in_motion ?j) )
          (once (in_motion ?j) )
        )
      )
    )
  )
)
(:terminal
  (>= (count-once preference2:dodgeball) 3 )
)
(:scoring
  (count preference1:dodgeball:dodgeball)
)
)(define (game mcmc-64-2-3-10-11-15-16-18-19-34-35-37-39-48-54-56-60-64-73-75-78-87-96-115-120-139-141) (:domain many-objects-room-v1)
(:setup
  (forall (?g - dodgeball)
    (exists (?a - (either doggie_bed cube_block))
      (and
        (game-optional
          (and
            (= (distance 5 desk) (distance ?a 0))
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
      (exists (?d - cube_block ?d - cube_block ?b - triangular_ramp)
        (at-end
          (agent_holds ?d)
        )
      )
    )
    (preference preference2
      (exists (?h - block)
        (at-end
          (in_motion green_golfball)
        )
      )
    )
    (forall (?n - doggie_bed)
      (and
        (preference preference3
          (exists (?f ?r - pillow)
            (then
              (once (not (in_motion ?r) ) )
              (once (is_setup_object ?f) )
              (hold-to-end (on ?n) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference1:beachball) (count preference3:dodgeball) )
)
(:scoring
  (count-once-per-external-objects preference2:dodgeball)
)
)(define (game mcmc-68-0-6-7-14-16-21-22-25-27-34-36-45-59-63-72-79-92-106-107-124-137-146) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (forall (?g - triangular_ramp ?e - shelf)
      (and
        (not
          (on ?e ?n)
        )
        (in ?e ?n)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?n - ball ?w - hexagonal_bin)
        (at-end
          (< (distance ?n ?n) 0.5)
        )
      )
    )
    (preference preference2
      (exists (?l ?w - dodgeball ?d - triangular_ramp ?j - building)
        (then
          (hold (in ?d ?j) )
          (hold (in_motion ?l) )
          (once (not (in ?w ?g) ) )
        )
      )
    )
  )
)
(:terminal
  (and
    (>= (count-once-per-objects preference1:alarm_clock) (- (count-once-per-objects preference2:green) (total-score) ) )
  )
)
(:scoring
  (count-once preference1)
)
)(define (game mcmc-29-0-7-11-15-21-22-29-30-43) (:domain few-objects-room-v1)
(:setup
  (exists (?q - (either desktop cd cd floor))
    (game-conserved
      (< 2 (distance 10 ?q))
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?z - pyramid_block)
        (at-end
          (not
            (not
              (agent_holds ?z)
            )
          )
        )
      )
    )
    (preference preference2
      (at-end
        (< 7 (distance agent agent))
      )
    )
    (forall (?i - book)
      (and
        (preference preference3
          (exists (?z - hexagonal_bin ?l - ball)
            (at-end
              (agent_holds ?i)
            )
          )
        )
      )
    )
  )
)
(:terminal
  (<= (count-shortest preference3:pink_dodgeball) (* (count preference2:doggie_bed) (count preference1:blue) )
  )
)
(:scoring
  (count-longest preference2:dodgeball)
)
)(define (game mcmc-46-0-2-5-12-20-28-38-39-41-55) (:domain many-objects-room-v1)
(:setup
  (exists (?i ?m - (either pyramid_block dodgeball))
    (game-conserved
      (agent_holds ?m)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?c - beachball)
        (then
          (hold (agent_holds ?c) )
        )
      )
    )
  )
)
(:terminal
  (>= (count-overlapping preference1:dodgeball:red) (count preference1:hexagonal_bin) )
)
(:scoring
  7
)
)(define (game mcmc-61-0-1-5-8-11-13-18) (:domain medium-objects-room-v1)
(:setup
  (exists (?k - hexagonal_bin ?u - dodgeball ?x - dodgeball)
    (game-optional
      (< 1 7)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?n - game_object ?q - pillow ?m - hexagonal_bin ?t - dodgeball ?h - hexagonal_bin)
        (at-end
          (and
            (in ?h ?q)
            (not
              (< 1 (distance ?m ?n))
            )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (<= (* 5 10 )
      2
    )
  )
)
(:scoring
  (* (* (count preference1:beachball) 1 )
    (count-once-per-objects preference1:orange)
    2
    (count preference1:beachball)
  )
)
)(define (game mcmc-3-0-3-7-8-9-10-14-16-17-20-23-24-26-31-36-38-39-40-45-55-56-62-65-66-69-71-73-83-84-88-89-91-96-103-106-114-116-121-129-131-140-141-156-158) (:domain medium-objects-room-v1)
(:setup
  (exists (?z - dodgeball)
    (exists (?i - hexagonal_bin)
      (exists (?k - (either dodgeball bridge_block))
        (not
          (game-conserved
            (and
              (= (distance ?i 2) (building_size ?z ?k) (distance ?k ?z door))
              (agent_holds ?z)
              (on bed ?i)
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
      (exists (?n - building ?j - block ?y - dodgeball)
        (then
          (hold-while (and (object_orientation ?y ?n) (in ?y) ) (agent_holds pink_dodgeball) (and (equal_x_position top_drawer ?y) (in_motion ?y) ) )
          (once (on rug ?y) )
          (once (touch ?j block) )
        )
      )
    )
    (preference preference2
      (exists (?o ?b ?a ?r - shelf)
        (then
          (hold (adjacent ?a ?b) )
          (hold (and (< 1 (distance back)) (not (between ?a ?o brown) ) ) )
          (once-measure (in ?o) (distance ?r ?o) )
        )
      )
    )
  )
)
(:terminal
  (>= 20 (count preference1:yellow_cube_block) )
)
(:scoring
  (count preference2:hexagonal_bin)
)
)(define (game mcmc-22-0-3-5-20-26-27-29-33-52-62-70-74-76-84) (:domain medium-objects-room-v1)
(:setup
  (exists (?k - shelf ?d - pyramid_block)
    (forall (?s - (either book key_chain cellphone))
      (game-conserved
        (and
          (on ?s)
          (not
            (agent_holds ?d)
          )
          (in ?k)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?q - building)
      (and
        (preference preference1
          (exists (?x - beachball)
            (then
              (hold (not (agent_holds ?x) ) )
              (hold (not (< 1 (building_size room_center ?q)) ) )
            )
          )
        )
      )
    )
    (preference preference2
      (exists (?g - dodgeball ?p - (either doggie_bed book))
        (at-end
          (and
            (and
              (open ?p)
              (in_motion ?g)
            )
            (in_motion ?g)
          )
        )
      )
    )
  )
)
(:terminal
  (>= (total-score) (+ (<= 2 (+ (count-once-per-objects preference2) (total-score) (count preference1:yellow) )
      )
      1
    )
  )
)
(:scoring
  1
)
)(define (game mcmc-21-1-3-4-5-6-8-10-19-24-26-28-30-40-42-47-57-66-67-75-77-78-81-98-119-122) (:domain medium-objects-room-v1)
(:setup
  (forall (?g - game_object)
    (and
      (or
        (game-conserved
          (in_motion ?g)
        )
        (game-conserved
          (agent_holds ?g)
        )
        (forall (?k - pillow)
          (and
            (game-conserved
              (not
                (agent_holds ?k)
              )
            )
            (game-conserved
              (agent_holds ?k)
            )
            (game-optional
              (in_motion ?k)
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
      (exists (?l - ball)
        (then
          (hold (rug_color_under ?l) )
          (once (not (agent_holds pink_dodgeball) ) )
          (hold (< (distance ) 1) )
        )
      )
    )
  )
)
(:terminal
  (>= 3 (count preference1:basketball) )
)
(:scoring
  (* 8 (+ (+ 7 3 )
    )
  )
)
)(define (game mcmc-65-0-1-2-5-17-24-27-28-29-30-31-32-35-36-37-39-40-48-53-55-61-64-71-76-78-79-80-89-91-96-99-100-103-104-110-112-115-116-121-123-127-128-135-136-139-143-153-158-169-171-187-189-200-208-221-232-239-251-254-257-262-268-281) (:domain many-objects-room-v1)
(:setup
  (exists (?s - wall ?i - teddy_bear)
    (and
      (forall (?h ?p - hexagonal_bin)
        (and
          (exists (?n - cube_block)
            (exists (?k - hexagonal_bin)
              (not
                (and
                  (forall (?o - hexagonal_bin)
                    (game-optional
                      (in_motion ?n)
                    )
                  )
                  (and
                    (and
                      (and
                        (game-conserved
                          (not
                            (agent_holds ?i)
                          )
                        )
                        (game-conserved
                          (or
                            (not
                              (not
                                (broken ?s)
                              )
                            )
                          )
                        )
                      )
                      (and
                        (game-conserved
                          (on rug ?h)
                        )
                      )
                    )
                  )
                  (forall (?m - cube_block)
                    (exists (?v - ball)
                      (game-optional
                        (and
                          (agent_holds ?v)
                          (not
                            (on ?k)
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
    (forall (?v - (either dodgeball cd))
      (and
        (preference preference1
          (exists (?r - hexagonal_bin ?g - teddy_bear)
            (then
              (once (agent_holds ?v) )
              (hold-to-end (agent_holds ?g) )
              (once (not (not (not (forall (?m - dodgeball ?s - (either book game_object cube_block)) (and (not (not (agent_holds ?s) ) ) (in_motion ?m) ) ) ) ) ) )
            )
          )
        )
      )
    )
    (forall (?s - (either golfball cd))
      (and
        (preference preference4
          (exists (?r - ball ?q - hexagonal_bin ?y - (either book cd lamp) ?p - bridge_block ?r - bridge_block)
            (then
              (once (in_motion ?s) )
              (hold (and (in_motion ?q ?y ?q) (in_motion ?r) ) )
              (hold (in_motion ?s) )
            )
          )
        )
      )
    )
    (preference preference7
      (exists (?p - teddy_bear ?b - hexagonal_bin ?o - (either dodgeball ball dodgeball pyramid_block))
        (then
          (hold (agent_holds ?p) )
          (once (> 1 2) )
          (once (agent_holds ?o) )
          (hold (in_motion ?o) )
          (hold-while (agent_holds ?p) (and (and (on ?b) (in_motion ?p) ) (touch ?o desk) ) )
          (once (agent_holds ?o) )
        )
      )
    )
  )
)
(:terminal
  (> (external-forall-maximize 1 ) (* 4 (count-overlapping preference1:white) )
  )
)
(:scoring
  20
)
)(define (game mcmc-25-1-14) (:domain many-objects-room-v1)
(:setup
  (exists (?t - pyramid_block)
    (game-optional
      (agent_holds ?t)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?j - game_object ?t - pillow)
        (then
          (once (agent_holds ?t) )
          (once (agent_holds ?j) )
          (once (object_orientation ?j rug) )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (count-once preference1:golfball) 2 )
    (>= 5 50 )
  )
)
(:scoring
  1
)
)(define (game mcmc-48-0-1-9-12-14-23-28-34-38-42-43-44-49-51-56-57-58-60-61-69-70-72-78-94-108-116-124-127-131-142-147-159-167-183-184) (:domain medium-objects-room-v1)
(:setup
  (game-optional
    (touch pink_dodgeball)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?x - block)
        (at-end
          (in_motion ?x)
        )
      )
    )
    (preference preference2
      (exists (?u - teddy_bear)
        (then
          (once (not (in_motion ?u) ) )
          (hold (in_motion ?u) )
          (hold (= 8 (distance ?u ?u)) )
        )
      )
    )
    (preference preference3
      (exists (?s - (either cd wall) ?p - color)
        (then
          (once (is_setup_object ?p) )
          (once (touch ?s upside_down) )
          (once (and (adjacent ?s ?p) (not (> 1 (distance ?s desk ?p)) ) ) )
        )
      )
    )
  )
)
(:terminal
  (>= 50 (count-once-per-objects preference1:red:purple) )
)
(:scoring
  (+ (total-score) 9 (count preference2:red:beachball) (count-shortest preference3:dodgeball:pink_dodgeball) 9 (- 10 )
  )
)
)(define (game mcmc-71-1-3-5-6-9-15-22-24-32) (:domain few-objects-room-v1)
(:setup
  (game-optional
    (is_setup_object agent)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?o - desk_shelf)
        (then
          (once (not (in_motion ?o) ) )
          (any)
          (hold (in top_shelf ?o) )
        )
      )
    )
  )
)
(:terminal
  (>= (count-overlapping preference1:basketball) 5 )
)
(:scoring
  (* (- 7 )
    (* 5 )
  )
)
)(define (game mcmc-40-0-1-3-5-9-12-13-17-18-19-22-23-27-30-35-36-38-40-42-49-59-60-65-66-67-71-76-80-81-90-94-97-99-109-111-113-120-122-123-125-134-137-140-142-143-145-152-157-170) (:domain few-objects-room-v1)
(:setup
  (forall (?i - hexagonal_bin)
    (game-conserved
      (agent_holds ?i)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?l - hexagonal_bin)
        (at-end
          (agent_holds ?l)
        )
      )
    )
  )
)
(:terminal
  (or
    (< (count preference1:hexagonal_bin) 2 )
    (or
      (>= 10 7 )
      (> 3 8 )
    )
  )
)
(:scoring
  0.7
)
)(define (game mcmc-93-0-1-3-7-12-16-20-30-31-32-47-50-52) (:domain many-objects-room-v1)
(:setup
  (and
    (or
      (exists (?s - (either golfball))
        (game-conserved
          (agent_holds ?s)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?k ?l - dodgeball)
      (and
        (preference preference1
          (exists (?p - hexagonal_bin)
            (then
              (once (on ?k ?p) )
              (hold (not (not (= 2 6) ) ) )
              (hold (and (agent_holds ?l) (and (on desk ?k) (in bed) ) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (= (* (+ (/ (* (total-time) 5 )
          (count preference1:golfball)
        )
        (and
          30
          3
          8
        )
      )
      (+ (- 2 )
        (* 3 (count preference1:beachball) 1 )
      )
    )
    3
  )
)
(:scoring
  (total-score)
)
)(define (game mcmc-77-0-4-8-12-27-28-30-31-32-34-36-39-40-41-47-49-50-51-52-53-54-57-58-66-68-73-82-84-92-96-101-102-103-111-114-115) (:domain few-objects-room-v1)
(:setup
  (exists (?t - cube_block)
    (game-optional
      (in_motion ?t)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?z - curved_wooden_ramp)
        (then
          (once (on agent front) )
          (hold (in_motion ?z) )
          (once (< 5 1) )
        )
      )
    )
    (preference preference2
      (exists (?n - ball ?k - building)
        (at-end
          (agent_holds upright)
        )
      )
    )
  )
)
(:terminal
  (or
    (= (count-once-per-objects preference2:beachball) (count preference2:dodgeball) )
    (or
      (> (count preference2:dodgeball) 15 )
      (or
        (< (+ (* 300 2 )
            (* (* 3 0.5 )
              (count preference2:pink_dodgeball)
              (-
                6
                (count preference2:dodgeball)
              )
              15
              (count-total preference1:rug)
              (* (+ (not (- (count-once-per-objects preference1:cube_block) )
                  )
                  8
                )
                3
                (count-once-per-objects preference2:pink)
              )
              (count-shortest preference1:yellow:dodgeball)
            )
            (count-once-per-objects preference1:pyramid_block)
            (count preference1:basketball)
            (* (- 5 )
              3
            )
            (count preference1:wall)
          )
          1
        )
      )
    )
    (>= (count-once-per-objects preference1:beachball) 30 )
  )
)
(:scoring
  8
)
)(define (game mcmc-88-0-3-11-14-23-32-40) (:domain few-objects-room-v1)
(:setup
  (exists (?m - ball)
    (exists (?a - hexagonal_bin)
      (game-conserved
        (in_motion ?m)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?h - hexagonal_bin ?g - hexagonal_bin)
        (then
          (hold (< 4 1) )
          (once (agent_holds ?h) )
          (hold (not (same_object ?g upright) ) )
        )
      )
    )
  )
)
(:terminal
  (> (count preference1:cylindrical_block) 100 )
)
(:scoring
  (count preference1:beachball:yellow)
)
)(define (game mcmc-69-0-1-4-5-6-11-12-15-16-20-24-25-32-36-37-38-45-60) (:domain few-objects-room-v1)
(:setup
  (exists (?o - dodgeball)
    (game-optional
      (agent_holds ?o)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?a - wall)
        (at-end
          (in ?a rug)
        )
      )
    )
    (forall (?f - wall)
      (and
        (preference preference2
          (exists (?b - dodgeball ?n ?o - dodgeball)
            (at-end
              (>= 1 (distance desk 8))
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference1:blue_dodgeball) 2 )
)
(:scoring
  (total-score)
)
)(define (game mcmc-31-1-3-4-5-6-8-9-12-15-17-18-22-28-31-35-36-37-39-58-63-69-78-83-87-91-102) (:domain medium-objects-room-v1)
(:setup
  (exists (?s - curved_wooden_ramp ?n - ball)
    (forall (?t - red_dodgeball)
      (and
        (exists (?t - hexagonal_bin ?f - red_dodgeball)
          (forall (?n - (either golfball watch triangular_ramp))
            (game-conserved
              (same_type ?n ?f)
            )
          )
        )
        (game-conserved
          (on ?n ?s ?t)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?r - curved_wooden_ramp ?i ?a - hexagonal_bin)
        (at-end
          (object_orientation desk)
        )
      )
    )
    (preference preference2
      (exists (?l - flat_block)
        (then
          (once (agent_holds ?l) )
          (hold-to-end (in_motion ?l) )
          (hold (and (not (in_motion ?l) ) (agent_holds ?l) ) )
        )
      )
    )
  )
)
(:terminal
  (< (count-once-per-objects preference1:golfball) 3 )
)
(:scoring
  (count-shortest preference2:cube_block)
)
)(define (game mcmc-58-1-3-5-9-18-20-21-22-29-35-37-42-45-53-60-63-66-83-85-90-91-102-105-109-117-119-120-129-137) (:domain medium-objects-room-v1)
(:setup
  (exists (?n - dodgeball)
    (forall (?y - chair)
      (game-conserved
        (< (distance ?y 3 agent) (distance ?n bed))
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?o - pillow)
        (then
          (hold (not (not (in_motion ?o) ) ) )
          (hold-while (in_motion ?o) (not (not (agent_holds ?o) ) ) )
          (hold (on ?o) )
        )
      )
    )
    (preference preference3
      (exists (?u - dodgeball)
        (then
          (hold (agent_holds upside_down) )
          (hold (and (on ?u) (or (agent_holds ?u) (same_color sideways block) ) ) )
          (hold (not (in_motion ?u) ) )
        )
      )
    )
  )
)
(:terminal
  (not
    (< 5 16 )
  )
)
(:scoring
  300
)
)(define (game mcmc-73-0-5-6-9-15-16) (:domain few-objects-room-v1)
(:setup
  (exists (?e - (either key_chain golfball))
    (game-conserved
      (agent_holds ?e)
    )
  )
)
(:constraints
  (and
    (forall (?a - rug)
      (and
        (preference preference1
          (exists (?b - dodgeball)
            (then
              (hold-while (in_motion ?a ?b) (agent_holds ?b) (in_motion ?b) (not (agent_holds ?b) ) )
              (once (on ?b) )
              (once (in_motion bed upright) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= 2 (count preference1:golfball:tan) )
)
(:scoring
  (count-once-per-objects preference1:basketball)
)
)(define (game mcmc-5-3-6-15-16-21-24-27-33-39-43-45-47-50-54-57-70-75) (:domain few-objects-room-v1)
(:setup
  (forall (?l - hexagonal_bin ?v - game_object ?n - beachball)
    (game-conserved
      (= 1 8 1 0)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?n - hexagonal_bin)
        (then
          (once (in rug front) )
          (once (in_motion ?n) )
          (once (and (on south_west_corner ?n) (agent_holds ?n) ) )
        )
      )
    )
    (preference preference2
      (exists (?f - game_object)
        (at-end
          (< 2 3)
        )
      )
    )
  )
)
(:terminal
  (>= (count-once-per-objects preference2) (* 2 (count-once-per-external-objects preference1:beachball:dodgeball) )
  )
)
(:scoring
  (external-forall-maximize
    (total-time)
  )
)
)(define (game mcmc-19-1-2-3-4-20-21-22-26-28-33-34) (:domain medium-objects-room-v1)
(:setup
  (forall (?x - teddy_bear ?b - chair ?m - chair)
    (game-conserved
      (and
        (< (distance ?x 2) (x_position ?m desk))
        (not
          (exists (?p - dodgeball)
            (agent_holds ?p)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?d - (either side_table mug) ?j - cube_block)
      (and
        (preference preference1
          (exists (?d - dodgeball)
            (then
              (once (open ?d) )
              (once (and (in_motion ?d) (> 1 (distance ?d ?d)) ) )
              (once (and (not (not (in_motion ?d rug back) ) ) (touch agent ?d) ) )
            )
          )
        )
        (preference preference2
          (exists (?q - (either teddy_bear alarm_clock))
            (at-end
              (and
                (in_motion ?d)
                (touch ?j ?d)
              )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count-shortest preference2:yellow_cube_block) 5 )
)
(:scoring
  6
)
)(define (game mcmc-7-1-4-8-11-12-17-19-23-32-40-42-53-56-60-61-63-67-69-70-76-87-88-94-106-109-115-117-130-142-156-173-191-201-214-221) (:domain many-objects-room-v1)
(:setup
  (forall (?c - hexagonal_bin)
    (game-conserved
      (in_motion ?c agent)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?v - block ?t - block)
        (then
          (hold (agent_holds ?v) )
        )
      )
    )
    (forall (?s - teddy_bear)
      (and
        (preference preference2
          (exists (?e - (either book cube_block))
            (then
              (hold (on ?w agent) )
              (once (agent_holds ?e ?s) )
              (hold (and (and (and (in_motion ?e) (not (faces ?s ?e) ) ) (agent_holds ?e) (in_motion ?s) (exists (?y - shelf ?p - (either dodgeball cylindrical_block)) (not (in_motion ?p) ) ) ) (in ?e) (in_motion ?e) ) )
            )
          )
        )
        (preference preference3
          (exists (?y ?j ?x ?a ?o ?t - doggie_bed)
            (then
              (hold (in_motion green_golfball) )
              (once (and (not (and (< (distance ?a 3) 1) (in_motion ?a ?x) ) ) (not (on ?j) ) ) )
              (hold-while (adjacent ?y floor) (not (and (and (object_orientation ?o ?t) (agent_holds ?o ?y) ) (is_setup_object ?o) ) ) (in_motion ?t ?a) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (> 3 1 )
)
(:scoring
  (* (count preference1:beachball) (+ (count preference3:purple) (count-once preference2:dodgeball) (count preference2:beachball) )
  )
)
)(define (game mcmc-20-2-3-6-9-11-15-20-24-25-28-35-36-43-63-64-80-98-115-116) (:domain medium-objects-room-v1)
(:setup
  (exists (?e - hexagonal_bin ?m - ball ?l - curved_wooden_ramp ?l - curved_wooden_ramp)
    (exists (?o - ball)
      (exists (?j - hexagonal_bin ?t - block)
        (game-optional
          (same_color ?o ?e)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?h - block ?j - dodgeball ?o - dodgeball ?t - ball ?u - doggie_bed)
        (then
          (once (agent_holds ?t) )
        )
      )
    )
  )
)
(:terminal
  (>= 5 10 )
)
(:scoring
  (count-longest preference1:basketball)
)
)(define (game mcmc-16-0-4-5-6-7-9-13-25-26-27-33-37-45-46) (:domain few-objects-room-v1)
(:setup
  (and
    (exists (?m - hexagonal_bin)
      (game-conserved
        (not
          (in ?m)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?a - pillow ?o - triangular_ramp ?v - cube_block)
        (then
          (hold (> (distance ?o 0 3) (distance ?a ?o)) )
          (once (not (< (distance ?a ?o) 1) ) )
          (once (not (and (not (not (in_motion ?g) ) ) (not (agent_holds ?g) ) ) ) )
          (hold-while (agent_holds ?a) (in_motion ?a) )
        )
      )
    )
    (forall (?b - hexagonal_bin)
      (and
        (preference preference2
          (exists (?v - dodgeball)
            (at-end
              (agent_holds ?v)
            )
          )
        )
      )
    )
  )
)
(:terminal
  (not
    (> (* (count preference1:green) (count-once-per-objects preference2:basketball) )
      1
    )
  )
)
(:scoring
  (count preference1:cylindrical_block)
)
)(define (game mcmc-38-1-4-6-7-8-9-10-12-16-17-18-19-21-24-26-31-36-37-38-55-56-63-65-70-74-76-79-90-92-96-103-105-114-116-122-133-142-159-165-167) (:domain medium-objects-room-v1)
(:setup
  (exists (?t - hexagonal_bin)
    (and
      (game-optional
        (touch agent)
      )
    )
  )
)
(:constraints
  (and
    (forall (?s - dodgeball)
      (and
        (preference preference1
          (exists (?j - ball ?e - doggie_bed)
            (then
              (once (agent_holds ?e) )
              (hold (in_motion ?s) )
              (once (agent_holds ?s) )
            )
          )
        )
        (preference preference2
          (exists (?k - hexagonal_bin)
            (at-end
              (in_motion ?s)
            )
          )
        )
      )
    )
    (forall (?x - shelf ?r - ball)
      (and
        (preference preference2
          (then
            (hold-while (and (in_motion ?j) (in_motion ?r) ) (equal_z_position ?r desk) )
            (hold (and (on ?x ?r) (adjacent ?x) ) )
            (once (agent_holds ?r) )
          )
        )
        (preference preference4
          (exists (?w - dodgeball)
            (at-end
              (agent_holds ?r)
            )
          )
        )
      )
    )
    (preference preference4
      (exists (?q - bridge_block ?j - block)
        (then
          (hold-for 7 (and (broken ?q) (in_motion ?q) ) )
          (once (agent_holds ?q) )
          (hold (agent_holds ?q agent) )
        )
      )
    )
  )
)
(:terminal
  (= (count preference1:hexagonal_bin) 1 )
)
(:scoring
  (total-time)
)
)(define (game mcmc-57-2-4-6-10-23-32-40) (:domain medium-objects-room-v1)
(:setup
  (exists (?n - building ?y - doggie_bed ?x - hexagonal_bin ?l - cube_block)
    (game-conserved
      (in ?y front)
    )
  )
)
(:constraints
  (and
    (forall (?p - doggie_bed)
      (and
        (preference preference1
          (exists (?i - hexagonal_bin)
            (then
              (once (agent_holds ?p) )
              (hold (above ?p ?i) )
              (once (rug_color_under ?p) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (* 3 180 )
    (count preference1:green)
  )
)
(:scoring
  (count preference1)
)
)(define (game mcmc-45-0-2-6-10-11-20-23-24-27) (:domain few-objects-room-v1)
(:setup
  (forall (?l - bridge_block ?e - hexagonal_bin ?e - cube_block)
    (and
      (game-conserved
        (< 1 1)
      )
      (forall (?o - (either cube_block key_chain))
        (game-conserved
          (in front ?l)
        )
      )
      (game-conserved
        (not
          (agent_holds ?e)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?h - game_object)
      (and
        (preference preference1
          (exists (?t - curved_wooden_ramp)
            (then
              (once (on front) )
              (once (not (< (distance ?t room_center) 5) ) )
              (hold-for 10 (on ?t ?h) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count-once-per-objects preference1:beachball) 1 )
)
(:scoring
  (count preference1:golfball)
)
)(define (game mcmc-54-4-5-7-9-12-16-19-21-25-26-36-37-40-50-52-56-58-65-69) (:domain few-objects-room-v1)
(:setup
  (forall (?f - ball)
    (and
      (game-optional
        (in_motion ?f)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?x - blinds)
        (then
          (hold (in_motion ?x) )
          (once (in_motion front) )
          (hold-while (>= (distance ?x ?x) (distance ?x ?x)) (not (on ?x agent) ) (agent_holds ?x) )
        )
      )
    )
    (forall (?x - hexagonal_bin)
      (and
        (preference preference2
          (exists (?l - (either pillow cube_block blue_cube_block pencil book doggie_bed pyramid_block) ?y - dodgeball)
            (then
              (hold (touch upright) )
              (hold (not (< 2 (distance ?y ?x)) ) )
              (once (in_motion ?p) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (* 4 3 )
      4
    )
    (< (count preference2:book:hexagonal_bin) 1 )
  )
)
(:scoring
  (/
    (* (count preference2:basketball) (count-unique-positions preference1:yellow) )
    (count-measure preference2:doggie_bed)
  )
)
)(define (game mcmc-52-0-1-3-6-10-11-26-29-32-46-54-60) (:domain few-objects-room-v1)
(:setup
  (exists (?u - game_object)
    (exists (?n - ball ?m - (either laptop pyramid_block))
      (game-conserved
        (and
          (not
            (in ?u)
          )
          (same_color ?m ?n)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?z - block)
        (then
          (hold (agent_holds ?z) )
          (once (in_motion ?z) )
          (once (not (and (agent_holds ?z) ) ) )
          (hold (in_motion ?z) )
        )
      )
    )
  )
)
(:terminal
  (= 3 1 )
)
(:scoring
  (count-same-positions preference1:yellow_cube_block)
)
)(define (game mcmc-66-0-1-2-4-5-7-9-10-12-13-20-21-22-23-30-31-34-36-49-52-53-59-61-67-83-86-93-99-100-104-110-126-127-137-144-146-152-163-169-175-181-185-206-210-211-214-228) (:domain many-objects-room-v1)
(:setup
  (game-conserved
    (in blinds)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?f - cube_block ?m - hexagonal_bin)
        (then
          (once (in_motion ?f) )
          (once (in_motion ?f ?m) )
          (hold (touch agent ?m) )
        )
      )
    )
    (preference preference2
      (exists (?p - (either beachball cube_block))
        (then
          (hold (agent_holds rug bed) )
          (hold (in_motion ?p) )
          (once (not (not (in_motion ?p) ) ) )
        )
      )
    )
    (preference preference3
      (exists (?m - block)
        (at-end
          (not
            (not
              (on ?m ?m)
            )
          )
        )
      )
    )
    (preference preference4
      (exists (?q - teddy_bear)
        (then
          (hold (and (not (in_motion ?q) ) (not (and (in_motion ?q) (adjacent ?q rug) ) ) ) )
          (once (not (exists (?u - hexagonal_bin) (and (in_motion ?q) (not (in_motion ?u ?q) ) ) ) ) )
          (once (agent_holds ?q) )
          (once (not (exists (?e - teddy_bear) (and (in_motion ?q) (and (agent_holds ?e) (same_type ?e) ) ) ) ) )
        )
      )
    )
    (preference preference5
      (exists (?f ?t - hexagonal_bin)
        (then
          (once (not (same_type ?t ?f) ) )
          (hold (not (agent_holds upside_down) ) )
          (once (not (>= 1 1) ) )
        )
      )
    )
    (preference preference6
      (exists (?w - dodgeball)
        (then
          (once (not (and (and (agent_holds ?w) (not (not (agent_holds ?w) ) ) ) (in_motion ?w) ) ) )
          (hold (agent_holds ?w desk) )
          (once (agent_holds ?w) )
        )
      )
    )
  )
)
(:terminal
  (> (count preference2) (total-score) )
)
(:scoring
  (count preference1:book)
)
)(define (game mcmc-8-3-4-5-8-13-18-19-26-28-29-33-39-41-44-52) (:domain few-objects-room-v1)
(:setup
  (game-optional
    (<= 4 1)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?y - cube_block)
        (then
          (hold-while (and (not (and (not (adjacent desk) ) (in_motion ?y) ) ) (object_orientation ?y yellow) ) (and (and (not (adjacent ?y) ) (in_motion ?y) ) (on ?y ?y) ) )
          (once (and (not (and (not (agent_holds ?y) ) (equal_x_position rug ?y) ) ) ) )
          (once (not (< 0.5 1) ) )
        )
      )
    )
    (forall (?y - cube_block ?p ?f - game_object)
      (and
        (preference preference2
          (exists (?v - (either golfball chair) ?i - cube_block ?q - curved_wooden_ramp)
            (then
              (hold (and (agent_holds ?i front) (agent_holds ?f) (in ?i ?p) ) )
              (hold (adjacent ?y ?q) )
              (once (< (distance ?v room_center) (distance room_center agent)) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (< (+ (- (total-score) )
        (+ (count preference2:beachball) (count-once-per-objects preference1:doggie_bed) )
      )
      (count-once preference2:yellow_cube_block)
    )
    (count preference1:dodgeball)
  )
)
(:scoring
  (-
    2
    (total-score)
  )
)
)(define (game mcmc-56-0-1-5-8-15-17-25-35-37-38-40-41-42-47-53-54-56) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (exists (?g - teddy_bear)
      (not
        (not
          (in_motion ?g)
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
          (once (or (in_motion ?g) (between ?g) ) )
          (once (< 4 9) )
          (once (agent_holds ?g) )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= (count preference1:golfball) 1 )
    (not
      (>= 3 0 )
    )
    (>= 1 0.5 )
  )
)
(:scoring
  3
)
)(define (game mcmc-51-0-1-2-9-11-12-13-14-31-40-49-52-53-54-60) (:domain few-objects-room-v1)
(:setup
  (exists (?d - building)
    (and
      (game-optional
        (not
          (in ?d)
        )
      )
    )
  )
)
(:constraints
  (and
    (forall (?h - chair)
      (and
        (preference preference1
          (exists (?d - dodgeball ?k - dodgeball)
            (then
              (once (and (in ?h ?k) (in_motion ?d) ) )
              (hold (agent_holds ?k) )
              (once (agent_holds ?h) )
            )
          )
        )
      )
    )
    (preference preference2
      (exists (?f - building)
        (then
          (once (agent_holds ?f) )
          (hold (and (in ?f ?q) (in_motion ?f) ) )
          (any)
        )
      )
    )
  )
)
(:terminal
  (>= (+ (* 5 20 )
      (count-overlapping preference2:pink)
    )
    (* (total-score) 2 )
  )
)
(:scoring
  (count preference1:blue_dodgeball)
)
)(define (game mcmc-70-0-1-6-12-13-15-20-24-34-35-44-49-50-56-59) (:domain many-objects-room-v1)
(:setup
  (and
    (forall (?n - hexagonal_bin)
      (exists (?y - dodgeball ?y - hexagonal_bin)
        (exists (?p - (either dodgeball pyramid_block cellphone))
          (game-optional
            (and
              (in_motion bed ?n)
              (not
                (in_motion ?p)
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
    (forall (?s - dodgeball)
      (and
        (preference preference1
          (exists (?q - red_dodgeball ?v - wall)
            (then
              (hold (in_motion ?v ?s) )
              (once (agent_holds ?s) )
              (once (on bed) )
            )
          )
        )
      )
    )
    (forall (?o - hexagonal_bin)
      (and
        (preference preference2
          (exists (?c ?z - sliding_door)
            (then
              (once (< 2 2) )
              (hold (< (distance ?o 7) 1) )
              (once (and (same_type ?c agent) (in_motion ?o ?z) (agent_holds ?o) ) )
            )
          )
        )
      )
    )
    (preference preference3
      (exists (?k - dodgeball)
        (then
          (hold-while (in rug) (not (on ?k) ) (agent_holds ?k) )
          (once (agent_holds ?k) )
          (hold (not (agent_holds ?k) ) )
        )
      )
    )
  )
)
(:terminal
  (>= (- (+ (/ 10 (= (+ (count-once preference1:beachball) (count-once-per-external-objects preference3:dodgeball) )
          )
        )
        (total-time)
        3
        (external-forall-maximize
          (count preference2)
        )
        3
        (* 4 (count-once preference3:dodgeball) )
      )
    )
    15
  )
)
(:scoring
  5
)
)(define (game mcmc-94-2-5-8-20-21-24-39-41-47-51-52-65-74-76) (:domain medium-objects-room-v1)
(:setup
  (exists (?n - hexagonal_bin ?q - hexagonal_bin)
    (forall (?i - beachball ?x - hexagonal_bin)
      (game-optional
        (in_motion rug bed)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?f - hexagonal_bin)
        (then
          (hold (< 8 (distance 4 1)) )
          (hold (same_type ?f agent) )
          (hold (not (adjacent desk) ) )
        )
      )
    )
    (preference preference2
      (exists (?x - game_object)
        (then
          (once (agent_holds ?x) )
          (once (agent_holds ?x) )
          (once (<= 1 2) )
        )
      )
    )
  )
)
(:terminal
  (>= 2 (count-once-per-objects preference1:dodgeball:beachball) )
)
(:scoring
  (* (count-once-per-objects preference2:yellow) 3 (external-forall-maximize (count preference2:dodgeball:bed) ) )
)
)(define (game mcmc-90-2-5-7-9-10-12-24-31-38-54-71) (:domain medium-objects-room-v1)
(:setup
  (and
    (forall (?v - cube_block ?c ?o - cube_block ?a - dodgeball ?o - pillow ?p - dodgeball)
      (forall (?h - ball)
        (forall (?y - (either golfball key_chain))
          (exists (?x - ball)
            (exists (?s - chair ?w - cube_block)
              (forall (?q - teddy_bear)
                (and
                  (game-optional
                    (and
                      (and
                        (and
                          (on ?v ?w)
                          (in_motion ?p)
                        )
                        (agent_holds ?o)
                      )
                      (forall (?o - teddy_bear)
                        (= (distance ?c ?c) 1)
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
      (< (distance ) 0)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?c ?h - chair)
        (at-end
          (< (distance ?h side_table) (distance bed))
        )
      )
    )
  )
)
(:terminal
  (not
    (and
      (>= (count-once-per-objects preference1:pink) 10 )
    )
  )
)
(:scoring
  1
)
)(define (game mcmc-1-1-3-5-7-9-10-12-23-27-35-40-45-56-61-64-78-80-93-107-124) (:domain few-objects-room-v1)
(:setup
  (exists (?g - (either dodgeball cd))
    (and
      (game-optional
        (agent_holds ?g)
      )
    )
  )
)
(:constraints
  (and
    (forall (?v - dodgeball)
      (and
        (preference preference1
          (exists (?i - yellow_pyramid_block)
            (then
              (hold (> (distance 2 ?v) (distance ?v 6)) )
              (hold-while (on ?i) (agent_holds ?i) )
              (any)
            )
          )
        )
      )
    )
    (forall (?r - ball)
      (and
        (preference preference4
          (exists (?g - dodgeball ?o - (either dodgeball cd))
            (then
              (hold (agent_holds ?r) )
              (once (in_motion ?o) )
              (once (agent_holds ?r) )
            )
          )
        )
        (preference preference5
          (exists (?q - hexagonal_bin)
            (then
              (once (touch blue) )
              (once (in_motion ?r) )
              (once (rug_color_under ?r) )
              (once (touch ?q ?r) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (not
    (>= 2 (* (- 18 )
        (count preference1:cube_block)
      )
    )
  )
)
(:scoring
  (- 10 )
)
)(define (game mcmc-74-2-4-7-14-18-20-25-38) (:domain many-objects-room-v1)
(:setup
  (forall (?i - building ?u - shelf)
    (and
      (game-conserved
        (not
          (< 6 1)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?q - chair)
        (then
          (once (is_setup_object ?q) )
          (any)
          (hold (and (not (in_motion ?q) ) (agent_holds ?q) ) )
        )
      )
    )
    (forall (?u - ball ?y ?s - (either pencil cd laptop))
      (and
        (preference preference2
          (exists (?t - book)
            (then
              (once (on ?s) )
              (hold (adjacent ?p ?t) )
              (hold (not (agent_holds ?y) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count-once-per-objects preference2:basketball) (+ (total-time) (* (+ (count-once-per-objects preference2:dodgeball) (count-overlapping preference1:white) (external-forall-maximize (count preference1:orange) ) )
        (count preference1:dodgeball)
      )
    )
  )
)
(:scoring
  (* 4 (* (count preference2:hexagonal_bin) )
  )
)
)(define (game mcmc-60-0-10-14-16-22-23-27-31-36-53-70-75-82-87-92-96-98) (:domain medium-objects-room-v1)
(:setup
  (exists (?v - green_triangular_ramp)
    (and
      (exists (?x - dodgeball)
        (exists (?h - building)
          (exists (?l - hexagonal_bin)
            (game-conserved
              (not
                (not
                  (and
                    (not
                      (not
                        (on door ?v)
                      )
                    )
                    (adjacent ?h)
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
      (exists (?p - building ?m - hexagonal_bin ?r - bridge_block ?x - flat_block ?f - dodgeball)
        (then
          (once (agent_holds ?r) )
          (once (not (in_motion ?x) ) )
          (hold (agent_holds ?x) )
        )
      )
    )
    (preference preference2
      (exists (?y - hexagonal_bin)
        (then
          (once (< 1 2) )
          (hold (not (agent_holds ?y) ) )
          (once (< (distance ?y 0) 1) )
        )
      )
    )
  )
)
(:terminal
  (>= 1 (count preference1:basketball) )
)
(:scoring
  (count preference2:tan)
)
)(define (game mcmc-82-6-7-10-15-25-29-32-33) (:domain few-objects-room-v1)
(:setup
  (exists (?j - wall)
    (and
      (forall (?e - hexagonal_bin)
        (and
          (and
            (and
              (game-optional
                (not
                  (< (distance ?e) (distance room_center ?j))
                )
              )
              (game-optional
                (same_type ?j)
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
    (forall (?w - block)
      (and
        (preference preference1
          (exists (?p - hexagonal_bin ?n - ball)
            (then
              (once (in_motion ?n) )
              (once (on ?p) )
              (once (in_motion ?w) )
            )
          )
        )
      )
    )
    (preference preference2
      (exists (?t - wall)
        (then
          (once (in_motion sideways) )
          (once (touch ?t) )
        )
      )
    )
  )
)
(:terminal
  (>= (* 3 )
    (/
      (* (+ (count-once-per-objects preference1:yellow) 10 )
        100
      )
      (* 2 3 )
    )
  )
)
(:scoring
  (count-once-per-objects preference2:yellow)
)
)(define (game mcmc-17-3-4-5-10-17-27-34-37-57-58) (:domain few-objects-room-v1)
(:setup
  (game-conserved
    (exists (?w - chair)
      (agent_holds ?w)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?g - chair)
        (then
          (once (on ?g agent) )
          (once (< 1 1) )
          (once (and (same_type ?g) (or (agent_holds ?g) ) ) )
        )
      )
    )
  )
)
(:terminal
  (>= (count-once preference1:yellow) (count-once-per-external-objects preference1:dodgeball:dodgeball) )
)
(:scoring
  (count-once preference1:pink:blue_dodgeball)
)
)(define (game mcmc-80-0-1-3-4-5-6-11-12-13-14-18-19-28-29-33-34-35-45-48-55-57-60-65-67-79-89-93-99-106-111) (:domain many-objects-room-v1)
(:setup
  (exists (?w - cube_block)
    (and
      (and
        (forall (?f - cylindrical_block ?u - red_dodgeball)
          (game-conserved
            (agent_holds ?f)
          )
        )
      )
      (and
        (and
          (exists (?t - game_object)
            (not
              (game-conserved
                (in_motion ?t)
              )
            )
          )
        )
      )
      (game-conserved
        (and
          (not
            (agent_holds ?w)
          )
          (in ?w)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?q - triangular_ramp)
        (at-end
          (not
            (not
              (adjacent_side ?q)
            )
          )
        )
      )
    )
    (preference preference2
      (exists (?u ?d ?i ?w - hexagonal_bin ?i - hexagonal_bin ?i - hexagonal_bin)
        (then
          (hold (on agent ?i) )
          (once (same_type bed) )
          (hold-for 6 (in ?w ?d) )
        )
      )
    )
    (preference preference3
      (exists (?c - dodgeball)
        (then
          (once (agent_holds ?c) )
          (once (agent_holds ?c) )
          (once (agent_holds ?c) )
        )
      )
    )
  )
)
(:terminal
  (>= (count-measure preference3:blue:hexagonal_bin) 5 )
)
(:scoring
  (count preference1:tall_cylindrical_block:cube_block)
)
)(define (game mcmc-27-0-1-7-12-14-15-17-25-29-30-32-36-39-40-44-49-53-57-59-61) (:domain many-objects-room-v1)
(:setup
  (forall (?z - hexagonal_bin)
    (and
      (and
        (or
          (game-conserved
            (> (distance 10 ?z) 1)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?i - ball)
        (at-end
          (not
            (not
              (adjacent ?i)
            )
          )
        )
      )
    )
    (forall (?f - (either dodgeball) ?z - hexagonal_bin ?o - beachball)
      (and
        (preference preference2
          (exists (?j - building ?m - cube_block)
            (then
              (hold (not (touch ?j pink_dodgeball) ) )
              (hold-to-end (in_motion ?o) )
              (once-measure (not (agent_holds ?m ?o) ) (distance ?m 7) )
            )
          )
        )
        (preference preference3
          (exists (?r - hexagonal_bin)
            (then
              (hold (not (in_motion ?o) ) )
              (once (not (in_motion ?z) ) )
              (hold (not (and (agent_holds green_golfball ?z) (on ?f) ) ) )
            )
          )
        )
        (preference preference4
          (exists (?f ?u - hexagonal_bin)
            (then
              (once (in ?f) )
              (once (on ?u) )
              (once (agent_holds bridge_block) )
            )
          )
        )
      )
    )
    (preference preference5
      (exists (?p - dodgeball ?m - curved_wooden_ramp)
        (then
          (hold (not (rug_color_under ?m) ) )
          (hold (not (not (and (is_setup_object ?p) (same_object ?m) ) ) ) )
          (hold (in_motion ?p) )
          (once (touch ?m) )
        )
      )
    )
    (preference preference6
      (exists (?n - dodgeball)
        (then
          (hold (in_motion ?n) )
          (once (< 1 (distance desk ?n)) )
          (hold (on block floor back) )
        )
      )
    )
  )
)
(:terminal
  (>= (count preference1:basketball) (* (* (* (count-once preference3:blue_dodgeball) (+ (count preference3:pink:dodgeball) (* (count-once-per-objects preference2) (- (count preference4:dodgeball) )
            )
          )
        )
        (* 4 (not (= (count preference6:golfball) 10 )
          )
        )
        (count preference1:beachball:beachball)
        (* 8 (count preference6:yellow) )
        (+ 10 (count-once-per-objects preference1:yellow) )
        (count-once-per-objects preference5:pink)
      )
      (* 1 4 (not (* (count preference2:purple) 6 )
        )
      )
    )
  )
)
(:scoring
  2
)
)(define (game mcmc-86-3-6-7-8-9-18-20-25-32-39-40-47-50-56-57-64-65-72-76-78-84-91-96) (:domain few-objects-room-v1)
(:setup
  (game-optional
    (in_motion agent)
  )
)
(:constraints
  (and
    (forall (?c ?m ?b - (either dodgeball) ?d - cube_block)
      (and
        (preference preference1
          (exists (?t - chair ?i - pillow)
            (then
              (once (agent_holds ?i ?b) )
              (once (agent_holds ?m) )
              (once (in_motion ?d) )
            )
          )
        )
      )
    )
    (forall (?c - dodgeball)
      (and
        (preference preference2
          (exists (?d - dodgeball)
            (at-end
              (between ?c)
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
      (>= 10 4 )
      (> (external-forall-maximize (count-once-per-objects preference1:yellow) ) (- 300 )
      )
    )
    (or
      (>= 10 5 )
      (> (total-time) 8 )
    )
  )
)
(:scoring
  (>= (count preference2:dodgeball) (count preference1:purple) )
)
)(define (game mcmc-0-5-8-17) (:domain medium-objects-room-v1)
(:setup
  (game-optional
    (in_motion side_table floor)
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?w - cube_block)
        (then
          (once (agent_holds ?w bed) )
          (hold (agent_holds ?w) )
          (once (and (faces ?w) (in_motion ?w) ) )
        )
      )
    )
  )
)
(:terminal
  (>= (+ (count preference1:book) )
    2
  )
)
(:scoring
  (count-increasing-measure preference1:blue_dodgeball)
)
)(define (game mcmc-42-0-8-10-11-12-13-14-19-25-40-43-44-51-60-68-73-76-87-91-94-95-97-99-100-102-105-123-124-126-132-140-150-154-162-182) (:domain many-objects-room-v1)
(:setup
  (exists (?l - wall)
    (game-conserved
      (not
        (not
          (< (distance 4 desk) (distance ?l ?l))
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?h - dodgeball ?s - cube_block)
        (then
          (once (not (and (= (distance 10 room_center) 1 1) (and (not (in_motion ?s) ) (exists (?y - dodgeball) (agent_holds ?y) ) ) ) ) )
          (once (< 2 1) )
          (once (not (and (agent_holds ?s) (and (in_motion ?h ?s) (agent_holds ?h) ) (not (not (in_motion ?h) ) ) (not (on ?s) ) ) ) )
        )
      )
    )
    (preference preference2
      (exists (?u - ball)
        (then
          (hold (agent_holds ?u) )
          (once (and (not (agent_holds ?u ?u) ) (same_color ?u) ) )
          (forall-sequence (?l - hexagonal_bin ?f - cylindrical_block)
            (then
              (hold (and (agent_holds ?l ?u) (not (on ?f) ) (or (agent_holds ?f) (faces ?f ?l) ) ) )
              (once (forall (?p - (either game_object dodgeball teddy_bear)) (agent_holds ?p) ) )
              (hold (and (and (open ?l) (on ?f ?u) ) (in ?u) ) )
            )
          )
        )
      )
    )
    (preference preference3
      (exists (?m - doggie_bed)
        (then
          (hold (in_motion ?m ?m) )
          (hold (and (< 1 (distance ?m desk)) (not (not (exists (?t - ball) (adjacent ?t) ) ) ) ) )
          (once (and (= (distance ?m ?m) 1) ) )
        )
      )
    )
  )
)
(:terminal
  (< (count preference1:golfball) (count preference3:pink) )
)
(:scoring
  (count preference2:golfball)
)
)(define (game mcmc-62-0-1-2-5-8-9-13-14-17-18-20-24-25-40-46-47-64-66-81-96-109) (:domain many-objects-room-v1)
(:setup
  (not
    (forall (?s - dodgeball)
      (game-conserved
        (agent_holds ?s)
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?r - hexagonal_bin)
        (then
          (once (agent_holds ?r) )
          (once (in ?r) )
          (once (not (in_motion ?r) ) )
        )
      )
    )
    (preference preference2
      (exists (?l - hexagonal_bin)
        (then
          (once (same_object ?l agent) )
          (once (in desk) )
          (hold (not (not (is_setup_object ?l) ) ) )
        )
      )
    )
    (preference preference3
      (exists (?m - hexagonal_bin)
        (at-end
          (agent_holds ?m)
        )
      )
    )
  )
)
(:terminal
  (= (+ (* (count preference3:red) (count-once preference2:hexagonal_bin) (+ (count preference3:pink) (count-overlapping preference1) (count-overlapping preference3:dodgeball) 10 10 6 3 30 7 )
      )
      2
      10
    )
    (- 1 )
  )
)
(:scoring
  (* 3 5 )
)
)(define (game mcmc-35-2-3-4-5-6-7-8-10-11-12-14-21) (:domain many-objects-room-v1)
(:setup
  (exists (?j - tall_cylindrical_block)
    (game-conserved
      (or
        (and
          (on ?j ?j)
          (object_orientation ?j)
        )
        (same_object front ?j)
        (open door)
      )
    )
  )
)
(:constraints
  (and
    (forall (?h - ball)
      (and
        (preference preference1
          (exists (?c - chair)
            (at-end
              (not
                (not
                  (and
                    (agent_holds ?h)
                    (< 1 9)
                  )
                )
              )
            )
          )
        )
      )
    )
    (preference preference2
      (exists (?s - dodgeball ?n - (either key_chain dodgeball) ?g - color ?g - wall ?l - dodgeball)
        (then
          (once (not (< 1 (distance ?l desk)) ) )
          (once (agent_holds ?s) )
          (once (not (agent_holds ?l) ) )
        )
      )
    )
    (preference preference3
      (exists (?c - hexagonal_bin)
        (then
          (once (on ?c) )
          (once (and (not (in_motion ?c) ) (on ?c upside_down) ) )
          (once (adjacent top_shelf) )
        )
      )
    )
  )
)
(:terminal
  (or
    (or
      (> 5 5 )
      (<= 2 (count-once-per-objects preference2:red:dodgeball) )
      (<= (count-once-per-external-objects preference1:block) 2 )
    )
    (>= 2 (count-once-per-objects preference3:dodgeball) )
    (>= (count-once-per-objects preference3:dodgeball:blue_cube_block) 3 )
    (>= 15 (count-once-per-objects preference1:cylindrical_block) )
  )
)
(:scoring
  (count-once-per-objects preference2:wall)
)
)(define (game mcmc-41-0-1-11-14-27-30-31-37-42-55-60-66-69-72-75) (:domain many-objects-room-v1)
(:setup
  (forall (?x - ball ?q - building)
    (and
      (and
        (exists (?x - curved_wooden_ramp ?g - golfball)
          (game-optional
            (agent_holds ?g)
          )
        )
      )
      (game-conserved
        (not
          (and
            (= (distance ?x 3 desk) (distance 10 agent) (distance ?x ?q desk))
            (in_motion desk)
          )
        )
      )
      (and
        (forall (?z - hexagonal_bin)
          (game-conserved
            (rug_color_under ?q agent)
          )
        )
      )
      (forall (?i - dodgeball)
        (and
          (forall (?b - hexagonal_bin)
            (game-optional
              (on ?i)
            )
          )
        )
      )
      (or
        (forall (?r - hexagonal_bin)
          (game-conserved
            (in ?x ?r)
          )
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?m - game_object)
        (then
          (once (not (not (< 1 1) ) ) )
          (once (in_motion ?m) )
        )
      )
    )
  )
)
(:terminal
  (>= (* (count preference1:dodgeball) (- 15 )
    )
    6
  )
)
(:scoring
  (* 5 (count-once-per-objects preference1:hexagonal_bin:block) )
)
)(define (game mcmc-53-6-10-11-13-15-20-21-23-27-34-35-38-45-48) (:domain few-objects-room-v1)
(:setup
  (exists (?q - dodgeball)
    (exists (?g - game_object)
      (game-optional
        (not
          (in_motion ?q)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?w - (either cellphone golfball) ?u - hexagonal_bin ?q - hexagonal_bin)
        (then
          (once (in ?q pink_dodgeball) )
          (once (exists (?l - golfball ?d - rug) (and (on ?d ?q) (in_motion ?u ?d) ) ) )
          (once (>= (distance ?w ?w) 5) )
        )
      )
    )
    (forall (?t - hexagonal_bin ?l - (either laptop golfball))
      (and
        (preference preference2
          (exists (?o - block ?d - hexagonal_bin)
            (then
              (once (not (in_motion ?l) ) )
              (hold-while (not (and (in_motion ?l) (on desk ?t) ) ) (on ?k ?l) (agent_holds ?l) (adjacent_side ?k) )
              (once (in ?l) )
            )
          )
        )
        (preference preference3
          (exists (?z - block ?a - dodgeball)
            (at-end
              (on ?l bed)
            )
          )
        )
        (preference preference4
          (exists (?o - curved_wooden_ramp ?c - (either book key_chain blue_cube_block))
            (then
              (once (not (not (agent_holds ?c pink_dodgeball) ) ) )
              (hold (not (not (in_motion ?t) ) ) )
              (hold (is_setup_object ?l ?l) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (or
    (> 5 3 )
    (or
      (or
        (or
          (> (* (+ 2 5 )
              (total-time)
              5
              (* (count preference3:pink) (total-score) )
            )
            (count-same-positions preference2:book)
          )
        )
        (or
          (>= 5 (count preference2:doggie_bed) )
          (= (count-once-per-objects preference4:dodgeball:purple) (count preference1) )
        )
      )
      (and
        (>= (count preference2:blue_pyramid_block:golfball) 10 )
        (>= 6 3 )
      )
    )
  )
)
(:scoring
  (* 5 6 )
)
)(define (game mcmc-87-0-8-12-23-33-43-45-57-61-68-70) (:domain few-objects-room-v1)
(:setup
  (exists (?y - hexagonal_bin)
    (game-optional
      (< (distance desk ?y) 1)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?o - dodgeball ?o - curved_wooden_ramp ?v - (either triangle_block cellphone) ?l - cube_block)
        (then
          (hold (exists (?f - hexagonal_bin) (on desk) ) )
          (once (agent_holds ?v) )
          (hold (same_color ?o ?l) )
        )
      )
    )
    (preference preference2
      (exists (?m - block)
        (then
          (once (on ?m) )
          (once (not (on ?m ?m) ) )
          (hold-while (and (exists (?s - teddy_bear) (between ?s) ) (touch ?m ?m) ) (and (equal_z_position ?m) (agent_holds ?m) ) )
        )
      )
    )
  )
)
(:terminal
  (>= 4 3 )
)
(:scoring
  (* (count preference1) 5 1 (count preference1:basketball) (* (count preference2:beachball) 10 (count preference2:dodgeball) 2 (- 20 )
      5
    )
  )
)
)(define (game mcmc-97-3-4-6-11-13-14) (:domain medium-objects-room-v1)
(:setup
  (and
    (forall (?v - wall)
      (game-conserved
        (not
          (not
            (not
              (not
                (in ?v)
              )
            )
          )
        )
      )
    )
    (exists (?t - pyramid_block)
      (and
        (and
          (exists (?w - hexagonal_bin)
            (game-conserved
              (not
                (in ?w ?t)
              )
            )
          )
        )
        (forall (?w - chair)
          (forall (?i - game_object ?u - (either bridge_block yellow_cube_block) ?v - blue_cube_block ?k - hexagonal_bin ?o - hexagonal_bin)
            (and
              (and
                (game-optional
                  (adjacent_side ?u ?v)
                )
                (and
                  (game-conserved
                    (object_orientation ?i ?v)
                  )
                )
              )
            )
          )
        )
        (game-conserved
          (agent_holds ?t)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?t - dodgeball)
        (at-end
          (in_motion ?t ?t ?t)
        )
      )
    )
  )
)
(:terminal
  (< (count preference1:beachball:yellow_pyramid_block) 4 )
)
(:scoring
  (+ 5 )
)
)(define (game mcmc-78-0-1-7-9-12-13-15-24-26-28-30-33-39-48) (:domain many-objects-room-v1)
(:setup
  (and
    (and
      (exists (?b - book)
        (game-conserved
          (in ?b)
        )
      )
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?s - tall_cylindrical_block)
        (then
          (once (in_motion ?s) )
          (once (on ?s) )
          (hold (and (agent_holds agent ?s) (and (agent_holds ?s) (and (on ?s) (not (and (in_motion ?s) (in_motion ?s) ) ) ) (and (or (or (on bed agent) (agent_holds ?s) ) (agent_holds ?s rug) ) (adjacent upright) ) (in_motion ?s) ) ) )
        )
      )
    )
  )
)
(:terminal
  (or
    (>= 3 180 )
  )
)
(:scoring
  (- (* (* (* (count-unique-positions preference1:pink) (count-once-per-objects preference1:beachball) )
        (* (* 3 2 )
          0.7
        )
        (- 5 )
      )
      8
    )
  )
)
)(define (game mcmc-50-2-3-5-6-10-15-16-18-27-29-30-41) (:domain many-objects-room-v1)
(:setup
  (and
    (game-optional
      (in_motion pink)
    )
  )
)
(:constraints
  (and
    (preference preference1
      (exists (?d - curved_wooden_ramp)
        (then
          (once (agent_holds ?d ?d) )
          (once (agent_holds ?d ?d) )
          (once (not (and (not (adjacent ?d floor) ) (adjacent ?d) ) ) )
        )
      )
    )
    (preference preference2
      (exists (?u - curved_wooden_ramp)
        (then
          (once (< (distance agent door) 1) )
          (once (and (in ?u) (not (not (in ?u) ) ) ) )
          (once (touch agent) )
        )
      )
    )
    (forall (?j - dodgeball)
      (and
        (preference preference3
          (exists (?n - hexagonal_bin ?z - beachball ?e - dodgeball)
            (then
              (once (and (agent_holds ?e) (not (on ?n) ) ) )
              (once-measure (same_color ?j ?z) (distance_side ?j ?e) )
              (once (on ?n ?e) )
            )
          )
        )
        (preference preference4
          (exists (?s - (either dodgeball yellow_cube_block) ?d - hexagonal_bin ?b - hexagonal_bin)
            (then
              (once (in_motion rug agent) )
              (once (not (agent_holds ?b ?j) ) )
              (hold (not (not (>= (building_size 5 6) (distance_side )) ) ) )
            )
          )
        )
      )
    )
  )
)
(:terminal
  (>= (count-once preference2:dodgeball) (count-once-per-objects preference4:pyramid_block) )
)
(:scoring
  (count preference3:golfball)
)
)

from deep_sort_realtime.deepsort_tracker import DeepSort

# STRONGER TRACKER

tracker = DeepSort(

    max_age=60,
    n_init=3,
    max_iou_distance=0.7

)

def track_students(detections, frame):

    tracks = tracker.update_tracks(
        detections,
        frame=frame
    )

    results = []

    for track in tracks:

        # IGNORE UNCONFIRMED TRACKS
        if not track.is_confirmed():
            continue

        track_id = track.track_id

        ltrb = track.to_ltrb()

        x1, y1, x2, y2 = map(int, ltrb)

        results.append({

            "id": track_id,

            "box": (
                x1,
                y1,
                x2,
                y2
            )
        })

    return results
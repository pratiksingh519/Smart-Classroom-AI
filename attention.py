import math

def get_attention(nose_x,
                  nose_y,
                  left_eye_x,
                  right_eye_x,
                  frame_width,
                  frame_height):

    # CENTER RANGE
    center_left = frame_width * 0.35
    center_right = frame_width * 0.65

    # HEAD DOWN CHECK
    head_down_limit = frame_height * 0.60

    # LOOKING SIDE
    if nose_x < center_left:
        return "Looking Left"

    if nose_x > center_right:
        return "Looking Right"

    # HEAD DOWN
    if nose_y > head_down_limit:
        return "Head Down"

    # EYE DISTANCE
    eye_distance = abs(right_eye_x - left_eye_x)

    if eye_distance < 20:
        return "Distracted"

    return "Attentive"
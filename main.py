import cv2
from utils import (read_video,
                   save_video,
                   measure_distance,
                   convert_pix_to_dist,
                   draw_player_stats,
                   convert_meters_to_pix_dist)
import constants
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
from copy import deepcopy
import pandas as pd

def main():
    # Read Video
    input_video_path = 'input_videos/input_video.mp4'
    video_frames = read_video(input_video_path)
    
    # Detect Players and ball
    player_tracker = PlayerTracker(model_path='yolov8x')
    player_detections = player_tracker.detect_frames(video_frames, 
                                                     read_from_stub=True, 
                                                     stub_path='tracker_stubs/player_detections.pk1'
                                                     )
    ball_tracker = BallTracker(model_path='models/yolo5_last.pt')
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                 read_from_stub=True,
                                                 stub_path='tracker_stubs/ball_detections.pk1'
                                                 )
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)


    # Court Line Detection
    court_model_path = 'models/keypoints_model.pth'
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[1])

    # Choose Players 
    player_detections = player_tracker.choose_filter_players(court_keypoints, player_detections)

    # Mini Court
    mini_court = MiniCourt(video_frames[0])

    # Convert positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bbox_to_mini_court_coordinates(player_detections, ball_detections, court_keypoints)
    
    player_stats_data = [{
        'frame_num': 0,
        'player_1_num_of_shots':0,
        'player_1_total_shot_speed':0,
        'player_1_last_shot_speed':0,
        'player_1_total_player_speed':0,
        'player_1_last_player_speed':0,

        'player_2_num_of_shots':0,
        'player_2_total_shot_speed':0,
        'player_2_last_shot_speed':0,
        'player_2_total_player_speed':0,
        'player_2_last_player_speed':0,
    } ]

    # Detect ball shots
    ball_hit_frames = ball_tracker.get_ball_ball_hit_frames(ball_detections)
    
    for ball_hit_ind in range(len(ball_hit_frames)-1):
        start_frame = ball_hit_frames[ball_hit_ind]
        end_frame = ball_hit_frames[ball_hit_ind+1]
        ball_hit_time_in_seconds = (end_frame - start_frame) /30

        # Get Distance covered by the ball
        distance_covered_by_ball_in_pixels = measure_distance(ball_mini_court_detections[start_frame][1],
                                                              ball_mini_court_detections[end_frame][1])
        distance_covered_by_ball_in_meters = convert_pix_to_dist(distance_covered_by_ball_in_pixels,
                                                                 constants.DOUBLE_LINE_WIDTH,
                                                                 mini_court.get_width_of_mini_court()
                                                                 )

        # Speed of the ball shot in km/h
        speed_of_ball_hit = distance_covered_by_ball_in_meters / ball_hit_time_in_seconds * 3.6

        # Player who hit the ball
        player_positions = player_mini_court_detections[start_frame]
        player_hit_ball = min(player_positions.keys(), key = lambda player_id: measure_distance(player_positions[player_id],
                                                                                                ball_mini_court_detections[start_frame][1]))
        
        # Opponent Player Speed
        oppenent_player_id = 1 if player_hit_ball == 2 else 2
        distance_covered_by_player_in_pixels = measure_distance(player_mini_court_detections[start_frame][oppenent_player_id],
                                                                player_mini_court_detections[end_frame][oppenent_player_id])
        distance_covered_by_player_in_meters = convert_pix_to_dist(distance_covered_by_player_in_pixels,
                                                                   constants.DOUBLE_LINE_WIDTH,
                                                                   mini_court.get_width_of_mini_court()
                                                                   )
        speed_of_opponent = distance_covered_by_player_in_meters / ball_hit_time_in_seconds * 3.6

        current_player_stats = deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{player_hit_ball}_num_of_shots'] += 1
        current_player_stats[f'player_{player_hit_ball}_total_shot_speed'] += speed_of_ball_hit
        current_player_stats[f'player_{player_hit_ball}_last_shot_speed'] += speed_of_ball_hit

        current_player_stats[f'player_{oppenent_player_id}_total_player_speed'] += speed_of_opponent
        current_player_stats[f'player_{player_hit_ball}_last_player_speed'] = speed_of_opponent

        player_stats_data.append(current_player_stats)

    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()

    player_stats_data_df['player_1_avg_shot_speed'] = player_stats_data_df['player_1_total_shot_speed'] / player_stats_data_df['player_1_num_of_shots']
    player_stats_data_df['player_2_avg_shot_speed'] = player_stats_data_df['player_2_total_shot_speed'] / player_stats_data_df['player_2_num_of_shots']
    player_stats_data_df['player_1_avg_player_speed'] = player_stats_data_df['player_1_total_player_speed'] / player_stats_data_df['player_2_num_of_shots']
    player_stats_data_df['player_2_avg_player_speed'] = player_stats_data_df['player_2_total_player_speed'] / player_stats_data_df['player_1_num_of_shots']

    # Detect ball shots
    ball_hit_frames = ball_tracker.get_ball_ball_hit_frames(ball_detections)

    # Draw Output
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    # Draw Court Keypoints
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    # Draw Mini Court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_mini_court_detections)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, ball_mini_court_detections, color=(0,255,255))
    
    # Draw Player Stats
    output_video_frames = draw_player_stats(output_video_frames, player_stats_data_df)
    

    # Draw Frame number on top left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Save Video
    save_video(output_video_frames, 'output_videos/output_video.avi')
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
from utils import read_video,save_video
from tracker import Tracker
def main():
    # Read Video 
    video_frames = read_video('input_data/08fd33_4.mp4')

    # Initialize Tracker
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(video_frames,
               read_from_stub=True,
               stub_path='stubs/track_stubs.pkl')
    
    # Draw output
    ## Draw Object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # Save the video
    save_video(output_video_frames, 'output_data/output_video.avi')


if __name__ == '__main__':
    main()


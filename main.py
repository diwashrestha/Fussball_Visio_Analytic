from utils import read_video,save_video
def main():
    # Read Video 
    video_frames = read_video('input_data/08fd33_4.mp4')
    

    # Save the video
    save_video(video_frames, 'output_data/output_video.avi')


if __name__ == '__main__':
    main()


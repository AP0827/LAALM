from load_env import load_env_file
from pipeline import run_mvp

if __name__ == "__main__":
    load_env_file()
    
    # Using MATCHING audio/video pair from samples folder
    # File naming: same base name means same content
    # Ground truth: "set white with p four please"
    result = run_mvp(
        video_file="samples/video/lwwz9s.mpg",
        audio_file="samples/audio/lwwz9s.wav",
        avsr_model_path="auto_avsr/pretrained_models/vsr_trlrs2lrs3vox2avsp_base.pth"
    )
    
    print(f"\n📄 Final Transcript: {result['final_transcript']}")
    print(f"\nAudio: {result['deepgram']['transcript']}")
    print(f"Video: {result['avsr']['transcript']}")
    print(f"Sources: audio, video, groq")


from load_env import load_env_file
from pipeline import run_mvp

if __name__ == "__main__":
    load_env_file()
    
    # Using files from samples folder
    result = run_mvp(
        video_file="samples/video/bbaf2n.mpg",
        audio_file="samples/audio/bbaf2n.wav",
        lipnet_weights="LipNet/evaluation/models/unseen-weights178.h5"
    )
    
    print(f"\n📄 Final Transcript: {result['final_transcript']}")
    print(f"\nDeepGram: {result['deepgram']['transcript']}")
    print(f"LipNet: {result['lipnet']['transcript']}")
    print(f"Sources: deepgram, lipnet, groq")

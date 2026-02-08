from ml.predict import predict_video
from ml.heuristic import heuristic_video_score

video = "test_video.mp4"

ml_score = predict_video(video)
heur_score = heuristic_video_score(video)

print("\n===== MODEL COMPARISON =====")
print(f"ML Model Score      : {ml_score:.3f}")
print(f"Heuristic Score     : {heur_score:.3f}")

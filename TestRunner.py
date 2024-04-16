from metaseg import SegAutoMaskPredictor, SegManualMaskPredictor, SahiAutoSegmentation, sahi_sliced_predict

def automask_video_app(video_path, model_type, points_per_side, points_per_batch, min_area):
    SegAutoMaskPredictor().video_predict(
        source=video_path,
        model_type=model_type,  # vit_l, vit_h, vit_b
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        min_area=min_area,
        output_path="output.mp4",
    )
    return "output.mp4"

automask_video_app("./test.mp4","vit_l",1,1,0)
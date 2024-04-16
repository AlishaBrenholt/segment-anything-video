import PIL
from metaseg import SegAutoMaskPredictor
import mtl
import numpy as np

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

# Load the data
def load_data(target_height=400,error_range=10):
    data_array = []
    for folder in range(0,32):
        try:
            data_path = f"./gen_ai_fml/frame_{folder}/mask_0.png"
            image = PIL.Image.open(data_path)
            data = list(image.getdata(0))
            data = np.array(data)
            data = data.reshape(512,512)
            # Function to find the first and last row that contains a non-zero value
            def find_nonzero_rows(data):
                nonzero_rows = np.nonzero(data)[0]
                return nonzero_rows[0], nonzero_rows[-1]
            # Find the first and last row that contains a non-zero value
            first_row, last_row = find_nonzero_rows(data)
            # Compute the height of the object
            height = last_row - first_row + 1
            # Get difference between height and target height
            diff = abs(target_height - height)
            meaningful_diff = error_range-diff
            data_array.append((folder, meaningful_diff))
        except:
            data_array.append((folder,-1))
    return data_array





# Run model and get the output video, this is the normal output of the model
# Model.run_model(....) or whatever the function is, we are assuming it is output to test.mp4
pred_y = "./test.mp4"
# The predicted output of the model, this is the video generated, we are reading it from a file right now so it is the path
automask_video_app("./test.mp4","vit_l",1,1,0)

# Get the data
phi = mtl.parse('G height') # The formula to be evaluated, it means height is always true or it is between the error range and target height
data = load_data(target_height=400,error_range=10) # error range is max error, target height is how tall we want man to be
data_dict = {}
data_dict['height'] = data # the data should be a list of tuples where the first value is the frame and the second is if it is true or not

# Compute the logic loss
logic_weight = 0.1 # The weight of the logic loss
tl_score = phi(data_dict, dt=0.2) # Evaluate the formula
logic_loss = np.log(1+np.exp(-tl_score)) # Compute the logic loss
weighted_logic_loss = logic_weight * logic_loss # Weight the logic loss
print(weighted_logic_loss)
# Combine loss functions ?
task_loss = 0 # temporary value so you can see what we are thinking it should look like?
total_task_loss = task_loss + logic_weight * logic_loss
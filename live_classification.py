import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models

cam_IP = "rtsp://192.168.60.101:8554/"

model = models.efficientnet_v2_s(weights=None)
num_classes = 4
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load("models/model7/model7.pth"))
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


original_width = 1280
original_height = 720

new_width = 640
new_height = 368

original_roi = [525, 215, 150, 400]
scaled_roi = [
    int(original_roi[0] * (new_width / original_width)),  # x
    int(original_roi[1] * (new_height / original_height)),  # y
    int(original_roi[2] * (new_width / original_width)),  # width = 75
    int(original_roi[3] * (new_height / original_height))  # height = 204.4
]
B = 269
A = 110


mask = cv2.imread("imgs/bin_mask_opt.jpg")
mask_resized = cv2.resize(mask, (scaled_roi[2], scaled_roi[3]))
def preprocess_frame(frame):
    """
    Preprocess the frame to match training transformations
    """
    img_cropped = frame[A:A+204, B:B+75]
    masked_img = cv2.bitwise_and(img_cropped, mask_resized)
    masked_img_rgb = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)

    def resize_to_height(image, height):
            aspect_ratio = image.shape[1] / image.shape[0]
            new_width = int(aspect_ratio * height)
            return cv2.resize(image, (new_width, height))

    img_cropped_resized = resize_to_height(img_cropped, new_height)
    masked_img_resized = resize_to_height(masked_img, new_height)
    concat_crop_mask = cv2.hconcat([img_cropped_resized, masked_img_resized])

    pil_frame = transforms.functional.to_pil_image(masked_img_rgb)  # Convert to PIL Image
    transformed_frame = transform(pil_frame)
    return transformed_frame.unsqueeze(0), concat_crop_mask


print(cv2.getBuildInformation())
camera = cv2.VideoCapture(cam_IP, cv2.CAP_FFMPEG)
if not camera.isOpened():
    print("Error: Could not open video stream.")
    exit()
pred_prev = 2
counter = 0
is_anomaly = False
timer = 0
while True:
    ret, frame = camera.read()
    if not ret:
        print("Failed to capture frame")
        break
    
    input_tensor, concat_crop_mask = preprocess_frame(frame)  # Define this based on your model's requirements
    #prediction = model(input_tensor).detach().numpy()
    with torch.no_grad():
        prediction = model(input_tensor)
    
    predicted_class = prediction.argmax(dim=1).item()  # Adjust based on your model output
    if pred_prev == predicted_class and predicted_class != 2:
        counter += 1
    else:
        counter = 0
   
   
    if predicted_class == 0:
        Prediction = "Fallen After"
        color=(0, 0, 255)
    elif predicted_class == 3:
        Prediction = "No Lid"
        color=(0, 0, 255)
    elif predicted_class == 2:
        Prediction = "No Anomaly"
        color=(0, 255, 0)
    elif predicted_class == 1:
        Prediction = "Fallen Before"
        color=(0, 0, 255)
    predicted_text = f"Prediction: {Prediction}"
    cv2.putText(frame, predicted_text, (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=1, color=color, thickness=2)


    error_message = "ERROR!"
    if counter >= 5 or (timer <= 18 and timer != 0):
        cv2.putText(frame, error_message, (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=2, color=(0, 0, 255), thickness=3)
        timer += 1
    pred_prev = predicted_class
    if timer == 19:
        counter = 0
        timer = 0
    # Display the prediction and frame
    cv2.imshow("Live Feed", frame)
    cv2.imshow("Crop + Masked", concat_crop_mask)
    print("Prediction:", prediction)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
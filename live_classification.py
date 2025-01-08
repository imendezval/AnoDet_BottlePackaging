import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models

cam_IP = ""

model = models.efficientnet_v2_s(weights=None)
num_classes = 4
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load("checkpoint_epoch_5.pth"))
model.eval()



transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_frame(frame):
    """
    Preprocess the frame to match the training transformations.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    pil_frame = transforms.functional.to_pil_image(rgb_frame)  # Convert to PIL Image
    transformed_frame = transform(pil_frame)
    return transformed_frame.unsqueeze(0)


camera = cv2.VideoCapture(cam_IP)

while True:
    ret, frame = camera.read()
    if not ret:
        break
    
    input_tensor = preprocess_frame(frame)  # Define this based on your model's requirements
    #prediction = model(input_tensor).detach().numpy()
    with torch.no_grad():
        prediction = model(input_tensor)

    
    predicted_class = prediction.argmax(dim=1).item()  # Adjust based on your model output
    predicted_text = f"Prediction: {predicted_class}"
    cv2.putText(frame, predicted_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=1, color=(0, 255, 0), thickness=2)

    # Display the prediction and frame
    cv2.imshow("Live Feed", frame)
    print("Prediction:", prediction)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
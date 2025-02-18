import os
import cv2 as cv

def select_roi(image_path):
    img = cv.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None
    
    roi = cv.selectROI("Select ROI", img, fromCenter=False, showCrosshair=True)
    cv.destroyAllWindows()
    return roi

def crop_images(input_folder, output_folder, x, y, w, h):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, filename)
        img = cv.imread(image_path)
        if img is None:
            print(f"Failed to load image: {filename}")
            continue
        
        img_cropped = img[y:y+h, x:x+w]   
        img_path_output = os.path.join(output_folder, filename)
        cv.imwrite(img_path_output, img_cropped)
        
        print(f'Cropped and saved: {filename}')

if __name__ == "__main__":
    input_folder = "C:/Users/Arian/OneDrive - Hochschule Heilbronn/data_seminararbeit/wrong_lid/sampled"
    output_folder = "C:/Users/Arian/OneDrive - Hochschule Heilbronn/data_seminararbeit/wrong_lid/cropped"
    
    # Manually input the coordinates after using select_roi()
    #coordinates = select_roi("C:/Users/Arian/OneDrive - Hochschule Heilbronn/data_seminararbeit/wrong_lid/sampled/nd_0000.jpg")
    #print(coordinates)
    x, y, w, h = 524, 555, 75, 65  # Replace with actual values
    crop_images(input_folder, output_folder, x, y, w, h)

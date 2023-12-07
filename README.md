# Car_Counter-YOLOv8-
• Used the pretrained YOLOv8 model from ultralytics to detect cars(vehicles) in a video.
• Used SORT(Simple Online and Realtime Tracking) to generate ID's for detected vehicles and keep track of it. This was done to track the vehicles across frames and count it
• Used opencv to draw a line across the video, which became reference to count the cars. Each bounding box was marked by it's centre using cv2.circle. 
• Using equality conditions, it was checked whether the circle entered a region close to the line, which became the logic to count cars.

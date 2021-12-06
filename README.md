# Social-Distancing-Alert
I alert two people if

They are closer and 1/6 of input image width
Their y-coordinates are closer than 1/10 of input image height
so this is the main steps:

find people using OpenCV HOGDescriptor
apply non-maxima suppression to the bounding boxes
calculate the center of each bounding box of detected people
check the distance of people
draw red bounding boxes for who are close to each other

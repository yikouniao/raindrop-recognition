# raindrop-recognition
Recognize the raindrops in digital images and count the quantity.

### Process
1. Read an image.
2. Convert it into a grey image.
3. Use anisotropic diffusion to erase noise.
4. Calculate the sobel derivatives.
5. Convert it into a binary image with a threshold.
6. Opening operation to seperate interference from raindrops and weaken it.
7. Remove small connected components.
8. Closing operation to make the edges of raindrops more continuous.
9. Remove the long straight edges interference and clear the numbers and characters on the top left corner.
10. Closing operation again.
11. Fill the holes inside the raindrops.
12. Calculate fittest ellipses for contours. Count the quantity of contours. Draw the ellipses with labels and quantity on the image. Save the result image with the same name in result file folder.

### Files
The source image files are in `images\`, and the result files with corresponding names are saved in `results\`.
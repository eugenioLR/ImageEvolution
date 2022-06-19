# ImageEvolution

This program uses evolutionary algorithms to generate images using some function to optimize.
One of those functions can be how close is it to a reference image, or the amount of different colors in the image.

There's a folder with reference images, but you are encouraged to add more.

To change some of the parameters of the program, a configuration file is provided. It has the following parameters:
- img_width/img_height: number of pixels of the image being evolved.
- screen_width/screen_height: size of the window that shows the current image.
- reference_img: path to a reference image, preferably located in "./references".
- img_init: starting point of the algorithm, options are:
  - black
  - white
  - random  
- display: show the current progress.
- verbose: show aditional information on the console.
- method: method used in the evolutionary algorithm, options are:
  - pixels
  - triangles

I got the idea from https://github.com/antirez/shapeme and I have plans to implement this with triangles in the same way as well.

To test this program run the "genImg.py" script and you should see a black window, it should slowly turn into the image "Pepe.png"

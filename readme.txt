
Code written and compiled on pyCharm 2019.1.3 running on macOS Mojave 10.14.6

Command to run the program
- python3 main.py

For the code
The program reads the original coloured image and the mosaic image.
I have used masks for creating colour channels. The pattern is totally followed as described in the assignment. Then
there are 3 kernels used as filters on each channel to perform linear interpolation.
As observed, there are some artifacts produced when the root squared differences due to estimation of the missing pixels (Interpolation).
To be precise, most of them are on edges of pencils, sharp places on crayons, and high frequency parts in old well.
As the kernels used are averaging kernels and do not take account into cases of edges, as the pixel value for each
channel changes drastically as we travel through each pixel in said channel. This produces quite different result
on nearby averaging pixels, making the difference larger than normal.
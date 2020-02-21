# CAPTCHA Breaking 
### Inspired by 15 minute CAPTCHA breaking guy and the guy who wrote my textbook (Adrian Rosenbrock I think)

## CNN 
It's all on a series of blogposts I made on Tumblr (*insert link here*) but here's an abridged version:

### The Process
Basically, I wrote and stole a bunch of python scripts that does the following:
1. Hit the CAPTCHA a bunch of times and downloads their images
2. Splits up the CAPTCHA into individual letters
3. Identifies the letters of each CAPTCHA 

### The Architecture 
Input -> (Convolutional layer -> Max Pooling Layer) -> (Convolutional layer -> Max Pooling Layer) -> Fully Connected Layer -> Output layer -> Output 

And so we have an incredibly basic ConvNet since my net is bad and we're basically just trying to solve MNIST anyway, which this network will do with ~99.8% accuracy. 

